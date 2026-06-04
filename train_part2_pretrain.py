"""
Part II Pretraining Script.

Pretrain both BERT (MLM) and GPT (CLM) from scratch on the Hindi-Marathi corpus.
Both models use: RoPE + GQA + RMSNorm (as required).

Usage:
    # Pretrain BERT (MLM) — full scale on MPS:
    python train_part2_pretrain.py --model bert --scale full --batch_size 32 --grad_accum 2 --epochs 20

    # Pretrain GPT (CLM) — full scale on MPS:
    python train_part2_pretrain.py --model gpt --scale full --batch_size 32 --grad_accum 2 --epochs 20

    # Scaled-down versions (for limited compute):
    python train_part2_pretrain.py --model bert --scale small
    python train_part2_pretrain.py --model gpt  --scale small
"""

import os
import sys
import json
import math
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import load_parallel_corpus, split_corpus, SPTokenizer
from part2.models import (
    HindiMarathiBERT, HindiMarathiGPT, BERTConfig, GPTConfig,
    BERT_FULL, GPT_FULL, BERT_SMALL, GPT_SMALL
)
from utils.training_utils import (
    build_optimizer, build_scheduler,
    save_checkpoint, load_checkpoint, MetricTracker
)

from torch.utils.data import Dataset, DataLoader


# ─── Pretraining Datasets ─────────────────────────────────────────────────────

class MLMDataset(Dataset):
    """
    Monolingual dataset for Masked Language Modeling.
    Combines both Hindi and Marathi sentences (shared vocabulary).
    """

    def __init__(self, sentences, tokenizer: SPTokenizer, max_len: int = 512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []
        for sent in sentences:
            ids = tokenizer.encode(sent, add_bos=False, add_eos=False)
            if len(ids) >= 4:  # skip very short sequences
                self.data.append(ids[:max_len])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class CLMDataset(Dataset):
    """
    Causal Language Modeling dataset.
    Each item is a token sequence; the model predicts token[i+1] from token[i].
    """

    def __init__(self, sentences, tokenizer: SPTokenizer, max_len: int = 1024):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []
        for sent in sentences:
            ids = tokenizer.encode(sent, add_bos=True, add_eos=True)
            if len(ids) >= 4:
                self.data.append(ids[:max_len])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def mlm_collate_fn(batch, pad_id=0, mask_id=6, mask_prob=0.15):
    """Pad batch and create MLM masks."""
    max_len = max(len(x) for x in batch)
    B = len(batch)

    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    attn_mask = torch.zeros(B, max_len, dtype=torch.long)

    for i, seq in enumerate(batch):
        input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        attn_mask[i, :len(seq)] = 1

    return input_ids, attn_mask


def clm_collate_fn(batch, pad_id=0):
    """Pad batch for CLM."""
    max_len = max(len(x) for x in batch)
    B = len(batch)

    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    attn_mask = torch.zeros(B, max_len, dtype=torch.long)

    for i, seq in enumerate(batch):
        input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        attn_mask[i, :len(seq)] = 1

    return input_ids, attn_mask



# ─── Args ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Part II Pretraining")

    parser.add_argument("--model",       type=str, default="bert", choices=["bert", "gpt"])
    parser.add_argument("--scale",       type=str, default="full", choices=["full", "small"])

    parser.add_argument("--data_dir",    type=str, default="data/corpus")
    parser.add_argument("--output_dir",  type=str, default="outputs/part2")
    parser.add_argument("--spm_model",   type=str, default="outputs/part1/spm_8000.model",
                        help="Reuse the SentencePiece model trained in Part I")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_len",     type=int, default=None,
                        help="Override max sequence length")

    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--weight_decay",type=float, default=1e-2)
    parser.add_argument("--clip_grad",   type=float, default=1.0)
    parser.add_argument("--warmup_ratio",type=float, default=0.05)
    parser.add_argument("--grad_accum",  type=int,   default=2,
                        help="Gradient accumulation steps — simulates larger batch")

    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--workers",     type=int,   default=2)
    parser.add_argument("--resume",      type=str,   default=None)

    # NOTE: fp16/AMP is NOT supported on MPS (Apple Silicon) and is skipped.

    return parser.parse_args()


def set_seed(seed):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    # ── Device detection (MPS > CUDA > CPU) ──────────────────────────────────
    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Select config ─────────────────────────────────────────────────────────
    if args.model == "bert":
        config = BERT_FULL if args.scale == "full" else BERT_SMALL
    else:
        config = GPT_FULL if args.scale == "full" else GPT_SMALL

    if args.max_len:
        config.max_seq_len = args.max_len

    run_name = f"{args.model}_{args.scale}"
    out_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    # ── Save config for reproducibility (Prompt 10) ──────────────────────────
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Config saved to {os.path.join(out_dir, 'config.json')}")

    # ── Load corpus ───────────────────────────────────────────────────────────
    # For pretraining, we use BOTH languages as monolingual data (no alignment needed)
    hi_src, hi_tgt = load_parallel_corpus(
        args.data_dir, src_lang="hi", tgt_lang="mr",
        max_samples=args.max_samples, seed=args.seed
    )
    # Combine all sentences (both languages) for pretraining
    all_sentences = hi_src + hi_tgt
    import random
    rng = random.Random(args.seed)
    rng.shuffle(all_sentences)
    print(f"Total pretraining sentences: {len(all_sentences):,}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = SPTokenizer(args.spm_model)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Update vocab size in config to match tokenizer
    config.vocab_size = tokenizer.vocab_size

    # ── Dataset ───────────────────────────────────────────────────────────────
    split = int(len(all_sentences) * 0.95)
    train_sents = all_sentences[:split]
    val_sents   = all_sentences[split:]

    if args.model == "bert":
        max_len = config.max_seq_len
        train_ds = MLMDataset(train_sents, tokenizer, max_len)
        val_ds   = MLMDataset(val_sents,   tokenizer, max_len)

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            collate_fn=mlm_collate_fn, num_workers=args.workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=mlm_collate_fn, num_workers=args.workers
        )
    else:
        max_len = config.max_seq_len
        train_ds = CLMDataset(train_sents, tokenizer, max_len)
        val_ds   = CLMDataset(val_sents,   tokenizer, max_len)

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            collate_fn=clm_collate_fn, num_workers=args.workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=clm_collate_fn, num_workers=args.workers
        )

    # ── Model ─────────────────────────────────────────────────────────────────
    if args.model == "bert":
        model = HindiMarathiBERT(config).to(device)
    else:
        model = HindiMarathiGPT(config, use_cross_attention=False).to(device)

    n_params = model.count_parameters()
    print(f"\nModel: {args.model.upper()} ({args.scale})")
    print(f"Parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    effective_batch = args.batch_size * args.grad_accum
    print(f"Effective batch size: {effective_batch}")

    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = len(train_loader) // args.grad_accum
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = build_scheduler(optimizer, warmup_steps, total_steps)

    # NOTE: fp16/AMP is not supported on MPS and is skipped.

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = load_checkpoint(model, args.resume, optimizer, scheduler, str(device))
        start_epoch = ckpt["epoch"] + 1

    # ── Tracker ───────────────────────────────────────────────────────────────
    tracker = MetricTracker(os.path.join(out_dir, "plots"))

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        grad_norms = []  # Prompt 9: gradient norm tracking

        for step, batch in tqdm(
            enumerate(train_loader), total=len(train_loader),
            desc=f"Epoch {epoch}", leave=False
        ):
            if args.model == "bert":
                input_ids, attn_mask = batch
                input_ids = input_ids.to(device)
                attn_mask = attn_mask.to(device)

                # Create MLM masks
                masked_ids, labels = model.create_mlm_batch(input_ids)

                out = model(masked_ids, attn_mask, labels)
                loss = out["loss"] / args.grad_accum
                loss.backward()

            else:  # GPT CLM
                input_ids, attn_mask = batch
                input_ids = input_ids.to(device)
                attn_mask = attn_mask.to(device)

                out = model(input_ids=input_ids, attention_mask=attn_mask, labels=input_ids)
                loss = out["loss"] / args.grad_accum
                loss.backward()

            total_loss += loss.item() * args.grad_accum

            if (step + 1) % args.grad_accum == 0:
                # Gradient clipping + norm logging (Prompt 9)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                grad_norms.append(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if (step // args.grad_accum) % 50 == 0:
                    lr_now = scheduler.get_last_lr()[0]
                    mean_gnorm = sum(grad_norms[-50:]) / max(len(grad_norms[-50:]), 1)
                    print(f"  E{epoch:02d} S{step // args.grad_accum:04d} "
                          f"loss={total_loss / max(step+1, 1):.4f}  lr={lr_now:.2e}  "
                          f"grad_norm={mean_gnorm:.4f}")

        avg_train_loss = total_loss / len(train_loader)
        train_ppl = math.exp(min(avg_train_loss, 20))  # clamp to avoid overflow

        # Log mean gradient norm for this epoch
        if grad_norms:
            epoch_mean_gnorm = sum(grad_norms) / len(grad_norms)
            print(f"  Epoch {epoch} mean gradient norm: {epoch_mean_gnorm:.4f}")

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if args.model == "bert":
                    input_ids, attn_mask = batch
                    input_ids = input_ids.to(device)
                    attn_mask = attn_mask.to(device)
                    masked_ids, labels = model.create_mlm_batch(input_ids)
                    out = model(masked_ids, attn_mask, labels)
                else:
                    input_ids, attn_mask = batch
                    input_ids = input_ids.to(device)
                    attn_mask = attn_mask.to(device)
                    out = model(input_ids=input_ids, attention_mask=attn_mask, labels=input_ids)
                val_loss_total += out["loss"].item()

        avg_val_loss = val_loss_total / len(val_loader)
        # Perplexity (Prompt 7): clamp to avoid overflow
        val_ppl = math.exp(min(avg_val_loss, 20))

        print(f"\nEpoch {epoch:02d} | train_loss={avg_train_loss:.4f} | "
              f"val_loss={avg_val_loss:.4f} | train_ppl={train_ppl:.2f} | val_ppl={val_ppl:.2f}")

        # Track metrics including perplexity (Prompt 7)
        tracker.update("train", {"loss": avg_train_loss, "ppl": train_ppl}, epoch)
        tracker.update("val",   {"loss": avg_val_loss,   "ppl": val_ppl},   epoch)
        tracker.save()

        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss

        save_checkpoint(
            model, optimizer, scheduler, epoch, 0,
            {"val_loss": avg_val_loss, "val_ppl": val_ppl},
            save_dir=out_dir, filename="checkpoint_latest.pt"
        )
        # Save numbered checkpoints for checkpoint averaging
        save_checkpoint(
            model, optimizer, scheduler, epoch, 0,
            {"val_loss": avg_val_loss, "val_ppl": val_ppl},
            save_dir=out_dir, filename=f"checkpoint_epoch{epoch:02d}.pt"
        )
        if is_best:
            save_checkpoint(
                model, optimizer, scheduler, epoch, 0,
                {"val_loss": avg_val_loss, "val_ppl": val_ppl},
                save_dir=out_dir, filename="checkpoint_best.pt"
            )
            print(f"  ★ New best val loss: {avg_val_loss:.4f}")

    # ── Final plots (including perplexity — Prompt 7) ─────────────────────────
    tracker.plot_loss()
    tracker.plot_ppl()
    print(f"\nPretraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved in: {out_dir}")


if __name__ == "__main__":
    main()
