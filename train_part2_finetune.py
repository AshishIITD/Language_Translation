"""
Part II Fine-tuning Script: Seq2Seq Translation with Pretrained BERT + GPT.

Two-phase fine-tuning strategy:
  Phase 1 (--phase 1): Only cross-attention + LM head trainable. Fast, stable.
  Phase 2 (--phase 2): All parameters unfrozen with lower LR. End-to-end.

Usage:
    # Phase 1 (train cross-attention only) — full scale on MPS:
    python train_part2_finetune.py --phase 1 --scale full --batch_size 32 --grad_accum 2 \\
        --bert_ckpt outputs/part2/bert_full/checkpoint_best.pt \\
        --gpt_ckpt  outputs/part2/gpt_full/checkpoint_best.pt

    # Phase 2 (full fine-tune, start from Phase 1 checkpoint):
    python train_part2_finetune.py --phase 2 --scale full --batch_size 16 --grad_accum 4 --lr 5e-5 \\
        --bert_ckpt outputs/part2/bert_full/checkpoint_best.pt \\
        --gpt_ckpt  outputs/part2/gpt_full/checkpoint_best.pt \\
        --resume    outputs/part2/finetune_full_hi2mr_phase1/checkpoint_best.pt
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import (
    load_parallel_corpus, split_corpus, SPTokenizer,
    TranslationDataset, make_dataloader, print_corpus_stats
)
from part2.models import (
    Seq2SeqTransformer, BERTConfig, GPTConfig,
    BERT_FULL, GPT_FULL, BERT_SMALL, GPT_SMALL
)
from evaluation.metrics import evaluate_model
from utils.training_utils import (
    LabelSmoothingLoss, build_optimizer, build_scheduler,
    save_checkpoint, load_checkpoint, MetricTracker
)


def parse_args():
    p = argparse.ArgumentParser(description="Part II: Seq2Seq Fine-tuning")

    p.add_argument("--phase",       type=int,   default=1, choices=[1, 2])
    p.add_argument("--scale",       type=str,   default="full", choices=["full", "small"])
    p.add_argument("--direction",   type=str,   default="hi2mr", choices=["hi2mr", "mr2hi"])

    p.add_argument("--bert_ckpt",   type=str,   required=True)
    p.add_argument("--gpt_ckpt",    type=str,   required=True)

    p.add_argument("--data_dir",    type=str,   default="data/corpus")
    p.add_argument("--output_dir",  type=str,   default="outputs/part2")
    p.add_argument("--spm_model",   type=str,   default="outputs/part1/spm_8000.model")
    p.add_argument("--max_samples", type=int,   default=None)
    p.add_argument("--max_len",     type=int,   default=128)

    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--weight_decay",type=float, default=1e-2)
    p.add_argument("--clip_grad",   type=float, default=1.0)
    p.add_argument("--warmup_ratio",type=float, default=0.1)
    p.add_argument("--label_smooth",type=float, default=0.1)
    p.add_argument("--grad_accum",  type=int,   default=2)

    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--workers",     type=int,   default=2)
    p.add_argument("--resume",      type=str,   default=None)
    p.add_argument("--eval_beam",   action="store_true")

    # NOTE: fp16/AMP is NOT supported on MPS (Apple Silicon) and is skipped.

    return p.parse_args()


def set_seed(seed):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    # ── Device detection (MPS > CUDA > CPU) ──────────────────────────────────
    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    # ── Config ────────────────────────────────────────────────────────────────
    bert_cfg = BERT_FULL if args.scale == "full" else BERT_SMALL
    gpt_cfg  = GPT_FULL  if args.scale == "full" else GPT_SMALL

    run_name = f"finetune_{args.scale}_{args.direction}_phase{args.phase}"
    out_dir  = os.path.join(args.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    # ── Save config for reproducibility (Prompt 10) ──────────────────────────
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Config saved to {os.path.join(out_dir, 'config.json')}")

    # ── Data ──────────────────────────────────────────────────────────────────
    src_lang = "hi" if args.direction == "hi2mr" else "mr"
    tgt_lang = "mr" if args.direction == "hi2mr" else "hi"

    all_src, all_tgt = load_parallel_corpus(
        args.data_dir, src_lang=src_lang, tgt_lang=tgt_lang,
        max_samples=args.max_samples, seed=args.seed
    )
    (train_src, train_tgt), (val_src, val_tgt), (test_src, test_tgt) = \
        split_corpus(all_src, all_tgt, seed=args.seed)

    tokenizer = SPTokenizer(args.spm_model)
    bert_cfg.vocab_size = tokenizer.vocab_size
    gpt_cfg.vocab_size  = tokenizer.vocab_size

    train_ds = TranslationDataset(train_src, train_tgt, tokenizer, args.max_len, args.max_len)
    val_ds   = TranslationDataset(val_src,   val_tgt,   tokenizer, args.max_len, args.max_len)
    test_ds  = TranslationDataset(test_src,  test_tgt,  tokenizer, args.max_len, args.max_len)

    train_loader = make_dataloader(train_ds, args.batch_size, shuffle=True,  num_workers=args.workers)
    val_loader   = make_dataloader(val_ds,   args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader  = make_dataloader(test_ds,  args.batch_size, shuffle=False, num_workers=args.workers)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = Seq2SeqTransformer(
        bert_config=bert_cfg,
        gpt_config=gpt_cfg,
        bert_pretrained_path=args.bert_ckpt,
        gpt_pretrained_path=args.gpt_ckpt,
    ).to(device)

    if args.phase == 1:
        model.freeze_for_phase1()
    else:
        model.unfreeze_all()

    # ── Optimizer + Scheduler ─────────────────────────────────────────────────
    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = len(train_loader) // args.grad_accum
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = build_scheduler(optimizer, warmup_steps, total_steps)

    V = tokenizer.vocab_size
    criterion = LabelSmoothingLoss(V, tokenizer.PAD_ID, smoothing=args.label_smooth)

    # NOTE: fp16/AMP is not supported on MPS and is skipped.

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = load_checkpoint(model, args.resume, optimizer, scheduler, str(device))
        start_epoch = ckpt["epoch"] + 1
        if args.phase == 2:
            model.unfreeze_all()  # ensure unfrozen after loading

    tracker = MetricTracker(os.path.join(out_dir, "plots"))
    tracker.load()
    best_val_bleu = tracker.best("val/bleu")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        grad_norms = []  # Prompt 9: gradient norm tracking

        for step, (src, tgt, src_lengths) in tqdm(
            enumerate(train_loader), total=len(train_loader),
            desc=f"Epoch {epoch}", leave=False
        ):
            src = src.to(device)
            tgt = tgt.to(device)

            src_mask = (src != tokenizer.PAD_ID).long()
            tgt_in   = tgt[:, :-1]
            tgt_out  = tgt[:, 1:]
            tgt_mask = (tgt_in != tokenizer.PAD_ID).long()

            out = model(
                src_ids=src, tgt_ids=tgt_in,
                src_mask=src_mask, tgt_mask=tgt_mask,
            )
            logits = out["logits"]   # (B, T-1, V)
            B_, T, V_ = logits.shape
            loss = criterion(logits.reshape(B_ * T, V_), tgt_out.reshape(B_ * T))
            loss = loss / args.grad_accum
            loss.backward()

            total_loss += loss.item() * args.grad_accum

            if (step + 1) % args.grad_accum == 0:
                # Gradient clipping + norm logging (Prompt 9)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                grad_norms.append(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if (step // args.grad_accum) % 100 == 0:
                    lr_now = scheduler.get_last_lr()[0]
                    mean_gnorm = sum(grad_norms[-100:]) / max(len(grad_norms[-100:]), 1)
                    print(f"  E{epoch:02d} S{step // args.grad_accum:04d} "
                          f"loss={total_loss / max(step+1, 1):.4f}  lr={lr_now:.2e}  "
                          f"grad_norm={mean_gnorm:.4f}")

        avg_train_loss = total_loss / len(train_loader)

        # Log mean gradient norm for this epoch
        if grad_norms:
            epoch_mean_gnorm = sum(grad_norms) / len(grad_norms)
            print(f"  Epoch {epoch} mean gradient norm: {epoch_mean_gnorm:.4f}")

        # ── Eval ──────────────────────────────────────────────────────────────
        # Use a wrapper to handle the Seq2SeqTransformer interface
        val_loss, val_bleu, val_chrf = _evaluate_seq2seq(
            model, val_loader, tokenizer, device, criterion, args.eval_beam
        )
        train_loss_eval, train_bleu, train_chrf = _evaluate_seq2seq(
            model,
            make_dataloader(
                TranslationDataset(train_src[:300], train_tgt[:300], tokenizer,
                                   args.max_len, args.max_len),
                args.batch_size, shuffle=False
            ),
            tokenizer, device, criterion,
        )

        print(f"\nEpoch {epoch:02d} | train={avg_train_loss:.4f} val={val_loss:.4f} | "
              f"val_BLEU={val_bleu:.2f}  val_CHRF++={val_chrf:.2f}")

        tracker.update("train", {"loss": avg_train_loss, "bleu": train_bleu, "chrf_pp": train_chrf}, epoch)
        tracker.update("val",   {"loss": val_loss,       "bleu": val_bleu,   "chrf_pp": val_chrf},   epoch)
        tracker.save()

        is_best = val_bleu > best_val_bleu
        if is_best:
            best_val_bleu = val_bleu
            print(f"  ★ New best val BLEU: {val_bleu:.2f}")

        save_checkpoint(model, optimizer, scheduler, epoch, 0,
                        {"val_bleu": val_bleu, "val_loss": val_loss},
                        out_dir, "checkpoint_latest.pt")
        # Save numbered checkpoints for checkpoint averaging
        save_checkpoint(model, optimizer, scheduler, epoch, 0,
                        {"val_bleu": val_bleu, "val_loss": val_loss},
                        out_dir, f"checkpoint_epoch{epoch:02d}.pt")
        if is_best:
            save_checkpoint(model, optimizer, scheduler, epoch, 0,
                            {"val_bleu": val_bleu, "val_loss": val_loss},
                            out_dir, "checkpoint_best.pt")

    tracker.plot_all(prefix=f"{run_name}_")

    # ── Test ──────────────────────────────────────────────────────────────────
    best_path = os.path.join(out_dir, "checkpoint_best.pt")
    if os.path.exists(best_path):
        load_checkpoint(model, best_path, device=str(device))

    test_loss, test_bleu, test_chrf = _evaluate_seq2seq(
        model, test_loader, tokenizer, device, criterion, use_beam=True
    )
    print(f"\n{'='*60}")
    print(f"TEST RESULTS [{run_name}]")
    print(f"  BLEU-100  : {test_bleu:.2f}")
    print(f"  CHRF++-100: {test_chrf:.2f}")
    print(f"{'='*60}")

    with open(os.path.join(out_dir, "test_results.json"), "w") as f:
        json.dump({"run": run_name, "bleu": test_bleu, "chrf_pp": test_chrf,
                   "loss": test_loss, "args": vars(args)}, f, indent=2)


# ─── Eval helper for Seq2SeqTransformer ──────────────────────────────────────

@torch.no_grad()
def _evaluate_seq2seq(model, dataloader, tokenizer, device, criterion, use_beam=False):
    from evaluation.metrics import compute_metrics
    model.eval()

    hypotheses, references = [], []
    total_loss, total_tokens = 0.0, 0

    for src, tgt, src_lengths in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        src_mask = (src != tokenizer.PAD_ID).long()
        tgt_in   = tgt[:, :-1]
        tgt_out  = tgt[:, 1:]
        tgt_mask = (tgt_in != tokenizer.PAD_ID).long()

        out = model(src_ids=src, tgt_ids=tgt_in,
                    src_mask=src_mask, tgt_mask=tgt_mask)
        logits = out["logits"]
        B, T, V = logits.shape
        loss = criterion(logits.reshape(B*T, V), tgt_out.reshape(B*T))
        n_tok = (tgt_out != tokenizer.PAD_ID).sum().item()
        total_loss += loss.item() * n_tok
        total_tokens += n_tok

        # Generate — use beam search if enabled
        if use_beam:
            for i in range(B):
                hyp_ids = model.beam_search(
                    src[i:i+1], src_mask[i:i+1],
                    tokenizer.BOS_ID, tokenizer.EOS_ID,
                    beam_size=4
                )
                hypotheses.append(tokenizer.decode(hyp_ids))
        else:
            pred_ids = model.greedy_decode(
                src, src_mask, tokenizer.BOS_ID, tokenizer.EOS_ID
            )
            for i in range(B):
                hypotheses.append(tokenizer.decode(pred_ids[i].tolist()))

        for i in range(B):
            references.append(tokenizer.decode(tgt[i, 1:].tolist()))

    avg_loss = total_loss / max(total_tokens, 1)
    metrics = compute_metrics(hypotheses, references)
    return avg_loss, metrics["bleu"], metrics["chrf_pp"]


if __name__ == "__main__":
    main()
