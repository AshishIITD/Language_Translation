"""
Part I Training Script: LSTM Seq2Seq with Bahdanau Attention.

Usage:
    # Train with random embeddings (baseline):
    python train_part1.py --embedding_type random --direction hi2mr --batch_size 64 --epochs 30

    # Train with BERT embeddings:
    python train_part1.py --embedding_type bert --direction hi2mr --batch_size 32 --freeze_emb --epochs 30

    # Both directions, BERT:
    python train_part1.py --embedding_type bert --direction mr2hi

    # Resume from checkpoint:
    python train_part1.py --embedding_type random --resume outputs/part1/random/checkpoint_best.pt

    # Curriculum learning (enabled by default):
    python train_part1.py --embedding_type random --direction hi2mr --curriculum
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import (
    load_parallel_corpus, split_corpus, SPTokenizer,
    TranslationDataset, make_dataloader, print_corpus_stats
)
from part1.model import Seq2SeqLSTM
from part1.bert_embeddings import (
    prepare_bert_embeddings_for_model, get_sp_vocab_list
)
from evaluation.metrics import evaluate_model
from utils.training_utils import (
    LabelSmoothingLoss, build_optimizer, build_scheduler,
    save_checkpoint, load_checkpoint, MetricTracker
)


# ─── Config ───────────────────────────────────────────────────────────────────

HINDI_BERT  = "l3cube-pune/hindi-bert-v2"
MARATHI_BERT = "l3cube-pune/marathi-bert-v2"


def parse_args():
    parser = argparse.ArgumentParser(description="Part I: LSTM Seq2Seq NMT")

    # Data
    parser.add_argument("--data_dir",    type=str,   default="data/corpus")
    parser.add_argument("--output_dir",  type=str,   default="outputs/part1")
    parser.add_argument("--max_samples", type=int,   default=None,
                        help="Subsample corpus. None = use all.")
    parser.add_argument("--vocab_size",  type=int,   default=8000)
    parser.add_argument("--max_len",     type=int,   default=128)

    # Direction
    parser.add_argument("--direction",   type=str,   default="hi2mr",
                        choices=["hi2mr", "mr2hi"])

    # Embedding
    parser.add_argument("--embedding_type", type=str, default="random",
                        choices=["random", "bert"])
    parser.add_argument("--freeze_emb",  action="store_true",
                        help="Freeze embeddings (relevant for BERT init)")
    parser.add_argument("--embed_dim",   type=int,   default=256)

    # Model
    parser.add_argument("--hidden_size", type=int,   default=512)
    parser.add_argument("--attn_dim",    type=int,   default=256)
    parser.add_argument("--num_layers",  type=int,   default=2)
    parser.add_argument("--dropout",     type=float, default=0.3)

    # Training
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--weight_decay",type=float, default=1e-2)
    parser.add_argument("--clip_grad",   type=float, default=1.0)
    parser.add_argument("--warmup_ratio",type=float, default=0.1)
    parser.add_argument("--label_smooth",type=float, default=0.1)
    parser.add_argument("--tf_ratio_start", type=float, default=1.0,
                        help="Teacher forcing ratio at epoch 0")
    parser.add_argument("--tf_ratio_end",   type=float, default=0.5,
                        help="Teacher forcing ratio at final epoch (linear decay)")

    # BERT unfreeze schedule
    parser.add_argument("--unfreeze_epoch", type=int, default=5,
                        help="Epoch at which to unfreeze BERT embeddings")

    # Curriculum learning
    parser.add_argument("--curriculum", action="store_true", default=True,
                        help="Enable length-based curriculum learning (default: True)")
    parser.add_argument("--no_curriculum", action="store_true",
                        help="Disable curriculum learning")
    parser.add_argument("--curriculum_max_len", type=int, default=40,
                        help="Max token length for curriculum easy examples")

    # Misc
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--workers",     type=int,   default=2)
    parser.add_argument("--resume",      type=str,   default=None)
    parser.add_argument("--eval_beam",   action="store_true",
                        help="Use beam search for eval (slower, more accurate)")

    # NOTE: fp16/AMP is NOT supported on MPS (Apple Silicon) and is skipped.
    # If running on CUDA, you can manually enable AMP in the code.

    return parser.parse_args()


def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Handle curriculum flag
    if args.no_curriculum:
        args.curriculum = False

    set_seed(args.seed)

    # ── Device detection (MPS > CUDA > CPU) ──────────────────────────────────
    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Output directory ──────────────────────────────────────────────────────
    run_name = f"{args.direction}_{args.embedding_type}"
    if args.freeze_emb and args.embedding_type == "bert":
        run_name += "_frozen"
    out_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output dir: {out_dir}")

    # ── Save config for reproducibility (Prompt 10) ──────────────────────────
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Config saved to {os.path.join(out_dir, 'config.json')}")

    # ── Load and split corpus ─────────────────────────────────────────────────
    src_lang = "hi" if args.direction == "hi2mr" else "mr"
    tgt_lang = "mr" if args.direction == "hi2mr" else "hi"

    all_src, all_tgt = load_parallel_corpus(
        args.data_dir, src_lang=src_lang, tgt_lang=tgt_lang,
        max_samples=args.max_samples, seed=args.seed
    )
    print_corpus_stats(all_src, all_tgt, name=f"{src_lang}→{tgt_lang}")

    (train_src, train_tgt), (val_src, val_tgt), (test_src, test_tgt) = \
        split_corpus(all_src, all_tgt, seed=args.seed)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    sp_model_path = os.path.join(args.output_dir, f"spm_{args.vocab_size}.model")
    if not os.path.exists(sp_model_path):
        print("Training SentencePiece model...")
        # Use both src and tgt for shared vocabulary (justified: same Devanagari script)
        all_sentences = train_src + train_tgt
        tokenizer = SPTokenizer.train(
            sentences=all_sentences,
            model_prefix=os.path.join(args.output_dir, f"spm_{args.vocab_size}"),
            vocab_size=args.vocab_size,
        )
    else:
        tokenizer = SPTokenizer(sp_model_path)
        print(f"Loaded tokenizer: {sp_model_path}  vocab={tokenizer.vocab_size}")

    V = tokenizer.vocab_size

    # ── Datasets & Loaders ────────────────────────────────────────────────────
    train_ds = TranslationDataset(train_src, train_tgt, tokenizer, args.max_len, args.max_len)
    val_ds   = TranslationDataset(val_src,   val_tgt,   tokenizer, args.max_len, args.max_len)
    test_ds  = TranslationDataset(test_src,  test_tgt,  tokenizer, args.max_len, args.max_len)

    train_loader = make_dataloader(train_ds, args.batch_size, shuffle=True,  num_workers=args.workers)
    val_loader   = make_dataloader(val_ds,   args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader  = make_dataloader(test_ds,  args.batch_size, shuffle=False, num_workers=args.workers)

    # ── Curriculum learning: create a "short pairs" dataloader (Prompt 5) ────
    curriculum_loader = None
    if args.curriculum:
        # Filter to pairs where max(src_len, tgt_len) <= curriculum_max_len tokens
        short_src, short_tgt = [], []
        for s, t in zip(train_src, train_tgt):
            if max(len(s.split()), len(t.split())) <= args.curriculum_max_len:
                short_src.append(s)
                short_tgt.append(t)
        if len(short_src) > 100:  # only use curriculum if enough short pairs
            short_ds = TranslationDataset(short_src, short_tgt, tokenizer, args.max_len, args.max_len)
            curriculum_loader = make_dataloader(short_ds, args.batch_size, shuffle=True, num_workers=args.workers)
            print(f"Curriculum learning: {len(short_src):,} short pairs "
                  f"(≤{args.curriculum_max_len} tokens) for first 30% of epochs")
        else:
            print("Not enough short pairs for curriculum learning, using full dataset")
            args.curriculum = False

    # ── Embeddings ────────────────────────────────────────────────────────────
    src_emb, tgt_emb = None, None

    if args.embedding_type == "bert":
        src_bert = HINDI_BERT  if src_lang == "hi" else MARATHI_BERT
        tgt_bert = MARATHI_BERT if tgt_lang == "mr" else HINDI_BERT

        sp_vocab = get_sp_vocab_list(sp_model_path)

        print("Preparing source BERT embeddings...")
        src_emb = prepare_bert_embeddings_for_model(
            sp_vocab, src_bert, args.embed_dim, device=str(device),
            save_path=os.path.join(args.output_dir, f"bert_emb_{src_lang}_{args.embed_dim}.pt")
        )

        print("Preparing target BERT embeddings...")
        tgt_emb = prepare_bert_embeddings_for_model(
            sp_vocab, tgt_bert, args.embed_dim, device=str(device),
            save_path=os.path.join(args.output_dir, f"bert_emb_{tgt_lang}_{args.embed_dim}.pt")
        )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = Seq2SeqLSTM(
        src_vocab_size=V,
        tgt_vocab_size=V,
        embed_dim=args.embed_dim,
        hidden_size=args.hidden_size,
        attn_dim=args.attn_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pad_idx=tokenizer.PAD_ID,
        src_pretrained_emb=src_emb,
        tgt_pretrained_emb=tgt_emb,
        freeze_embeddings=args.freeze_emb,
    ).to(device)

    n_params = model.count_parameters()
    print(f"\nModel parameters: {n_params:,}")

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = build_scheduler(optimizer, warmup_steps, total_steps)

    criterion = LabelSmoothingLoss(V, tokenizer.PAD_ID, smoothing=args.label_smooth)

    # NOTE: fp16/AMP is not supported on MPS and is skipped.

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    if args.resume:
        ckpt = load_checkpoint(model, args.resume, optimizer, scheduler, str(device))
        start_epoch = ckpt["epoch"] + 1

    # ── Tracker ───────────────────────────────────────────────────────────────
    tracker = MetricTracker(os.path.join(out_dir, "plots"))
    tracker.load()

    # ── Training Loop ─────────────────────────────────────────────────────────
    best_val_bleu = tracker.best("val/bleu")

    # Curriculum: use short pairs for first 30% of epochs
    curriculum_cutoff = int(args.epochs * 0.3)

    for epoch in range(start_epoch, args.epochs):

        # Scheduled teacher forcing: linear decay
        tf_ratio = args.tf_ratio_start - (args.tf_ratio_start - args.tf_ratio_end) * \
                   (epoch / max(args.epochs - 1, 1))

        # Unfreeze BERT embeddings after warmup
        if args.embedding_type == "bert" and args.freeze_emb and epoch == args.unfreeze_epoch:
            print(f"\nEpoch {epoch}: Unfreezing embeddings")
            for p in model.encoder.embedding.parameters():
                p.requires_grad_(True)
            for p in model.decoder.embedding.parameters():
                p.requires_grad_(True)

        # ── Select loader (curriculum learning) ───────────────────────────
        if args.curriculum and curriculum_loader and epoch < curriculum_cutoff:
            active_loader = curriculum_loader
            loader_desc = f"Epoch {epoch} [curriculum]"
        else:
            active_loader = train_loader
            loader_desc = f"Epoch {epoch}"

        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        total_tokens = 0
        grad_norms = []  # Prompt 9: gradient norm tracking

        for step, (src, tgt, src_lengths) in tqdm(
            enumerate(active_loader), total=len(active_loader),
            desc=loader_desc, leave=False
        ):
            src = src.to(device)
            tgt = tgt.to(device)
            src_lengths = src_lengths.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(src, tgt, src_lengths, tf_ratio)
            tgt_out = tgt[:, 1:]
            B, T, V_ = logits.size()
            loss = criterion(logits.reshape(B * T, V_), tgt_out.reshape(B * T))
            loss.backward()

            # Gradient clipping + norm logging (Prompt 9)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            grad_norms.append(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)

            optimizer.step()
            scheduler.step()

            n_tok = (tgt_out != tokenizer.PAD_ID).sum().item()
            total_loss += loss.item() * n_tok
            total_tokens += n_tok

            if step % 100 == 0:
                lr_now = scheduler.get_last_lr()[0]
                mean_gnorm = sum(grad_norms[-100:]) / len(grad_norms[-100:])
                print(f"  E{epoch:02d} S{step:04d}/{len(active_loader)} "
                      f"loss={loss.item():.4f}  lr={lr_now:.2e}  tf={tf_ratio:.2f}  "
                      f"grad_norm={mean_gnorm:.4f}")

        train_loss = total_loss / max(total_tokens, 1)

        # Log mean gradient norm for this epoch
        epoch_mean_gnorm = sum(grad_norms) / max(len(grad_norms), 1)
        print(f"  Epoch {epoch} mean gradient norm: {epoch_mean_gnorm:.4f}")

        # ── Evaluate ──────────────────────────────────────────────────────
        print(f"\nEpoch {epoch} evaluation...")
        val_loss, val_bleu, val_chrf = evaluate_model(
            model, val_loader, tokenizer, device,
            use_beam=args.eval_beam
        )
        train_loss_eval, train_bleu, train_chrf = evaluate_model(
            model,
            # Use first 500 samples from train for quick train metric
            make_dataloader(
                TranslationDataset(train_src[:500], train_tgt[:500], tokenizer,
                                   args.max_len, args.max_len),
                args.batch_size, shuffle=False
            ),
            tokenizer, device,
        )

        print(f"\nEpoch {epoch:02d} | "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f} | "
              f"val_BLEU={val_bleu:.2f}  val_CHRF++={val_chrf:.2f}")

        # Track
        tracker.update("train", {"loss": train_loss, "bleu": train_bleu, "chrf_pp": train_chrf}, epoch)
        tracker.update("val",   {"loss": val_loss,   "bleu": val_bleu,   "chrf_pp": val_chrf},   epoch)
        tracker.save()

        # Save checkpoint
        is_best = val_bleu > best_val_bleu
        if is_best:
            best_val_bleu = val_bleu

        save_checkpoint(
            model, optimizer, scheduler, epoch, epoch * steps_per_epoch,
            {"val_bleu": val_bleu, "val_chrf": val_chrf, "val_loss": val_loss},
            save_dir=out_dir,
            filename="checkpoint_latest.pt"
        )
        # Save numbered checkpoints for checkpoint averaging
        save_checkpoint(
            model, optimizer, scheduler, epoch, epoch * steps_per_epoch,
            {"val_bleu": val_bleu, "val_chrf": val_chrf, "val_loss": val_loss},
            save_dir=out_dir,
            filename=f"checkpoint_epoch{epoch:02d}.pt"
        )
        if is_best:
            save_checkpoint(
                model, optimizer, scheduler, epoch, epoch * steps_per_epoch,
                {"val_bleu": val_bleu, "val_chrf": val_chrf, "val_loss": val_loss},
                save_dir=out_dir,
                filename="checkpoint_best.pt"
            )
            print(f"  ★ New best val BLEU: {val_bleu:.2f}")

    # ── Final plots ───────────────────────────────────────────────────────────
    tracker.plot_all(prefix=f"{run_name}_")

    # ── Test evaluation ───────────────────────────────────────────────────────
    print("\nLoading best checkpoint for test evaluation...")
    best_ckpt_path = os.path.join(out_dir, "checkpoint_best.pt")
    if os.path.exists(best_ckpt_path):
        load_checkpoint(model, best_ckpt_path, device=str(device))

    test_loss, test_bleu, test_chrf = evaluate_model(
        model, test_loader, tokenizer, device, use_beam=True, beam_size=4
    )
    print(f"\n{'='*60}")
    print(f"TEST RESULTS [{run_name}]")
    print(f"  BLEU-100  : {test_bleu:.2f}")
    print(f"  CHRF++-100: {test_chrf:.2f}")
    print(f"  Loss      : {test_loss:.4f}")
    print(f"{'='*60}\n")

    # Save test results
    with open(os.path.join(out_dir, "test_results.json"), "w") as f:
        json.dump({
            "run_name": run_name,
            "test_bleu": test_bleu,
            "test_chrf_pp": test_chrf,
            "test_loss": test_loss,
            "n_params": model.count_parameters(),
            "args": vars(args),
        }, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
