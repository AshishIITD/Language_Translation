"""
Unified Conversational GPT Training Script.

Features:
    - Target device fallback detection (MPS > CUDA > CPU) without AMP
    - Dual-channel conversational data loading:
        - Parallel cross-lingual turns (e.g., hi -> mr)
        - Monolingual conversational continuation (e.g., ur_N -> ur_N+1)
    - Autoregressive prompt-loss masking (training only on assistant turns)
    - Cosine learning rate scheduling with warmup
    - Gradient norm tracking & periodic perplexity logging
    - Automatic config serialization & training curves plotting
"""

import os
import sys
import argparse
import math
import random
import json
import torch
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Dict, Any

sys.path.insert(0, str(Path(__file__).parent))

from data.chat_dataset import (
    SPChatTokenizer, MultilingualChatDataset, make_chat_dataloader, normalize_chat_text
)
from model.gpt import MultilingualGPT, GPT_SMALL, GPT_FULL
from utils.training_utils import CosineWarmupScheduler, ChatMetricTracker

LANGUAGES = ["hi", "mr", "ta", "te", "ur", "or", "pa", "ml", "mai", "gu", "as", "bn", "bho"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scale",         default="small", choices=["small", "full"])
    p.add_argument("--data_dir",      default="Dataset")
    p.add_argument("--output_dir",    default="outputs/chat")
    p.add_argument("--epochs",        type=int, default=10)
    p.add_argument("--batch_size",    type=int, default=32)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--warmup_steps",  type=int, default=100)
    p.add_argument("--grad_accum",    type=int, default=1)
    p.add_argument("--max_len",       type=int, default=256)
    p.add_argument("--clip",          type=float, default=1.0)
    p.add_argument("--workers",       type=int, default=1)
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--max_samples",   type=int, default=None)
    return p.parse_args()


def load_conversational_data(data_dir: str, seed: int = 42) -> List[Tuple[str, str]]:
    """
    Intelligent Dual-Channel Conversational Data Loader.
    
    1. Channel A (Monolingual Continuation):
       For each language, if `{lang}.txt` or `train.{lang}` exists, pairs consecutive
       sentences as dialogue turns: Prompt = Sentence N, Response = Sentence N+1.
       
    2. Channel B (Cross-Lingual dialogue):
       For parallel pairs (like `train.hi` and `train.mr`), constructs translation-dialogue turns.
    """
    data_dir = Path(data_dir)
    dialogues: List[Tuple[str, str]] = []

    print("\nScanning dataset for dialogue channels...")

    # ── Channel A: Monolingual Continuation ──────────────────────────────────
    for lang in LANGUAGES:
        lang_paths = [
            data_dir / f"{lang}.txt",
            data_dir / f"train.{lang}",
            data_dir / f"corpus.{lang}",
        ]
        lines = []
        for path in lang_paths:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    lines = [l.strip() for l in f if l.strip()]
                print(f"  [Channel A] Found {len(lines):,} lines for monolingual '{lang}'")
                break

        if len(lines) >= 2:
            # Pair consecutive lines: Turn N -> Turn N+1
            for i in range(0, len(lines) - 1, 2):
                dialogues.append((lines[i], lines[i+1]))

    # ── Channel B: Cross-lingual turns ───────────────────────────────────────
    # Scan for parallel language pairs: train.{lang1} and train.{lang2}
    for i, lang1 in enumerate(LANGUAGES):
        for lang2 in LANGUAGES[i+1:]:
            path1 = data_dir / f"train.{lang1}"
            path2 = data_dir / f"train.{lang2}"
            if path1.exists() and path2.exists():
                with open(path1, "r", encoding="utf-8") as f1, open(path2, "r", encoding="utf-8") as f2:
                    lines1 = f1.read().splitlines()
                    lines2 = f2.read().splitlines()
                n = min(len(lines1), len(lines2))
                if n > 0:
                    print(f"  [Channel B] Found {n:,} parallel turns between '{lang1}' and '{lang2}'")
                    for k in range(n):
                        if lines1[k].strip() and lines2[k].strip():
                            dialogues.append((lines1[k], lines2[k]))
                            dialogues.append((lines2[k], lines1[k]))  # Bidirectional

    if not dialogues:
        raise FileNotFoundError(
            f"Could not construct conversational turns. Place text files under '{data_dir}' "
            f"named like train.hi, train.mr, train.ur, etc."
        )

    # Shuffle conversational pairs
    rng = random.Random(seed)
    rng.shuffle(dialogues)

    print(f"Total conversational dialogue pairs constructed: {len(dialogues):,}\n")
    return dialogues


@torch.no_grad()
def evaluate_chat_model(model, loader, device: torch.device) -> float:
    """Evaluate loss on validation set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        _, loss = model(inputs, targets)

        # Count active non-masked targets
        active = (targets != -100).sum().item()
        total_loss += loss.item() * active
        total_tokens += active

    return total_loss / max(total_tokens, 1)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Target Device Dispatch (MPS > CUDA > CPU) ────────────────────────────
    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")
    print(f"Output directory: {args.output_dir}")

    # Serialize Run Config
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Config saved to {config_path}")

    # ── Load and Split Data ──────────────────────────────────────────────────
    dialogues = load_conversational_data(args.data_dir, seed=args.seed)
    if args.max_samples and args.max_samples < len(dialogues):
        dialogues = dialogues[:args.max_samples]
        print(f"Subsampled to {len(dialogues):,} dialogue turns")

    # Split: 90% train, 10% validation
    n_val = int(len(dialogues) * 0.10)
    train_turns = dialogues[n_val:]
    val_turns = dialogues[:n_val] if n_val > 0 else dialogues

    print(f"Train split: {len(train_turns):,} turns | Val split: {len(val_turns):,} turns")

    # ── Prepare joint Conversational Tokenizer ────────────────────────────────
    # We search if a pre-trained tokenizer is cached under output, otherwise train one
    spm_prefix = os.path.join(args.output_dir, "spm_chat_2000")
    spm_model_path = spm_prefix + ".model"

    if not os.path.exists(spm_model_path):
        print("\nTraining Joint Conversational Tokenizer...")
        # Train on all available dialogue prompts and replies
        corpus = [prompt for prompt, _ in dialogues] + [reply for _, reply in dialogues]
        tokenizer = SPChatTokenizer.train(corpus, spm_prefix, vocab_size=2000)
    else:
        print(f"\nLoading cached Conversational Tokenizer: {spm_model_path}")
        tokenizer = SPChatTokenizer(spm_model_path)

    # ── Datasets and Loaders ──────────────────────────────────────────────────
    train_ds = MultilingualChatDataset(train_turns, tokenizer, args.max_len)
    val_ds = MultilingualChatDataset(val_turns, tokenizer, args.max_len)

    train_loader = make_chat_dataloader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = make_chat_dataloader(val_ds, args.batch_size, shuffle=False, num_workers=args.workers)

    # ── Initialize Causal GPT Model ──────────────────────────────────────────
    cfg = GPT_SMALL if args.scale == "small" else GPT_FULL
    cfg.vocab_size = tokenizer.vocab_size
    cfg.max_seq_len = args.max_len

    model = MultilingualGPT(cfg).to(device)
    print(f"Conversational GPT Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ── Optimizer and LR Warmup ──────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = CosineWarmupScheduler(
        optimizer, warmup_steps=args.warmup_steps, total_steps=total_steps, base_lr=args.lr
    )

    tracker = ChatMetricTracker(args.output_dir, "conversational_gpt")

    # ── Training Loop ────────────────────────────────────────────────────────
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        total_tokens = 0
        grad_norms = []

        loader_desc = f"Epoch {epoch:02d} [Train]"
        loader = tqdm(enumerate(train_loader), total=len(train_loader), desc=loader_desc, leave=False)

        optimizer.zero_grad()

        for step, (inputs, targets) in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            _, loss = model(inputs, targets)
            loss = loss / args.grad_accum
            loss.backward()

            # Active target tokens count
            active = (targets != -100).sum().item()
            train_loss += loss.item() * args.grad_accum * active
            total_tokens += active

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                # Gradient clipping with tracking
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                grad_norms.append(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)

                optimizer.step()
                lr = scheduler.step()
                optimizer.zero_grad()

            if step % 50 == 0:
                cur_loss = (train_loss / max(total_tokens, 1))
                cur_ppl = math.exp(min(cur_loss, 20))
                mean_gn = sum(grad_norms[-10:]) / max(len(grad_norms[-10:]), 1)
                loader.set_postfix(loss=f"{cur_loss:.3f}", ppl=f"{cur_ppl:.1f}", gn=f"{mean_gn:.2f}")

        # Compute Epoch Metrics
        avg_train_loss = train_loss / max(total_tokens, 1)
        train_ppl = math.exp(min(avg_train_loss, 20))
        mean_grad_norm = sum(grad_norms) / max(len(grad_norms), 1)

        print(f"\nEpoch {epoch:02d} | Train Loss={avg_train_loss:.4f} | Train PPL={train_ppl:.2f} | Mean GradNorm={mean_grad_norm:.3f}")

        # Validation Step
        print("Evaluating validation set...")
        avg_val_loss = evaluate_chat_model(model, val_loader, device)
        val_ppl = math.exp(min(avg_val_loss, 20))

        print(f"Epoch {epoch:02d} | Val Loss={avg_val_loss:.4f} | Val PPL={val_ppl:.2f}")

        # Update History & Training Curves
        tracker.update(epoch, avg_train_loss, avg_val_loss, train_ppl, val_ppl, optimizer.param_groups[0]["lr"])

        # Checkpoint Saving
        checkpoint_path = os.path.join(args.output_dir, "checkpoint_latest.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "cfg": cfg,
        }, checkpoint_path)

        # Save numbered checkpoints for Checkpoint Averaging
        epoch_checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch:02d}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "cfg": cfg,
        }, epoch_checkpoint_path)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_checkpoint_path = os.path.join(args.output_dir, "checkpoint_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "cfg": cfg,
            }, best_checkpoint_path)
            print(f"  ★ New best validation loss achieved! Checkpoint saved.")

    print("\nTraining complete! Causal Chat GPT is ready to converse.")


if __name__ == "__main__":
    main()
