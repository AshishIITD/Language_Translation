"""
Training utilities shared across Part I and Part II:
  - Optimizer + LR scheduler setup
  - Checkpoint save/load
  - Metric tracking + plotting
  - Label smoothing loss
  - Gradient clipping wrapper
"""

import os
import json
import math
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server environments
import matplotlib.pyplot as plt


# ─── Loss ─────────────────────────────────────────────────────────────────────

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing cross-entropy.

    Why label smoothing?
      Hard labels (one-hot) can cause the model to become overconfident.
      Label smoothing (Szegedy et al., 2016) softens targets, improving
      calibration and regularizing training — especially useful when the
      parallel corpus is noisy.

      smoothed target = (1 - ε) * one_hot + ε / (V - 1)
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.criterion = nn.KLDivLoss(reduction="sum")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: (N, V)
        target: (N,)
        """
        log_probs = torch.log_softmax(logits, dim=-1)

        # Build smoothed distribution
        with torch.no_grad():
            smooth_dist = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 2))
            smooth_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            smooth_dist[:, self.pad_idx] = 0
            # Zero out rows where target is padding
            mask = target == self.pad_idx
            smooth_dist[mask] = 0

        loss = self.criterion(log_probs, smooth_dist)

        # Normalize by non-padding tokens
        n_tokens = (~mask).sum()
        return loss / n_tokens.clamp(min=1)


# ─── Optimizer & Scheduler ────────────────────────────────────────────────────

def build_optimizer(
    model: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    betas: Tuple[float, float] = (0.9, 0.999),
    no_decay_params: Optional[List[str]] = None,
) -> torch.optim.AdamW:
    """
    AdamW optimizer with weight decay applied only to non-bias, non-norm params.

    Why separate decay groups?
      Applying L2 regularization to bias terms and LayerNorm parameters is
      generally harmful — they don't overfit in the same way as weight matrices.
    """
    if no_decay_params is None:
        no_decay_params = ["bias", "LayerNorm.weight", "layer_norm", "norm"]

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in no_decay_params):
            no_decay.append(param)
        else:
            decay.append(param)

    param_groups = [
        {"params": decay,    "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=1e-8)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Linear warmup + cosine decay schedule.

    Why warmup?
      Early training has unstable gradients. A warmup period prevents
      large initial LR updates from pushing the model into a poor basin.
      Especially important when using pretrained embeddings.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─── Checkpoint ───────────────────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    step: int,
    metrics: dict,
    save_dir: str,
    filename: str = "checkpoint.pt",
):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save({
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
    }, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(
    model: nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    device: str = "cpu",
) -> dict:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    print(f"Loaded checkpoint from {path} (epoch {ckpt['epoch']})")
    return ckpt


def average_checkpoints(paths: List[str], model: nn.Module) -> nn.Module:
    """
    Average the parameters of N checkpoint files and load into model.

    Checkpoint averaging (Vaswani et al., 2017) typically improves BLEU by
    0.5–1.5 points by smoothing out parameter noise from SGD oscillations
    near the end of training.

    Args:
        paths: list of checkpoint file paths
        model: model instance to load averaged weights into
    Returns:
        model with averaged weights
    """
    assert len(paths) > 0, "Must provide at least one checkpoint path"

    # Load first checkpoint as base
    avg_state = torch.load(paths[0], map_location="cpu")["model_state_dict"]
    avg_state = {k: v.float() for k, v in avg_state.items()}

    # Accumulate remaining checkpoints
    for path in paths[1:]:
        state = torch.load(path, map_location="cpu")["model_state_dict"]
        for k in avg_state:
            avg_state[k] += state[k].float()

    # Divide by N
    n = len(paths)
    for k in avg_state:
        avg_state[k] /= n

    model.load_state_dict(avg_state)
    print(f"Averaged {n} checkpoints: {[os.path.basename(p) for p in paths]}")
    return model


# ─── Metric Tracker ───────────────────────────────────────────────────────────

class MetricTracker:
    """Tracks train/val metrics across epochs for plotting and early stopping."""

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.history: Dict[str, List[float]] = {}

    def update(self, split: str, metrics: dict, epoch: int):
        for key, val in metrics.items():
            full_key = f"{split}/{key}"
            if full_key not in self.history:
                self.history[full_key] = []
            self.history[full_key].append(val)

    def save(self, filename: str = "metrics.json"):
        with open(os.path.join(self.save_dir, filename), "w") as f:
            json.dump(self.history, f, indent=2)

    def load(self, filename: str = "metrics.json"):
        path = os.path.join(self.save_dir, filename)
        if os.path.exists(path):
            with open(path) as f:
                self.history = json.load(f)

    def plot_loss(self, output_path: Optional[str] = None):
        """Plot train and validation loss curves."""
        fig, ax = plt.subplots(figsize=(10, 5))
        if "train/loss" in self.history:
            ax.plot(self.history["train/loss"], label="Train Loss", color="steelblue")
        if "val/loss" in self.history:
            ax.plot(self.history["val/loss"], label="Val Loss", color="orangered")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training and Validation Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = output_path or os.path.join(self.save_dir, "loss_curves.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Loss plot saved: {path}")

    def plot_bleu(self, output_path: Optional[str] = None):
        """Plot BLEU-100 curves."""
        fig, ax = plt.subplots(figsize=(10, 5))
        if "train/bleu" in self.history:
            ax.plot(self.history["train/bleu"], label="Train BLEU-100", color="steelblue")
        if "val/bleu" in self.history:
            ax.plot(self.history["val/bleu"], label="Val BLEU-100", color="orangered")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BLEU-100")
        ax.set_title("BLEU-100 Curves")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = output_path or os.path.join(self.save_dir, "bleu_curves.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"BLEU plot saved: {path}")

    def plot_chrf(self, output_path: Optional[str] = None):
        """Plot CHRF++-100 curves."""
        fig, ax = plt.subplots(figsize=(10, 5))
        if "train/chrf_pp" in self.history:
            ax.plot(self.history["train/chrf_pp"], label="Train CHRF++-100", color="steelblue")
        if "val/chrf_pp" in self.history:
            ax.plot(self.history["val/chrf_pp"], label="Val CHRF++-100", color="orangered")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("CHRF++-100")
        ax.set_title("CHRF++-100 Curves")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = output_path or os.path.join(self.save_dir, "chrf_curves.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"CHRF++ plot saved: {path}")

    def plot_ppl(self, output_path: Optional[str] = None):
        """Plot perplexity curves (used for LM pretraining evaluation)."""
        fig, ax = plt.subplots(figsize=(10, 5))
        if "train/ppl" in self.history:
            ax.plot(self.history["train/ppl"], label="Train PPL", color="steelblue")
        if "val/ppl" in self.history:
            ax.plot(self.history["val/ppl"], label="Val PPL", color="orangered")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Perplexity")
        ax.set_title("Pretraining Perplexity")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = output_path or os.path.join(self.save_dir, "ppl_curves.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Perplexity plot saved: {path}")

    def plot_all(self, prefix: str = ""):
        """Generate all required plots."""
        self.plot_loss(os.path.join(self.save_dir, f"{prefix}loss_curves.png"))
        self.plot_bleu(os.path.join(self.save_dir, f"{prefix}bleu_curves.png"))
        self.plot_chrf(os.path.join(self.save_dir, f"{prefix}chrf_curves.png"))
        # Generate perplexity plot if ppl data exists
        if "val/ppl" in self.history or "train/ppl" in self.history:
            self.plot_ppl(os.path.join(self.save_dir, f"{prefix}ppl_curves.png"))

    def best(self, key: str = "val/bleu") -> float:
        return max(self.history.get(key, [0.0]))

    def is_best(self, key: str = "val/bleu") -> bool:
        vals = self.history.get(key, [])
        return len(vals) > 0 and vals[-1] == max(vals)
