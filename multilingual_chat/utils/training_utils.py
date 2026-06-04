"""
Isolated training utilities for Causal Chat GPT.

Includes:
    - Cosine annealing learning rate scheduler with linear warmup
    - Metric tracking for validation loss and perplexities
    - Matplotlib plotting helpers for training curves
"""

import os
import json
import math
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for macOS/headless environments
import matplotlib.pyplot as plt
from typing import List, Dict, Any


class CosineWarmupScheduler:
    """Cosine learning rate scheduler with linear warmup."""
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        base_lr: float,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            # Linear warmup
            lr = self.min_lr + (self.base_lr - self.min_lr) * (self.current_step / max(1, self.warmup_steps))
        elif self.current_step > self.total_steps:
            lr = self.min_lr
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr


class ChatMetricTracker:
    """Track training metrics (Loss, Perplexity) and generate curves."""
    def __init__(self, output_dir: str, name: str):
        self.output_dir = output_dir
        self.name = name
        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_ppl": [],
            "val_ppl": [],
            "lr": [],
        }

    def update(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_ppl: float,
        val_ppl: float,
        lr: float,
    ):
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["train_ppl"].append(train_ppl)
        self.history["val_ppl"].append(val_ppl)
        self.history["lr"].append(lr)

        self.save_logs()
        self.plot_curves()

    def save_logs(self):
        log_path = os.path.join(self.output_dir, f"{self.name}_metrics.json")
        with open(log_path, "w") as f:
            json.dump(self.history, f, indent=2)

    def plot_curves(self):
        epochs = self.history["epoch"]
        if len(epochs) < 2:
            return  # Need at least two data points to generate meaningful curves

        # 1. Plot Loss Curves
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, self.history["train_loss"], label="Train Loss", color="#3b82f6", marker="o")
        plt.plot(epochs, self.history["val_loss"], label="Val Loss", color="#ef4444", marker="s")
        plt.title(f"{self.name.upper()} - Training & Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Cross-Entropy Loss")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{self.name}_loss_curves.png"), dpi=150)
        plt.close()

        # 2. Plot Perplexity Curves
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, self.history["train_ppl"], label="Train PPL", color="#8b5cf6", marker="o")
        plt.plot(epochs, self.history["val_ppl"], label="Val PPL", color="#10b981", marker="s")
        plt.title(f"{self.name.upper()} - Training & Validation Perplexity")
        plt.xlabel("Epochs")
        plt.ylabel("Perplexity")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{self.name}_ppl_curves.png"), dpi=150)
        plt.close()
