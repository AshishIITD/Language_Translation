"""
Comparison plotting script.
Generates side-by-side plots comparing random vs BERT embeddings (Part I)
and LSTM vs Transformer results (Part I vs Part II).

Usage:
    python scripts/plot_comparisons.py --output_dir outputs/
"""

import os
import sys
import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_metrics(metrics_json: str) -> dict:
    if not os.path.exists(metrics_json):
        return {}
    with open(metrics_json) as f:
        return json.load(f)


def plot_part1_comparison(output_dir: str):
    """Compare random vs BERT embeddings for Part I."""
    runs = {
        "Random (hi→mr)":  os.path.join(output_dir, "part1/hi2mr_random/plots/metrics.json"),
        "BERT (hi→mr)":    os.path.join(output_dir, "part1/hi2mr_bert/plots/metrics.json"),
        "Random (mr→hi)":  os.path.join(output_dir, "part1/mr2hi_random/plots/metrics.json"),
        "BERT (mr→hi)":    os.path.join(output_dir, "part1/mr2hi_bert/plots/metrics.json"),
    }

    colors = ["steelblue", "orangered", "seagreen", "purple"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].set_title("Validation Loss",      fontsize=13)
    axes[1].set_title("Validation BLEU-100",  fontsize=13)
    axes[2].set_title("Validation CHRF++-100",fontsize=13)

    for (name, path), color in zip(runs.items(), colors):
        data = load_metrics(path)
        if not data:
            continue
        val_loss = data.get("val/loss", [])
        val_bleu = data.get("val/bleu", [])
        val_chrf = data.get("val/chrf_pp", [])

        epochs = range(1, len(val_loss) + 1)
        if val_loss: axes[0].plot(epochs, val_loss, label=name, color=color)
        epochs = range(1, len(val_bleu) + 1)
        if val_bleu: axes[1].plot(epochs, val_bleu, label=name, color=color)
        epochs = range(1, len(val_chrf) + 1)
        if val_chrf: axes[2].plot(epochs, val_chrf, label=name, color=color)

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Part I: Random vs BERT Embeddings", fontsize=15, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(output_dir, "comparison_part1.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Part I comparison plot saved: {out}")


def plot_part2_comparison(output_dir: str):
    """Compare LSTM baseline vs Seq2Seq Transformer (Part II) for hi→mr."""
    runs = {
        "LSTM+Random (hi→mr)":  os.path.join(output_dir, "part1/hi2mr_random/plots/metrics.json"),
        "LSTM+BERT (hi→mr)":    os.path.join(output_dir, "part1/hi2mr_bert/plots/metrics.json"),
        "Transformer (hi→mr)":  os.path.join(output_dir, "part2/finetune_small_hi2mr_phase2/plots/metrics.json"),
    }

    colors = ["steelblue", "orangered", "seagreen"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].set_title("Validation BLEU-100",  fontsize=13)
    axes[1].set_title("Validation CHRF++-100",fontsize=13)

    for (name, path), color in zip(runs.items(), colors):
        data = load_metrics(path)
        if not data:
            continue
        val_bleu = data.get("val/bleu", [])
        val_chrf = data.get("val/chrf_pp", [])

        if val_bleu:
            axes[0].plot(range(1, len(val_bleu)+1), val_bleu, label=name, color=color)
        if val_chrf:
            axes[1].plot(range(1, len(val_chrf)+1), val_chrf, label=name, color=color)

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Part I vs Part II: Translation Quality", fontsize=15, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(output_dir, "comparison_part1_vs_part2.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Part I vs II comparison saved: {out}")


def generate_results_table(output_dir: str):
    """Print a results table from all test_results.json files."""
    import glob
    print("\n" + "="*70)
    print(f"{'Run':<40} {'BLEU-100':>10} {'CHRF++-100':>12}")
    print("-"*70)

    for path in sorted(glob.glob(os.path.join(output_dir, "**", "test_results.json"), recursive=True)):
        with open(path) as f:
            data = json.load(f)
        name = data.get("run_name") or data.get("run") or Path(path).parent.name
        bleu = data.get("test_bleu") or data.get("bleu", 0)
        chrf = data.get("test_chrf_pp") or data.get("chrf_pp", 0)
        print(f"{name:<40} {bleu:>10.2f} {chrf:>12.2f}")
    print("="*70)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="outputs")
    args = p.parse_args()

    plot_part1_comparison(args.output_dir)
    plot_part2_comparison(args.output_dir)
    generate_results_table(args.output_dir)


if __name__ == "__main__":
    main()
