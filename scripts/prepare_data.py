"""
Data preparation script.
Run this FIRST after downloading the corpus.

Usage:
    python scripts/prepare_data.py --data_dir data/corpus --output_dir outputs/part1

This script:
  1. Loads and inspects the corpus
  2. Cleans and filters sentence pairs
  3. Trains a shared SentencePiece BPE tokenizer
  4. Prints corpus statistics
  5. Saves train/val/test splits as text files (for inspection)
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import (
    load_parallel_corpus, split_corpus, SPTokenizer,
    print_corpus_stats
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    default="data/corpus")
    p.add_argument("--output_dir",  default="outputs/part1")
    p.add_argument("--vocab_size",  type=int, default=8000)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("DATA PREPARATION")
    print("=" * 60)

    # Load corpus
    src, tgt = load_parallel_corpus(
        args.data_dir, src_lang="hi", tgt_lang="mr",
        max_samples=args.max_samples, seed=args.seed
    )
    print_corpus_stats(src, tgt, "Full corpus")

    # Split
    (tr_s, tr_t), (v_s, v_t), (te_s, te_t) = split_corpus(src, tgt, seed=args.seed)

    # Save splits as text for inspection
    splits_dir = os.path.join(args.output_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)
    for name, (s_list, t_list) in [("train", (tr_s, tr_t)), ("val", (v_s, v_t)), ("test", (te_s, te_t))]:
        with open(os.path.join(splits_dir, f"{name}.hi"), "w", encoding="utf-8") as f:
            f.write("\n".join(s_list))
        with open(os.path.join(splits_dir, f"{name}.mr"), "w", encoding="utf-8") as f:
            f.write("\n".join(t_list))
    print(f"\nSplit files saved to {splits_dir}/")

    # Train tokenizer (shared Hindi + Marathi vocabulary)
    sp_model_path = os.path.join(args.output_dir, f"spm_{args.vocab_size}.model")
    if not os.path.exists(sp_model_path):
        print(f"\nTraining SentencePiece BPE (vocab={args.vocab_size})...")
        tokenizer = SPTokenizer.train(
            sentences=tr_s + tr_t,
            model_prefix=os.path.join(args.output_dir, f"spm_{args.vocab_size}"),
            vocab_size=args.vocab_size,
        )
    else:
        tokenizer = SPTokenizer(sp_model_path)
        print(f"\nTokenizer already exists: {sp_model_path}")

    # Verify tokenizer
    sample_hi = src[0] if src else "नमस्ते"
    sample_mr = tgt[0] if tgt else "नमस्कार"
    print(f"\nTokenizer verification:")
    print(f"  Hindi   : '{sample_hi}' → {tokenizer.encode(sample_hi)[:10]}...")
    print(f"  Marathi : '{sample_mr}' → {tokenizer.encode(sample_mr)[:10]}...")
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # Token coverage check
    from collections import Counter
    unk_count = 0
    total_count = 0
    for sent in tr_s[:1000] + tr_t[:1000]:
        ids = tokenizer.encode(sent)
        unk_count  += ids.count(tokenizer.UNK_ID)
        total_count += len(ids)

    unk_rate = unk_count / max(total_count, 1) * 100
    print(f"  UNK rate on train sample: {unk_rate:.2f}%  (< 1% is good)")

    print("\nData preparation complete!")
    print(f"Tokenizer: {sp_model_path}")
    print(f"Splits:    {splits_dir}/")


if __name__ == "__main__":
    main()
