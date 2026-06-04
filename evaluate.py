"""
Standalone evaluation script.

Usage:
    # Evaluate Part I model:
    python evaluate.py --model_type lstm \
        --checkpoint outputs/part1/hi2mr_random/checkpoint_best.pt \
        --spm_model outputs/part1/spm_8000.model \
        --data_dir data/corpus --split test

    # Evaluate Part II model:
    python evaluate.py --model_type transformer \
        --checkpoint outputs/part2/finetune_full_hi2mr_phase2/checkpoint_best.pt \
        --bert_ckpt outputs/part2/bert_full/checkpoint_best.pt \
        --gpt_ckpt  outputs/part2/gpt_full/checkpoint_best.pt \
        --scale full --spm_model outputs/part1/spm_8000.model \
        --data_dir data/corpus --split test

    # Qualitative analysis with markdown output:
    python evaluate.py --model_type lstm \
        --checkpoint outputs/part1/hi2mr_random/checkpoint_best.pt \
        --spm_model outputs/part1/spm_8000.model \
        --data_dir data/corpus --split test \
        --qualitative --n_examples 20 \
        --qualitative_output outputs/part1/qualitative_results.md

    # With checkpoint averaging (last 5 checkpoints):
    python evaluate.py --model_type lstm \
        --checkpoint outputs/part1/hi2mr_random/checkpoint_best.pt \
        --spm_model outputs/part1/spm_8000.model \
        --data_dir data/corpus --split test \
        --avg_checkpoints
"""

import os
import sys
import glob
import argparse
import json
import statistics
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import (
    load_parallel_corpus, split_corpus, SPTokenizer,
    TranslationDataset, make_dataloader
)
from evaluation.metrics import compute_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_type",  required=True, choices=["lstm", "transformer"])
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--spm_model",   required=True)
    p.add_argument("--data_dir",    default="data/corpus")
    p.add_argument("--split",       default="test", choices=["train", "val", "test"])
    p.add_argument("--direction",   default="hi2mr", choices=["hi2mr", "mr2hi"])
    p.add_argument("--scale",       default="full", choices=["full", "small"])  # for transformer
    p.add_argument("--bert_ckpt",   default=None)
    p.add_argument("--gpt_ckpt",    default=None)
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--max_len",     type=int, default=128)
    p.add_argument("--beam_size",   type=int, default=4)
    p.add_argument("--qualitative", action="store_true")
    p.add_argument("--qualitative_output", type=str, default=None,
                   help="Path to save qualitative examples as markdown table")
    p.add_argument("--n_examples",  type=int, default=20)
    p.add_argument("--output_file", default=None)
    p.add_argument("--seed",        type=int, default=42)

    # Checkpoint averaging (Prompt 6)
    p.add_argument("--avg_checkpoints", action="store_true",
                   help="Average last 5 checkpoints before evaluation (improves BLEU 0.5-1.5)")
    p.add_argument("--n_avg",       type=int, default=5,
                   help="Number of last checkpoints to average")

    return p.parse_args()


def load_lstm_model(checkpoint_path, tokenizer, device):
    from part1.model import Seq2SeqLSTM
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("args", {})
    V = tokenizer.vocab_size
    model = Seq2SeqLSTM(
        src_vocab_size=V, tgt_vocab_size=V,
        embed_dim=cfg.get("embed_dim", 256),
        hidden_size=cfg.get("hidden_size", 512),
        attn_dim=cfg.get("attn_dim", 256),
        num_layers=cfg.get("num_layers", 2),
        dropout=0.0,
        pad_idx=tokenizer.PAD_ID,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_transformer_model(checkpoint_path, bert_ckpt, gpt_ckpt, scale, tokenizer, device):
    from part2.models import (
        Seq2SeqTransformer,
        BERT_FULL, GPT_FULL, BERT_SMALL, GPT_SMALL
    )
    bert_cfg = BERT_FULL if scale == "full" else BERT_SMALL
    gpt_cfg  = GPT_FULL  if scale == "full" else GPT_SMALL
    bert_cfg.vocab_size = tokenizer.vocab_size
    gpt_cfg.vocab_size  = tokenizer.vocab_size

    model = Seq2SeqTransformer(
        bert_config=bert_cfg, gpt_config=gpt_cfg,
        bert_pretrained_path=bert_ckpt,
        gpt_pretrained_path=gpt_ckpt,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def translate_lstm(model, src_ids, src_lengths, tokenizer, device, beam_size):
    src = src_ids.to(device)
    src_lengths = src_lengths.to(device)
    if beam_size > 1:
        results = []
        for i in range(src.size(0)):
            hyp = model.beam_search(
                src[i:i+1], src_lengths[i:i+1],
                bos_id=tokenizer.BOS_ID, eos_id=tokenizer.EOS_ID,
                beam_size=beam_size
            )
            results.append(tokenizer.decode(hyp))
        return results
    else:
        pred = model.greedy_decode(src, src_lengths, tokenizer.BOS_ID, tokenizer.EOS_ID)
        return [tokenizer.decode(pred[i].tolist()) for i in range(src.size(0))]


@torch.no_grad()
def translate_transformer(model, src_ids, src_lengths, tokenizer, device, beam_size=1):
    src = src_ids.to(device)
    src_mask = (src != tokenizer.PAD_ID).long()
    if beam_size > 1:
        results = []
        for i in range(src.size(0)):
            hyp_ids = model.beam_search(
                src[i:i+1], src_mask[i:i+1],
                tokenizer.BOS_ID, tokenizer.EOS_ID,
                beam_size=beam_size
            )
            results.append(tokenizer.decode(hyp_ids))
        return results
    else:
        pred = model.greedy_decode(src, src_mask, tokenizer.BOS_ID, tokenizer.EOS_ID)
        return [tokenizer.decode(pred[i].tolist()) for i in range(src.size(0))]


def main():
    args = parse_args()

    # ── Device detection (MPS > CUDA > CPU) ──────────────────────────────────
    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    src_lang = "hi" if args.direction == "hi2mr" else "mr"
    tgt_lang = "mr" if args.direction == "hi2mr" else "hi"

    all_src, all_tgt = load_parallel_corpus(
        args.data_dir, src_lang=src_lang, tgt_lang=tgt_lang, seed=args.seed
    )
    (train_src, train_tgt), (val_src, val_tgt), (test_src, test_tgt) = \
        split_corpus(all_src, all_tgt, seed=args.seed)

    splits = {"train": (train_src, train_tgt), "val": (val_src, val_tgt), "test": (test_src, test_tgt)}
    eval_src, eval_tgt = splits[args.split]
    print(f"Evaluating on {args.split}: {len(eval_src):,} pairs")

    tokenizer = SPTokenizer(args.spm_model)
    ds = TranslationDataset(eval_src, eval_tgt, tokenizer, args.max_len, args.max_len)
    loader = make_dataloader(ds, args.batch_size, shuffle=False)

    # ── Load model ────────────────────────────────────────────────────────────
    if args.model_type == "lstm":
        model = load_lstm_model(args.checkpoint, tokenizer, device)
    else:
        model = load_transformer_model(
            args.checkpoint, args.bert_ckpt, args.gpt_ckpt,
            args.scale, tokenizer, device
        )

    # ── Checkpoint averaging (Prompt 6) ───────────────────────────────────────
    if args.avg_checkpoints:
        from utils.training_utils import average_checkpoints
        ckpt_dir = os.path.dirname(args.checkpoint)
        # Find numbered checkpoints: checkpoint_epoch*.pt
        ckpt_paths = sorted(glob.glob(os.path.join(ckpt_dir, "checkpoint_epoch*.pt")))
        if len(ckpt_paths) >= 2:
            # Take the last N checkpoints
            avg_paths = ckpt_paths[-args.n_avg:]
            print(f"\nAveraging {len(avg_paths)} checkpoints...")
            model = average_checkpoints(avg_paths, model)
            model = model.to(device)
        else:
            print(f"Only {len(ckpt_paths)} numbered checkpoints found, skipping averaging")

    # ── Generate translations ─────────────────────────────────────────────────
    hypotheses = []
    references = []

    for src, tgt, src_lengths in loader:
        if args.model_type == "lstm":
            hyps = translate_lstm(model, src, src_lengths, tokenizer, device, args.beam_size)
        else:
            hyps = translate_transformer(model, src, src_lengths, tokenizer, device, args.beam_size)

        hypotheses.extend(hyps)
        for i in range(tgt.size(0)):
            references.append(tokenizer.decode(tgt[i, 1:].tolist()))

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = compute_metrics(hypotheses, references)
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"  Model     : {args.model_type} ({args.checkpoint})")
    print(f"  Split     : {args.split}  ({len(eval_src):,} pairs)")
    print(f"  Direction : {args.direction}")
    print(f"  BLEU-100  : {metrics['bleu']:.2f}")
    print(f"  CHRF++-100: {metrics['chrf_pp']:.2f}")
    if args.avg_checkpoints:
        print(f"  (using checkpoint averaging)")
    print(f"{'='*60}\n")

    # ── Qualitative Analysis (Prompt 8 — Report-quality) ─────────────────────
    if args.qualitative:
        import random
        from sacrebleu.metrics import BLEU as SacreBLEU, CHRF as SacreCHRF

        rng = random.Random(args.seed)
        n = min(args.n_examples, len(eval_src))
        indices = rng.sample(range(len(eval_src)), n)

        bleu_scorer = SacreBLEU(effective_order=True)
        chrf_scorer = SacreCHRF(word_order=2)

        print(f"\nQUALITATIVE EXAMPLES ({n} samples):")
        print("─" * 80)
        examples = []
        sent_bleus = []

        for idx in indices:
            ex = {
                "source"    : eval_src[idx],
                "reference" : eval_tgt[idx],
                "hypothesis": hypotheses[idx],
            }

            # Sentence-level BLEU and CHRF++ (Prompt 8)
            try:
                sent_bleu = bleu_scorer.sentence_score(
                    ex["hypothesis"], [ex["reference"]]
                ).score
                ex["sent_bleu"] = round(sent_bleu, 2)
            except Exception:
                ex["sent_bleu"] = 0.0

            try:
                sent_chrf = chrf_scorer.sentence_score(
                    ex["hypothesis"], [ex["reference"]]
                ).score
                ex["sent_chrf"] = round(sent_chrf, 2)
            except Exception:
                ex["sent_chrf"] = 0.0

            # Flag good/bad examples (Prompt 8)
            if ex["sent_bleu"] > 30:
                ex["quality"] = "✓"
            elif ex["sent_bleu"] < 10:
                ex["quality"] = "✗"
            else:
                ex["quality"] = "~"

            sent_bleus.append(ex["sent_bleu"])
            examples.append(ex)

            print(f"{ex['quality']} SRC : {ex['source']}")
            print(f"  REF : {ex['reference']}")
            print(f"  HYP : {ex['hypothesis']}")
            print(f"  sBLEU: {ex['sent_bleu']:.1f}  sCHRF++: {ex['sent_chrf']:.1f}")
            print("─" * 80)

        # Aggregate statistics (Prompt 8)
        if sent_bleus:
            mean_sbleu = sum(sent_bleus) / len(sent_bleus)
            median_sbleu = statistics.median(sent_bleus)
            pct_above_20 = sum(1 for b in sent_bleus if b > 20) / len(sent_bleus) * 100

            print(f"\n{'─'*40}")
            print(f"AGGREGATE STATISTICS ({n} examples)")
            print(f"  Mean sentence BLEU   : {mean_sbleu:.2f}")
            print(f"  Median sentence BLEU : {median_sbleu:.2f}")
            print(f"  % above 20 BLEU      : {pct_above_20:.1f}%")
            print(f"  Good (✓ >30 BLEU)    : {sum(1 for e in examples if e['quality']=='✓')}")
            print(f"  Bad  (✗ <10 BLEU)    : {sum(1 for e in examples if e['quality']=='✗')}")
            print(f"{'─'*40}\n")

        # Save as markdown table (Prompt 8)
        if args.qualitative_output:
            md_lines = [
                f"# Qualitative Translation Examples\n",
                f"**Model:** {args.model_type} | **Split:** {args.split} | **Direction:** {args.direction}\n",
                f"**Corpus BLEU-100:** {metrics['bleu']:.2f} | **Corpus CHRF++-100:** {metrics['chrf_pp']:.2f}\n",
                "",
                "| # | Quality | sBLEU | sCHRF++ | Source | Reference | Hypothesis |",
                "|---|---------|-------|---------|--------|-----------|------------|",
            ]
            for i, ex in enumerate(examples, 1):
                md_lines.append(
                    f"| {i} | {ex['quality']} | {ex['sent_bleu']:.1f} | {ex['sent_chrf']:.1f} "
                    f"| {ex['source'][:60]}{'...' if len(ex['source'])>60 else ''} "
                    f"| {ex['reference'][:60]}{'...' if len(ex['reference'])>60 else ''} "
                    f"| {ex['hypothesis'][:60]}{'...' if len(ex['hypothesis'])>60 else ''} |"
                )

            md_lines.extend([
                "",
                f"## Aggregate Statistics",
                f"- Mean sentence BLEU: **{mean_sbleu:.2f}**",
                f"- Median sentence BLEU: **{median_sbleu:.2f}**",
                f"- % above 20 BLEU: **{pct_above_20:.1f}%**",
                f"- Good examples (✓ sBLEU>30): **{sum(1 for e in examples if e['quality']=='✓')}**",
                f"- Bad examples (✗ sBLEU<10): **{sum(1 for e in examples if e['quality']=='✗')}**",
            ])

            os.makedirs(os.path.dirname(args.qualitative_output) or ".", exist_ok=True)
            with open(args.qualitative_output, "w", encoding="utf-8") as f:
                f.write("\n".join(md_lines))
            print(f"Qualitative markdown saved to {args.qualitative_output}")

    # ── Save results ──────────────────────────────────────────────────────────
    if args.output_file:
        results = {
            "metrics": metrics,
            "split": args.split,
            "direction": args.direction,
            "model_type": args.model_type,
            "checkpoint": args.checkpoint,
            "n_pairs": len(eval_src),
            "avg_checkpoints": args.avg_checkpoints,
        }
        if args.qualitative:
            results["examples"] = examples
            results["aggregate"] = {
                "mean_sent_bleu": mean_sbleu,
                "median_sent_bleu": median_sbleu,
                "pct_above_20": pct_above_20,
            }
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
