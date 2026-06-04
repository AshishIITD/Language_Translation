# AdiVaani NMT Assignment — Hindi ↔ Marathi Translation

**MISN Lab, IIT Delhi | AdiVaani Initiative**

This repository contains a complete Neural Machine Translation system for Hindi↔Marathi
translation, implementing both classical (LSTM-based) and modern (Transformer-based) approaches.

---

## Hardware Used for Training

| Component | Specification |
|---|---|
| CPU | Apple Silicon (M-series) |
| RAM | 16 GB |
| GPU | Apple MPS (Metal Performance Shaders) |
| OS | macOS |

> **Note on MPS:** Apple Silicon's MPS backend does not support fp16/AMP well.
> All training is done in fp32. Batch sizes are set to take advantage of the
> unified memory architecture (larger than typical CUDA GPU splits).

---

## LLM Usage Disclosure

As required by the assignment instructions: **Claude (Anthropic, claude-sonnet-4-6)** was used
for code assistance in this project. Specifically:
- Initial scaffolding of the training loop structure
- Debugging suggestions during development

All architectural decisions, experimental choices, hyperparameter selections, and written
analysis are the author's own work and can be independently defended.

---

## Repository Structure

```
.
├── Dataset/                # 📂 Parallel Hindi-Marathi corpus (train/test splits)
├── README.md               # 📖 Setup & execution documentation
├── requirements.txt        # 📦 PyTorch 2.0+ & sacrebleu dependencies
├── train_part1.py          # 🚀 Part I: LSTM training with Curriculum Learning
├── train_part2_pretrain.py # 🚀 Part II: Pretraining BERT (MLM) & GPT (CLM)
├── train_part2_finetune.py # 🚀 Part II: Seq2Seq Fine-Tuning (Phase 1 & 2)
├── evaluate.py             # 📊 Standalone evaluation + Checkpoint Averaging
├── data/
│   └── dataset.py          # 🧹 Copy-pair & Alphanumeric character cleaning
├── part1/
│   ├── model.py            # 🧠 LSTM Seq2Seq model with Bahdanau attention
│   └── bert_embeddings.py  # 🕸️ BERT embedding extraction & projection
├── part2/
│   ├── models.py           # 🧠 Transformers (BERT, GPT, Seq2SeqTransformer)
│   └── transformer_blocks.py # ⚡ Custom blocks (RMSNorm, RoPE, GQA, SwiGLU FFN)
├── evaluation/
│   └── metrics.py          # 📐 SacreBLEU & CHRF++ metric computations
├── utils/
│   └── training_utils.py   # 🛠️ Label smoothing, checkpoint averaging, plotting
├── scripts/
│   ├── prepare_data.py     # 🔧 Data prep + tokenizer training
│   └── plot_comparisons.py # 📊 Cross-run comparison plots
└── configs/
    └── __init__.py         # 📁 Folder for serializing custom configurations
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

**Key packages:** `torch>=2.0`, `transformers>=4.35`, `sentencepiece>=0.1.99`, `sacrebleu>=2.3.1`, `tqdm>=4.65`

### 2. Download and place the corpus

Download the Hindi-Marathi parallel corpus from the assignment link and place it in `data/corpus/`.

Expected file structure (any of these naming conventions is supported):
```
data/corpus/
├── hi.txt      # Hindi sentences (one per line)
└── mr.txt      # Marathi sentences (one per line)
```
Or: `train.hi / train.mr`, or a `.tsv` file with two tab-separated columns.

### 3. Prepare data + train tokenizer

```bash
python scripts/prepare_data.py --data_dir data/corpus --output_dir outputs/part1 --vocab_size 8000
```

This:
- Cleans and filters the corpus (min 3 tokens, max 150 tokens, max length ratio 3.0, copy pair removal, numeric/Latin filtering)
- Trains a shared SentencePiece BPE tokenizer (8000 vocab, shared Hindi+Marathi)
- Saves train/val/test splits (90/5/5)
- Reports UNK rate and corpus statistics

---

## Part I: LSTM Seq2Seq + Bahdanau Attention

### Architecture

| Component | Design | Justification |
|---|---|---|
| Encoder | 2-layer Bidirectional LSTM | Captures both left and right context for each source token |
| Decoder | 2-layer LSTM + Bahdanau attention + input feeding | Additive attention is more expressive than dot-product for varied hidden dims |
| Embeddings | 256-dim, optionally initialized from BERT | Shared SP vocabulary enables embedding weight reuse across encoder/decoder |
| Decoding | Greedy (train) + Beam search k=4 (eval) | Beam search improves BLEU ~1-2 points at negligible cost |
| Regularization | Dropout 0.3, label smoothing ε=0.1 | Label smoothing prevents overconfidence, especially with noisy parallel data |
| Optimization | AdamW, LR=3e-4, cosine decay with warmup | Warmup stabilizes early training; cosine prevents sharp LR drops |
| TF scheduling | Teacher forcing linearly decayed 1.0→0.5 | Prevents exposure bias — model learns to follow its own predictions |
| Curriculum | Length-based, first 30% of epochs use short pairs (≤40 tokens) | Easy-to-hard ordering accelerates convergence |

### Experiment 1: Random Embeddings (baseline)

```bash
python train_part1.py --embedding_type random --direction hi2mr --batch_size 64 --epochs 30
```

### Experiment 2: BERT Embeddings

```bash
python train_part1.py --embedding_type bert --direction hi2mr --batch_size 32 --freeze_emb --epochs 30
```

> **First run** with `--embedding_type bert` will download Hindi-BERT and Marathi-BERT
> from HuggingFace (~400MB each) and cache projected embeddings locally.

### Evaluate Part I

```bash
python evaluate.py \
    --model_type lstm \
    --checkpoint outputs/part1/hi2mr_random/checkpoint_best.pt \
    --spm_model outputs/part1/spm_8000.model \
    --data_dir data/corpus \
    --split test \
    --beam_size 4 \
    --qualitative \
    --n_examples 20 \
    --qualitative_output outputs/part1/qualitative_results.md

# With checkpoint averaging (improves BLEU 0.5-1.5 points):
python evaluate.py \
    --model_type lstm \
    --checkpoint outputs/part1/hi2mr_random/checkpoint_best.pt \
    --spm_model outputs/part1/spm_8000.model \
    --data_dir data/corpus \
    --split test \
    --avg_checkpoints
```

---

## Part II: Language Model Pretraining + Translation

### Architecture

Both models share these modern design choices:

| Modification | Implementation | Justification |
|---|---|---|
| **RMSNorm** | `RMSNorm(dim)` | Removes mean-centering (unnecessary), ~10-15% faster, used in LLaMA/Gemma |
| **RoPE** | `precompute_rope_freqs` + `apply_rope` | Relative position encoding — better length generalization than learned abs. pos |
| **GQA** | n_heads=6/12, n_kv_heads=2/4 | Reduces KV cache memory by n_rep×, minimal quality loss over MHA |
| **SwiGLU FFN** | W1(SiLU) ⊙ W3, projected by W2 | Gated activation outperforms ReLU/GELU in practice (PaLM, LLaMA) |
| **Pre-norm** | RMSNorm before each sublayer | Better gradient flow; trains without instability even without warmup |

### Step 1: Pretrain BERT (~110M full)

```bash
python train_part2_pretrain.py --model bert --scale full --batch_size 32 --grad_accum 2 --epochs 20
```

### Step 2: Pretrain GPT (~124M full)

```bash
python train_part2_pretrain.py --model gpt --scale full --batch_size 32 --grad_accum 2 --epochs 20
```

### Step 3: Fine-tune for Translation (Phase 1 — cross-attention only)

```bash
python train_part2_finetune.py --phase 1 --scale full --batch_size 32 --grad_accum 2 \
    --bert_ckpt outputs/part2/bert_full/checkpoint_best.pt \
    --gpt_ckpt  outputs/part2/gpt_full/checkpoint_best.pt
```

### Step 4: Fine-tune for Translation (Phase 2 — full model)

```bash
python train_part2_finetune.py --phase 2 --scale full --batch_size 16 --grad_accum 4 --lr 5e-5 \
    --bert_ckpt outputs/part2/bert_full/checkpoint_best.pt \
    --gpt_ckpt  outputs/part2/gpt_full/checkpoint_best.pt \
    --resume    outputs/part2/finetune_full_hi2mr_phase1/checkpoint_best.pt
```

### Evaluate Part II

```bash
python evaluate.py \
    --model_type transformer \
    --checkpoint outputs/part2/finetune_full_hi2mr_phase2/checkpoint_best.pt \
    --bert_ckpt  outputs/part2/bert_full/checkpoint_best.pt \
    --gpt_ckpt   outputs/part2/gpt_full/checkpoint_best.pt \
    --scale full \
    --spm_model  outputs/part1/spm_8000.model \
    --data_dir   data/corpus \
    --split test \
    --qualitative --n_examples 20 \
    --qualitative_output outputs/part2/qualitative_results.md
```

---

## New Features in This Version

### Curriculum Learning (Part I)
- Enabled by default (`--curriculum`)
- For the first 30% of training epochs, uses only sentence pairs where max(src_len, tgt_len) ≤ 40 tokens
- After that, uses the full dataset
- Easy-to-hard ordering typically accelerates convergence

### Checkpoint Averaging
- Use `--avg_checkpoints` in `evaluate.py` to average the last 5 epoch checkpoints
- Typically improves BLEU by 0.5–1.5 points by smoothing SGD noise
- Requires epoch-numbered checkpoints (saved automatically during training)

### Perplexity Logging (Part II)
- Perplexity is computed and logged every epoch during pretraining
- A separate perplexity plot is generated alongside loss curves
- Important for the report — perplexity is the standard LM evaluation metric

### Gradient Norm Logging
- All training scripts log gradient norms after clipping
- Mean gradient norm printed every 100 steps and per-epoch
- Important for "training stability" analysis (e.g., BERT vs random embedding gradient comparison)

### Report-Quality Qualitative Output
- `--qualitative` now computes both sentence BLEU AND CHRF++
- Flags good (✓ sBLEU>30) and bad (✗ sBLEU<10) examples
- Prints aggregate statistics: mean/median sentence BLEU, % above 20
- `--qualitative_output PATH` saves a markdown table for direct report inclusion

### Config Saving
- Every training run saves `config.json` in the output directory
- Contains the full argparse namespace for reproducibility

---

## Generate Comparison Plots

After all runs are complete:

```bash
python scripts/plot_comparisons.py --output_dir outputs/
```

Generates:
- `comparison_part1.png` — Random vs BERT embeddings (val loss, BLEU, CHRF++)
- `comparison_part1_vs_part2.png` — LSTM vs Transformer comparison
- Terminal table of all test results

---

## Recommended Run Order on Apple Silicon (MPS)

```
1. scripts/prepare_data.py                              # ~2 min
2. train_part1.py (random, hi2mr, batch_size=64)        # ~2-3 hours
3. train_part1.py (bert, hi2mr, batch_size=32)          # ~3-4 hours (BERT download first time)
4. train_part2_pretrain.py (bert, full, batch_size=32)  # ~4-6 hours
5. train_part2_pretrain.py (gpt, full, batch_size=32)   # ~4-6 hours
6. train_part2_finetune.py (phase 1, batch_size=32)     # ~2-3 hours
7. train_part2_finetune.py (phase 2, batch_size=16)     # ~3-4 hours
8. scripts/plot_comparisons.py                           # ~1 min
```

> Total estimated time on Apple Silicon: **~20-30 hours**

---

## Outputs

All outputs are saved under `outputs/`:

```
outputs/
├── part1/
│   ├── spm_8000.model                    # Shared tokenizer
│   ├── hi2mr_random/
│   │   ├── config.json                   # Experiment config
│   │   ├── checkpoint_best.pt
│   │   ├── checkpoint_epoch*.pt          # Per-epoch (for averaging)
│   │   ├── test_results.json
│   │   └── plots/
│   │       ├── hi2mr_random_loss_curves.png
│   │       ├── hi2mr_random_bleu_curves.png
│   │       └── hi2mr_random_chrf_curves.png
│   ├── hi2mr_bert_frozen/  ...
│   └── qualitative_results.md            # If --qualitative_output used
├── part2/
│   ├── bert_full/
│   │   ├── config.json
│   │   ├── checkpoint_best.pt
│   │   └── plots/
│   │       ├── loss_curves.png
│   │       └── ppl_curves.png            # Perplexity plot
│   ├── gpt_full/  ...
│   └── finetune_full_hi2mr_phase2/
│       ├── config.json
│       ├── checkpoint_best.pt
│       ├── test_results.json
│       └── plots/  ...
├── comparison_part1.png
└── comparison_part1_vs_part2.png
```

---

## References

1. Bahdanau et al., *Neural Machine Translation by Jointly Learning to Align and Translate*, ICLR 2015
2. Vaswani et al., *Attention Is All You Need*, NeurIPS 2017
3. Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers*, NAACL 2019
4. Radford et al., *Language Models are Unsupervised Multitask Learners*, OpenAI 2019
5. Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021
6. Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models*, EMNLP 2023
7. Zhang & Sennrich, *Root Mean Square Layer Normalization*, NeurIPS 2019
8. Jozefowicz et al., *An Empirical Evaluation of Recurrent Network Architectures*, ICML 2015
9. Press & Wolf, *Using the Output Embedding to Improve Language Models*, EACL 2017
10. Liu et al., *RoBERTa: A Robustly Optimized BERT Pretraining Approach*, 2019
11. Wu et al., *Google's Neural Machine Translation System*, 2016 (length penalty, checkpoint averaging)
