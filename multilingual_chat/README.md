# 💬 Multilingual Causal Chat GPT — 13 Indian Languages

This directory contains a completely self-contained, isolated **Decoder-Only Causal Conversational GPT Model** (Chatbot) trained jointly on **13 regional Indian languages**. 

If a user writes a prompt in *any* of the supported languages, the model understands the semantic context and generates a natural conversational reply back in that same language.

---

## 🇮🇳 Supported Languages

| Language | Code | Script |
|---|---|---|
| **Hindi** | `hi` | Devanagari |
| **Marathi** | `mr` | Devanagari |
| **Tamil** | `ta` | Tamil |
| **Telugu** | `te` | Telugu |
| **Urdu** | `ur` | Perso-Arabic (Nastaliq) |
| **Odia** | `or` | Odia |
| **Punjabi** | `pa` | Gurmukhi |
| **Malayalam** | `ml` | Malayalam |
| **Maithili** | `mai` | Devanagari |
| **Gujarati** | `gu` | Gujarati |
| **Assamese** | `as` | Bengali-Assamese |
| **Bengali** | `bn` | Bengali-Assamese |
| **Bhojpuri** | `bho` | Devanagari |

---

## 🧠 Architectural Specifications

The chatbot employs a cutting-edge, custom-built **Decoder-only Causal GPT Transformer** incorporating state-of-the-art parameters for convergence and generation quality:

| Component | Design Choice | Technical Justification |
|---|---|---|
| **RMSNorm** | `RMSNorm(dim)` | Faster normalization path by eliminating mean-centering. |
| **RoPE** | Rotary Position Embeddings | Achieves much better relative token sequence coherence over long conversation turns than abs. position embeddings. |
| **GQA** | Grouped-Query Attention (Ratio=3) | Combines multi-head query attention with grouped key-value heads to optimize live text streaming speed. |
| **SwiGLU** | Gated Swish Feed-Forward | Outperforms standard GELU/ReLU activations in causal text continuation. |
| **Prompt Masking** | Target Masking with `-100` | Ignores the user's prompt tokens in CrossEntropy calculation, ensuring backpropagation focuses strictly on model-generated replies. |
| **Interactive Sampling** | Top-K + Top-P Nucleus | Generates creative, fluid regional text without falling into repetitive generation loops. |

---

## 📁 Repository Structure

```
multilingual_chat/
├── README.md                      # 📖 Setup and execution documentation
├── requirements.txt               # 📦 PyTorch & SentencePiece dependencies
├── train_chat_gpt.py              # 🚀 Causal Pretraining & Conversational Prompt-loss Training
├── chat.py                        # 💬 Live Interactive Command-line Chatbot Terminal
├── data/
│   ├── __init__.py
│   └── chat_dataset.py            # 🧹 Loads dialogue corpora, BPE tokenizer, masks prompts
├── model/
│   ├── __init__.py
│   ├── gpt.py                     # 🧠 Decoder-only Conversational GPT (with top-k/top-p sampling)
│   └── blocks.py                  # ⚡ Custom blocks (RMSNorm, RoPE, GQA, SwiGLU FFN)
└── utils/
    ├── __init__.py
    └── training_utils.py          # 🛠️ LR schedulers, metric trackers, plotting helpers
```

---

## ⚡ Setup & Execution Instructions

Run these exact command pipelines from your terminal:

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Conversational Dialogue Corpus
Ensure your regional text files (e.g. `train.hi`, `train.mr`, `train.ur`, `train.ta`, etc.) are placed inside the `Dataset/` directory in your workspace root.

*The script will automatically execute dual-channel dialogue data loading: monolingual conversational continuation (sentence N $\rightarrow$ sentence N+1) and cross-lingual conversational turns.*

### 3. Train the Conversational GPT Model
Train a lightweight **~10M parameter small variant** (highly optimized for rapid convergence and local live verification) or the **~124M parameter full variant**:

```bash
# Train small model (Highly recommended for testing and rapid training)
python3 train_chat_gpt.py --scale small --data_dir ../Dataset --epochs 10 --batch_size 32

# Train full model (Requires longer training times)
python3 train_chat_gpt.py --scale full --data_dir ../Dataset --epochs 10 --batch_size 16 --grad_accum 2
```

At startup, the training script:
* Automatically creates target loss masks ignoring user turns.
* Jointly trains a **SentencePiece conversational tokenizer** on all 13 active languages.
* Saves `config.json` inside output folder.
* Outputs training loss curves (`conversational_gpt_loss_curves.png`) and perplexity curves (`conversational_gpt_ppl_curves.png`).

### 4. Live Chat Terminal (Autoregressive Response Streaming)
Once training is complete, start the real-time interactive chatbot session:

```bash
python3 chat.py --checkpoint outputs/chat/checkpoint_best.pt --spm_model outputs/chat/spm_chat_8000.model --temp 0.7
```

*   Type your prompt in **Hindi, Urdu, Tamil, Marathi, Punjabi**, or any of the 13 supported languages.
*   The model will analyze the prompt and reply back in that same language, streamed to your console with an immersive, live **typing animation**!
*   Type `exit`, `quit`, or `bye` to end the conversation.
