"""
Fine-tuning Pipeline for State-of-the-Art Local Multilingual LLM.

Uses:
    - Model: 'Qwen/Qwen2-0.5B-Instruct' (highly optimized for Asian/Indian languages,
             occupies ~1.0 GB VRAM, making it exceptionally fast on 16GB Macs).
    - Acceleration: PyTorch MPS (Metal Performance Shaders) for Apple Silicon.
    - Dataset: Dual-channel conversational dialogue formatting with prompt-loss masking.
"""

import os
import sys
import argparse
import math
import random
import torch
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Dict, Any

from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from train_chat_gpt import load_conversational_data


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id",      default="Qwen/Qwen2-0.5B-Instruct")
    p.add_argument("--data_dir",      default="../Dataset")
    p.add_argument("--output_dir",    default="outputs/local_llm")
    p.add_argument("--epochs",        type=int, default=3)
    p.add_argument("--batch_size",    type=int, default=8)
    p.add_argument("--lr",            type=float, default=2e-5)
    p.add_argument("--max_len",       type=int, default=256)
    p.add_argument("--clip",          type=float, default=1.0)
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--max_samples",   type=int, default=500)
    return p.parse_args()


class LLMChatDataset(torch.utils.data.Dataset):
    """
    Hugging Face Dataset wrapping conversational templates for Qwen2.
    
    Qwen2 Chat Template:
        <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
        <|im_start|>user\n{prompt}<|im_end|>\n
        <|im_start|>assistant\n{reply}<|im_end|>\n
    """
    def __init__(self, dialogues: List[Tuple[str, str]], tokenizer, max_len: int = 256):
        self.input_ids = []
        self.labels = []

        for prompt, reply in dialogues:
            # Manual Qwen2 Chat Template Formatting to avoid Jinja2 compatibility bugs!
            system_part = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            user_part = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            assistant_part = f"{reply}<|im_end|>\n"
            
            prompt_text = system_part + user_part
            full_text = prompt_text + assistant_part
            
            prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
            full_tokens = tokenizer.encode(full_text, add_special_tokens=False)

            if len(full_tokens) > max_len:
                full_tokens = full_tokens[:max_len]

            # Shifted labels: mask user prompt with -100 so LLM only learns assistant replies
            prompt_len = min(len(prompt_tokens), len(full_tokens))
            label = [-100] * prompt_len + full_tokens[prompt_len:]

            self.input_ids.append(full_tokens)
            self.labels.append(label)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class LLMCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        labels = [item["labels"] for item in batch]

        max_len = max(len(x) for x in input_ids)
        B = len(input_ids)

        padded_inputs = torch.full((B, max_len), self.pad_token_id, dtype=torch.long)
        padded_labels = torch.full((B, max_len), -100, dtype=torch.long)

        for i, (inp, lab) in enumerate(zip(input_ids, labels)):
            padded_inputs[i, :len(inp)] = inp
            padded_labels[i, :len(lab)] = lab

        return {
            "input_ids": padded_inputs,
            "attention_mask": padded_inputs.ne(self.pad_token_id).long(),
            "labels": padded_labels
        }


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Device Dispatch (Apple Silicon GPU)
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"\nTraining Local LLM on device: {device}")
    print(f"Loading pre-trained model: {args.model_id}...")

    # 2. Load Local Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=torch.float32  # Standard Float32 for maximum compatibility on MPS
    ).to(device)

    print("Model loaded successfully!")

    # 3. Load Conversational Dataset
    try:
        dialogues = load_conversational_data(args.data_dir, seed=args.seed)
    except Exception as e:
        print(f"Error loading conversational data: {e}")
        print("Please ensure your parallel data files are inside the 'Dataset' folder.")
        sys.exit(1)

    if args.max_samples and len(dialogues) > args.max_samples:
        dialogues = dialogues[:args.max_samples]
        print(f"Subsampled to {len(dialogues):,} dialogue turns")

    n_val = int(len(dialogues) * 0.10)
    train_turns = dialogues[n_val:]
    val_turns = dialogues[:n_val] if n_val > 0 else dialogues

    train_ds = LLMChatDataset(train_turns, tokenizer, args.max_len)
    val_ds = LLMChatDataset(val_turns, tokenizer, args.max_len)

    collator = LLMCollator(tokenizer.pad_token_id)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    # 4. Optimizer and Training Loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"Starting Fine-Tuning of {args.model_id}...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        total_steps = 0

        loader = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            epoch_loss += loss.item()
            total_steps += 1
            loader.set_postfix(loss=f"{loss.item():.4f}", ppl=f"{math.exp(min(loss.item(), 20)):.2f}")

        # Compute Validation Loss
        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
                val_steps += 1

        avg_train_loss = epoch_loss / max(total_steps, 1)
        avg_val_loss = val_loss / max(val_steps, 1)

        print(f"\nEpoch {epoch+1} Complete | Train Loss: {avg_train_loss:.4f} (PPL: {math.exp(min(avg_train_loss, 20)):.2f}) | Val Loss: {avg_val_loss:.4f} (PPL: {math.exp(min(avg_val_loss, 20)):.2f})")

    # 5. Save Fine-Tuned Model locally
    print(f"\nSaving fine-tuned LLM locally to: {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Fine-tuning completed successfully! Your local LLM is now highly specialized.")


if __name__ == "__main__":
    main()
