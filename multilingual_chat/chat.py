"""
Interactive Terminal Chatbot for Multilingual Conversational GPT.

Allows users to type queries in any of the 13 regional Indian languages
(e.g., Hindi, Urdu, Tamil, Bengali) and receive creative, context-aware
responses in real time with a streaming typing effect.
"""

import os
import sys
import time
import argparse
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data.chat_dataset import SPChatTokenizer
from model.gpt import MultilingualGPT


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  default="outputs/chat/checkpoint_best.pt")
    p.add_argument("--spm_model",   default="outputs/chat/spm_chat_2000.model")
    p.add_argument("--temp",        type=float, default=0.7, help="Generation temperature")
    p.add_argument("--top_k",       type=int, default=50, help="Top-K sampling limit")
    p.add_argument("--top_p",       type=float, default=0.9, help="Top-P nucleus sampling threshold")
    p.add_argument("--max_tokens",  type=int, default=150, help="Maximum new tokens to generate")
    return p.parse_args()


def print_banner():
    banner = """
========================================================================
💬 MULTILINGUAL CONVERSATIONAL GPT CHATBOT
========================================================================
  An isolated, self-contained Causal Dialogue Agent supporting 
  13 regional Indian languages:
  
  🇮🇳 Hindi    🇮🇳 Marathi   🇮🇳 Tamil      🇮🇳 Telugu
  🇮🇳 Urdu     🇮🇳 Odia      🇮🇳 Punjabi    🇮🇳 Malayalam
  🇮🇳 Maithili 🇮🇳 Gujarati  🇮🇳 Assamese   🇮🇳 Bengali
  🇮🇳 Bhojpuri
  
  Type a message in any of these languages, and the model will 
  understand and reply back in that same language!
  
  (Type 'exit', 'quit', or 'bye' to end the conversation)
========================================================================
"""
    print(banner)


def main():
    args = parse_args()

    # Device Dispatch (MPS > CUDA > CPU)
    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if not os.path.exists(args.checkpoint):
        print(f"Error: Trained checkpoint not found at: {args.checkpoint}")
        print("Please train your conversational GPT model first using 'train_chat_gpt.py'.")
        sys.exit(1)

    if not os.path.exists(args.spm_model):
        print(f"Error: SentencePiece conversational model not found at: {args.spm_model}")
        sys.exit(1)

    print("Loading tokenizer...")
    tokenizer = SPChatTokenizer(args.spm_model)

    print(f"Loading trained conversational checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    
    # Reconstruct architecture configuration from checkpoint
    model = MultilingualGPT(ckpt["cfg"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print_banner()

    while True:
        try:
            user_input = input("\nYou ➔ ")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        cleaned_input = user_input.strip()
        if not cleaned_input:
            continue

        if cleaned_input.lower() in ("exit", "quit", "bye", "बाय"):
            print("Chatbot ➔ अलविदा! फिर मिलेंगे। (Goodbye! See you again.)")
            break

        # Encode user prompt
        u_tokens = tokenizer.encode(cleaned_input)

        # Construct conversational turn sequence:
        # [BOS, USER_ID] + prompt_tokens + [END_TURN_ID, ASSISTANT_ID]
        prompt_seq = (
            [tokenizer.BOS_ID, tokenizer.USER_ID]
            + u_tokens
            + [tokenizer.END_TURN_ID, tokenizer.ASSISTANT_ID]
        )

        # Autoregressive generation
        response_tokens = model.generate(
            prompt_seq=prompt_seq,
            max_new_tokens=args.max_tokens,
            temperature=args.temp,
            top_k=args.top_k,
            top_p=args.top_p,
            eos_id=tokenizer.EOS_ID,
        )

        # Isolate the newly generated assistant tokens (anything after our prompt prefix!)
        assistant_tokens = response_tokens[len(prompt_seq):]

        # Decode tokens back to readable text
        reply_text = tokenizer.decode(assistant_tokens)

        # Stream generated text with typing animation
        print("Chatbot ➔ ", end="", flush=True)
        for char in reply_text:
            print(char, end="", flush=True)
            time.sleep(0.015)  # Interactive typing delay
        print()


if __name__ == "__main__":
    main()
