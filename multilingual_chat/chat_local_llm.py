"""
Interactive Chat Console for Pre-trained or Fine-tuned Local Multilingual LLM.

Allows you to converse with the highly-optimized 'Qwen2-0.5B' model locally
on Apple Silicon MPS. Supports all 13 regional Indian languages out-of-the-box,
streaming responses to the terminal using a smooth character typing animation.
"""

import os
import sys
import time
import argparse
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",  default="outputs/local_llm", help="Path to fine-tuned LLM directory")
    p.add_argument("--base_model",  default="Qwen/Qwen2-0.5B-Instruct", help="Default base model fallback")
    p.add_argument("--temp",        type=float, default=0.7, help="Generation temperature")
    p.add_argument("--top_p",       type=float, default=0.9, help="Top-P nucleus sampling")
    p.add_argument("--max_tokens",  type=int, default=200, help="Maximum response tokens")
    return p.parse_args()


def print_banner(model_name: str):
    banner = f"""
========================================================================
🤖 LOCAL LLM INTERACTIVE CHATBOT ({model_name})
========================================================================
  Running locally on Apple Silicon MPS (Metal Performance Shaders) GPU!
  
  Supporting 13 regional Indian languages:
  🇮🇳 Hindi    🇮🇳 Marathi   🇮🇳 Tamil      🇮🇳 Telugu
  🇮🇳 Urdu     🇮🇳 Odia      🇮🇳 Punjabi    🇮🇳 Malayalam
  🇮🇳 Maithili 🇮🇳 Gujarati  🇮🇳 Assamese   🇮🇳 Bengali
  🇮🇳 Bhojpuri
  
  Type a message, and the pre-trained/fine-tuned local model will
  reply back creatively and natively in that same language!
  
  (Type 'exit', 'quit', or 'bye' to end the conversation)
========================================================================
"""
    print(banner)


def main():
    args = parse_args()

    # Device Dispatch (Apple Silicon GPU)
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Determine which model to load (fine-tuned vs base model fallback)
    if os.path.exists(args.model_path) and os.path.exists(os.path.join(args.model_path, "config.json")):
        load_path = args.model_path
        model_display_name = "Fine-Tuned Qwen2-0.5B"
    else:
        load_path = args.base_model
        model_display_name = "Base Qwen2-0.5B-Instruct"

    print(f"\nInitializing model and tokenizer ({model_display_name}) on device: {device}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            load_path,
            trust_remote_code=True,
            torch_dtype=torch.float32  # Standard Float32 for maximum compatibility on MPS
        ).to(device)
    except Exception as e:
        print(f"\nError loading model: {e}")
        print(f"Could not load pre-trained model '{load_path}'.")
        print("Please check your internet connection for the first download from Hugging Face.")
        sys.exit(1)

    print_banner(model_display_name)

    # Keep conversation history to support multi-turn dialogues
    history = [{"role": "system", "content": "You are a helpful and polite conversational assistant. Speak fluently in regional Indian languages."}]

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

        # Append user turn to history
        history.append({"role": "user", "content": cleaned_input})

        # Manual Qwen2 multi-turn chat template formatting to avoid Jinja2 compatibility bugs!
        prompt_text = ""
        for turn in history:
            role = turn["role"]
            content = turn["content"]
            prompt_text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        prompt_text += "<|im_start|>assistant\n"
        
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

        # Autoregressive generation
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=args.max_tokens,
                temperature=args.temp,
                top_p=args.top_p,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )

        # Extract only the newly generated response tokens
        new_tokens = output_ids[0, inputs.input_ids.shape[1]:]
        reply_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Append assistant turn to history to maintain conversational memory
        history.append({"role": "assistant", "content": reply_text})

        # Keep history compact to avoid exceeding max sequence length
        if len(history) > 7:
            history = [history[0]] + history[-6:]

        # Stream response text with character-by-character typing delay
        print("Chatbot ➔ ", end="", flush=True)
        for char in reply_text:
            print(char, end="", flush=True)
            time.sleep(0.012)
        print()


if __name__ == "__main__":
    main()
