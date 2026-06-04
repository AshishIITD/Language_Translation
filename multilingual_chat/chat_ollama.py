"""
Ollama-Powered Interactive Chatbot for 13 Indian Languages.

Integrates with Ollama (http://localhost:11434) to run local, highly optimized
Qwen2 models (0.5B, 1.5B, or 7B) at blazing-fast inference speeds on Apple Silicon.
Supports streaming responses, automatic model pulling, and full conversation history.
"""

import os
import sys
import time
import json
import argparse
import urllib.request
import urllib.error


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",    default="qwen2:0.5b", help="Ollama model name (e.g. qwen2:0.5b, qwen2:1.5b)")
    p.add_argument("--url",      default="http://localhost:11434", help="Ollama local host url")
    p.add_argument("--temp",     type=float, default=0.7, help="Generation temperature")
    p.add_argument("--max_len",  type=int, default=200, help="Maximum response tokens")
    return p.parse_args()


def print_banner(model_name: str, url: str):
    banner = f"""
========================================================================
🦙 OLLAMA-POWERED MULTILINGUAL CHATBOT ({model_name})
========================================================================
  Running on top-tier llama.cpp local inference engine!
  Connection: {url}
  
  Supporting 13 regional Indian languages:
  🇮🇳 Hindi    🇮🇳 Marathi   🇮🇳 Tamil      🇮🇳 Telugu
  🇮🇳 Urdu     🇮🇳 Odia      🇮🇳 Punjabi    🇮🇳 Malayalam
  🇮🇳 Maithili 🇮🇳 Gujarati  🇮🇳 Assamese   🇮🇳 Bengali
  🇮🇳 Bhojpuri
  
  Type a message, and Ollama will generate contextually accurate,
  highly fluid responses in that same language at blazing speeds!
  
  (Type 'exit', 'quit', or 'bye' to end the conversation)
========================================================================
"""
    print(banner)


def check_ollama_running(url: str) -> bool:
    """Check if the local Ollama service is active."""
    try:
        urllib.request.urlopen(url, timeout=3)
        return True
    except Exception:
        return False


def pull_ollama_model(url: str, model_name: str):
    """Proactively request Ollama to pull/download the specified model."""
    print(f"\nVerifying/Pulling model '{model_name}' from Ollama registry...")
    pull_url = f"{url}/api/pull"
    data = json.dumps({"name": model_name, "stream": False}).encode("utf-8")
    
    req = urllib.request.Request(
        pull_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            res = json.loads(response.read().decode("utf-8"))
            if res.get("status") == "success":
                print("Model is verified and ready to run!")
            else:
                print(f"Status: {res.get('status')}")
    except urllib.error.URLError as e:
        print(f"Error pulling model: {e.reason}")
        print("Please check that Ollama is currently running and you have internet access.")
        sys.exit(1)


def main():
    args = parse_args()

    # 1. Verify Ollama Connection
    if not check_ollama_running(args.url):
        print(f"\nError: Could not connect to Ollama local host at {args.url}")
        print("Please ensure Ollama is installed and running on your Mac.")
        print("Download link: https://ollama.com/")
        sys.exit(1)

    # 2. Pull model if not already cached
    pull_ollama_model(args.url, args.model)

    print_banner(args.model, args.url)

    # 3. Conversational History
    history = [
        {"role": "system", "content": "You are a helpful and polite conversational assistant. Speak fluently in regional Indian languages."}
    ]

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

        # Append turn to history
        history.append({"role": "user", "content": cleaned_input})

        # 4. Stream response using Ollama API
        chat_url = f"{args.url}/api/chat"
        payload = {
            "model": args.model,
            "messages": history,
            "stream": True,
            "options": {
                "temperature": args.temp,
                "num_predict": args.max_len
            }
        }
        data = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            chat_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        print("Chatbot ➔ ", end="", flush=True)
        assistant_reply = ""

        try:
            with urllib.request.urlopen(req) as response:
                for line in response:
                    if line:
                        chunk = json.loads(line.decode("utf-8"))
                        message = chunk.get("message", {})
                        content = message.get("content", "")
                        
                        assistant_reply += content
                        # Live print the character streaming
                        for char in content:
                            print(char, end="", flush=True)
                            time.sleep(0.005)  # Immersive streaming delay
            print()
        except Exception as e:
            print(f"\nError communicating with Ollama: {e}")
            break

        # Append assistant turn to conversational memory
        history.append({"role": "assistant", "content": assistant_reply})

        # Compact history to avoid memory bloat
        if len(history) > 9:
            history = [history[0]] + history[-8:]


if __name__ == "__main__":
    main()
