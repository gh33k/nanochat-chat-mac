#!/usr/bin/env python3
# chat_gradio.py — simple Gradio UI for local NanoChat checkpoints (CPU/MPS)
# Works with your mid/sft checkpoints produced by the training scripts.

import os
import argparse
import torch
import gradio as gr

from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

# ----------------------------
# Model / tokenizer loader
# ----------------------------
def build_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_nanochat(source: str, device, model_tag: str, step: int):
    """
    Uses NanoChat's checkpoint_manager to build the model + tokenizer correctly.
    `source` should be one of: 'base', 'mid', 'sft' (we’ll default to 'sft').
    """
    model, tokenizer, _meta = load_model(
        source=source,
        device=device,
        phase="eval",
        model_tag=model_tag,
        step=step,
    )
    model.eval()
    return model, tokenizer

# ----------------------------
# Text generation
# ----------------------------
@torch.inference_mode()
def generate_reply(engine, tokenizer, history, message, temperature=0.8, top_k=50, max_new_tokens=256):
    """
    history: list[dict(role, content)] from gradio (type="messages")
    message: current user message (string)
    """
    # Get special tokens
    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    
    # Build conversation tokens like chat_cli.py does
    conversation_tokens = [bos]
    
    # Add system message for first interaction to encourage greeting behavior
    if not history:
        system_msg = "You are a helpful AI assistant. When greeted, respond with a friendly greeting and ask how you can help."
        conversation_tokens.append(user_start)
        conversation_tokens.extend(tokenizer.encode(system_msg))
        conversation_tokens.append(user_end)
        conversation_tokens.append(assistant_start)
        conversation_tokens.extend(tokenizer.encode("Hello! I'm NanoChat, your AI assistant. How can I help you today?"))
        conversation_tokens.append(assistant_end)
    
    # Add all previous messages
    for msg in history:
        if msg["role"] == "user":
            conversation_tokens.append(user_start)
            conversation_tokens.extend(tokenizer.encode(msg["content"]))
            conversation_tokens.append(user_end)
        elif msg["role"] == "assistant":
            conversation_tokens.append(assistant_start)
            conversation_tokens.extend(tokenizer.encode(msg["content"]))
            conversation_tokens.append(assistant_end)
    
    # Add current user message
    conversation_tokens.append(user_start)
    conversation_tokens.extend(tokenizer.encode(message))
    conversation_tokens.append(user_end)
    
    # Start assistant response
    conversation_tokens.append(assistant_start)
    
    # Generate using Engine
    response_tokens = []
    for token_column, token_masks in engine.generate(
        conversation_tokens,
        num_samples=1,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    ):
        token = token_column[0]  # Get the token for the single sample
        response_tokens.append(token)
    
    # Decode the response
    reply = tokenizer.decode(response_tokens)
    
    # Clean up special tokens that shouldn't be shown to the user
    reply = reply.replace("<|assistant_end|>", "").replace("<|assistant_start|>", "")
    reply = reply.replace("<|user_start|>", "").replace("<|user_end|>", "")
    reply = reply.replace("<|bos|>", "")
    
    return reply.strip()

# ----------------------------
# CLI + Gradio app
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)

    # Where to load from — use NanoChat's logical sources so it finds ~/.cache/nanochat/...
    parser.add_argument("--source", choices=["sft", "mid", "base"], default="sft")
    parser.add_argument("--model-tag", default="d20")  # your run used d20
    parser.add_argument("--step", type=int, default=650)  # adjust to your checkpoint (e.g. 765 for mid)

    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=256)

    args = parser.parse_args()

    device = build_device()
    print(f"Using device: {device}")

    print(f"Loading model: source={args.source} tag={args.model_tag} step={args.step}")
    model, tokenizer = load_nanochat(args.source, device, args.model_tag, args.step)
    
    # Create Engine for proper generation
    engine = Engine(model, tokenizer)

    def gradio_chat(message, history, temperature, top_k, max_tokens):
        # message: str, history: list[{"role": "...", "content": "..."}]
        return generate_reply(engine, tokenizer, history, message,
                              temperature=temperature, top_k=top_k, max_new_tokens=max_tokens)

    demo = gr.ChatInterface(
        fn=gradio_chat,
        title="NanoChat (local)",
        description="Chat with your NanoChat checkpoint (MPS/CPU).",
        type="messages",  # IMPORTANT: this makes the arg order (message, history, ...)
        additional_inputs=[
            gr.Slider(0.0, 1.5, value=args.temperature, step=0.05, label="Temperature"),
            gr.Slider(1, 200, value=args.top_k, step=1, label="Top-k"),
            gr.Slider(8, 1024, value=args.max_tokens, step=8, label="Max new tokens"),
        ],
    )

    demo.launch(server_name=args.host, server_port=args.port, show_error=True)

if __name__ == "__main__":
    main()