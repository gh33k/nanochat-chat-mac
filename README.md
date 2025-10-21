# NanoChat Chat Interface for Mac

🍎 **Mac-optimized chat interface for NanoChat models with MPS support**

A focused, lightweight chat interface for running NanoChat models on Mac with Apple Silicon GPU acceleration.

## Features

- **🍎 MPS Support**: Automatic Apple Silicon GPU acceleration via Metal Performance Shaders
- **💬 Gradio Interface**: Modern web-based chat interface with real-time responses
- **🖥️ Local CLI**: Simple command-line chat interface
- **⚡ CPU Fallback**: Automatic fallback to CPU on non-Mac systems
- **✨ Clean Interface**: Professional chat experience without technical tokens

## Quick Start

### 1. Install Dependencies
```bash
uv sync
source .venv/bin/activate
```

### 2. Run Gradio Chat Interface
```bash
python chat_gradio.py --port 8001
```
Open http://localhost:8001 in your browser.

### 3. Run Local CLI Chat
```bash
python chat_local.py
```

### 4. Generate Text (CPU)
```bash
python generate_cpu.py --model-dir ~/.cache/nanochat/chatsft_checkpoints/d20 --prompt "Hello, how are you?"
```

## Requirements

- **Mac with Apple Silicon** (M1/M2/M3) for MPS acceleration
- **Python 3.10+**
- **NanoChat model checkpoint** (download from original nanochat training)

## Attribution

This chat interface is based on [nanochat](https://github.com/karpathy/nanochat) by [Andrej Karpathy](https://github.com/karpathy).

**Original nanochat**: The best ChatGPT that $100 can buy.

## Mac Improvements

- **MPS Device Detection**: Automatically uses Apple Silicon GPU when available
- **Enhanced Gradio Interface**: Proper conversation formatting and Engine usage
- **Special Token Cleanup**: Filters internal tokens from user responses
- **Better Greeting Behavior**: Improved initial responses with system messages

## Files

- `chat_gradio.py` - Web-based chat interface with Gradio
- `chat_local.py` - Command-line chat interface
- `generate_cpu.py` - Simple text generation script
- `nanochat/` - Core NanoChat modules (engine, tokenizer, etc.)

## License

MIT (inherited from original nanochat)
