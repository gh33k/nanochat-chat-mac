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
- **NanoChat model checkpoint** (see Model Setup below)

## Model Setup

### Download Pre-trained Models

NanoChat provides several pre-trained models. You can download them from:

1. **Official NanoChat Models**: Visit [nanochat.karpathy.ai](https://nanochat.karpathy.ai/) for available models
2. **GitHub Releases**: Check the [original nanochat repository](https://github.com/karpathy/nanochat) for model downloads
3. **Hugging Face**: Some models may be available on Hugging Face Hub

### Model Placement

Place your downloaded model checkpoints in the following directory structure:

```
~/.cache/nanochat/chatsft_checkpoints/
├── d20/                    # Example: d20 model (20 layers)
│   ├── model_0.pt         # Model weights
│   ├── meta.json          # Model metadata
│   └── tokenizer.json     # Tokenizer configuration
└── d32/                    # Example: d32 model (32 layers)
    ├── model_0.pt
    ├── meta.json
    └── tokenizer.json
```

### Model Directory Structure

Each model should contain:
- `model_*.pt` - PyTorch model weights
- `meta.json` - Model configuration and metadata
- `tokenizer.json` - Tokenizer configuration (if using custom tokenizer)

### Example Model Paths

- **d20 model**: `~/.cache/nanochat/chatsft_checkpoints/d20/`
- **d32 model**: `~/.cache/nanochat/chatsft_checkpoints/d32/`
- **Custom model**: `~/.cache/nanochat/chatsft_checkpoints/my_model/`

## Troubleshooting

### Model Not Found
If you get "Model not found" errors:
1. Check that your model directory exists: `ls ~/.cache/nanochat/chatsft_checkpoints/`
2. Verify the model files are present: `ls ~/.cache/nanochat/chatsft_checkpoints/d20/`
3. Ensure you're using the correct model path in commands

### MPS Not Available
If MPS acceleration isn't working:
1. Verify you're on Apple Silicon: `uname -m` should show `arm64`
2. Check PyTorch MPS support: `python -c "import torch; print(torch.backends.mps.is_available())"`
3. The system will automatically fall back to CPU if MPS isn't available

### Memory Issues
For large models on Mac:
1. Close other applications to free up memory
2. Use smaller models (d20 instead of d32) if you have limited RAM
3. Consider using CPU-only mode if GPU memory is insufficient

## Attribution

This chat interface is based on [nanochat](https://github.com/karpathy/nanochat) by [Andrej Karpathy](https://github.com/karpathy).

**Original nanochat**: The best ChatGPT that $100 can buy.

**Mac improvements**: Built using vibe coding techniques with Mac MPS support and chat interface optimization.

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
