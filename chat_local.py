import os
import torch
import gradio as gr
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import RustBPETokenizer, get_token_bytes

MODEL_KIND = "sft"
STEP = 650  # match your checkpoint file

def load_model_and_tokenizer():
    import torch, os
    from nanochat.gpt import GPT, GPTConfig
    from nanochat.tokenizer import RustBPETokenizer, get_token_bytes

    # === Model ===
    model_path = os.path.expanduser("~/.cache/nanochat/chatsft_checkpoints/d20/model_000650.pt")
    assert os.path.isfile(model_path), f"Model file not found: {model_path}"

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading model from {model_path} on {device}...")

    state = torch.load(model_path, map_location=device)
    config = GPTConfig(
        vocab_size=65536,
        n_layer=20,
        n_head=10,
        n_kv_head=10,
        n_embd=1280,
        sequence_len=2048,
    )

    model = GPT(config)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # === Tokenizer ===
    token_path = os.path.expanduser("~/.cache/nanochat/tokenizer/token_bytes.pt")
    assert os.path.isfile(token_path), f"Tokenizer file not found: {token_path}"

    # Use NanoChat’s built-in helper to load the tokenizer correctly
    token_bytes = get_token_bytes()
    tok = RustBPETokenizer(token_bytes, bos_token="<BOS>")

    return model, tok, device

def respond(message, history, model, tok, device):
    prompt = ""
    for user, bot in history:
        prompt += f"User: {user}\nAssistant: {bot}\n"
    prompt += f"User: {message}\nAssistant:"

    input_ids = torch.tensor(tok.encode(prompt), dtype=torch.long)[None, :].to(device)
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=128, temperature=0.8)
    response = tok.decode(output[0].tolist())
    if "Assistant:" in response:
        response = response.split("Assistant:")[-1].strip()
    return response


def main():
    model, tok, device = load_model_and_tokenizer()

    with gr.Blocks() as demo:
        gr.Markdown("## 💬 NanoChat Local (Mac MPS)")
        chatbot = gr.Chatbot()
        msg = gr.Textbox(label="Message")

        def user_input(message, chat_history):
            reply = respond(message, chat_history, model, tok, device)
            chat_history.append((message, reply))
            return "", chat_history

        msg.submit(user_input, [msg, chatbot], [msg, chatbot])
    demo.launch(server_name="127.0.0.1", server_port=8000)


if __name__ == "__main__":
    main()