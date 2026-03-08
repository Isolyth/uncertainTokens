# Uncertain Tokens

A chat interface that visualizes token-level uncertainty during LLM generation. Instead of hiding the model's indecision, each token is color-coded by how confident the model was when producing it, and hovering reveals the top-10 candidate tokens with their probabilities.

## How it works

The app runs **Qwen3.5-2B** (8-bit quantized) and performs custom autoregressive generation:

- **Confident tokens** (large gap between top-1 and top-2 probability): sampled normally via temperature + top-p.
- **Mix tokens** (small gap): the top-10 token embeddings are blended by their probability weights, and the nearest real token to that mixed embedding is used. This is fed back into the model as a continuous embedding rather than a discrete token ID, letting the model "hedge" between alternatives.

In the UI:
- **Orange highlight** = uncertainty (higher opacity = less confident).
- **Purple highlight + dashed underline** = mix token (the model blended multiple candidates).
- **Hover any token** to see the top-10 candidates and their probabilities.

## Requirements

- Python 3.10+
- CUDA-capable GPU (model runs in 8-bit via bitsandbytes)

## Setup

```bash
pip install torch transformers huggingface-hub fastapi uvicorn bitsandbytes accelerate
```

## Run

```bash
python app.py
```

The model will be downloaded on first run (~4 GB). The server starts at `http://localhost:7860`.

## Configuration

The `MIX_THRESHOLD` variable in `app.py` (default `1`) controls when token blending kicks in. It compares the probability gap between the top-1 and top-2 candidates:

- **Lower values** (e.g. `0.1`) — blend only when the model is very undecided, so most tokens are sampled normally.
- **Higher values** (e.g. `1`) — blend more aggressively; nearly all tokens become mix tokens since the gap rarely exceeds 1.

Adjust it at the top of `app.py`:

```python
MIX_THRESHOLD = 0.3  # smaller = less mixing, larger = more mixing
```
