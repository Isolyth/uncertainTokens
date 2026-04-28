import gc
import os

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from config import MODEL_ID

# Reduce CUDA memory fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

print(f"Downloading {MODEL_ID}...")
snapshot_download(MODEL_ID)
print("Download complete. Loading model in 8-bit...")

# Flush any stale CUDA allocations before loading
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto",
)
print("Model loaded.")

embed_layer = model.get_input_embeddings()

# Pre-compute normalized embedding matrix on CPU for nearest-token lookup
_embed_cpu = embed_layer.weight.detach().float().cpu()
_embed_cpu_norm = F.normalize(_embed_cpu, dim=-1)


def find_nearest_token(mixed_embed: torch.Tensor) -> int:
    """Find the token whose embedding is closest to the mixed embedding."""
    vec = F.normalize(mixed_embed.detach().float().cpu().unsqueeze(0), dim=-1)
    sims = (vec @ _embed_cpu_norm.T).squeeze(0)
    return sims.argmax().item()


def decode_single_token(tid: int) -> str:
    """Decode a single token for display, falling back to vocab entry for byte-level tokens."""
    tokenizer = processor.tokenizer
    text = tokenizer.decode([tid])
    if text and '\ufffd' not in text:
        return text
    return tokenizer.convert_ids_to_tokens([tid])[0]
