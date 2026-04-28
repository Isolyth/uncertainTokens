import os
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"

MODEL_ID = "Qwen/Qwen3.5-4B"

# Generation parameters
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_NEW_TOKENS = 2048

# Mix token: blend top-k embeddings when top1-top2 gap < threshold
MIX_THRESHOLD = 1

# Multi-mix token: when uncertainty is above this, start recording a sequence
# of uncertain tokens, then collapse them into a single weighted embedding.
MULTIMIX_THRESHOLD = 0.5  # top1-top2 gap below this triggers sequence recording
MULTIMIX_MAX_TOKENS = 5   # max tokens to record in an uncertain sequence
MULTIMIX_CERTAINTY_GAP = 1.1  # gap above this stops the sequence (certain token found)
MULTIMIX_COOLDOWN = 1         # min tokens after a multimix before another can start
MULTIMIX_UNIQUE_ONLY = True   # stop a multimix sequence if a duplicate token appears
