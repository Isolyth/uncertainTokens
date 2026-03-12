import os
import tomllib
from pathlib import Path

# Load config: config.toml values, overridden by env vars
_config = {}
_config_path = Path(__file__).parent / "config.toml"
if _config_path.exists():
    with open(_config_path, "rb") as f:
        _config = tomllib.load(f)


def _cfg(key: str, default=None):
    """Get config value: env var takes precedence, then config.toml, then default."""
    return os.environ.get(key.upper(), _config.get(key, default))


os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")

import json
import html
import secrets
import torch
import torch.nn.functional as F
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from huggingface_hub import snapshot_download
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

MODEL_ID = _cfg("model_id", "Qwen/Qwen3.5-4B")
QUANTIZE = _cfg("quantize", "8bit")

print(f"Downloading {MODEL_ID}...")
snapshot_download(MODEL_ID)

load_kwargs = dict(device_map="auto", attn_implementation="sdpa")
if QUANTIZE == "8bit":
    print("Loading model in 8-bit...")
    load_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True
    )
else:
    print("Loading model in fp16...")
    load_kwargs["torch_dtype"] = torch.float16

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **load_kwargs)
print("Model loaded.")

app = FastAPI()
AUTH_PASSWORD = _cfg("auth_password")


@app.middleware("http")
async def basic_auth(request: Request, call_next):
    if not AUTH_PASSWORD:
        return await call_next(request)
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Basic "):
        import base64
        try:
            decoded = base64.b64decode(auth[6:]).decode()
            _, password = decoded.split(":", 1)
            if secrets.compare_digest(password, AUTH_PASSWORD):
                return await call_next(request)
        except Exception:
            pass
    return Response(
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="uncertainTokens"'},
    )


class ChatRequest(BaseModel):
    messages: list[dict]


embed_layer = model.get_input_embeddings()
# Pre-compute normalized embedding matrix on CPU for nearest-token lookup
_embed_cpu = embed_layer.weight.detach().float().cpu()
_embed_cpu_norm = F.normalize(_embed_cpu, dim=-1)

mix_threshold = float(_cfg("mix_threshold", 0.65))


def _decode_single_token(tokenizer, tid):
    """Decode a single token for display, falling back to vocab entry for byte-level tokens."""
    text = tokenizer.decode([tid])
    if text and '\ufffd' not in text:
        return text
    return tokenizer.convert_ids_to_tokens([tid])[0]


def find_nearest_token(mixed_embed: torch.Tensor) -> int:
    """Find the token whose embedding is closest to the mixed embedding."""
    vec = F.normalize(mixed_embed.detach().float().cpu().unsqueeze(0), dim=-1)
    sims = (vec @ _embed_cpu_norm.T).squeeze(0)
    return sims.argmax().item()


def generate_tokens(messages: list[dict]):
    """Autoregressive generation yielding token data as SSE."""
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, return_tensors="pt").to(model.device)

    tokenizer = processor.tokenizer
    eos_ids = set()
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)
    if hasattr(model.config, "eos_token_id"):
        val = model.config.eos_token_id
        if isinstance(val, list):
            eos_ids.update(val)
        elif val is not None:
            eos_ids.add(val)

    temperature = 0.7
    top_p = 0.9
    max_new_tokens = 2048

    past_key_values = None
    model_inputs = inputs
    # Accumulate token IDs for proper multi-byte Unicode decoding
    all_token_ids = []
    prev_text = ""

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(**model_inputs, past_key_values=past_key_values, use_cache=True)

        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]

        probs = F.softmax(logits[0].float(), dim=-1)
        top10_probs, top10_ids = torch.topk(probs, 10)
        top_tokens = [
            {"token": _decode_single_token(tokenizer, tid), "prob": round(tp, 4)}
            for tid, tp in zip(top10_ids.tolist(), top10_probs.tolist())
        ]
        uncertainty = 1.0 - top10_probs[0].item()

        gap = (top10_probs[0] - top10_probs[1]).item()
        is_mix = mix_threshold > 0 and gap < mix_threshold

        if is_mix:
            # Mix token: weighted average of top-10 embeddings
            with torch.no_grad():
                top10_embeds = embed_layer(top10_ids).float()  # [10, hidden_dim]
            weights = top10_probs / top10_probs.sum()  # normalize to sum=1
            mixed_embed = (weights.unsqueeze(-1) * top10_embeds).sum(dim=0)  # [hidden_dim]

            nearest_id = find_nearest_token(mixed_embed)

            # Check EOS on nearest token (best proxy)
            if nearest_id in eos_ids:
                break

            all_token_ids.append(nearest_id)
            full_text = tokenizer.decode(all_token_ids, skip_special_tokens=True)
            display_text = full_text[len(prev_text):]

            if not display_text or '\ufffd' in display_text:
                # Incomplete multi-byte character, buffer until next token completes it
                model_inputs = {"inputs_embeds": mixed_embed.unsqueeze(0).unsqueeze(0).to(model.dtype)}
                continue
            prev_text = full_text

            data = json.dumps({
                "token": display_text,
                "uncertainty": round(uncertainty, 4),
                "top": top_tokens,
                "mix": True,
            })
            yield f"data: {data}\n\n"

            # Feed the mixed embedding — this is what lives in the KV cache
            model_inputs = {"inputs_embeds": mixed_embed.unsqueeze(0).unsqueeze(0).to(model.dtype)}

        else:
            # Confident: sample a discrete token
            scaled_logits = logits[0] / temperature
            sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits.float(), dim=-1), dim=-1)
            cutoff = cumulative_probs > top_p
            cutoff[..., 1:] = cutoff[..., :-1].clone()
            cutoff[..., 0] = False
            sorted_logits[cutoff] = float("-inf")
            sample_probs = F.softmax(sorted_logits.float(), dim=-1)
            sampled_idx = torch.multinomial(sample_probs, 1)
            next_token_id = sorted_indices[sampled_idx[0]].unsqueeze(0)

            if next_token_id.item() in eos_ids:
                break

            all_token_ids.append(next_token_id.item())
            full_text = tokenizer.decode(all_token_ids, skip_special_tokens=True)
            token_text = full_text[len(prev_text):]

            if not token_text or '\ufffd' in token_text:
                # Incomplete multi-byte character, buffer until next token completes it
                model_inputs = {"input_ids": next_token_id.unsqueeze(0)}
                continue
            prev_text = full_text

            data = json.dumps({
                "token": token_text,
                "uncertainty": round(uncertainty, 4),
                "top": top_tokens,
                "mix": False,
            })
            yield f"data: {data}\n\n"

            model_inputs = {"input_ids": next_token_id.unsqueeze(0)}

    yield "data: [DONE]\n\n"


@app.post("/chat")
async def chat(req: ChatRequest):
    return StreamingResponse(
        generate_tokens(req.messages),
        media_type="text/event-stream",
    )


class ThresholdRequest(BaseModel):
    value: float


@app.post("/mix-threshold")
async def set_mix_threshold(req: ThresholdRequest):
    global mix_threshold
    mix_threshold = max(0.0, min(1.0, req.value))
    return JSONResponse({"mix_threshold": mix_threshold})


@app.get("/")
async def index():
    return HTMLResponse(INDEX_HTML)


INDEX_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Uncertain Tokens</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #0a0a0f;
    color: #e0e0e0;
    height: 100vh;
    display: flex;
    flex-direction: column;
}
#chat {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 16px;
}
.msg {
    max-width: 80%;
    padding: 10px 14px;
    border-radius: 12px;
    line-height: 1.8;
}
.msg.user {
    align-self: flex-end;
    background: #1a3a5c;
    color: #d0e8ff;
}
.msg.assistant {
    align-self: flex-start;
    background: #1a1a2e;
}
#input-bar {
    display: flex;
    padding: 12px 20px;
    gap: 10px;
    background: #111118;
    border-top: 1px solid #222;
}
#input-bar input {
    flex: 1;
    padding: 10px 14px;
    border-radius: 8px;
    border: 1px solid #333;
    background: #1a1a2e;
    color: #e0e0e0;
    font-size: 15px;
    outline: none;
}
#input-bar input:focus { border-color: #556; }
#input-bar button {
    padding: 10px 20px;
    border-radius: 8px;
    border: none;
    background: #2a4a6c;
    color: #d0e8ff;
    font-size: 15px;
    cursor: pointer;
}
#input-bar button:hover { background: #3a5a7c; }
#input-bar button:disabled { opacity: 0.5; cursor: default; }
#mix-controls { display: flex; align-items: center; gap: 6px; font-size: 12px; color: #8cf; white-space: nowrap; }
#mix-slider { width: 100px; accent-color: #6448c8; }
#mix-label { min-width: 70px; }

/* Tokens */
.tok {
    cursor: pointer;
    border-radius: 3px;
    padding: 1px 0;
    white-space: pre-wrap;
}
.tok.mix {
    border-bottom: 1px dashed rgba(150, 100, 255, 0.5);
}

/* Tooltip */
#tooltip {
    display: none;
    position: fixed;
    background: #12121f;
    color: #e0e0e0;
    border: 1px solid #444;
    border-radius: 6px;
    padding: 6px 8px;
    font-size: 12px;
    font-family: monospace;
    white-space: nowrap;
    z-index: 10000;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    pointer-events: none;
}
#tooltip table { border-collapse: collapse; }
#tooltip td { padding: 1px 6px; }
#tooltip .tok-name { text-align: left; color: #8cf; }
#tooltip .tok-prob { text-align: right; color: #fc8; }
#tooltip tr:first-child td { font-weight: bold; }
</style>
</head>
<body>
<div id="chat"></div>
<div id="input-bar">
    <div id="mix-controls">
        <span id="mix-label">Mix: 0.65</span>
        <input id="mix-slider" type="range" min="0" max="100" value="65" title="0 = off, 100 = mix everything">
    </div>
    <input id="msg" type="text" placeholder="Type a message..." autofocus>
    <button id="send">Send</button>
</div>
<div id="tooltip"></div>
<script>
const chat = document.getElementById('chat');
const msgInput = document.getElementById('msg');
const sendBtn = document.getElementById('send');
const tooltip = document.getElementById('tooltip');
let history = [];

function escapeHtml(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
}

function addMessage(role, content) {
    const div = document.createElement('div');
    div.className = 'msg ' + role;
    if (typeof content === 'string') {
        div.textContent = content;
    }
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
    return div;
}

async function send() {
    const text = msgInput.value.trim();
    if (!text) return;
    msgInput.value = '';
    sendBtn.disabled = true;

    history.push({role: 'user', content: text});
    addMessage('user', text);

    const assistantDiv = addMessage('assistant', '');
    let fullText = '';

    try {
        const res = await fetch('/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({messages: history}),
        });
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const {done, value} = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, {stream: true});

            const lines = buffer.split('\\n');
            buffer = lines.pop();

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const payload = line.slice(6);
                if (payload === '[DONE]') continue;

                const data = JSON.parse(payload);
                fullText += data.token;

                const span = document.createElement('span');
                span.className = data.mix ? 'tok mix' : 'tok';
                const color = data.mix
                    ? `rgba(150, 100, 255, ${data.uncertainty})`
                    : `rgba(255, 150, 0, ${data.uncertainty})`;
                span.style.background = color;
                span.textContent = data.token;
                span.dataset.top = JSON.stringify(data.top);
                span.dataset.mix = data.mix ? '1' : '';
                assistantDiv.appendChild(span);
                chat.scrollTop = chat.scrollHeight;
            }
        }
    } catch (e) {
        assistantDiv.textContent = 'Error: ' + e.message;
    }

    history.push({role: 'assistant', content: fullText});
    sendBtn.disabled = false;
    msgInput.focus();
}

sendBtn.addEventListener('click', send);
msgInput.addEventListener('keydown', e => { if (e.key === 'Enter') send(); });

const mixSlider = document.getElementById('mix-slider');
const mixLabel = document.getElementById('mix-label');
mixSlider.addEventListener('input', () => {
    const v = (mixSlider.value / 100).toFixed(2);
    mixLabel.textContent = mixSlider.value == 0 ? 'Mix: OFF' : 'Mix: ' + v;
});
mixSlider.addEventListener('change', async () => {
    const v = mixSlider.value / 100;
    await fetch('/mix-threshold', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({value: v}),
    });
});

// Tooltip
document.addEventListener('mouseover', e => {
    const tok = e.target.closest('.tok');
    if (!tok || !tok.dataset.top) return;

    const top = JSON.parse(tok.dataset.top);
    const isMix = tok.dataset.mix === '1';
    const label = isMix ? '<div style="color:#b88cff;font-weight:bold;margin-bottom:4px">MIX TOKEN (weighted blend)</div>' : '';
    let rows = top.map(t =>
        `<tr><td class="tok-name">${escapeHtml(t.token)}</td><td class="tok-prob">${(t.prob * 100).toFixed(1)}%</td></tr>`
    ).join('');
    tooltip.innerHTML = label + '<table>' + rows + '</table>';
    tooltip.style.display = 'block';

    const rect = tok.getBoundingClientRect();
    let y = rect.bottom + 4;
    let x = rect.left;
    if (y + tooltip.offsetHeight > window.innerHeight) {
        y = rect.top - tooltip.offsetHeight - 4;
    }
    if (x + tooltip.offsetWidth > window.innerWidth) {
        x = window.innerWidth - tooltip.offsetWidth - 8;
    }
    tooltip.style.top = y + 'px';
    tooltip.style.left = x + 'px';
});

document.addEventListener('mouseout', e => {
    if (e.target.closest('.tok')) {
        tooltip.style.display = 'none';
    }
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
