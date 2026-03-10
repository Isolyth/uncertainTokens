import json
import torch
import torch.nn.functional as F

from config import (
    TEMPERATURE, TOP_P, MAX_NEW_TOKENS,
    MIX_THRESHOLD, MULTIMIX_THRESHOLD, MULTIMIX_MAX_TOKENS, MULTIMIX_CERTAINTY_GAP,
)
from model import model, processor, embed_layer, find_nearest_token, decode_single_token


def _get_eos_ids():
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
    return eos_ids


def _top_tokens_info(tokenizer, top10_probs, top10_ids):
    return [
        {"token": decode_single_token(tid), "prob": round(tp, 4)}
        for tid, tp in zip(top10_ids.tolist(), top10_probs.tolist())
    ]


def _compute_mix_embed(top10_probs, top10_ids):
    """Create a weighted-average embedding from top-10 tokens."""
    with torch.no_grad():
        top10_embeds = embed_layer(top10_ids).float()
    weights = top10_probs / top10_probs.sum()
    return (weights.unsqueeze(-1) * top10_embeds).sum(dim=0)


def _sample_discrete(logits):
    """Sample a token using temperature + top-p nucleus sampling."""
    scaled_logits = logits / TEMPERATURE
    sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits.float(), dim=-1), dim=-1)
    cutoff = cumulative_probs > TOP_P
    cutoff[..., 1:] = cutoff[..., :-1].clone()
    cutoff[..., 0] = False
    sorted_logits[cutoff] = float("-inf")
    sample_probs = F.softmax(sorted_logits.float(), dim=-1)
    sampled_idx = torch.multinomial(sample_probs, 1)
    return sorted_indices[sampled_idx[0]].unsqueeze(0)


def _forward_step(model_inputs, past_key_values):
    """Run one forward pass, return outputs."""
    with torch.no_grad():
        return model(**model_inputs, past_key_values=past_key_values, use_cache=True)


def _try_decode(tokenizer, all_token_ids, prev_text):
    """Attempt to decode newly added tokens. Returns (display_text, new_prev_text) or (None, prev_text) if incomplete."""
    full_text = tokenizer.decode(all_token_ids, skip_special_tokens=True)
    display_text = full_text[len(prev_text):]
    if not display_text or '\ufffd' in display_text:
        return None, prev_text
    return display_text, full_text


def _build_multimix_embed(sequence_embeds, sequence_probs, anchor_attention):
    """
    Build a multi-mix embedding from a sequence of uncertain token embeddings.

    sequence_embeds: list of [hidden_dim] tensors (the mix/token embeddings in the sequence)
    sequence_probs: list of [10] tensors (top-10 probs for each position, used as base weight)
    anchor_attention: [seq_len] tensor — attention weights from the certain token after the
                      sequence, over the positions corresponding to the sequence tokens.
                      Used to weight each position's contribution.

    Returns a single [hidden_dim] mixed embedding.
    """
    n = len(sequence_embeds)
    if n == 0:
        raise ValueError("Empty sequence for multimix")

    # Stack embeddings: [n, hidden_dim]
    embeds = torch.stack(sequence_embeds, dim=0).float()

    # Attention-based weights for each position in the sequence
    # anchor_attention should be length n — attention from the anchor token to each sequence position
    if anchor_attention is not None and anchor_attention.numel() >= n:
        attn_weights = anchor_attention[:n].float()
    else:
        # Fallback: uniform
        attn_weights = torch.ones(n, device=embeds.device)

    attn_weights = attn_weights / attn_weights.sum()

    # Weighted average across positions
    mixed = (attn_weights.unsqueeze(-1) * embeds).sum(dim=0)
    return mixed


def _get_kv_len(past_key_values) -> int:
    """Get the current sequence length from a KV cache."""
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return past_key_values.get_seq_length()
    if hasattr(past_key_values, "key_cache"):
        for k in past_key_values.key_cache:
            if k is not None:
                return k.shape[2]
        return 0
    # Plain tuple fallback
    return past_key_values[0][0].shape[2]


def _trim_kv_cache(past_key_values, target_len):
    """Trim a KV cache to target_len sequence positions, in-place when possible."""
    if hasattr(past_key_values, "crop"):
        past_key_values.crop(target_len)
        return past_key_values
    if hasattr(past_key_values, "key_cache"):
        # Mutate in-place — works for any DynamicCache variant
        for i in range(len(past_key_values.key_cache)):
            if past_key_values.key_cache[i] is not None:
                past_key_values.key_cache[i] = past_key_values.key_cache[i][:, :, :target_len, :]
            if past_key_values.value_cache[i] is not None:
                past_key_values.value_cache[i] = past_key_values.value_cache[i][:, :, :target_len, :]
        return past_key_values
    # Plain tuple of (key, value) per layer
    trimmed = []
    for layer_kv in past_key_values:
        k, v = layer_kv
        trimmed.append((
            k[:, :, :target_len, :],
            v[:, :, :target_len, :],
        ))
    return tuple(trimmed)


def generate_tokens(messages: list[dict]):
    """Autoregressive generation yielding token data as SSE, with multi-mix support."""
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, return_tensors="pt").to(model.device)

    tokenizer = processor.tokenizer
    eos_ids = _get_eos_ids()

    past_key_values = None
    model_inputs = inputs
    all_token_ids = []
    prev_text = ""

    # Multi-mix state
    multimix_active = False
    multimix_embeds = []       # embeddings at each uncertain position
    multimix_top_info = []     # top-token info for each position (for display)
    multimix_token_ids = []    # nearest-token IDs added during sequence (to roll back)
    multimix_uncertainties = []
    # Track KV cache length at start of multimix sequence for rollback
    multimix_kv_start_len = 0

    for step in range(MAX_NEW_TOKENS):
        outputs = _forward_step(model_inputs, past_key_values)
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]

        probs = F.softmax(logits[0].float(), dim=-1)
        top10_probs, top10_ids = torch.topk(probs, 10)
        top_tokens = _top_tokens_info(tokenizer, top10_probs, top10_ids)
        uncertainty = 1.0 - top10_probs[0].item()
        gap = (top10_probs[0] - top10_probs[1]).item()

        is_mix = gap < MIX_THRESHOLD
        is_very_uncertain = gap < MULTIMIX_THRESHOLD

        # --- Multi-mix sequence handling ---

        if multimix_active:
            is_certain_enough = gap >= MULTIMIX_CERTAINTY_GAP
            if not is_certain_enough and is_very_uncertain and len(multimix_embeds) < MULTIMIX_MAX_TOKENS:
                # Still uncertain — continue recording the sequence
                mix_embed = _compute_mix_embed(top10_probs, top10_ids)
                multimix_embeds.append(mix_embed)
                multimix_top_info.append(top_tokens)
                multimix_uncertainties.append(uncertainty)

                nearest_id = find_nearest_token(mix_embed)
                if nearest_id in eos_ids:
                    break
                multimix_token_ids.append(nearest_id)
                all_token_ids.append(nearest_id)

                # Feed mix embed to continue generation
                model_inputs = {"inputs_embeds": mix_embed.unsqueeze(0).unsqueeze(0).to(model.dtype)}
                continue

            else:
                # Sequence ended: either we hit a certain token or max length.
                # This current token is the "anchor" — the certain token after the sequence.
                # We need its attention over the sequence positions to weight the multimix.

                # Get attention from the last layer for the current (anchor) token
                # outputs.attentions requires output_attentions=True, which is expensive.
                # Instead, we use the logit probabilities as a proxy: re-run a single
                # forward pass with output_attentions on just the last step.
                # Actually, to keep things efficient, we'll use the softmax probs of
                # each sequence position as a rough proxy for importance (higher confidence
                # positions contribute more).

                # Proxy: use inverse-uncertainty (top1 prob) at each position as weight
                seq_top1_probs = []
                for info in multimix_top_info:
                    seq_top1_probs.append(info[0]["prob"])
                anchor_attention = torch.tensor(seq_top1_probs, device=model.device)

                # Build the collapsed multimix embedding
                multimix_embed = _build_multimix_embed(
                    multimix_embeds, None, anchor_attention
                )
                multimix_nearest = find_nearest_token(multimix_embed)

                # Roll back: remove the sequence tokens from all_token_ids and recompute prev_text
                for _ in multimix_token_ids:
                    all_token_ids.pop()
                prev_text = tokenizer.decode(all_token_ids, skip_special_tokens=True) if all_token_ids else ""

                # Roll back KV cache: trim to before the sequence started
                past_key_values = _trim_kv_cache(past_key_values, multimix_kv_start_len)

                # Now feed the multimix embedding as a single token replacement
                mm_input = {"inputs_embeds": multimix_embed.unsqueeze(0).unsqueeze(0).to(model.dtype)}
                mm_outputs = _forward_step(mm_input, past_key_values)
                past_key_values = mm_outputs.past_key_values

                # Add the nearest token for decoding purposes
                if multimix_nearest in eos_ids:
                    break
                all_token_ids.append(multimix_nearest)

                display_text, prev_text = _try_decode(tokenizer, all_token_ids, prev_text)
                if display_text is None:
                    # Incomplete char — will resolve on next token
                    # Now process the current (anchor) token normally below
                    pass
                else:
                    # Emit the multimix token event
                    # Collect all unique top tokens across the sequence for tooltip
                    combined_top = multimix_top_info[0] if multimix_top_info else top_tokens
                    avg_uncertainty = sum(multimix_uncertainties) / len(multimix_uncertainties) if multimix_uncertainties else uncertainty

                    data = json.dumps({
                        "token": display_text,
                        "uncertainty": round(avg_uncertainty, 4),
                        "top": combined_top,
                        "mix": True,
                        "multimix": True,
                        "sequence_length": len(multimix_embeds),
                    })
                    yield f"data: {data}\n\n"

                # Reset multimix state
                multimix_active = False
                multimix_embeds = []
                multimix_top_info = []
                multimix_token_ids = []
                multimix_uncertainties = []

                # Now handle the current anchor token (the certain one that ended the sequence)
                # Fall through to normal processing below, but we need to re-run forward
                # since we rewound the KV cache and fed the multimix embed.
                # The mm_outputs already gave us a new logits — use those.
                logits = mm_outputs.logits[:, -1, :]
                probs = F.softmax(logits[0].float(), dim=-1)
                top10_probs, top10_ids = torch.topk(probs, 10)
                top_tokens = _top_tokens_info(tokenizer, top10_probs, top10_ids)
                uncertainty = 1.0 - top10_probs[0].item()
                gap = (top10_probs[0] - top10_probs[1]).item()
                is_mix = gap < MIX_THRESHOLD
                is_very_uncertain = gap < MULTIMIX_THRESHOLD
                # Fall through to normal token handling

        # --- Check if we should START a new multimix sequence ---
        if not multimix_active and is_very_uncertain and is_mix:
            # Start a multimix sequence
            multimix_active = True
            multimix_embeds = []
            multimix_top_info = []
            multimix_token_ids = []
            multimix_uncertainties = []
            # Record KV cache length *before* this uncertain token was processed.
            # past_key_values already includes the current step, so subtract 1.
            multimix_kv_start_len = _get_kv_len(past_key_values) - 1

            # Record this first uncertain token
            mix_embed = _compute_mix_embed(top10_probs, top10_ids)
            multimix_embeds.append(mix_embed)
            multimix_top_info.append(top_tokens)
            multimix_uncertainties.append(uncertainty)

            nearest_id = find_nearest_token(mix_embed)
            if nearest_id in eos_ids:
                break
            multimix_token_ids.append(nearest_id)
            all_token_ids.append(nearest_id)

            model_inputs = {"inputs_embeds": mix_embed.unsqueeze(0).unsqueeze(0).to(model.dtype)}
            continue

        # --- Normal token handling (mix or discrete) ---
        if is_mix:
            mix_embed = _compute_mix_embed(top10_probs, top10_ids)
            nearest_id = find_nearest_token(mix_embed)

            if nearest_id in eos_ids:
                break

            all_token_ids.append(nearest_id)
            display_text, prev_text = _try_decode(tokenizer, all_token_ids, prev_text)

            if display_text is None:
                model_inputs = {"inputs_embeds": mix_embed.unsqueeze(0).unsqueeze(0).to(model.dtype)}
                continue

            data = json.dumps({
                "token": display_text,
                "uncertainty": round(uncertainty, 4),
                "top": top_tokens,
                "mix": True,
            })
            yield f"data: {data}\n\n"
            model_inputs = {"inputs_embeds": mix_embed.unsqueeze(0).unsqueeze(0).to(model.dtype)}

        else:
            next_token_id = _sample_discrete(logits[0])

            if next_token_id.item() in eos_ids:
                break

            all_token_ids.append(next_token_id.item())
            display_text, prev_text = _try_decode(tokenizer, all_token_ids, prev_text)

            if display_text is None:
                model_inputs = {"input_ids": next_token_id.unsqueeze(0)}
                continue

            data = json.dumps({
                "token": display_text,
                "uncertainty": round(uncertainty, 4),
                "top": top_tokens,
                "mix": False,
            })
            yield f"data: {data}\n\n"
            model_inputs = {"input_ids": next_token_id.unsqueeze(0)}

    yield "data: [DONE]\n\n"
