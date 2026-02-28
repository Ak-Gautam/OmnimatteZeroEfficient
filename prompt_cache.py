"""Prompt embedding cache utilities.

Goal: avoid re-loading + re-running the heavy T5 text encoder when the same prompt
strings are used repeatedly.

We cache the *outputs* that Diffusers pipelines accept at inference time:
- prompt_embeds
- prompt_attention_mask
- negative_prompt_embeds
- negative_prompt_attention_mask

The cache file is intentionally model/checkpoint-specific and parameterized by
max_sequence_length so we don't accidentally reuse incompatible embeddings.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import gc
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


T5_ENCODER_REPO = "Lightricks/LTX-Video"


def _stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _dtype_str(dtype: torch.dtype) -> str:
    if dtype is None:
        return "none"
    # torch.dtype string repr is stable enough across torch versions
    return str(dtype).replace("torch.", "")


def build_prompt_cache_key(
    *,
    prompt: str,
    negative_prompt: Optional[str],
    checkpoint_path: str,
    max_sequence_length: int,
    num_videos_per_prompt: int,
    dtype: torch.dtype,
) -> str:
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "checkpoint_basename": os.path.basename(checkpoint_path) if checkpoint_path else None,
        "checkpoint_path": os.path.abspath(checkpoint_path) if checkpoint_path else None,
        "max_sequence_length": int(max_sequence_length),
        "num_videos_per_prompt": int(num_videos_per_prompt),
        "dtype": _dtype_str(dtype),
    }
    digest = hashlib.sha256(_stable_json_dumps(payload).encode("utf-8")).hexdigest()
    return digest[:16]


def get_prompt_cache_path(cache_dir: str, cache_key: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{cache_key}.pt")


def save_prompt_cache(path: str, tensors: Dict[str, torch.Tensor]) -> None:
    # Always store on CPU to keep cache portable.
    to_save: Dict[str, torch.Tensor] = {}
    for k, v in tensors.items():
        if v is None:
            continue
        if not isinstance(v, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor for {k}, got {type(v)}")
        to_save[k] = v.detach().to(device="cpu")
    torch.save(to_save, path)


def load_prompt_cache(path: str) -> Dict[str, torch.Tensor]:
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        raise ValueError(f"Invalid cache file: {path}")
    return data


def _extract_prompt_from_cache(data: Dict[str, Any]) -> Optional[str]:
    prompt = data.get("prompt")
    if isinstance(prompt, str):
        return prompt

    meta = data.get("meta")
    if isinstance(meta, dict):
        prompt = meta.get("prompt")
        if isinstance(prompt, str):
            return prompt
    return None


def find_legacy_prompt_cache(cache_dir: str, prompt: str) -> Optional[str]:
    """Find a compatible cache file created by older scripts.

    Strategy:
    1) exact prompt match if metadata is available
    2) otherwise, any file that contains prompt_embeds + prompt_attention_mask
    """
    if not os.path.isdir(cache_dir):
        return None

    candidates = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))

    # Prefer exact prompt matches when metadata exists.
    for path in candidates:
        try:
            data = torch.load(path, map_location="cpu")
            if not isinstance(data, dict):
                continue
            cached_prompt = _extract_prompt_from_cache(data)
            if cached_prompt == prompt and "prompt_embeds" in data and "prompt_attention_mask" in data:
                return path
        except Exception:
            continue

    # Fallback: any minimally compatible cache.
    for path in candidates:
        try:
            data = torch.load(path, map_location="cpu")
            if isinstance(data, dict) and "prompt_embeds" in data and "prompt_attention_mask" in data:
                return path
        except Exception:
            continue

    return None


def normalize_cached_prompt_tensors(
    cached: Dict[str, torch.Tensor],
    *,
    require_negative: bool,
) -> Dict[str, torch.Tensor]:
    """Normalize cache contents from multiple formats.

    Some legacy caches may not include negative prompt tensors.
    For CFG runs, synthesize safe defaults to keep runtime path functional.
    """
    out = dict(cached)

    if "prompt_embeds" not in out or "prompt_attention_mask" not in out:
        raise ValueError("Cache is missing required prompt tensors")

    if require_negative:
        if "negative_prompt_embeds" not in out or out.get("negative_prompt_embeds") is None:
            out["negative_prompt_embeds"] = torch.zeros_like(out["prompt_embeds"])
        if "negative_prompt_attention_mask" not in out or out.get("negative_prompt_attention_mask") is None:
            out["negative_prompt_attention_mask"] = out["prompt_attention_mask"].clone()

    return out


def move_prompt_cache_to_device(
    cached: Dict[str, torch.Tensor],
    *,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in cached.items():
        if not isinstance(v, torch.Tensor):
            continue
        if dtype is not None and v.is_floating_point():
            out[k] = v.to(device=device, dtype=dtype)
        else:
            out[k] = v.to(device=device)
    return out


@torch.no_grad()
def encode_prompts_with_t5_only(
    *,
    prompt: str,
    negative_prompt: Optional[str],
    max_sequence_length: int,
    num_videos_per_prompt: int,
    dtype: torch.dtype,
    cache_dir: Optional[str],
    device: torch.device,
    do_classifier_free_guidance: bool,
) -> Dict[str, torch.Tensor]:
    """Encode prompts using only T5 tokenizer+encoder (no video model).

    Returns CPU tensors suitable for save_prompt_cache().
    """
    from transformers import T5EncoderModel, T5TokenizerFast

    tokenizer = T5TokenizerFast.from_pretrained(
        T5_ENCODER_REPO,
        subfolder="tokenizer",
        cache_dir=cache_dir,
    )
    text_encoder = T5EncoderModel.from_pretrained(
        T5_ENCODER_REPO,
        subfolder="text_encoder",
        torch_dtype=dtype,
        cache_dir=cache_dir,
    )
    text_encoder = text_encoder.to(device)
    text_encoder.eval()

    def _encode_one(texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        text_inputs = tokenizer(
            texts,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.bool().to(device)

        prompt_embeds = text_encoder(input_ids, attention_mask=attention_mask)[0]
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)

        batch_size, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        attention_mask = attention_mask.view(batch_size, -1)
        attention_mask = attention_mask.repeat(num_videos_per_prompt, 1)

        return prompt_embeds, attention_mask

    prompt_list = [prompt]
    prompt_embeds, prompt_attention_mask = _encode_one(prompt_list)

    negative_prompt_embeds = None
    negative_prompt_attention_mask = None
    if do_classifier_free_guidance:
        negative_text = "" if negative_prompt is None else negative_prompt
        negative_list = [negative_text]
        negative_prompt_embeds, negative_prompt_attention_mask = _encode_one(negative_list)

    out: Dict[str, torch.Tensor] = {
        "prompt_embeds": prompt_embeds.detach().to("cpu"),
        "prompt_attention_mask": prompt_attention_mask.detach().to("cpu"),
    }
    if negative_prompt_embeds is not None:
        out["negative_prompt_embeds"] = negative_prompt_embeds.detach().to("cpu")
    if negative_prompt_attention_mask is not None:
        out["negative_prompt_attention_mask"] = negative_prompt_attention_mask.detach().to("cpu")

    del text_encoder
    del tokenizer
    from device_utils import clear_memory
    clear_memory()

    return out
