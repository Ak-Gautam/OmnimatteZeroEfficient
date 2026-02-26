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
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


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
