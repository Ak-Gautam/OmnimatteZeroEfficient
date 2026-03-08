from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import imageio.v2 as imageio
import numpy as np
from PIL import Image
import torch

from memory_utils import round_frames_to_vae_compatible, round_to_vae_compatible
from prompt_cache import (
    build_prompt_cache_key,
    encode_prompts_with_t5_only,
    find_legacy_prompt_cache,
    get_prompt_cache_path,
    load_prompt_cache,
    move_prompt_cache_to_device,
    normalize_cached_prompt_tensors,
    save_prompt_cache,
)


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT_DIR = os.path.join(PROJECT_ROOT, "example_videos")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results")
DEFAULT_PROMPT_CACHE_DIR = os.path.join(PROJECT_ROOT, "cached_embeddings")
DEFAULT_FPS = 24.0

_VAE_SPATIAL_RATIO = 32
_VAE_TEMPORAL_RATIO = 8
_MIN_VAE_COMPATIBLE_FRAMES = _VAE_TEMPORAL_RATIO + 1

_REFERENCE_HEIGHT = 480
_REFERENCE_WIDTH = 704
_REFERENCE_WINDOW_FRAMES = 97
_REFERENCE_FRAME_PIXEL_BUDGET = _REFERENCE_HEIGHT * _REFERENCE_WIDTH * _REFERENCE_WINDOW_FRAMES
_DEFAULT_OVERLAP_FRAMES = 17


@dataclass(frozen=True)
class VideoInfo:
    path: str
    width: int
    height: int
    frame_count: int
    fps: float = DEFAULT_FPS


@dataclass(frozen=True)
class TemporalWindow:
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class ProcessingPlan:
    width: int
    height: int
    total_frames: int
    window_frames: int
    overlap_frames: int
    num_inference_steps: int
    fps: float
    warnings: tuple[str, ...] = field(default_factory=tuple)


def resolve_relative_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def resolve_video_folder(base_dir: str, video: str) -> str:
    if os.path.isdir(video):
        return video
    return os.path.join(base_dir, video)


def inspect_video(path: str) -> VideoInfo:
    reader = imageio.get_reader(path)
    try:
        meta = reader.get_meta_data()
        first_frame = reader.get_data(0)
        height, width = first_frame.shape[:2]
        frame_count = meta.get("nframes")
        if not isinstance(frame_count, int) or frame_count <= 0:
            try:
                frame_count = reader.count_frames()
            except Exception:
                frame_count = sum(1 for _ in reader)
        fps = float(meta.get("fps") or DEFAULT_FPS)
        return VideoInfo(
            path=path,
            width=int(width),
            height=int(height),
            frame_count=int(frame_count),
            fps=fps,
        )
    finally:
        reader.close()


def load_video_frames(path: str, start: int = 0, end: Optional[int] = None) -> List[Image.Image]:
    if start < 0:
        raise ValueError(f"start must be non-negative, got {start}")
    if end is not None and end < start:
        raise ValueError(f"end must be >= start, got start={start}, end={end}")

    frames: List[Image.Image] = []
    reader = imageio.get_reader(path)
    try:
        for frame_idx, frame in enumerate(reader):
            if frame_idx < start:
                continue
            if end is not None and frame_idx >= end:
                break
            frames.append(Image.fromarray(np.asarray(frame, dtype=np.uint8)))
    finally:
        reader.close()
    return frames


def next_vae_compatible_frame_count(num_frames: int, temporal_ratio: int = _VAE_TEMPORAL_RATIO) -> int:
    if num_frames <= 1:
        return 1
    return ((num_frames - 1 + temporal_ratio - 1) // temporal_ratio) * temporal_ratio + 1


def pad_frames_to_length(frames: Sequence[Image.Image], target_length: int) -> List[Image.Image]:
    if not frames:
        raise ValueError("Cannot pad an empty frame sequence")
    if target_length <= len(frames):
        return list(frames[:target_length])

    padded = list(frames)
    last_frame = frames[-1]
    for _ in range(target_length - len(frames)):
        padded.append(last_frame.copy())
    return padded


def _round_dimension(value: float) -> int:
    rounded = int(value) // _VAE_SPATIAL_RATIO * _VAE_SPATIAL_RATIO
    return max(_VAE_SPATIAL_RATIO, rounded)


def _fit_size_to_budget(
    *,
    input_width: int,
    input_height: int,
    requested_width: Optional[int],
    requested_height: Optional[int],
    max_pixels: float,
) -> tuple[int, int, Optional[str]]:
    aspect_ratio = input_width / input_height

    if requested_width and requested_height:
        width = requested_width
        height = requested_height
    elif requested_width:
        width = requested_width
        height = max(1, round(width / aspect_ratio))
    elif requested_height:
        height = requested_height
        width = max(1, round(height * aspect_ratio))
    else:
        scale = min(1.0, math.sqrt(max_pixels / max(1, input_width * input_height)))
        width = max(_VAE_SPATIAL_RATIO, int(input_width * scale))
        height = max(_VAE_SPATIAL_RATIO, int(input_height * scale))

    height, width = round_to_vae_compatible(_round_dimension(height), _round_dimension(width))
    warning = None
    if width * height > max_pixels:
        scale = math.sqrt(max_pixels / max(1, width * height))
        width = _round_dimension(width * scale)
        height = _round_dimension(height * scale)
        height, width = round_to_vae_compatible(height, width)
        warning = (
            f"Requested resolution exceeded the memory budget and was reduced to {width}x{height}."
        )

    width = min(width, _round_dimension(input_width))
    height = min(height, _round_dimension(input_height))
    height, width = round_to_vae_compatible(height, width)
    return width, height, warning


def plan_processing(
    video_info: VideoInfo,
    *,
    requested_width: Optional[int],
    requested_height: Optional[int],
    requested_total_frames: Optional[int],
    requested_window_frames: Optional[int],
    requested_overlap_frames: Optional[int],
    requested_num_inference_steps: int,
) -> ProcessingPlan:
    total_frames = video_info.frame_count
    warnings: List[str] = []

    if requested_total_frames is not None:
        total_frames = min(video_info.frame_count, requested_total_frames)
        if requested_total_frames > video_info.frame_count:
            warnings.append(
                f"Requested {requested_total_frames} frames, but the video only has {video_info.frame_count}; using {total_frames}."
            )

    if total_frames <= 0:
        raise ValueError(f"Video {video_info.path} does not contain any frames")

    size_budget_frames = requested_window_frames or min(total_frames, _REFERENCE_WINDOW_FRAMES)
    size_budget_frames = max(1, min(size_budget_frames, _REFERENCE_WINDOW_FRAMES))
    max_pixels = _REFERENCE_FRAME_PIXEL_BUDGET / size_budget_frames

    width, height, size_warning = _fit_size_to_budget(
        input_width=video_info.width,
        input_height=video_info.height,
        requested_width=requested_width,
        requested_height=requested_height,
        max_pixels=max_pixels,
    )
    if size_warning is not None:
        warnings.append(size_warning)

    max_window_frames = max(
        _MIN_VAE_COMPATIBLE_FRAMES,
        round_frames_to_vae_compatible(max(1, int(_REFERENCE_FRAME_PIXEL_BUDGET / max(1, width * height)))),
    )
    if total_frames >= _MIN_VAE_COMPATIBLE_FRAMES:
        max_window_frames = min(max_window_frames, round_frames_to_vae_compatible(total_frames))

    if requested_window_frames is not None:
        requested_window_frames = max(_MIN_VAE_COMPATIBLE_FRAMES, requested_window_frames)
        compatible_requested_window = round_frames_to_vae_compatible(requested_window_frames)
        if compatible_requested_window > max_window_frames:
            warnings.append(
                f"Requested {requested_window_frames} frames per window, but {width}x{height} only fits {max_window_frames}; using {max_window_frames}."
            )
        window_frames = min(compatible_requested_window, max_window_frames)
    else:
        if total_frames < _MIN_VAE_COMPATIBLE_FRAMES:
            window_frames = _MIN_VAE_COMPATIBLE_FRAMES
            warnings.append(
                f"Input clip is shorter than {_MIN_VAE_COMPATIBLE_FRAMES} frames; windows will be padded and trimmed back to {total_frames}."
            )
        else:
            window_frames = min(max_window_frames, round_frames_to_vae_compatible(total_frames))

    overlap_frames = requested_overlap_frames if requested_overlap_frames is not None else _DEFAULT_OVERLAP_FRAMES
    overlap_frames = max(0, min(overlap_frames, max(0, window_frames - 1)))

    return ProcessingPlan(
        width=width,
        height=height,
        total_frames=total_frames,
        window_frames=window_frames,
        overlap_frames=overlap_frames,
        num_inference_steps=requested_num_inference_steps,
        fps=video_info.fps,
        warnings=tuple(warnings),
    )


def plan_temporal_windows(total_frames: int, window_frames: int, overlap_frames: int) -> List[TemporalWindow]:
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if window_frames <= 0:
        raise ValueError("window_frames must be positive")

    if total_frames <= window_frames:
        return [TemporalWindow(0, total_frames)]

    step = max(1, window_frames - overlap_frames)
    starts = list(range(0, max(1, total_frames - window_frames + 1), step))
    last_start = total_frames - window_frames
    if starts[-1] != last_start:
        starts.append(last_start)

    windows: List[TemporalWindow] = []
    seen = set()
    for start in starts:
        if start in seen:
            continue
        seen.add(start)
        windows.append(TemporalWindow(start=start, end=min(total_frames, start + window_frames)))
    return windows


def load_prompt_tensors(
    *,
    prompt: str,
    negative_prompt: Optional[str],
    checkpoint_path: str,
    dtype: torch.dtype,
    device: torch.device,
    use_prompt_cache: bool,
    max_sequence_length: int,
    cache_dir_for_t5: Optional[str],
    require_negative: bool,
    prompt_cache_dir: str = DEFAULT_PROMPT_CACHE_DIR,
) -> Dict[str, torch.Tensor]:
    cache_key = build_prompt_cache_key(
        prompt=prompt,
        negative_prompt=negative_prompt,
        checkpoint_path=checkpoint_path,
        max_sequence_length=max_sequence_length,
        num_videos_per_prompt=1,
        dtype=dtype,
    )
    cache_path = get_prompt_cache_path(prompt_cache_dir, cache_key)

    cached = None
    if use_prompt_cache:
        if os.path.exists(cache_path):
            print(f"Loading prompt embeddings cache: {os.path.basename(cache_path)}")
            cached = load_prompt_cache(cache_path)
        else:
            legacy_path = find_legacy_prompt_cache(prompt_cache_dir, prompt)
            if legacy_path is not None:
                print(f"Using legacy cache: {os.path.basename(legacy_path)}")
                cached = load_prompt_cache(legacy_path)

    if cached is None:
        print("Prompt cache miss; loading T5 encoder to compute embeddings...")
        cached = encode_prompts_with_t5_only(
            prompt=prompt,
            negative_prompt=negative_prompt,
            max_sequence_length=max_sequence_length,
            num_videos_per_prompt=1,
            dtype=dtype,
            cache_dir=cache_dir_for_t5,
            device=device,
            do_classifier_free_guidance=require_negative,
        )
        if use_prompt_cache:
            save_prompt_cache(cache_path, cached)
            print(f"Saved prompt cache: {os.path.basename(cache_path)}")

    cached = normalize_cached_prompt_tensors(cached, require_negative=require_negative)
    return move_prompt_cache_to_device(cached, device=device, dtype=dtype)


def build_prompt_kwargs(prompt_tensors: Dict[str, torch.Tensor], max_sequence_length: int) -> Dict[str, torch.Tensor]:
    return {
        "prompt": None,
        "negative_prompt": None,
        "prompt_embeds": prompt_tensors.get("prompt_embeds"),
        "prompt_attention_mask": prompt_tensors.get("prompt_attention_mask"),
        "negative_prompt_embeds": prompt_tensors.get("negative_prompt_embeds"),
        "negative_prompt_attention_mask": prompt_tensors.get("negative_prompt_attention_mask"),
        "max_sequence_length": max_sequence_length,
    }


def _merge_overlap_frames(
    left_frames: Sequence[Image.Image],
    right_frames: Sequence[Image.Image],
    *,
    mode: str,
) -> List[Image.Image]:
    if len(left_frames) != len(right_frames):
        raise ValueError("Overlap frame sequences must have the same length")

    merged: List[Image.Image] = []
    total = len(left_frames)
    for index, (left_frame, right_frame) in enumerate(zip(left_frames, right_frames)):
        left = np.asarray(left_frame, dtype=np.float32)
        right = np.asarray(right_frame, dtype=np.float32)

        if mode == "blend":
            alpha = (index + 1) / (total + 1)
            merged_frame = left * (1.0 - alpha) + right * alpha
        elif mode == "max":
            merged_frame = np.maximum(left, right)
        else:
            raise ValueError(f"Unsupported overlap merge mode: {mode}")

        merged.append(Image.fromarray(np.clip(merged_frame, 0, 255).astype(np.uint8)))
    return merged


class WindowedFrameAssembler:
    def __init__(self, merge_mode: str = "blend"):
        self.merge_mode = merge_mode
        self._assembled: List[Image.Image] = []
        self._pending_frames: Optional[List[Image.Image]] = None
        self._pending_window: Optional[TemporalWindow] = None

    def add_window(self, window: TemporalWindow, frames: Sequence[Image.Image]) -> None:
        window_frames = list(frames[:window.length])
        if len(window_frames) != window.length:
            raise ValueError(
                f"Window {window.start}:{window.end} expected {window.length} frames but received {len(window_frames)}"
            )

        if self._pending_frames is None:
            self._pending_frames = window_frames
            self._pending_window = window
            return

        previous_window = self._pending_window
        previous_frames = self._pending_frames
        if previous_window is None:
            raise RuntimeError("Pending window metadata is missing")

        overlap = max(0, previous_window.end - window.start)
        overlap = min(overlap, len(previous_frames), len(window_frames))

        if overlap == 0:
            self._assembled.extend(previous_frames)
        else:
            self._assembled.extend(previous_frames[:-overlap])
            self._assembled.extend(
                _merge_overlap_frames(
                    previous_frames[-overlap:],
                    window_frames[:overlap],
                    mode=self.merge_mode,
                )
            )

        self._pending_frames = window_frames[overlap:]
        self._pending_window = TemporalWindow(start=window.start + overlap, end=window.end)

    def finalize(self) -> List[Image.Image]:
        if self._pending_frames is not None:
            self._assembled.extend(self._pending_frames)
            self._pending_frames = None
            self._pending_window = None
        return self._assembled
