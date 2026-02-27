"""Object removal (non-optimized reference implementation).

This version is refactored to run on:
- Apple Silicon (MPS) with limited unified memory
- CUDA GPUs

Key constraints for Apple Silicon runs:
- Use FP16 for the main pipeline
- Keep the VAE in FP32 for stability
- Optionally cache prompt embeddings to skip loading/running the T5 text encoder

Example (M4 Pro 24GB, 704x480, 97 frames):
    /path/to/python object_removal.py --preset mps_24gb --video swan_lake --height 480 --width 704 --num_frames 97 \
        --prompt "Empty" --negative_prompt "worst quality, inconsistent motion, blurry, jittery, distorted" \
        --use_prompt_cache
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import torch
from diffusers.utils import export_to_video, load_video
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition

from device_utils import (
    DEFAULT_CHECKPOINT,
    get_device,
    get_optimal_dtype,
    get_generator,
    clear_memory,
    print_device_info,
    print_memory_stats,
    load_pipeline,
)
from memory_utils import MemoryConfig
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


_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_PROMPT_CACHE_DIR = os.path.join(_PROJECT_ROOT, "cached_embeddings")


def _round_to_vae_grid(height: int, width: int, vae_ratio: int) -> tuple[int, int]:
    return height - (height % vae_ratio), width - (width % vae_ratio)


def _resolve_video_folder(base_dir: str, video: str) -> str:
    # Accept either a folder name under base_dir or an explicit path.
    if os.path.isdir(video):
        return video
    return os.path.join(base_dir, video)


@torch.no_grad()
def run_object_removal(
    *,
    base_dir: str,
    video: str,
    out_dir: str,
    checkpoint: str,
    cache_dir: str,
    preset: Optional[str],
    height: int,
    width: int,
    num_frames: Optional[int],
    num_inference_steps: int,
    seed: int,
    prompt: str,
    negative_prompt: str,
    use_prompt_cache: bool,
    max_sequence_length: int,
    offload_mode: str,
):
    device = get_device()
    dtype = get_optimal_dtype()  # float16 on MPS

    if preset:
        config = MemoryConfig(preset)
        # Respect user-provided height/width/frames if set; otherwise use preset.
        if height is None:
            height = config.max_resolution[0]
        if width is None:
            width = config.max_resolution[1]
        if num_frames is None:
            num_frames = config.max_frames

    if num_frames is None:
        raise ValueError("num_frames must be provided (or via preset)")

    video_folder = _resolve_video_folder(base_dir, video)
    video_path = os.path.join(video_folder, "video.mp4")
    mask_path = os.path.join(video_folder, "total_mask.mp4")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Missing input video: {video_path}")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(
            f"Missing total mask video: {mask_path}\n"
            f"Expected: {mask_path}\n"
            f"If you only have object_mask.mp4, generate total_mask.mp4 using self_attention_map.py"
        )

    os.makedirs(out_dir, exist_ok=True)

    print_device_info()

    # 1) Build/load prompt embeddings first (before loading video model).
    prompt_tensors_cpu = None
    prompt_cache_path = None
    if use_prompt_cache:
        cache_key = build_prompt_cache_key(
            prompt=prompt,
            negative_prompt=negative_prompt,
            checkpoint_path=checkpoint,
            max_sequence_length=max_sequence_length,
            num_videos_per_prompt=1,
            dtype=dtype,
        )
        prompt_cache_path = get_prompt_cache_path(_DEFAULT_PROMPT_CACHE_DIR, cache_key)
        if os.path.exists(prompt_cache_path):
            print(f"Loading prompt embeddings cache: {os.path.basename(prompt_cache_path)}")
            prompt_tensors_cpu = load_prompt_cache(prompt_cache_path)
        else:
            legacy_path = find_legacy_prompt_cache(_DEFAULT_PROMPT_CACHE_DIR, prompt)
            if legacy_path is not None:
                print(f"Prompt cache miss for keyed file; using legacy cache: {os.path.basename(legacy_path)}")
                prompt_tensors_cpu = load_prompt_cache(legacy_path)
            else:
                print("Prompt cache miss; loading T5 encoder only to compute embeddings...")
                prompt_tensors_cpu = encode_prompts_with_t5_only(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    max_sequence_length=max_sequence_length,
                    num_videos_per_prompt=1,
                    dtype=dtype,
                    cache_dir=cache_dir,
                    device=device,
                    do_classifier_free_guidance=True,
                )
                save_prompt_cache(prompt_cache_path, prompt_tensors_cpu)
                print(f"Saved prompt embeddings cache: {os.path.basename(prompt_cache_path)}")
                clear_memory()
    else:
        print("Prompt cache disabled; loading T5 encoder only for one-time prompt encoding...")
        prompt_tensors_cpu = encode_prompts_with_t5_only(
            prompt=prompt,
            negative_prompt=negative_prompt,
            max_sequence_length=max_sequence_length,
            num_videos_per_prompt=1,
            dtype=dtype,
            cache_dir=cache_dir,
            device=device,
            do_classifier_free_guidance=True,
        )
        clear_memory()

    # 2) Load pipeline.
    # Always skip loading T5 in the video pipeline.
    load_text_encoder = False

    print("Loading OmnimatteZero pipeline...")
    pipe = load_pipeline(
        checkpoint_path=checkpoint,
        cache_dir=cache_dir,
        dtype=dtype,
        load_text_encoder=load_text_encoder,
        force_vae_fp32=True,
    )

    # Memory-saving toggles (safe across backends)
    if hasattr(pipe, "enable_attention_slicing"):
        # "max" lowers peak memory further than default slicing
        try:
            pipe.enable_attention_slicing("max")
        except Exception:
            pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    pipe.vae.enable_tiling()

    # CPU offload configuration.
    # NOTE: On MPS (unified memory), cpu offload is counterproductive - it doesn't
    # actually free memory (same pool) and adds hook overhead. Keep model resident.
    resolved_offload_mode = offload_mode
    if resolved_offload_mode == "auto":
        # MPS: don't use cpu offload (unified memory). CUDA: optionally use it.
        resolved_offload_mode = "none" if str(device) == "mps" else "none"

    if resolved_offload_mode == "model":
        print("Enabling model CPU offload (CUDA only)...")
        if str(device) != "mps":
            pipe.enable_model_cpu_offload()
        else:
            print("  Skipping on MPS (unified memory - no benefit)")
            pipe.to(device)
    elif resolved_offload_mode == "sequential":
        print("Enabling sequential CPU offload (lowest memory, slowest)...")
        if str(device) != "mps":
            pipe.enable_sequential_cpu_offload()
        else:
            print("  Skipping on MPS (unified memory - no benefit)")
            pipe.to(device)
    else:
        # Standard fully-resident mode
        pipe.to(device)

    clear_memory()
    print_memory_stats()

    # 3) Ensure input dimensions are compatible with the VAE.
    height, width = _round_to_vae_grid(height, width, pipe.vae_spatial_compression_ratio)

    # 4) Load video + mask.
    # NOTE: load_video returns list of PIL images. We'll clear these after creating conditions.
    raw_video = load_video(video_path)
    raw_mask = load_video(mask_path)

    if num_frames is not None:
        raw_video = raw_video[:num_frames]
        raw_mask = raw_mask[:num_frames]

    print(f"Loaded {len(raw_video)} video frames, {len(raw_mask)} mask frames")

    condition1 = LTXVideoCondition(video=raw_video, frame_index=0)
    condition2 = LTXVideoCondition(video=raw_mask, frame_index=0)
    
    # Store frame count before clearing raw data
    video_num_frames = len(raw_video)
    
    # Clear raw frame data - conditions hold copies, we don't need originals.
    del raw_video, raw_mask
    clear_memory()

    generator = get_generator(seed, device=device)

    # 5) Prompt embeddings (always precomputed before video model load).
    if prompt_tensors_cpu is None:
        raise RuntimeError("Prompt tensors are missing before video model run.")

    prompt_tensors_cpu = normalize_cached_prompt_tensors(prompt_tensors_cpu, require_negative=True)
    prompt_tensors = move_prompt_cache_to_device(prompt_tensors_cpu, device=device, dtype=dtype)
    prompt_kwargs = {
        "prompt": None,
        "negative_prompt": None,
        "prompt_embeds": prompt_tensors.get("prompt_embeds"),
        "prompt_attention_mask": prompt_tensors.get("prompt_attention_mask"),
        "negative_prompt_embeds": prompt_tensors.get("negative_prompt_embeds"),
        "negative_prompt_attention_mask": prompt_tensors.get("negative_prompt_attention_mask"),
        "max_sequence_length": max_sequence_length,
    }

    # 6) Run inpainting / object removal.
    print("Running diffusion...")
    result = pipe.my_call(
        conditions=[condition1, condition2],
        width=width,
        height=height,
        num_frames=video_num_frames,
        num_inference_steps=num_inference_steps,
        generator=generator,
        output_type="pil",
        **prompt_kwargs,
    )

    frames = result.frames[0]
    out_path = os.path.join(out_dir, f"{os.path.basename(video_folder)}.mp4")
    export_to_video(frames, out_path, fps=24)

    print_memory_stats()
    print(f"Wrote: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="OmnimatteZero object removal")
    parser.add_argument("--base_dir", type=str, default="example_videos", help="Base directory containing video folders")
    parser.add_argument("--video", type=str, required=True, help="Video folder name under base_dir, or a full path")
    parser.add_argument("--out_dir", type=str, default="results", help="Output directory")

    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Path to ltx-video safetensors checkpoint")
    parser.add_argument("--cache_dir", type=str, default="", help="HuggingFace cache dir (optional)")

    parser.add_argument("--preset", type=str, default="mps_24gb", choices=["mps_24gb", "16gb", "24gb", "32gb"],
                        help="Memory preset (affects default max frames/resolution)")

    parser.add_argument("--height", type=int, default=480, help="Processing height")
    parser.add_argument("--width", type=int, default=704, help="Processing width")
    parser.add_argument("--num_frames", type=int, default=97, help="Number of frames to process")

    parser.add_argument("--num_inference_steps", type=int, default=30, help="Denoising steps")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")

    parser.add_argument("--prompt", type=str, default="Empty", help="Prompt")
    parser.add_argument("--negative_prompt", type=str,
                        default="worst quality, inconsistent motion, blurry, jittery, distorted",
                        help="Negative prompt")

    parser.add_argument("--no_prompt_cache", action="store_true", help="Disable prompt cache file reuse/save (still uses T5-only pre-encoding)")
    parser.add_argument("--max_sequence_length", type=int, default=256, help="Max sequence length for T5")
    parser.add_argument(
        "--offload",
        type=str,
        default="auto",
        choices=["auto", "none", "model", "sequential"],
        help="CPU offload mode: auto(model on MPS), none, model, sequential",
    )

    args = parser.parse_args()

    run_object_removal(
        base_dir=args.base_dir,
        video=args.video,
        out_dir=args.out_dir,
        checkpoint=args.checkpoint,
        cache_dir=args.cache_dir,
        preset=args.preset,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        use_prompt_cache=not args.no_prompt_cache,
        max_sequence_length=args.max_sequence_length,
        offload_mode=args.offload,
    )


if __name__ == "__main__":
    main()
