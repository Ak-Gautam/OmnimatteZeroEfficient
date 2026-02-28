"""Object removal using OmnimatteZero.

Removes objects and their effects (shadows, reflections) from videos using
latent diffusion conditioning. Optimized for Apple Silicon with 24 GB unified memory.

Example:
    python object_removal.py --video swan_lake --use_prompt_cache
"""

from __future__ import annotations

import argparse
import os

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
    MemoryTracker,
)
from memory_utils import (
    MemoryConfig,
    apply_memory_optimizations,
    round_to_vae_compatible,
    round_frames_to_vae_compatible,
)
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


def _resolve_video_folder(base_dir: str, video: str) -> str:
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
    height: int = 480,
    width: int = 704,
    num_frames: int = 97,
    num_inference_steps: int = 12,
    seed: int = 1,
    prompt: str = "Empty",
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted",
    use_prompt_cache: bool = True,
    max_sequence_length: int = 256,
):
    device = get_device()
    dtype = get_optimal_dtype()

    video_folder = _resolve_video_folder(base_dir, video)
    video_path = os.path.join(video_folder, "video.mp4")
    mask_path = os.path.join(video_folder, "total_mask.mp4")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Missing input video: {video_path}")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(
            f"Missing total mask video: {mask_path}\n"
            f"If you only have object_mask.mp4, generate total_mask.mp4 using self_attention_map.py"
        )

    os.makedirs(out_dir, exist_ok=True)
    print_device_info()

    # --- 1) Build/load prompt embeddings (before loading video model) ---
    prompt_tensors_cpu = None
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
                print(f"Using legacy cache: {os.path.basename(legacy_path)}")
                prompt_tensors_cpu = load_prompt_cache(legacy_path)
            else:
                print("Prompt cache miss; loading T5 encoder to compute embeddings...")
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
                print(f"Saved prompt cache: {os.path.basename(prompt_cache_path)}")
                clear_memory()
    else:
        print("Prompt cache disabled; loading T5 encoder for one-time encoding...")
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

    # --- 2) Load pipeline (skip T5 since we have cached embeddings) ---
    print("Loading OmnimatteZero pipeline...")
    pipe = load_pipeline(
        checkpoint_path=checkpoint,
        cache_dir=cache_dir,
        dtype=dtype,
        load_text_encoder=False,
        force_vae_fp32=True,
    )

    apply_memory_optimizations(pipe)
    print_memory_stats()

    # --- 3) Ensure dimensions are VAE-compatible ---
    height, width = round_to_vae_compatible(height, width)
    num_frames = round_frames_to_vae_compatible(num_frames)

    # --- 4) Load video + mask ---
    raw_video = load_video(video_path)[:num_frames]
    raw_mask = load_video(mask_path)[:num_frames]
    print(f"Loaded {len(raw_video)} video frames, {len(raw_mask)} mask frames")

    condition1 = LTXVideoCondition(video=raw_video, frame_index=0)
    condition2 = LTXVideoCondition(video=raw_mask, frame_index=0)
    video_num_frames = len(raw_video)

    del raw_video, raw_mask
    clear_memory()

    # --- 5) Prepare prompt embeddings ---
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

    # --- 6) Run diffusion ---
    generator = get_generator(seed, device=device)
    print(f"Running diffusion ({num_inference_steps} steps, {width}x{height}, {video_num_frames} frames)...")

    with MemoryTracker("Generation"):
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
    parser.add_argument("--base_dir", type=str, default="example_videos")
    parser.add_argument("--video", type=str, required=True, help="Video folder name under base_dir, or a full path")
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cache_dir", type=str, default="")

    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=704)
    parser.add_argument("--num_frames", type=int, default=97)
    parser.add_argument("--num_inference_steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--prompt", type=str, default="Empty")
    parser.add_argument("--negative_prompt", type=str,
                        default="worst quality, inconsistent motion, blurry, jittery, distorted")
    parser.add_argument("--use_prompt_cache", action="store_true", help="Cache T5 embeddings to disk")
    parser.add_argument("--max_sequence_length", type=int, default=256)

    args = parser.parse_args()

    run_object_removal(
        base_dir=args.base_dir,
        video=args.video,
        out_dir=args.out_dir,
        checkpoint=args.checkpoint,
        cache_dir=args.cache_dir,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        use_prompt_cache=args.use_prompt_cache,
        max_sequence_length=args.max_sequence_length,
    )


if __name__ == "__main__":
    main()
