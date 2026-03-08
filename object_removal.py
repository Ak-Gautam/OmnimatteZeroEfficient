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
from diffusers.utils import export_to_video
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
    apply_memory_optimizations,
)
from runtime_utils import (
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    WindowedFrameAssembler,
    build_prompt_kwargs,
    load_prompt_tensors,
    load_video_frames,
    pad_frames_to_length,
    plan_processing,
    plan_temporal_windows,
    resolve_video_folder,
    inspect_video,
)


@torch.no_grad()
def run_object_removal(
    *,
    base_dir: str,
    video: str,
    out_dir: str,
    checkpoint: str,
    cache_dir: str,
    height: int | None = None,
    width: int | None = None,
    num_frames: int | None = None,
    window_frames: int | None = None,
    overlap_frames: int | None = None,
    num_inference_steps: int = 12,
    seed: int = 1,
    prompt: str = "Empty",
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted",
    use_prompt_cache: bool = True,
    max_sequence_length: int = 256,
):
    device = get_device()
    dtype = get_optimal_dtype()

    video_folder = resolve_video_folder(base_dir, video)
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

    video_info = inspect_video(video_path)
    plan = plan_processing(
        video_info,
        requested_width=width,
        requested_height=height,
        requested_total_frames=num_frames,
        requested_window_frames=window_frames,
        requested_overlap_frames=overlap_frames,
        requested_num_inference_steps=num_inference_steps,
    )
    windows = plan_temporal_windows(plan.total_frames, plan.window_frames, plan.overlap_frames)

    print(
        f"Processing {plan.total_frames} frames at {plan.width}x{plan.height} "
        f"with {len(windows)} window(s) of up to {plan.window_frames} frames"
    )
    for warning in plan.warnings:
        print(f"Note: {warning}")

    prompt_tensors = load_prompt_tensors(
        prompt=prompt,
        negative_prompt=negative_prompt,
        checkpoint_path=checkpoint,
        dtype=dtype,
        device=device,
        use_prompt_cache=use_prompt_cache,
        max_sequence_length=max_sequence_length,
        cache_dir_for_t5=cache_dir,
        require_negative=True,
    )
    prompt_kwargs = build_prompt_kwargs(prompt_tensors, max_sequence_length=max_sequence_length)
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

    assembler = WindowedFrameAssembler(merge_mode="blend")
    for window_index, window in enumerate(windows, start=1):
        raw_video = load_video_frames(video_path, start=window.start, end=window.end)
        raw_mask = load_video_frames(mask_path, start=window.start, end=window.end)
        source_frame_count = len(raw_video)

        if source_frame_count == 0 or len(raw_mask) == 0:
            raise RuntimeError(f"Empty frame window loaded for {window.start}:{window.end}")

        padded_video = pad_frames_to_length(raw_video, plan.window_frames)
        padded_mask = pad_frames_to_length(raw_mask, plan.window_frames)

        condition1 = LTXVideoCondition(video=padded_video, frame_index=0)
        condition2 = LTXVideoCondition(video=padded_mask, frame_index=0)
        generator = get_generator(seed + window.start, device=device)

        print(
            f"Window {window_index}/{len(windows)}: frames {window.start}:{window.end} "
            f"({source_frame_count} source, {plan.window_frames} model)"
        )
        with MemoryTracker(f"Generation window {window_index}"):
            result = pipe.my_call(
                conditions=[condition1, condition2],
                width=plan.width,
                height=plan.height,
                num_frames=plan.window_frames,
                num_inference_steps=plan.num_inference_steps,
                generator=generator,
                output_type="pil",
                unload_transformer_after_generation=False,
                **prompt_kwargs,
            )

        frames = result.frames[0][:source_frame_count]
        assembler.add_window(window, frames)

        del raw_video, raw_mask, padded_video, padded_mask, condition1, condition2, result, frames
        clear_memory()

    frames = assembler.finalize()
    out_path = os.path.join(out_dir, f"{os.path.basename(video_folder)}.mp4")
    export_to_video(frames, out_path, fps=max(1, int(round(plan.fps or 24))))

    print_memory_stats()
    print(f"Wrote: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="OmnimatteZero object removal")
    parser.add_argument("--base_dir", type=str, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--video", type=str, required=True, help="Video folder name under base_dir, or a full path")
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cache_dir", type=str, default="")

    parser.add_argument("--height", type=int, default=None, help="Output height. Defaults to auto-fit input.")
    parser.add_argument("--width", type=int, default=None, help="Output width. Defaults to auto-fit input.")
    parser.add_argument("--num_frames", type=int, default=None, help="Optional cap on total frames to process.")
    parser.add_argument("--window_frames", type=int, default=None, help="Frames per generation window.")
    parser.add_argument("--overlap_frames", type=int, default=None, help="Overlap between generation windows.")
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
        window_frames=args.window_frames,
        overlap_frames=args.overlap_frames,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        use_prompt_cache=args.use_prompt_cache,
        max_sequence_length=args.max_sequence_length,
    )


if __name__ == "__main__":
    main()
