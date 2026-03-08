"""Foreground extraction + layer composition for OmnimatteZero.

Extracts foreground layers and composes them onto new backgrounds using
latent-space operations. Optimized for Apple Silicon with 24 GB unified memory.

Usage:
    python foreground_composition.py \
      --video_folder three_swans_lake \
      --new_bg ./results/cat_reflection.mp4 \
      --use_prompt_cache

Notes:
- This script expects the object-removed background video to exist under results/
- It expects object_mask.mp4 and total_mask.mp4 in the example_videos/<video_folder>/ folder
"""

from __future__ import annotations

import argparse
import os
from typing import Optional, Union

import torch
from diffusers import AutoencoderKLLTXVideo
from diffusers.pipelines.ltx.pipeline_ltx_condition import retrieve_latents
from diffusers.utils import export_to_video
from PIL import Image

from device_utils import (
    DEFAULT_CHECKPOINT,
    clear_memory,
    get_device,
    get_generator,
    get_optimal_dtype,
    load_pipeline,
    print_device_info,
    print_memory_stats,
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
    inspect_video,
    load_prompt_tensors,
    load_video_frames,
    pad_frames_to_length,
    plan_processing,
    plan_temporal_windows,
    resolve_relative_path,
    resolve_video_folder,
)


def tensor_video_to_pil_images(video_tensor: torch.Tensor):
    video_tensor = video_tensor.squeeze(0)
    video_numpy = video_tensor.detach().cpu().numpy()
    return [Image.fromarray(frame.astype("uint8")) for frame in video_numpy]


class MyAutoencoderKLLTXVideo(AutoencoderKLLTXVideo):
    """Custom VAE that encodes multiple videos sequentially to reduce peak memory."""

    def _encode_single(self, video, sample_posterior, generator):
        """Encode a single video tensor."""
        posterior = self.encode(video).latent_dist
        if sample_posterior:
            return posterior.sample(generator=generator)
        return posterior.mode()

    def forward(
        self,
        sample: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[torch.Tensor, torch.Tensor]:
        all_v, bg, mask, mask2, new_bg = sample

        # Encode each video sequentially, clearing memory between
        z_all = self._encode_single(all_v, sample_posterior, generator)
        clear_memory()

        z_bg = self._encode_single(bg, sample_posterior, generator)
        clear_memory()

        z_mask = self._encode_single(mask, sample_posterior, generator)
        clear_memory()

        z_mask2 = self._encode_single(mask2, sample_posterior, generator)
        clear_memory()

        z_new_bg = self._encode_single(new_bg, sample_posterior, generator)
        clear_memory()

        # Latent arithmetic
        z_diff = z_all - z_bg
        z = z_new_bg + z_diff

        del z_all, z_bg, z_new_bg
        clear_memory()

        # Decode sequentially
        dec = self.decode(z, temb)
        clear_memory()

        dec2 = self.decode(z_diff, temb)
        clear_memory()

        dec_mask = self.decode(z_mask, temb)
        clear_memory()

        dec_mask2 = self.decode(z_mask2, temb)

        if not return_dict:
            return (dec,)
        return dec, dec2, dec_mask, dec_mask2


def _load_windowed_video_tensor(
    pipe,
    path: str,
    window,
    *,
    target_length: int,
    width: int,
    height: int,
    device: torch.device,
):
    raw_frames = load_video_frames(path, start=window.start, end=window.end)
    source_frame_count = len(raw_frames)
    if source_frame_count == 0:
        raise RuntimeError(f"No frames were loaded from {path} for window {window.start}:{window.end}")
    padded_frames = pad_frames_to_length(raw_frames, target_length)
    tensor = pipe.video_processor.preprocess_video(padded_frames, width=width, height=height).to(
        device=device,
        dtype=torch.float32,
    )
    return tensor, source_frame_count


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="OmnimatteZero foreground extraction + composition")
    parser.add_argument("--video_folder", type=str, required=True,
                        help="Folder name under example_videos/ (contains video.mp4, object_mask.mp4, total_mask.mp4)")
    parser.add_argument("--new_bg", type=str, required=True,
                        help="Path to new background video (mp4)")

    parser.add_argument("--base_dir", type=str, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--height", type=int, default=None, help="Output height. Defaults to auto-fit input.")
    parser.add_argument("--width", type=int, default=None, help="Output width. Defaults to auto-fit input.")
    parser.add_argument("--num_frames", type=int, default=None, help="Optional cap on total frames to process.")
    parser.add_argument("--window_frames", type=int, default=None, help="Frames per processing window.")
    parser.add_argument("--overlap_frames", type=int, default=None, help="Overlap between processing windows.")
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cache_dir", type=str, default="", help="HuggingFace cache dir (optional)")

    parser.add_argument("--use_prompt_cache", action="store_true", help="Cache T5 embeddings to disk")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--negative_prompt", type=str,
                        default="worst quality, inconsistent motion, blurry, jittery, distorted")
    parser.add_argument("--max_sequence_length", type=int, default=256)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_refinement", action="store_true",
                        help="Skip the optional diffusion refinement step")

    args = parser.parse_args()

    device = get_device()
    dtype = get_optimal_dtype()

    print_device_info()

    video_folder = resolve_video_folder(args.base_dir, args.video_folder)
    source_video_path = os.path.join(video_folder, "video.mp4")
    object_mask_path = os.path.join(video_folder, "object_mask.mp4")
    total_mask_path = os.path.join(video_folder, "total_mask.mp4")
    clean_bg_path = os.path.join(args.out_dir, f"{os.path.basename(video_folder)}.mp4")
    new_bg_path = resolve_relative_path(args.new_bg)

    for required_path in [source_video_path, object_mask_path, total_mask_path, clean_bg_path, new_bg_path]:
        if not os.path.exists(required_path):
            raise FileNotFoundError(f"Missing required input: {required_path}")

    source_info = inspect_video(source_video_path)
    available_frames = min(
        source_info.frame_count,
        inspect_video(clean_bg_path).frame_count,
        inspect_video(object_mask_path).frame_count,
        inspect_video(total_mask_path).frame_count,
        inspect_video(new_bg_path).frame_count,
    )
    requested_total_frames = available_frames if args.num_frames is None else min(available_frames, args.num_frames)
    plan = plan_processing(
        source_info,
        requested_width=args.width,
        requested_height=args.height,
        requested_total_frames=requested_total_frames,
        requested_window_frames=args.window_frames,
        requested_overlap_frames=args.overlap_frames,
        requested_num_inference_steps=10,
    )
    windows = plan_temporal_windows(plan.total_frames, plan.window_frames, plan.overlap_frames)

    print(
        f"Processing {plan.total_frames} frames at {plan.width}x{plan.height} "
        f"with {len(windows)} window(s) of up to {plan.window_frames} frames"
    )
    for warning in plan.warnings:
        print(f"Note: {warning}")

    # --- 1) Load pipeline (skip T5 since we'll use cached embeddings) ---
    pipe = load_pipeline(
        checkpoint_path=args.checkpoint,
        cache_dir=args.cache_dir,
        dtype=dtype,
        load_text_encoder=False,
        force_vae_fp32=True,
    )

    # Replace VAE with custom subclass that encodes sequentially
    custom_vae = MyAutoencoderKLLTXVideo.from_config(pipe.vae.config)
    custom_vae.load_state_dict(pipe.vae.state_dict())
    custom_vae = custom_vae.to(device=device, dtype=torch.float32)
    pipe.vae = custom_vae

    apply_memory_optimizations(pipe)
    print_memory_stats()

    os.makedirs(args.out_dir, exist_ok=True)

    prompt_kwargs = None
    if not args.skip_refinement:
        prompt_kwargs = build_prompt_kwargs(
            load_prompt_tensors(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                checkpoint_path=args.checkpoint,
                dtype=dtype,
                device=device,
                use_prompt_cache=args.use_prompt_cache,
                max_sequence_length=args.max_sequence_length,
                cache_dir_for_t5=args.cache_dir,
                require_negative=True,
            ),
            max_sequence_length=args.max_sequence_length,
        )
        clear_memory()

    foreground_assembler = WindowedFrameAssembler(merge_mode="blend")
    latent_add_assembler = WindowedFrameAssembler(merge_mode="blend")
    refined_assembler = WindowedFrameAssembler(merge_mode="blend") if not args.skip_refinement else None

    for window_index, window in enumerate(windows, start=1):
        print(f"Window {window_index}/{len(windows)}: frames {window.start}:{window.end}")

        video_p, source_frame_count = _load_windowed_video_tensor(
            pipe,
            source_video_path,
            window,
            target_length=plan.window_frames,
            width=plan.width,
            height=plan.height,
            device=device,
        )
        video_bg, _ = _load_windowed_video_tensor(
            pipe,
            clean_bg_path,
            window,
            target_length=plan.window_frames,
            width=plan.width,
            height=plan.height,
            device=device,
        )
        video_mask, _ = _load_windowed_video_tensor(
            pipe,
            object_mask_path,
            window,
            target_length=plan.window_frames,
            width=plan.width,
            height=plan.height,
            device=device,
        )
        video_mask2, _ = _load_windowed_video_tensor(
            pipe,
            total_mask_path,
            window,
            target_length=plan.window_frames,
            width=plan.width,
            height=plan.height,
            device=device,
        )
        video_new_bg, _ = _load_windowed_video_tensor(
            pipe,
            new_bg_path,
            window,
            target_length=plan.window_frames,
            width=plan.width,
            height=plan.height,
            device=device,
        )

        temb = torch.tensor(0.0, device=device, dtype=torch.float32)

        with MemoryTracker(f"VAE Encoding/Decoding window {window_index}"):
            x, foreground, z_mask, z_mask2 = pipe.vae(
                [video_p, video_bg, video_mask, video_mask2, video_new_bg],
                temb=temb,
            )

        noise = x.sample
        foreground = foreground.sample
        z_mask = z_mask.sample
        z_mask2 = z_mask2.sample
        del x

        video_mask_bin = (z_mask.detach().cpu().float() > 0).to(device=device, dtype=video_bg.dtype)
        video_mask2_bin = (z_mask2.detach().cpu().float() > 0).to(device=device, dtype=video_bg.dtype)
        del z_mask, z_mask2, video_bg, video_mask, video_mask2, video_new_bg
        clear_memory()

        foreground = foreground * (1 - video_mask_bin) + video_p * video_mask_bin
        foreground = foreground * video_mask2_bin
        foreground_frames = tensor_video_to_pil_images(
            (pipe.video_processor.postprocess_video(foreground, output_type="pt")[0] * 255)
            .long()
            .permute(0, 2, 3, 1)
        )[:source_frame_count]
        foreground_assembler.add_window(window, foreground_frames)
        del foreground_frames, foreground
        clear_memory()

        noise = noise * (1 - video_mask_bin) + video_p * video_mask_bin
        del video_mask_bin, video_mask2_bin, video_p
        clear_memory()

        latent_add_frames = tensor_video_to_pil_images(
            (pipe.video_processor.postprocess_video(noise, output_type="pt")[0] * 255)
            .long()
            .permute(0, 2, 3, 1)
        )[:source_frame_count]
        latent_add_assembler.add_window(window, latent_add_frames)
        del latent_add_frames
        clear_memory()

        if refined_assembler is not None and prompt_kwargs is not None:
            condition_latents = retrieve_latents(pipe.vae.encode(noise), generator=None)
            condition_latents = pipe._normalize_latents(condition_latents, pipe.vae.latents_mean, pipe.vae.latents_std)
            condition_latents = condition_latents.to(device=device, dtype=dtype)
            del noise
            clear_memory()

            generator = get_generator(args.seed + window.start, device=device)
            with MemoryTracker(f"Refinement window {window_index}"):
                refined_frames = pipe.my_call(
                    width=plan.width,
                    height=plan.height,
                    num_frames=plan.window_frames,
                    denoise_strength=0.3,
                    num_inference_steps=10,
                    latents=condition_latents,
                    decode_timestep=0.05,
                    image_cond_noise_scale=0.025,
                    generator=generator,
                    output_type="pil",
                    unload_transformer_after_generation=False,
                    **prompt_kwargs,
                ).frames[0][:source_frame_count]

            refined_assembler.add_window(window, refined_frames)
            del condition_latents, refined_frames
            clear_memory()
        else:
            del noise
            clear_memory()

    video_folder_name = os.path.basename(video_folder)
    foreground_path = os.path.join(args.out_dir, f"{video_folder_name}_foreground.mp4")
    latent_add_path = os.path.join(args.out_dir, f"{video_folder_name}_latent_addition.mp4")
    export_to_video(foreground_assembler.finalize(), foreground_path, fps=max(1, int(round(plan.fps or 24))))
    export_to_video(latent_add_assembler.finalize(), latent_add_path, fps=max(1, int(round(plan.fps or 24))))
    print(f"Saved foreground: {foreground_path}")
    print(f"Saved latent addition: {latent_add_path}")

    if refined_assembler is None:
        print("Skipping refinement step.")
    else:
        refined_path = os.path.join(args.out_dir, f"{video_folder_name}_refined.mp4")
        export_to_video(refined_assembler.finalize(), refined_path, fps=max(1, int(round(plan.fps or 24))))
        print(f"Saved refined: {refined_path}")

    print_memory_stats()


if __name__ == "__main__":
    main()
