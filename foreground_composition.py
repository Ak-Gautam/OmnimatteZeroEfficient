"""Foreground extraction + layer composition (non-optimized reference implementation).

This refactors the original CUDA-only script to:
- load the local LTX-Video safetensors checkpoint (0.9.5) via `from_single_file`
- run on MPS (Apple Silicon) or CUDA
- keep VAE in FP32 for stability; keep the rest of the pipeline in FP16 on MPS
- optionally cache prompt embeddings for the refinement stage

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
from diffusers.utils import export_to_video, load_video
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
)
from prompt_cache import (
    build_prompt_cache_key,
    find_legacy_prompt_cache,
    get_prompt_cache_path,
    load_prompt_cache,
    move_prompt_cache_to_device,
    normalize_cached_prompt_tensors,
    save_prompt_cache,
)


def tensor_video_to_pil_images(video_tensor: torch.Tensor):
    video_tensor = video_tensor.squeeze(0)
    video_numpy = video_tensor.detach().cpu().numpy()
    return [Image.fromarray(frame.astype("uint8")) for frame in video_numpy]


class MyAutoencoderKLLTXVideo(AutoencoderKLLTXVideo):
    def forward(
        self,
        sample: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[torch.Tensor, torch.Tensor]:
        all_v, bg, mask, mask2, new_bg = sample

        posterior = self.encode(all_v).latent_dist
        z_all = posterior.sample(generator=generator) if sample_posterior else posterior.mode()

        posterior = self.encode(bg).latent_dist
        z_bg = posterior.sample(generator=generator) if sample_posterior else posterior.mode()

        posterior = self.encode(mask).latent_dist
        z_mask = posterior.sample(generator=generator) if sample_posterior else posterior.mode()

        posterior = self.encode(mask2).latent_dist
        z_mask2 = posterior.sample(generator=generator) if sample_posterior else posterior.mode()

        posterior = self.encode(new_bg).latent_dist
        z_new_bg = posterior.sample(generator=generator) if sample_posterior else posterior.mode()

        z_diff = z_all - z_bg
        z = z_new_bg + z_diff

        dec = self.decode(z, temb)
        dec2 = self.decode(z_diff, temb)
        if not return_dict:
            return (dec,)
        return dec, dec2, self.decode(z_mask, temb), self.decode(z_mask2, temb)

    def forward_encode(
        self,
        sample: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[torch.Tensor, torch.Tensor]:
        posterior = self.encode(sample).latent_dist
        z = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        if not return_dict:
            return (z,)
        return z


def _load_or_build_prompt_embeds(
    *,
    pipe,
    prompt: str,
    negative_prompt: str,
    checkpoint: str,
    dtype: torch.dtype,
    device: torch.device,
    use_prompt_cache: bool,
    max_sequence_length: int,
) -> dict:
    if not use_prompt_cache:
        return {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "max_sequence_length": max_sequence_length,
        }

    project_root = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(project_root, "cached_embeddings")
    cache_key = build_prompt_cache_key(
        prompt=prompt,
        negative_prompt=negative_prompt,
        checkpoint_path=checkpoint,
        max_sequence_length=max_sequence_length,
        num_videos_per_prompt=1,
        dtype=dtype,
    )
    cache_path = get_prompt_cache_path(cache_dir, cache_key)

    cached = None
    if os.path.exists(cache_path):
        cached = load_prompt_cache(cache_path)
    else:
        legacy_path = find_legacy_prompt_cache(cache_dir, prompt)
        if legacy_path is not None:
            cached = load_prompt_cache(legacy_path)

    if cached is None:
        # Compute once (requires text encoder)
        if pipe.text_encoder is None or pipe.tokenizer is None:
            raise RuntimeError(
                "Prompt cache miss, but pipeline was loaded without text encoder. "
                "Re-run once with text encoder enabled to populate the cache."
            )

        pe, pam, ne, nam = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=True,
            num_videos_per_prompt=1,
            device=pipe._execution_device,
            max_sequence_length=max_sequence_length,
        )
        cached = {
            "prompt_embeds": pe,
            "prompt_attention_mask": pam,
            "negative_prompt_embeds": ne,
            "negative_prompt_attention_mask": nam,
        }
        save_prompt_cache(cache_path, cached)

        pipe.text_encoder = None
        pipe.tokenizer = None
        clear_memory()

    cached = normalize_cached_prompt_tensors(cached, require_negative=True)

    moved = move_prompt_cache_to_device(cached, device=device, dtype=dtype)
    return {
        "prompt": None,
        "negative_prompt": None,
        "prompt_embeds": moved.get("prompt_embeds"),
        "prompt_attention_mask": moved.get("prompt_attention_mask"),
        "negative_prompt_embeds": moved.get("negative_prompt_embeds"),
        "negative_prompt_attention_mask": moved.get("negative_prompt_attention_mask"),
        "max_sequence_length": max_sequence_length,
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="OmnimatteZero foreground extraction + composition")
    parser.add_argument("--video_folder", type=str, required=True,
                        help="Folder name under example_videos/ (contains video.mp4, object_mask.mp4, total_mask.mp4)")
    parser.add_argument("--new_bg", type=str, required=True,
                        help="Path to new background video (mp4)")

    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=704)

    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cache_dir", type=str, default="", help="HuggingFace cache dir (optional)")

    parser.add_argument("--use_prompt_cache", action="store_true")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--negative_prompt", type=str,
                        default="worst quality, inconsistent motion, blurry, jittery, distorted")
    parser.add_argument("--max_sequence_length", type=int, default=256)

    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    device = get_device()
    dtype = get_optimal_dtype()

    print_device_info()

    project_root = os.path.dirname(os.path.abspath(__file__))
    prompt_cache_dir = os.path.join(project_root, "cached_embeddings")
    cache_key = build_prompt_cache_key(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        checkpoint_path=args.checkpoint,
        max_sequence_length=args.max_sequence_length,
        num_videos_per_prompt=1,
        dtype=dtype,
    )
    keyed_cache_path = get_prompt_cache_path(prompt_cache_dir, cache_key)
    legacy_cache_path = find_legacy_prompt_cache(prompt_cache_dir, args.prompt) if args.use_prompt_cache else None
    has_compatible_cache = args.use_prompt_cache and (
        os.path.exists(keyed_cache_path) or legacy_cache_path is not None
    )

    # Skip T5 load entirely when prompt cache already exists.
    pipe = load_pipeline(
        checkpoint_path=args.checkpoint,
        cache_dir=args.cache_dir,
        dtype=dtype,
        load_text_encoder=not has_compatible_cache,
        force_vae_fp32=True,
    )

    # Replace VAE with a subclass that can encode multiple videos in one forward.
    custom_vae = MyAutoencoderKLLTXVideo.from_config(pipe.vae.config)
    custom_vae.load_state_dict(pipe.vae.state_dict())
    custom_vae = custom_vae.to(dtype=torch.float32)
    pipe.vae = custom_vae

    # Memory reducers
    if hasattr(pipe, "enable_attention_slicing"):
        try:
            pipe.enable_attention_slicing("max")
        except Exception:
            pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    pipe.vae.enable_tiling()

    pipe.to(device)
    clear_memory()
    print_memory_stats()

    w, h = args.width, args.height
    video_folder = args.video_folder

    video_p = load_video(os.path.join("example_videos", video_folder, "video.mp4"))
    video_bg = load_video(os.path.join("results", f"{video_folder}.mp4"))
    video_mask = load_video(os.path.join("example_videos", video_folder, "object_mask.mp4"))
    video_mask2 = load_video(os.path.join("example_videos", video_folder, "total_mask.mp4"))
    video_new_bg = load_video(args.new_bg)

    # Preprocess to tensors for VAE (VAE is FP32)
    video_p = pipe.video_processor.preprocess_video(video_p, width=w, height=h).to(device=device, dtype=torch.float32)
    video_bg = pipe.video_processor.preprocess_video(video_bg, width=w, height=h).to(device=device, dtype=torch.float32)
    video_mask = pipe.video_processor.preprocess_video(video_mask, width=w, height=h).to(device=device, dtype=torch.float32)
    video_mask2 = pipe.video_processor.preprocess_video(video_mask2, width=w, height=h).to(device=device, dtype=torch.float32)
    video_new_bg = pipe.video_processor.preprocess_video(video_new_bg, width=w, height=h).to(device=device, dtype=torch.float32)

    nframes = min(video_new_bg.shape[2], video_p.shape[2])
    video_p = video_p[:, :, :nframes]
    video_bg = video_bg[:, :, :nframes]
    video_mask = video_mask[:, :, :nframes]
    video_mask2 = video_mask2[:, :, :nframes]
    video_new_bg = video_new_bg[:, :, :nframes]

    temb = torch.tensor(0.0, device=device, dtype=torch.float32)
    x, foreground, z_mask, z_mask2 = pipe.vae(
        [video_p, video_bg, video_mask, video_mask2, video_new_bg],
        temb=temb,
    )

    noise = x.sample
    foreground = foreground.sample
    z_mask = z_mask.sample
    z_mask2 = z_mask2.sample

    video_mask_bin = (z_mask.detach().cpu().float() > 0).to(device=device, dtype=video_bg.dtype)
    video_mask2_bin = (z_mask2.detach().cpu().float() > 0).to(device=device, dtype=video_bg.dtype)

    # Foreground extraction with pixel injection
    foreground = foreground * (1 - video_mask_bin) + video_p * video_mask_bin
    foreground = foreground * video_mask2_bin

    out_foreground = tensor_video_to_pil_images(
        (pipe.video_processor.postprocess_video(foreground, output_type="pt")[0] * 255)
        .long()
        .permute(0, 2, 3, 1)
    )
    export_to_video(out_foreground, os.path.join("results", f"{video_folder}_foreground.mp4"), fps=24)

    # Latent addition to new background
    noise = noise * (1 - video_mask_bin) + video_p * video_mask_bin
    out_latent_add = tensor_video_to_pil_images(
        (pipe.video_processor.postprocess_video(noise, output_type="pt")[0] * 255)
        .long()
        .permute(0, 2, 3, 1)
    )
    export_to_video(out_latent_add, os.path.join("results", f"{video_folder}_latent_addition.mp4"), fps=24)

    # Refinement (few noising-denoising steps)
    condition_latents = retrieve_latents(pipe.vae.encode(noise), generator=None)
    condition_latents = pipe._normalize_latents(condition_latents, pipe.vae.latents_mean, pipe.vae.latents_std)
    condition_latents = condition_latents.to(device=device, dtype=dtype)

    prompt_kwargs = _load_or_build_prompt_embeds(
        pipe=pipe,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        checkpoint=args.checkpoint,
        dtype=dtype,
        device=device,
        use_prompt_cache=args.use_prompt_cache,
        max_sequence_length=args.max_sequence_length,
    )

    generator = get_generator(args.seed, device=device)
    refined = pipe(
        width=w,
        height=h,
        num_frames=nframes,
        denoise_strength=0.3,
        num_inference_steps=10,
        latents=condition_latents,
        decode_timestep=0.05,
        image_cond_noise_scale=0.025,
        generator=generator,
        output_type="pil",
        **prompt_kwargs,
    ).frames[0]

    export_to_video(refined, os.path.join("results", f"{video_folder}_refined.mp4"), fps=24)

    print_memory_stats()


if __name__ == "__main__":
    main()
