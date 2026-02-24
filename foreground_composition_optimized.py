"""
Memory-optimized Foreground Composition for OmnimatteZero.

Extracts foreground layers and composes them onto new backgrounds using
latent-space operations. Optimized for consumer GPUs and Apple Silicon (MPS).

Usage:
    python foreground_composition_optimized.py [--preset mps_24gb|16gb|24gb|32gb]
"""

import os
import argparse
from typing import Optional, Union
import torch
from diffusers import AutoencoderKLLTXVideo
from diffusers.pipelines.ltx.pipeline_ltx_condition import retrieve_latents
from diffusers.utils import export_to_video, load_video
from PIL import Image

from OmnimatteZero import OmnimatteZero
from device_utils import (
    get_device, get_device_type, is_mps,
    get_optimal_dtype, get_generator,
    clear_memory, print_memory_stats, MemoryTracker
)
from memory_utils import (
    MemoryConfig,
    apply_memory_optimizations,
    round_frames_to_vae_compatible,
    auto_configure
)


def tensor_video_to_pil_images(video_tensor):
    """
    Converts a PyTorch tensor representing a video to a list of PIL Images.
    
    Args:
        video_tensor: Tensor of shape (1, frames, height, width, 3).
    Returns:
        List of PIL Images.
    """
    video_tensor = video_tensor.squeeze(0)
    video_numpy = video_tensor.cpu().numpy()
    pil_images = [Image.fromarray(frame.astype('uint8')) for frame in video_numpy]
    return pil_images


class MyAutoencoderKLLTXVideoOptimized(AutoencoderKLLTXVideo):
    """
    Memory-optimized VAE that processes video chunks sequentially.
    Reduces peak memory by encoding/decoding one video at a time.
    Cross-platform: supports CUDA and MPS.
    """

    def forward(
            self,
            sample,
            temb: Optional[torch.Tensor] = None,
            sample_posterior: bool = False,
            return_dict: bool = True,
            generator: Optional[torch.Generator] = None,
    ) -> Union[torch.Tensor, torch.Tensor]:
        """Optimized forward that processes videos sequentially to save memory."""
        all_video, bg, mask, mask2, new_bg = sample

        # Process each encoding sequentially, clearing memory between
        z_all = self._encode_single(all_video, sample_posterior, generator)
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

        # Free unused latents
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

    def _encode_single(self, video, sample_posterior, generator):
        """Encode a single video, clearing memory after."""
        posterior = self.encode(video).latent_dist
        if sample_posterior:
            return posterior.sample(generator=generator)
        return posterior.mode()

    def forward_encode(
            self,
            sample: torch.Tensor,
            temb: Optional[torch.Tensor] = None,
            sample_posterior: bool = False,
            return_dict: bool = True,
            generator: Optional[torch.Generator] = None,
    ) -> Union[torch.Tensor, torch.Tensor]:
        posterior = self.encode(sample).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()

        if not return_dict:
            return (z,)
        return z


def parse_args():
    parser = argparse.ArgumentParser(description="Memory-optimized foreground composition")
    parser.add_argument("--preset", type=str, default=None,
                       choices=["mps_24gb", "16gb", "24gb", "32gb"])
    parser.add_argument("--video_folder", type=str, default="swan_lake",
                       help="Video folder name within example_videos/")
    parser.add_argument("--new_bg_path", type=str, default="./results/cat_reflection.mp4",
                       help="Path to the new background video")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Output directory")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--denoise_strength", type=float, default=0.3,
                       help="Strength for refinement denoising (0=none, 1=full)")
    parser.add_argument("--refine_steps", type=int, default=10,
                       help="Number of refinement steps")
    parser.add_argument("--skip_refinement", action="store_true",
                       help="Skip the optional refinement step")
    parser.add_argument("--cache_dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("OmnimatteZero - Memory Optimized Foreground Composition")
    print("=" * 60)

    # Initialize configuration
    if args.preset:
        config = MemoryConfig(args.preset)
    else:
        config = auto_configure()
    
    device = get_device()
    dtype = get_optimal_dtype()
    
    height = args.height or config.max_resolution[0]
    width = args.width or config.max_resolution[1]
    
    print(f"\nConfiguration:")
    print(f"  Preset: {config.preset}")
    print(f"  Device: {get_device_type()}")
    print(f"  Resolution: {width}x{height}")

    # Load pipeline with optimized VAE
    print("\nLoading pipeline from local checkpoint...")
    from device_utils import load_pipeline as load_base_pipeline, load_vae
    pipe = load_base_pipeline(cache_dir=args.cache_dir)
    
    # Replace VAE with optimized version that processes sequentially
    optimized_vae = MyAutoencoderKLLTXVideoOptimized.from_config(pipe.vae.config)
    optimized_vae.load_state_dict(pipe.vae.state_dict())
    optimized_vae = optimized_vae.to(dtype=dtype)
    pipe.vae = optimized_vae
    
    if not config.enable_model_cpu_offload:
        pipe.to(device)
    pipe = apply_memory_optimizations(pipe, config)
    
    print_memory_stats()

    # Load all input videos
    w, h = width, height
    video_folder = args.video_folder

    print(f"\nLoading videos from: {video_folder}")
    
    video_p = load_video(f"./example_videos/{video_folder}/video.mp4")
    video_p = pipe.video_processor.preprocess_video(video_p, width=w, height=h).to(dtype=dtype, device=device)

    video_bg = load_video(f"./results/{video_folder}.mp4")
    video_bg = pipe.video_processor.preprocess_video(video_bg, width=w, height=h).to(dtype=dtype, device=device)

    video_mask = load_video(f"./example_videos/{video_folder}/object_mask.mp4")
    video_mask = pipe.video_processor.preprocess_video(video_mask, width=w, height=h).to(dtype=dtype, device=device)

    video_mask2 = load_video(f"./example_videos/{video_folder}/total_mask.mp4")
    video_mask2 = pipe.video_processor.preprocess_video(video_mask2, width=w, height=h).to(dtype=dtype, device=device)

    video_new_bg = load_video(args.new_bg_path)
    video_new_bg = pipe.video_processor.preprocess_video(video_new_bg, width=w, height=h).to(dtype=dtype, device=device)

    # Align frame counts
    nframes = min(video_new_bg.shape[2], video_p.shape[2])
    nframes = round_frames_to_vae_compatible(nframes)
    video_p = video_p[:, :, :nframes, :, :]
    video_bg = video_bg[:, :, :nframes, :, :]
    video_mask = video_mask[:, :, :nframes, :, :]
    video_mask2 = video_mask2[:, :, :nframes, :, :]
    video_new_bg = video_new_bg[:, :, :nframes, :, :]

    print(f"  Frames: {nframes}, Resolution: {w}x{h}")

    os.makedirs(args.output_dir, exist_ok=True)

    with torch.no_grad():
        print("\nEncoding and composing in latent space...")
        with MemoryTracker("VAE Encoding/Decoding"):
            x, foreground, z_mask, z_mask2 = pipe.vae(
                [video_p, video_bg, video_mask, video_mask2, video_new_bg],
                temb=torch.tensor(0.0, device=device, dtype=dtype)
            )
        
        noise = x.sample
        foreground = foreground.sample
        video_mask_decoded = z_mask.sample
        video_mask2_decoded = z_mask2.sample
        
        del x, z_mask, z_mask2
        clear_memory()
        
        # Binarize masks
        video_mask_decoded = (video_mask_decoded.cpu().float() > 0).to(dtype=dtype, device=device)
        video_mask2_decoded = (video_mask2_decoded.cpu().float() > 0).to(dtype=dtype, device=device)

        # Extract foreground layer with pixel injection
        foreground = foreground * (1 - video_mask_decoded) + video_p * video_mask_decoded
        foreground = foreground * video_mask2_decoded
        video_foreground = tensor_video_to_pil_images(
            ((pipe.video_processor.postprocess_video(foreground, output_type='pt')[0] * 255)
             .long().permute(0, 2, 3, 1)))
        export_to_video(video_foreground, os.path.join(args.output_dir, "foreground.mp4"), fps=24)
        print(f"  Saved foreground to: {args.output_dir}/foreground.mp4")
        
        del foreground, video_mask_decoded, video_mask2_decoded
        clear_memory()

        # Latent addition to new background
        noise = noise * (1 - video_mask_decoded if 'video_mask_decoded' in dir() 
                         else (video_mask.cpu().float() > 0).to(dtype=dtype, device=device).mean(dim=1, keepdim=True)) + \
                video_p * (video_mask.mean(dim=1, keepdim=True) > 0.5).float().to(device)
        
        # Simplified: reload mask for latent addition
        video_mask_bin = (video_mask.cpu().float().mean(dim=1, keepdim=True) > 0).to(dtype=dtype, device=device)
        noise_out = noise * (1 - video_mask_bin) + video_p * video_mask_bin
        
        video_composed = tensor_video_to_pil_images(
            ((pipe.video_processor.postprocess_video(noise_out, output_type='pt')[0] * 255)
             .long().permute(0, 2, 3, 1)))
        export_to_video(video_composed, os.path.join(args.output_dir, "latent_addition.mp4"), fps=24)
        print(f"  Saved latent addition to: {args.output_dir}/latent_addition.mp4")

        # Optional refinement
        if not args.skip_refinement:
            print("\nRefining composed video...")
            clear_memory()
            
            condition_latents = retrieve_latents(pipe.vae.encode(noise_out), generator=None)
            condition_latents = pipe._normalize_latents(
                condition_latents, pipe.vae.latents_mean, pipe.vae.latents_std
            ).to(device=device, dtype=dtype)
            
            del noise_out
            clear_memory()
            
            prompt = ""
            negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"
            expected_height, expected_width = video_composed[0].size[1], video_composed[0].size[0]
            num_frames = len(video_composed)

            with MemoryTracker("Refinement"):
                video_refined = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=expected_width,
                    height=expected_height,
                    num_frames=num_frames,
                    denoise_strength=args.denoise_strength,
                    num_inference_steps=args.refine_steps,
                    latents=condition_latents,
                    decode_timestep=0.05,
                    image_cond_noise_scale=0.025,
                    generator=get_generator(seed=0),
                    output_type="pil",
                ).frames[0]
            
            export_to_video(video_refined, os.path.join(args.output_dir, "refinement.mp4"), fps=24)
            print(f"  Saved refinement to: {args.output_dir}/refinement.mp4")

    print("\n=== Complete ===")
    print_memory_stats()


if __name__ == "__main__":
    main()
