"""
Memory-optimized Foreground Layer Composition for OmnimatteZero.

This script extracts a foreground layer and its effects (shadows, reflections)
and composites them onto a new background video.

Optimized for 16GB VRAM GPUs.

Usage:
    python foreground_composition_optimized.py --preset 16gb
"""

from typing import Optional, Union
import argparse
import torch
from diffusers import AutoencoderKLLTXVideo
from diffusers.pipelines.ltx.pipeline_ltx_condition import retrieve_latents
from diffusers.utils import export_to_video, load_video
from PIL import Image
import os

from OmnimatteZero import OmnimatteZero
from memory_utils import (
    MemoryConfig,
    apply_memory_optimizations,
    apply_memory_optimizations_vae_only,
    clear_memory,
    print_memory_stats,
    round_frames_to_vae_compatible,
    MemoryTracker
)


def tensor_video_to_pil_images(video_tensor):
    """
    Converts a PyTorch tensor representing a video to a list of PIL Images.
    Memory-optimized version that processes frames incrementally.
    """
    # Remove the batch dimension
    video_tensor = video_tensor.squeeze(0)
    
    # Process frames one at a time to save memory
    pil_images = []
    for i in range(video_tensor.shape[0]):
        frame = video_tensor[i].cpu().numpy().astype('uint8')
        pil_images.append(Image.fromarray(frame))
    
    return pil_images


class MyAutoencoderKLLTXVideoOptimized(AutoencoderKLLTXVideo):
    """
    Memory-optimized VAE that processes videos in chunks.
    """

    def forward(
            self,
            sample: torch.Tensor,
            temb: Optional[torch.Tensor] = None,
            sample_posterior: bool = False,
            return_dict: bool = True,
            generator: Optional[torch.Generator] = None,
    ) -> Union[torch.Tensor, torch.Tensor]:
        all_vid, bg, mask, mask2, new_bg = sample
        
        # Process sequentially to reduce memory
        clear_memory()
        
        posterior = self.encode(all_vid).latent_dist
        z_all = posterior.mode() if not sample_posterior else posterior.sample(generator=generator)
        del posterior
        clear_memory()

        posterior = self.encode(bg).latent_dist
        z_bg = posterior.mode() if not sample_posterior else posterior.sample(generator=generator)
        del posterior
        clear_memory()

        posterior = self.encode(mask).latent_dist
        z_mask = posterior.mode() if not sample_posterior else posterior.sample(generator=generator)
        del posterior
        clear_memory()

        posterior = self.encode(mask2).latent_dist
        z_mask2 = posterior.mode() if not sample_posterior else posterior.sample(generator=generator)
        del posterior
        clear_memory()

        posterior = self.encode(new_bg).latent_dist
        z_new_bg = posterior.mode() if not sample_posterior else posterior.sample(generator=generator)
        del posterior
        clear_memory()

        z_diff = z_all - z_bg
        z = z_new_bg + z_diff
        
        del z_all, z_bg, z_new_bg
        clear_memory()

        dec = self.decode(z, temb)
        del z
        clear_memory()
        
        dec2 = self.decode(z_diff, temb)
        del z_diff
        clear_memory()
        
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
    ) -> torch.Tensor:
        posterior = self.encode(sample).latent_dist
        z = posterior.mode() if not sample_posterior else posterior.sample(generator=generator)
        if not return_dict:
            return (z,)
        return z


def parse_args():
    parser = argparse.ArgumentParser(description="Memory-optimized foreground composition")
    parser.add_argument("--preset", type=str, default="16gb", choices=["16gb", "24gb", "32gb"],
                       help="Memory optimization preset")
    parser.add_argument("--video_folder", type=str, default="swan_lake",
                       help="Video folder name (in example_videos/)")
    parser.add_argument("--new_bg_video", type=str, default="cat_reflection",
                       help="New background video (in results/)")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Output directory")
    parser.add_argument("--height", type=int, default=None,
                       help="Output height (auto-selected based on preset if not specified)")
    parser.add_argument("--width", type=int, default=None,
                       help="Output width (auto-selected based on preset if not specified)")
    parser.add_argument("--max_frames", type=int, default=None,
                       help="Maximum frames to process")
    parser.add_argument("--skip_refinement", action="store_true",
                       help="Skip the refinement step (faster, slightly lower quality)")
    parser.add_argument("--cache_dir", type=str, default=None,
                       help="Directory for model cache")
    return parser.parse_args()


def load_and_preprocess_video(pipe, video_path, width, height, max_frames=None):
    """Load and preprocess a video with proper frame handling."""
    video = load_video(video_path)
    
    # Limit frames
    if max_frames is not None and len(video) > max_frames:
        video = video[:max_frames]
    
    # Ensure VAE-compatible frame count
    num_frames = round_frames_to_vae_compatible(len(video))
    video = video[:num_frames]
    
    # Preprocess
    video_tensor = pipe.video_processor.preprocess_video(video, width=width, height=height)
    video_tensor = video_tensor.to(dtype=torch.bfloat16, device="cuda")
    
    return video_tensor


def main():
    args = parse_args()
    
    print("=" * 60)
    print("OmnimatteZero - Memory Optimized Foreground Composition")
    print("=" * 60)
    
    # Initialize configuration
    config = MemoryConfig(args.preset)
    
    # Get preset defaults or use command line overrides
    height = args.height or config.max_resolution[0]
    width = args.width or config.max_resolution[1]
    max_frames = args.max_frames or config.max_frames
    
    print(f"\nConfiguration:")
    print(f"  Preset: {args.preset}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Max frames: {max_frames}")
    print(f"  Refinement: {'No' if args.skip_refinement else 'Yes'}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load pipeline
    print("\nLoading pipeline...")
    pipe = OmnimatteZero.from_pretrained(
        "a-r-r-o-w/LTX-Video-0.9.7-diffusers",
        torch_dtype=torch.bfloat16,
        cache_dir=args.cache_dir
    )
    
    # Load optimized VAE
    pipe.vae = MyAutoencoderKLLTXVideoOptimized.from_pretrained(
        "a-r-r-o-w/LTX-Video-0.9.7-diffusers",
        subfolder="vae",
        torch_dtype=torch.bfloat16,
        cache_dir=args.cache_dir
    )
    
    # Apply memory optimizations
    if not config.enable_model_cpu_offload:
        pipe.to("cuda")
    pipe = apply_memory_optimizations(pipe, config)
    
    print_memory_stats()
    
    # Define paths
    example_dir = "example_videos"
    video_folder = args.video_folder
    
    # Load videos
    print("\nLoading videos...")
    
    video_p_path = f"./{example_dir}/{video_folder}/video.mp4"
    video_bg_path = f"./{args.output_dir}/{video_folder}.mp4"
    video_mask_path = f"./{example_dir}/{video_folder}/object_mask.mp4"
    video_mask2_path = f"./{example_dir}/{video_folder}/total_mask.mp4"
    video_new_bg_path = f"./{args.output_dir}/{args.new_bg_video}.mp4"
    
    # Check required files exist
    for path, desc in [
        (video_p_path, "Original video"),
        (video_bg_path, "Background (run object_removal first)"),
        (video_mask_path, "Object mask"),
        (video_mask2_path, "Total mask"),
        (video_new_bg_path, "New background video")
    ]:
        if not os.path.exists(path):
            print(f"Error: {desc} not found at {path}")
            return
    
    # Load and preprocess videos
    with MemoryTracker("Loading videos"):
        video_p = load_and_preprocess_video(pipe, video_p_path, width, height, max_frames)
        video_bg = load_and_preprocess_video(pipe, video_bg_path, width, height, max_frames)
        video_mask = load_and_preprocess_video(pipe, video_mask_path, width, height, max_frames)
        video_mask2 = load_and_preprocess_video(pipe, video_mask2_path, width, height, max_frames)
        video_new_bg = load_and_preprocess_video(pipe, video_new_bg_path, width, height, max_frames)
    
    # Ensure all videos have the same number of frames
    nframes = min(
        video_new_bg.shape[2], video_p.shape[2], video_bg.shape[2],
        video_mask.shape[2], video_mask2.shape[2]
    )
    nframes = round_frames_to_vae_compatible(nframes)
    
    video_p = video_p[:, :, :nframes, :, :]
    video_bg = video_bg[:, :, :nframes, :, :]
    video_mask = video_mask[:, :, :nframes, :, :]
    video_mask2 = video_mask2[:, :, :nframes, :, :]
    video_new_bg = video_new_bg[:, :, :nframes, :, :]
    
    print(f"Processing {nframes} frames at {width}x{height}")
    
    # Perform composition
    print("\nCompositing foreground layer...")
    with torch.no_grad():
        with MemoryTracker("VAE encoding/decoding"):
            x, foreground, z_mask, z_mask2 = pipe.vae(
                [video_p, video_bg, video_mask, video_mask2, video_new_bg],
                temb=torch.tensor(0.0, device="cuda", dtype=torch.bfloat16)
            )
        
        noise = x.sample
        foreground = foreground.sample
        video_mask_dec = z_mask.sample
        video_mask2_dec = z_mask2.sample
        
        # Free decoded mask tensors after binarization
        video_mask_bin = (video_mask_dec.cpu().float() > 0).type(video_bg.dtype).cuda()
        video_mask2_bin = (video_mask2_dec.cpu().float() > 0).type(video_bg.dtype).cuda()
        
        del x, z_mask, z_mask2, video_mask_dec, video_mask2_dec
        clear_memory()
        
        # Extract foreground layer with pixel injection
        foreground = foreground * (1 - video_mask_bin) + video_p * video_mask_bin
        foreground = foreground * video_mask2_bin
        
        # Save foreground
        video_foreground = tensor_video_to_pil_images(
            ((pipe.video_processor.postprocess_video(foreground, output_type='pt')[0] * 255).long().permute(0, 2, 3, 1))
        )
        foreground_path = os.path.join(args.output_dir, f"{video_folder}_foreground.mp4")
        export_to_video(video_foreground, foreground_path, fps=24)
        print(f"  Saved foreground to: {foreground_path}")
        
        del foreground
        clear_memory()
        
        # Latent addition to new background
        noise = noise * (1 - video_mask_bin) + video_p * video_mask_bin
        
        video_output = tensor_video_to_pil_images(
            ((pipe.video_processor.postprocess_video(noise, output_type='pt')[0] * 255).long().permute(0, 2, 3, 1))
        )
        latent_add_path = os.path.join(args.output_dir, f"{video_folder}_latent_addition.mp4")
        export_to_video(video_output, latent_add_path, fps=24)
        print(f"  Saved latent addition to: {latent_add_path}")
        
        del video_mask_bin, video_mask2_bin, video_p, video_bg, video_mask, video_mask2, video_new_bg
        clear_memory()
        
        # Apply refinement (optional)
        if not args.skip_refinement:
            print("\nApplying refinement...")
            with MemoryTracker("Refinement"):
                condition_latents = retrieve_latents(pipe.vae.encode(noise), generator=None)
                condition_latents = pipe._normalize_latents(
                    condition_latents, pipe.vae.latents_mean, pipe.vae.latents_std
                ).to(noise.device, dtype=noise.dtype)
                
                del noise
                clear_memory()
                
                prompt = ""
                negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"
                expected_height, expected_width = video_output[0].size[1], video_output[0].size[0]
                num_frames = len(video_output)
                
                # Fewer steps for memory efficiency
                num_steps = 8 if args.preset == "16gb" else 10
                
                video_refined = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=expected_width,
                    height=expected_height,
                    num_frames=num_frames,
                    denoise_strength=0.3,
                    num_inference_steps=num_steps,
                    latents=condition_latents,
                    decode_timestep=0.05,
                    image_cond_noise_scale=0.025,
                    generator=torch.Generator().manual_seed(0),
                    output_type="pil",
                ).frames[0]
                
                refined_path = os.path.join(args.output_dir, f"{video_folder}_refined.mp4")
                export_to_video(video_refined, refined_path, fps=24)
                print(f"  Saved refined output to: {refined_path}")
        else:
            print("\nSkipping refinement (--skip_refinement)")
    
    print("\n=== Complete ===")
    print_memory_stats()


if __name__ == "__main__":
    main()
