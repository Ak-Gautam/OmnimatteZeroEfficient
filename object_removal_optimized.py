"""
Optimized Object Removal for OmnimatteZero.

This script removes objects and their effects (shadows, reflections) from videos
using memory-efficient techniques optimized for consumer GPUs (16GB+ VRAM)
and Apple Silicon (MPS).

Usage:
    python object_removal_optimized.py [--preset mps_24gb|16gb|24gb|32gb] [--video VIDEO_FOLDER]
"""

import os
import argparse
from tqdm import tqdm
import torch
from diffusers import LTXLatentUpsamplePipeline
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from diffusers.utils import export_to_video, load_video

from OmnimatteZero import OmnimatteZero
from device_utils import (
    get_device, get_device_type, is_mps,
    get_optimal_dtype, get_generator,
    clear_memory, print_memory_stats, MemoryTracker,
    load_pipeline as load_base_pipeline
)
from memory_utils import (
    MemoryConfig, 
    apply_memory_optimizations,
    round_to_vae_compatible,
    round_frames_to_vae_compatible,
    auto_configure
)


def parse_args():
    parser = argparse.ArgumentParser(description="Memory-optimized object removal")
    parser.add_argument("--preset", type=str, default=None,
                       choices=["mps_24gb", "16gb", "24gb", "32gb"],
                       help="Memory optimization preset (auto-detected if not specified)")
    parser.add_argument("--video", type=str, default=None,
                       help="Specific video folder to process (processes all if not specified)")
    parser.add_argument("--base_dir", type=str, default="example_videos",
                       help="Base directory containing video folders")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Output directory for processed videos")
    parser.add_argument("--height", type=int, default=None,
                       help="Output height (auto-selected based on preset if not specified)")
    parser.add_argument("--width", type=int, default=None,
                       help="Output width (auto-selected based on preset if not specified)")
    parser.add_argument("--max_frames", type=int, default=None,
                       help="Maximum frames to process (auto-selected based on preset if not specified)")
    parser.add_argument("--num_inference_steps", type=int, default=None,
                       help="Number of inference steps (fewer = faster but lower quality)")
    parser.add_argument("--skip_upscale", action="store_true",
                       help="Skip latent upscaling (faster, smaller output)")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Directory for model cache")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .safetensors checkpoint (default: model_checkpoint/ltx-video-2b-v0.9.5.safetensors)")
    return parser.parse_args()


def load_pipeline(config: MemoryConfig, cache_dir: str = None, checkpoint: str = None):
    """Load and optimize the OmnimatteZero pipeline from local checkpoint."""
    print("\n=== Loading Pipeline ===")
    
    device = get_device()
    
    # Load from local safetensors checkpoint
    pipe = load_base_pipeline(
        checkpoint_path=checkpoint,
        cache_dir=cache_dir
    )
    
    # Apply memory optimizations
    # For MPS: enable_model_cpu_offload handles device placement
    # For CUDA without model_cpu_offload: move to device first
    if not config.enable_model_cpu_offload:
        pipe.to(device)
    
    pipe = apply_memory_optimizations(pipe, config)
    
    print_memory_stats()
    return pipe


def load_upscaler(config: MemoryConfig, vae, cache_dir: str = None):
    """Load and optimize the latent upscaler pipeline."""
    print("\n=== Loading Upscaler ===")
    
    dtype = get_optimal_dtype()
    
    pipe_upsample = LTXLatentUpsamplePipeline.from_pretrained(
        "a-r-r-o-w/LTX-Video-0.9.7-Latent-Spatial-Upsampler-diffusers",
        vae=vae,
        torch_dtype=dtype,
        cache_dir=cache_dir
    )
    
    # Always use model CPU offload for upscaler to save memory
    pipe_upsample.enable_model_cpu_offload()
    
    print_memory_stats()
    return pipe_upsample


def process_video(
    pipe,
    video_path: str,
    mask_path: str,
    output_path: str,
    config: MemoryConfig,
    height: int,
    width: int,
    max_frames: int,
    num_inference_steps: int,
    pipe_upsample=None
):
    """Process a single video for object removal."""
    
    # Load video
    try:
        video = load_video(video_path)
        mask = load_video(mask_path)
    except Exception as e:
        print(f"  Error loading video: {e}")
        return False
    
    # Limit frames
    num_frames = min(len(video), max_frames)
    num_frames = round_frames_to_vae_compatible(num_frames)
    video = video[:num_frames]
    mask = mask[:num_frames]
    
    # Prepare conditions
    condition1 = LTXVideoCondition(video=video, frame_index=0)
    condition2 = LTXVideoCondition(video=mask, frame_index=0)
    
    prompt = "Empty"  # Minimal prompt for inpainting
    negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"
    
    # Round dimensions to VAE-compatible values
    downscaled_height, downscaled_width = round_to_vae_compatible(height, width)
    
    print(f"  Resolution: {downscaled_width}x{downscaled_height}")
    print(f"  Frames: {num_frames}")
    print(f"  Steps: {num_inference_steps}")
    
    # Clear memory before generation
    clear_memory()
    
    with MemoryTracker("Generation"):
        # Generate with optimized settings
        generator = get_generator(seed=1)
        
        output = pipe.my_call(
            conditions=[condition1, condition2],
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=downscaled_width,
            height=downscaled_height,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            generator=generator,
            output_type="pil",
        )
        video_output = output.frames[0]
    
    # Optional upscaling
    if pipe_upsample is not None:
        print("  Upscaling...")
        clear_memory()
        with MemoryTracker("Upscaling"):
            video_output = pipe_upsample(
                video=video_output,
                output_type="pil"
            ).frames[0]
    
    # Export
    export_to_video(video_output, output_path, fps=24)
    print(f"  Saved to: {output_path}")
    
    # Clean up
    clear_memory()
    return True


def main():
    args = parse_args()
    
    print("=" * 60)
    print("OmnimatteZero - Memory Optimized Object Removal")
    print("=" * 60)
    
    # Initialize configuration
    if args.preset:
        config = MemoryConfig(args.preset)
    else:
        config = auto_configure()
    
    # Get preset defaults or use command line overrides
    height = args.height or config.max_resolution[0]
    width = args.width or config.max_resolution[1]
    max_frames = args.max_frames or config.max_frames
    
    # Inference steps based on preset
    default_steps = {
        "mps_24gb": 30,  # User-proven: 30 steps works well
        "16gb": 20,
        "24gb": 25,
        "32gb": 30
    }
    num_inference_steps = args.num_inference_steps or default_steps.get(config.preset, 25)
    
    print(f"\nConfiguration:")
    print(f"  Preset: {config.preset}")
    print(f"  Device: {get_device_type()}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Max frames: {max_frames}")
    print(f"  Inference steps: {num_inference_steps}")
    print(f"  Upscaling: {'No' if args.skip_upscale else 'Yes'}")
    
    # Load pipeline
    pipe = load_pipeline(config, args.cache_dir, args.checkpoint)
    
    # Load upscaler if needed (skip on MPS by default to save memory)
    pipe_upsample = None
    if not args.skip_upscale:
        if is_mps():
            print("Note: Skipping upscaler on Apple Silicon to conserve memory.")
            print("  Use a CUDA GPU or run upscaling separately for higher resolution.")
        else:
            try:
                pipe_upsample = load_upscaler(config, pipe.vae, args.cache_dir)
            except Exception as e:
                print(f"Warning: Could not load upscaler: {e}")
                print("Continuing without upscaling...")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of videos to process
    if args.video:
        video_folders = [args.video]
    else:
        video_folders = os.listdir(args.base_dir)
    
    # Process videos
    print(f"\n=== Processing {len(video_folders)} video(s) ===")
    
    for f_name in tqdm(video_folders, desc="Videos"):
        video_dir = os.path.join(args.base_dir, f_name)
        
        if not os.path.isdir(video_dir):
            continue
            
        print(f"\nProcessing: {f_name}")
        
        video_path = os.path.join(video_dir, "video.mp4")
        mask_path = os.path.join(video_dir, "total_mask.mp4")
        output_path = os.path.join(args.output_dir, f"{f_name}_mps.mp4")
        
        if not os.path.exists(video_path):
            print(f"  Skipping: video.mp4 not found")
            continue
            
        if not os.path.exists(mask_path):
            print(f"  Skipping: total_mask.mp4 not found")
            continue
        
        success = process_video(
            pipe=pipe,
            video_path=video_path,
            mask_path=mask_path,
            output_path=output_path,
            config=config,
            height=height,
            width=width,
            max_frames=max_frames,
            num_inference_steps=num_inference_steps,
            pipe_upsample=pipe_upsample
        )
        
        if not success:
            print(f"  Failed to process {f_name}")
    
    print("\n=== Complete ===")
    print_memory_stats()


if __name__ == "__main__":
    main()
