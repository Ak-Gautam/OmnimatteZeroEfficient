"""
Optimized Object Removal for OmnimatteZero.

This script removes objects and their effects (shadows, reflections) from videos
using memory-efficient techniques optimized for consumer GPUs (16GB+ VRAM).

Usage:
    python object_removal_optimized.py [--preset 16gb|24gb|32gb] [--video VIDEO_FOLDER]
"""

import os
import argparse
from tqdm import tqdm
import torch
from diffusers import LTXLatentUpsamplePipeline
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from diffusers.utils import export_to_video, load_video
from transformers import AutoTokenizer, T5EncoderModel

from OmnimatteZero_mps import OmnimatteZero
from memory_utils_mps import (
    MemoryConfig, 
    apply_memory_optimizations,
    clear_memory,
    print_memory_stats,
    round_to_vae_compatible,
    round_frames_to_vae_compatible,
    MemoryTracker,
    auto_configure
)


def parse_args():
    parser = argparse.ArgumentParser(description="Memory-optimized object removal")
    parser.add_argument("--preset", type=str, default="16gb", choices=["16gb", "24gb", "32gb"],
                       help="Memory optimization preset")
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
    return parser.parse_args()


def precompute_text_embeddings(prompt: str, negative_prompt: str, cache_dir: str = None):
    """Load text encoder, compute embeddings, and unload it to free memory."""
    print("\n=== Precomputing Text Embeddings ===")
    
    tokenizer = AutoTokenizer.from_pretrained(
        "a-r-r-o-w/LTX-Video-0.9.7-diffusers", 
        subfolder="tokenizer",
        cache_dir=cache_dir
    )
    
    text_encoder = T5EncoderModel.from_pretrained(
        "a-r-r-o-w/LTX-Video-0.9.7-diffusers", 
        subfolder="text_encoder",
        torch_dtype=torch.float16,
        cache_dir=cache_dir
    )
    
    if torch.backends.mps.is_available():
        text_encoder.to("mps")
        device = "mps"
    else:
        device = "cpu"
        
    def encode_text(text):
        text_inputs = tokenizer(
            text,
            padding="max_length",
            max_length=256,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        attention_mask = text_inputs.attention_mask.to(device)
        prompt_embeds = text_encoder(text_inputs.input_ids.to(device), attention_mask=attention_mask)[0]
        return prompt_embeds.detach().cpu(), attention_mask.detach().cpu()

    with torch.no_grad():
        prompt_embeds, prompt_attention_mask = encode_text(prompt)
        negative_prompt_embeds, negative_prompt_attention_mask = encode_text(negative_prompt)
        
    del text_encoder
    del tokenizer
    clear_memory()
    
    return prompt_embeds.to("mps" if torch.backends.mps.is_available() else "cpu"), prompt_attention_mask.to("mps" if torch.backends.mps.is_available() else "cpu"), negative_prompt_embeds.to("mps" if torch.backends.mps.is_available() else "cpu"), negative_prompt_attention_mask.to("mps" if torch.backends.mps.is_available() else "cpu")


def load_pipeline(config: MemoryConfig, cache_dir: str = None):
    """Load and optimize the OmnimatteZero pipeline."""
    print("\n=== Loading Pipeline ===")
    
    # Load with float16 for memory efficiency and skip text encoder
    pipe = OmnimatteZero.from_pretrained(
        "a-r-r-o-w/LTX-Video-0.9.7-diffusers",
        text_encoder=None,
        tokenizer=None,
        torch_dtype=torch.float16,
        cache_dir=cache_dir
    )
    
    # Apply memory optimizations BEFORE moving to GPU
    # This is crucial for group offloading to work properly
    if not config.enable_model_cpu_offload:
        pipe.to("mps")
    
    pipe = apply_memory_optimizations(pipe, config)
    
    print_memory_stats()
    return pipe


def load_upscaler(config: MemoryConfig, vae, cache_dir: str = None):
    """Load and optimize the latent upscaler pipeline."""
    print("\n=== Loading Upscaler ===")
    
    pipe_upsample = LTXLatentUpsamplePipeline.from_pretrained(
        "a-r-r-o-w/LTX-Video-0.9.7-Latent-Spatial-Upsampler-diffusers",
        vae=vae,
        torch_dtype=torch.float16,
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
    prompt_embeds,
    prompt_attention_mask,
    negative_prompt_embeds,
    negative_prompt_attention_mask,
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
    
    # prompt = "Empty"  # Minimal prompt for inpainting
    # negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"
    
    # Round dimensions to VAE-compatible values
    downscaled_height, downscaled_width = round_to_vae_compatible(height, width)
    
    print(f"  Resolution: {downscaled_width}x{downscaled_height}")
    print(f"  Frames: {num_frames}")
    print(f"  Steps: {num_inference_steps}")
    
    # Clear memory before generation
    clear_memory()
    
    with MemoryTracker("Generation"):
        # Generate with optimized settings
        output = pipe.my_call(
            conditions=[condition1, condition2],
            prompt=None,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            width=downscaled_width,
            height=downscaled_height,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator(device="mps" if torch.backends.mps.is_available() else "cpu").manual_seed(1),
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
    config = MemoryConfig(args.preset)
    
    # Get preset defaults or use command line overrides
    height = args.height or config.max_resolution[0]
    width = args.width or config.max_resolution[1]
    max_frames = args.max_frames or config.max_frames
    
    # Inference steps based on preset
    default_steps = {
        "16gb": 20,  # Reduced for memory
        "24gb": 25,
        "32gb": 30
    }
    num_inference_steps = args.num_inference_steps or default_steps.get(args.preset, 25)
    
    print(f"\nConfiguration:")
    print(f"  Preset: {args.preset}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Max frames: {max_frames}")
    print(f"  Inference steps: {num_inference_steps}")
    print(f"  Upscaling: {'No' if args.skip_upscale else 'Yes'}")
    
    prompt = "Empty"
    negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"
    pe, pam, ne, nam = precompute_text_embeddings(prompt, negative_prompt, args.cache_dir)
    
    # Load pipeline
    pipe = load_pipeline(config, args.cache_dir)
    
    # Load upscaler if needed
    pipe_upsample = None
    if not args.skip_upscale:
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
        output_path = os.path.join(args.output_dir, f"{f_name}.mp4")
        
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
            prompt_embeds=pe,
            prompt_attention_mask=pam,
            negative_prompt_embeds=ne,
            negative_prompt_attention_mask=nam,
            pipe_upsample=pipe_upsample
        )
        
        if not success:
            print(f"  Failed to process {f_name}")
    
    print("\n=== Complete ===")
    print_memory_stats()


if __name__ == "__main__":
    main()
