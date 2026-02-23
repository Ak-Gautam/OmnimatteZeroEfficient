"""
Object Removal for OmnimatteZero — Apple Silicon (MPS) version.

Uses the 2B distilled model (ltxv-2b-0.9.8-distilled) for efficient inference.
Optimized for MacBook M4 Pro with 24GB unified memory.
Supports 704x480 @ 97 frames on 24GB.

Strategy:
  1. Load text encoder separately, encode prompts, then delete it
  2. Load 2B distilled transformer via from_single_file
  3. Build pipeline with transformer + VAE (no text encoder)
  4. Run inference with pre-computed embeddings
  5. No upscaler (saves ~4GB)

Usage:
    python object_removal_optimized_mps.py --preset 24gb --video cat_reflection
    python object_removal_optimized_mps.py --preset 16gb --video cat_reflection
"""

import os
import argparse
from tqdm import tqdm
import torch
from diffusers import LTXVideoTransformer3DModel, AutoencoderKLLTXVideo
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
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
)

# The checkpoint repo containing model files
CHECKPOINT_REPO = "Lightricks/LTX-Video"
# The specific 2B distilled checkpoint file
TRANSFORMER_FILE = "ltxv-2b-0.9.8-distilled.safetensors"
# Reference diffusers-format repo for VAE, scheduler, tokenizer, text encoder config
# (the 2B model shares the same VAE/scheduler as 0.9.7)
DIFFUSERS_REPO = "a-r-r-o-w/LTX-Video-0.9.7-diffusers"


def parse_args():
    parser = argparse.ArgumentParser(description="MPS-optimized object removal (2B distilled)")
    parser.add_argument("--preset", type=str, default="24gb",
                        choices=["16gb", "24gb", "32gb"],
                        help="Memory preset (default: 24gb for M4 Pro)")
    parser.add_argument("--video", type=str, default=None,
                        help="Specific video folder to process")
    parser.add_argument("--base_dir", type=str, default="example_videos",
                        help="Base directory containing video folders")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="HuggingFace cache directory")
    return parser.parse_args()


def precompute_text_embeddings(prompt: str, negative_prompt: str,
                                cache_dir: str = None):
    """
    Load text encoder, compute embeddings, then DELETE it to free memory.

    This is the key memory optimization: the T5 text encoder (~4.5GB in FP16)
    never coexists in memory with the transformer (~3.2GB in FP16 for 2B).
    """
    print("\n=== Precomputing Text Embeddings ===")

    tokenizer = AutoTokenizer.from_pretrained(
        DIFFUSERS_REPO, subfolder="tokenizer", cache_dir=cache_dir
    )

    text_encoder = T5EncoderModel.from_pretrained(
        DIFFUSERS_REPO, subfolder="text_encoder",
        torch_dtype=torch.float16, cache_dir=cache_dir
    )

    # Encode on MPS if available, else CPU
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    text_encoder.to(device)
    print(f"  Text encoder loaded on {device}")
    print_memory_stats()

    def encode_text(text):
        inputs = tokenizer(
            text,
            padding="max_length",
            max_length=256,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        attention_mask = inputs.attention_mask.to(device)
        embeds = text_encoder(
            inputs.input_ids.to(device), attention_mask=attention_mask
        )[0]
        return embeds.detach().cpu(), attention_mask.detach().cpu()

    with torch.no_grad():
        prompt_embeds, prompt_mask = encode_text(prompt)
        neg_embeds, neg_mask = encode_text(negative_prompt)

    # Free the text encoder — this is the whole point
    del text_encoder
    del tokenizer
    clear_memory()

    print("  ✓ Text encoder unloaded")
    print_memory_stats()

    return prompt_embeds, prompt_mask, neg_embeds, neg_mask


def load_pipeline(config: MemoryConfig, cache_dir: str = None):
    """
    Load OmnimatteZero pipeline with the 2B distilled transformer.

    Steps:
      1. Load the 2B distilled transformer from single file
      2. Load VAE and scheduler from the diffusers-format repo
      3. Assemble the pipeline without text_encoder/tokenizer
    """
    print("\n=== Loading Pipeline (2B distilled, no text encoder) ===")

    # Load the 2B distilled transformer from the checkpoint repo
    print(f"  Loading transformer: {CHECKPOINT_REPO}/{TRANSFORMER_FILE}")
    transformer = LTXVideoTransformer3DModel.from_single_file(
        f"https://huggingface.co/{CHECKPOINT_REPO}/blob/main/{TRANSFORMER_FILE}",
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
    )
    print(f"  ✓ Transformer loaded ({sum(p.numel() for p in transformer.parameters()) / 1e9:.2f}B params)")

    # Load VAE from the diffusers-format repo (shared across LTX versions)
    print(f"  Loading VAE from {DIFFUSERS_REPO}")
    vae = AutoencoderKLLTXVideo.from_pretrained(
        DIFFUSERS_REPO,
        subfolder="vae",
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
    )

    # Load scheduler from the diffusers-format repo
    print(f"  Loading scheduler from {DIFFUSERS_REPO}")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        DIFFUSERS_REPO,
        subfolder="scheduler",
        cache_dir=cache_dir,
    )

    # Assemble pipeline
    pipe = OmnimatteZero(
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        text_encoder=None,
        tokenizer=None,
    )

    pipe.to("mps")
    pipe = apply_memory_optimizations(pipe, config)

    print_memory_stats()
    return pipe


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
    prompt_mask,
    neg_embeds,
    neg_mask,
):
    """Process a single video for object removal."""

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

    condition1 = LTXVideoCondition(video=video, frame_index=0)
    condition2 = LTXVideoCondition(video=mask, frame_index=0)

    downscaled_height, downscaled_width = round_to_vae_compatible(height, width)

    print(f"  Resolution: {downscaled_width}x{downscaled_height}")
    print(f"  Frames: {num_frames}")
    print(f"  Steps: {num_inference_steps}")

    clear_memory()

    # Move embeddings to MPS for inference
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pe = prompt_embeds.to(device)
    pm = prompt_mask.to(device)
    ne = neg_embeds.to(device)
    nm = neg_mask.to(device)

    with MemoryTracker("Generation"):
        output = pipe.my_call(
            conditions=[condition1, condition2],
            prompt=None,
            prompt_embeds=pe,
            prompt_attention_mask=pm,
            negative_prompt_embeds=ne,
            negative_prompt_attention_mask=nm,
            width=downscaled_width,
            height=downscaled_height,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            # Use CPU generator — more reliable with MPS
            generator=torch.Generator(device="cpu").manual_seed(1),
            output_type="pil",
        )
        video_output = output.frames[0]

    export_to_video(video_output, output_path, fps=24)
    print(f"  Saved to: {output_path}")

    clear_memory()
    return True


def main():
    args = parse_args()

    print("=" * 60)
    print("OmnimatteZero — MPS Object Removal (2B Distilled)")
    print("=" * 60)

    if not torch.backends.mps.is_available():
        print("WARNING: MPS not available. This will run on CPU and be very slow.")

    config = MemoryConfig(args.preset)

    height = args.height or config.max_resolution[0]
    width = args.width or config.max_resolution[1]
    max_frames = args.max_frames or config.max_frames
    num_inference_steps = args.num_inference_steps or config.default_inference_steps

    print(f"\nConfiguration:")
    print(f"  Preset: {args.preset}")
    print(f"  Model: {CHECKPOINT_REPO}/{TRANSFORMER_FILE}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Max frames: {max_frames}")
    print(f"  Inference steps: {num_inference_steps}")

    # Step 1: Encode prompts (loads & unloads text encoder)
    prompt = "Empty"
    negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"
    pe, pm, ne, nm = precompute_text_embeddings(
        prompt, negative_prompt, args.cache_dir
    )

    # Step 2: Load pipeline (2B distilled transformer + VAE only)
    pipe = load_pipeline(config, args.cache_dir)

    # Step 3: Process videos
    os.makedirs(args.output_dir, exist_ok=True)

    if args.video:
        video_folders = [args.video]
    else:
        video_folders = sorted(os.listdir(args.base_dir))

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
            prompt_mask=pm,
            neg_embeds=ne,
            neg_mask=nm,
        )

        if not success:
            print(f"  Failed to process {f_name}")

    print("\n=== Complete ===")
    print_memory_stats()


if __name__ == "__main__":
    main()
