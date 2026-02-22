"""
Foreground composition for OmnimatteZero — Apple Silicon (MPS) version.

Extracts the foreground layer (object + effects) and composes onto new background.
Uses text encoder decoupling for memory efficiency.

Usage:
    python foreground_composition_mps.py --video_folder swan_lake --new_bg results/cat_reflection.mp4
"""

from typing import Optional, Union
import torch
import argparse
import os
from diffusers import AutoencoderKLLTXVideo
from diffusers.pipelines.ltx.pipeline_ltx_condition import retrieve_latents
from diffusers.utils import export_to_video, load_video
from PIL import Image
from transformers import AutoTokenizer, T5EncoderModel

from OmnimatteZero_mps import OmnimatteZero
from memory_utils_mps import (
    MemoryConfig,
    apply_memory_optimizations,
    clear_memory,
    print_memory_stats,
)

MODEL_ID = "a-r-r-o-w/LTX-Video-0.9.7-diffusers"


def tensor_video_to_pil_images(video_tensor):
    """Converts (1, frames, height, width, 3) tensor to list of PIL Images."""
    video_tensor = video_tensor.squeeze(0)
    video_numpy = video_tensor.cpu().numpy()
    return [Image.fromarray(frame.astype('uint8')) for frame in video_numpy]


class MyAutoencoderKLLTXVideo(AutoencoderKLLTXVideo):

    def forward(self, sample, temb=None, sample_posterior=False,
                return_dict=True, generator=None):
        all_vid, bg, mask, mask2, new_bg = sample
        posterior = self.encode(all_vid).latent_dist
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

    def forward_encode(self, sample, temb=None, sample_posterior=False,
                       return_dict=True, generator=None):
        posterior = self.encode(sample).latent_dist
        z = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        if not return_dict:
            return (z,)
        return z


def precompute_text_embeddings(prompt, negative_prompt, cache_dir=None):
    """Load text encoder, encode, delete."""
    print("\n=== Precomputing Text Embeddings ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer", cache_dir=cache_dir)
    text_encoder = T5EncoderModel.from_pretrained(
        MODEL_ID, subfolder="text_encoder", torch_dtype=torch.float16, cache_dir=cache_dir)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    text_encoder.to(device)

    def encode_text(text):
        inputs = tokenizer(text, padding="max_length", max_length=256, truncation=True,
                           return_attention_mask=True, add_special_tokens=True, return_tensors="pt")
        mask = inputs.attention_mask.to(device)
        embeds = text_encoder(inputs.input_ids.to(device), attention_mask=mask)[0]
        return embeds.detach().cpu(), mask.detach().cpu()

    with torch.no_grad():
        pe, pm = encode_text(prompt)
        ne, nm = encode_text(negative_prompt)

    del text_encoder, tokenizer
    clear_memory()
    print("  ✓ Text encoder unloaded")
    return pe, pm, ne, nm


def main():
    parser = argparse.ArgumentParser(description="Foreground composition (MPS)")
    parser.add_argument("--video_folder", type=str, required=True)
    parser.add_argument("--new_bg", type=str, required=True,
                        help="Path to new background video")
    parser.add_argument("--preset", type=str, default="24gb", choices=["16gb", "24gb", "32gb"])
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--skip_refinement", action="store_true")
    parser.add_argument("--cache_dir", type=str, default=None)
    args = parser.parse_args()

    config = MemoryConfig(args.preset)
    w = args.width or config.max_resolution[1]
    h = args.height or config.max_resolution[0]
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Step 1: Precompute embeddings for refinement
    if not args.skip_refinement:
        prompt = ""
        negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"
        pe, pm, ne, nm = precompute_text_embeddings(prompt, negative_prompt, args.cache_dir)

    # Step 2: Load pipeline
    print("\nLoading pipeline...")
    pipe = OmnimatteZero.from_pretrained(
        MODEL_ID, text_encoder=None, tokenizer=None,
        torch_dtype=torch.float16, cache_dir=args.cache_dir)
    pipe.vae = MyAutoencoderKLLTXVideo.from_pretrained(
        MODEL_ID, subfolder="vae", torch_dtype=torch.float16, cache_dir=args.cache_dir)
    pipe.to(device)
    pipe = apply_memory_optimizations(pipe, config)

    # Step 3: Load videos
    folder = args.video_folder
    base_dir = "example_videos"

    video_p = load_video(f"./{base_dir}/{folder}/video.mp4")
    video_p = pipe.video_processor.preprocess_video(video_p, width=w, height=h).to(device, torch.float16)

    video_bg = load_video(f"./results/{folder}.mp4")
    video_bg = pipe.video_processor.preprocess_video(video_bg, width=w, height=h).to(device, torch.float16)

    video_mask = load_video(f"./{base_dir}/{folder}/object_mask.mp4")
    video_mask = pipe.video_processor.preprocess_video(video_mask, width=w, height=h).to(device, torch.float16)

    video_mask2 = load_video(f"./{base_dir}/{folder}/total_mask.mp4")
    video_mask2 = pipe.video_processor.preprocess_video(video_mask2, width=w, height=h).to(device, torch.float16)

    video_new_bg = load_video(args.new_bg)
    video_new_bg = pipe.video_processor.preprocess_video(video_new_bg, width=w, height=h).to(device, torch.float16)

    nframes = min(video_new_bg.shape[2], video_p.shape[2])
    video_p = video_p[:, :, :nframes, :, :]
    video_bg = video_bg[:, :, :nframes, :, :]
    video_mask = video_mask[:, :, :nframes, :, :]
    video_mask2 = video_mask2[:, :, :nframes, :, :]
    video_new_bg = video_new_bg[:, :, :nframes, :, :]

    os.makedirs(args.output_dir, exist_ok=True)

    with torch.no_grad():
        x, foreground, z_mask, z_mask2 = pipe.vae(
            [video_p, video_bg, video_mask, video_mask2, video_new_bg],
            temb=torch.tensor(0.0, device=device, dtype=torch.float16))

        noise = x.sample
        foreground = foreground.sample
        video_mask_dec = z_mask.sample
        video_mask2_dec = z_mask2.sample
        video_mask_dec = (video_mask_dec.cpu().float() > 0).to(torch.float16).to(device)
        video_mask2_dec = (video_mask2_dec.cpu().float() > 0).to(torch.float16).to(device)

        # Foreground extraction with pixel injection
        foreground = foreground * (1 - video_mask_dec) + video_p * video_mask_dec
        foreground = foreground * video_mask2_dec
        video_foreground = tensor_video_to_pil_images(
            ((pipe.video_processor.postprocess_video(foreground, output_type='pt')[0] * 255).long().permute(0, 2, 3, 1)))
        export_to_video(video_foreground, os.path.join(args.output_dir, "foreground.mp4"), fps=24)

        # Latent addition
        noise = noise * (1 - video_mask_dec) + video_p * video_mask_dec
        video_out = tensor_video_to_pil_images(
            ((pipe.video_processor.postprocess_video(noise, output_type='pt')[0] * 255).long().permute(0, 2, 3, 1)))
        export_to_video(video_out, os.path.join(args.output_dir, "latent_addition.mp4"), fps=24)

        # Refinement (optional)
        if not args.skip_refinement:
            condition_latents = retrieve_latents(pipe.vae.encode(noise), generator=None)
            condition_latents = pipe._normalize_latents(
                condition_latents, pipe.vae.latents_mean, pipe.vae.latents_std
            ).to(noise.device, dtype=noise.dtype)

            expected_height, expected_width = video_out[0].size[1], video_out[0].size[0]
            num_frames = len(video_out)

            clear_memory()

            video_refined = pipe(
                prompt=None,
                prompt_embeds=pe.to(device),
                prompt_attention_mask=pm.to(device),
                negative_prompt=None,
                negative_prompt_embeds=ne.to(device),
                negative_prompt_attention_mask=nm.to(device),
                width=expected_width,
                height=expected_height,
                num_frames=num_frames,
                denoise_strength=0.3,
                num_inference_steps=10,
                latents=condition_latents,
                decode_timestep=0.05,
                image_cond_noise_scale=0.025,
                generator=torch.Generator(device="cpu").manual_seed(0),
                output_type="pil",
            ).frames[0]
            export_to_video(video_refined, os.path.join(args.output_dir, "refinement.mp4"), fps=24)

    print("\nDone!")
    print_memory_stats()


if __name__ == "__main__":
    main()
