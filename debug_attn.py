"""Quick diagnostic: check attention map shapes with minimal frames."""
import torch
from device_utils import load_pipeline, get_device, get_optimal_dtype, clear_memory
from memory_utils import apply_memory_optimizations, round_frames_to_vae_compatible
from runtime_utils import load_video_frames, pad_frames_to_length, load_prompt_tensors
from self_attention_map import SelfAttentionMapExtraction

device = get_device()
dtype = get_optimal_dtype()

# Load prompt tensors
prompt_tensors = load_prompt_tensors(
    prompt="Empty", negative_prompt=None,
    checkpoint_path="model_checkpoint/ltx-video-2b-v0.9.5.safetensors",
    dtype=dtype, device=device, use_prompt_cache=True,
    max_sequence_length=256, cache_dir_for_t5="", require_negative=False,
)
clear_memory()

pipe = load_pipeline(
    checkpoint_path="model_checkpoint/ltx-video-2b-v0.9.5.safetensors",
    cache_dir="", dtype=dtype, load_text_encoder=False, force_vae_fp32=True,
)
apply_memory_optimizations(pipe)

# Load just 33 frames (= 4 latent frames after VAE temporal compression of 8, +1)
video_frames = load_video_frames("./example_videos/test_street_pair/video.mp4", start=0, end=33)
video_frames = pad_frames_to_length(video_frames, 33)
print(f"Loaded {len(video_frames)} frames")

video_tensor = pipe.video_processor.preprocess_video(video_frames, 480, 704)
video_tensor = video_tensor.to(device=device, dtype=dtype)
print(f"Video tensor: {video_tensor.shape}, dtype={video_tensor.dtype}")

B, C, T, H, W = video_tensor.shape
vae_tc = pipe.vae_temporal_compression_ratio
valid_frames = round_frames_to_vae_compatible(T, vae_tc)
print(f"Valid frames for VAE: {valid_frames} (from {T})")

extractor = SelfAttentionMapExtraction(
    pipe, extraction_timestep=0.5, max_layers=4,
    prompt_embeds=prompt_tensors.get("prompt_embeds"),
    prompt_attention_mask=prompt_tensors.get("prompt_attention_mask"),
)
extractor.setup_extractor()

print("\nRunning attention extraction...")
attention_maps, latent_dims = extractor.extract_from_video(
    video_tensor, height=H, width=W, generator=None,
    prompt_embeds=prompt_tensors.get("prompt_embeds"),
    prompt_attention_mask=prompt_tensors.get("prompt_attention_mask"),
)

num_latent_frames, latent_height, latent_width = latent_dims
spatial_size = latent_height * latent_width
expected_seq_len = num_latent_frames * spatial_size
print(f"\nLatent dims: frames={num_latent_frames}, h={latent_height}, w={latent_width}")
print(f"Spatial size: {spatial_size}")
print(f"Expected seq_len: {expected_seq_len}")

print(f"\nGot {len(attention_maps)} attention map layers")
for layer_idx, attn_map in attention_maps.items():
    print(f"  Layer {layer_idx}: shape={attn_map.shape}, dtype={attn_map.dtype}")
    seq_len = attn_map.shape[-1]
    print(f"    seq_len={seq_len}, expected={expected_seq_len}, match={seq_len == expected_seq_len}")
    if attn_map.dim() == 4:
        m = attn_map.mean(dim=1)
    else:
        m = attn_map
    print(f"    min={m.min():.6f}, max={m.max():.6f}, mean={m.mean():.6f}, std={m.std():.6f}")

extractor.cleanup()
del video_tensor
clear_memory()
print("\nDone.")
