"""
Memory optimization utilities for OmnimatteZero on Apple Silicon.

Key insight: On unified memory (MPS), standard diffusers optimizations like
enable_model_cpu_offload() and enable_attention_slicing() are ineffective:
- CPU offload creates 2x peak memory during transitions (same physical memory)
- Attention slicing is a no-op for LTX-Video's custom LTXAttention class

Instead, we use:
- Head-sliced attention: chunks SDPA over attention heads to reduce peak memory
- VAE tiling/slicing: reduces VAE encode/decode memory (actually works)
- Direct MPS loading: no CPU↔MPS transitions
"""

import torch
import torch.nn.functional as F
from device_utils import (
    get_device, get_device_type, is_mps,
    get_optimal_dtype,
    clear_memory, get_memory_usage,
    get_total_memory, get_device_name,
    print_memory_stats,
)


# Default configuration for Apple Silicon 24 GB unified memory
_DEFAULT_CONFIG = {
    "enable_vae_tiling": True,
    "enable_vae_slicing": True,
    "attention_slice_size": 4,  # heads per chunk (32 total, 4 = 8 chunks)
    "max_resolution": (480, 704),
    "max_frames": 97,  # ~4 seconds at 24fps
}


class MemoryConfig:
    """Configuration for memory optimizations."""

    PRESETS = {
        "default": _DEFAULT_CONFIG,
        "mps_24gb": _DEFAULT_CONFIG,
    }

    def __init__(self, preset: str = "default"):
        if preset not in self.PRESETS:
            raise ValueError(f"Unknown preset: {preset}. Choose from {list(self.PRESETS.keys())}")

        config = self.PRESETS[preset]
        for key, value in config.items():
            setattr(self, key, value)

        self.preset = preset


class LTXVideoAttnProcessorSliced:
    """
    Memory-efficient attention processor for LTX-Video that chunks over heads.

    LTX-Video uses its own LTXAttention class (not diffusers' standard Attention),
    so diffusers' enable_attention_slicing() is a no-op. On MPS, PyTorch's SDPA
    falls back to the math path which materializes the full S×S attention matrix
    per head. With 32 heads and seq_len=4290, that's ~4.7 GB for batch=2.

    This processor chunks Q/K/V over the head dimension and processes
    `slice_size` heads at a time, reducing peak memory by heads/slice_size.
    """

    def __init__(self, slice_size: int = 4):
        self.slice_size = slice_size

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
    ) -> torch.Tensor:
        from diffusers.models.transformers.transformer_ltx import apply_rotary_emb

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # Reshape to (batch, seq, heads, head_dim)
        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        # Process in chunks of heads to reduce peak memory
        output_chunks = []
        for i in range(0, attn.heads, self.slice_size):
            end = min(i + self.slice_size, attn.heads)
            q_chunk = query[:, :, i:end, :].transpose(1, 2)
            k_chunk = key[:, :, i:end, :].transpose(1, 2)
            v_chunk = value[:, :, i:end, :].transpose(1, 2)

            mask_chunk = None
            if attention_mask is not None:
                mask_chunk = attention_mask[:, i:end, :, :]

            attn_output = F.scaled_dot_product_attention(
                q_chunk, k_chunk, v_chunk,
                attn_mask=mask_chunk,
                dropout_p=0.0,
                is_causal=False,
            )
            output_chunks.append(attn_output.transpose(1, 2))

        # Concatenate head chunks: (batch, seq, heads, head_dim)
        hidden_states = torch.cat(output_chunks, dim=2)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def set_sliced_attention(transformer, slice_size: int = 4):
    """
    Replace all LTXAttention processors in the transformer with sliced versions.

    Args:
        transformer: LTXVideoTransformer3DModel instance
        slice_size: Number of attention heads to process per chunk (default: 4)
    """
    count = 0
    for module in transformer.modules():
        if type(module).__name__ == "LTXAttention":
            module.set_processor(LTXVideoAttnProcessorSliced(slice_size=slice_size))
            count += 1
    return count


def apply_memory_optimizations(pipe, config: MemoryConfig = None, verbose: bool = True):
    """
    Apply memory optimizations to a diffusion pipeline for Apple Silicon.

    - VAE tiling: processes large videos in spatial tiles
    - VAE slicing: processes video frames one at a time
    - Head-sliced attention: chunks SDPA over heads (actual memory reduction)

    Does NOT use enable_model_cpu_offload() or enable_attention_slicing()
    because they are ineffective on unified memory / LTX-Video respectively.
    """
    if config is None:
        config = MemoryConfig()

    if verbose:
        print(f"Applying memory optimizations ({config.preset})...")

    if config.enable_vae_tiling:
        pipe.vae.enable_tiling()
        if verbose:
            print("  - VAE tiling enabled")

    if config.enable_vae_slicing and hasattr(pipe, 'enable_vae_slicing'):
        pipe.enable_vae_slicing()
        if verbose:
            print("  - VAE slicing enabled")

    # Head-sliced attention (replaces non-functional enable_attention_slicing)
    if hasattr(pipe, 'transformer') and pipe.transformer is not None:
        num_replaced = set_sliced_attention(pipe.transformer, config.attention_slice_size)
        if verbose:
            print(f"  - Head-sliced attention: {num_replaced} modules, {config.attention_slice_size} heads/chunk")

    clear_memory()

    if verbose:
        mem = get_memory_usage()
        print(f"  Memory after optimization: {mem:.2f} GB")

    return pipe


def get_recommended_settings() -> dict:
    """Get recommended settings based on available Apple Silicon memory."""
    total_memory = get_total_memory()
    effective_memory = total_memory * 0.65  # ~65% available for ML after macOS overhead

    if effective_memory >= 16:
        return {
            "height": 480,
            "width": 704,
            "num_frames": 97,
            "num_inference_steps": 12,
        }
    else:
        return {
            "height": 384,
            "width": 576,
            "num_frames": 65,
            "num_inference_steps": 20,
            "warning": "Limited unified memory. Results may be degraded.",
        }


def round_to_vae_compatible(height: int, width: int, vae_ratio: int = 32) -> tuple:
    """Round dimensions to be compatible with VAE compression ratio."""
    height = height - (height % vae_ratio)
    width = width - (width % vae_ratio)
    return height, width


def round_frames_to_vae_compatible(num_frames: int, temporal_ratio: int = 8) -> int:
    """Round frame count to be compatible with VAE temporal compression.
    LTX-Video requires (k * 8 + 1) frames.
    """
    k = (num_frames - 1) // temporal_ratio
    return k * temporal_ratio + 1


def auto_configure() -> MemoryConfig:
    """Auto-detect device and return the appropriate memory configuration."""
    device_name = get_device_name()
    total_memory = get_total_memory()
    device_type = get_device_type()

    print(f"Detected device: {device_name}")
    print(f"Total memory: {total_memory:.1f} GB")
    print(f"Backend: {device_type}")

    if is_mps():
        print("Apple Silicon detected — using default preset")
    else:
        print("No MPS device found — using CPU (will be slow)")

    return MemoryConfig("default")
