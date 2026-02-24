"""
Memory optimization utilities for OmnimatteZero.
Enables running on consumer GPUs (16GB VRAM) and Apple Silicon (MPS).
Cross-platform: supports CUDA, MPS, and CPU backends.
"""

import torch
import gc
from typing import Optional, Literal
from device_utils import (
    get_device, get_device_type, is_cuda, is_mps, is_cpu,
    get_optimal_dtype, get_compute_dtype,
    clear_memory, get_memory_usage, get_memory_reserved,
    get_total_memory, get_device_name,
    print_memory_stats, print_device_info,
    MemoryTracker
)


class MemoryConfig:
    """Configuration for memory optimizations."""
    
    # Presets for different VRAM targets
    PRESETS = {
        # Apple Silicon 24GB unified memory — proven config (user tested)
        "mps_24gb": {
            "enable_layerwise_casting": False,  # FP8 not supported on MPS
            "storage_dtype": torch.float16,
            "compute_dtype": torch.float16,
            "enable_group_offload": False,  # No CUDA streams on MPS
            "offload_type": None,
            "use_stream": False,
            "enable_vae_tiling": True,
            "enable_vae_slicing": True,
            "enable_model_cpu_offload": True,  # Effective on unified memory
            "enable_attention_slicing": True,  # Critical for MPS memory
            "max_resolution": (480, 704),  # Proven on M4 Pro
            "max_frames": 97,  # ~4 seconds at 24fps
        },
        # 16GB VRAM - Aggressive optimizations (CUDA)
        "16gb": {
            "enable_layerwise_casting": True,
            "storage_dtype": torch.float8_e4m3fn,
            "compute_dtype": torch.bfloat16,
            "enable_group_offload": True,
            "offload_type": "leaf_level",
            "use_stream": True,
            "enable_vae_tiling": True,
            "enable_vae_slicing": True,
            "enable_model_cpu_offload": True,
            "enable_attention_slicing": False,
            "max_resolution": (480, 704),
            "max_frames": 97,
        },
        # 24GB VRAM - Moderate optimizations (CUDA)
        "24gb": {
            "enable_layerwise_casting": True,
            "storage_dtype": torch.float8_e4m3fn,
            "compute_dtype": torch.bfloat16,
            "enable_group_offload": True,
            "offload_type": "leaf_level",
            "use_stream": True,
            "enable_vae_tiling": True,
            "enable_vae_slicing": False,
            "enable_model_cpu_offload": False,
            "enable_attention_slicing": False,
            "max_resolution": (512, 768),
            "max_frames": 121,
        },
        # 32GB+ VRAM - Minimal optimizations for quality (CUDA)
        "32gb": {
            "enable_layerwise_casting": False,
            "storage_dtype": torch.bfloat16,
            "compute_dtype": torch.bfloat16,
            "enable_group_offload": False,
            "offload_type": None,
            "use_stream": False,
            "enable_vae_tiling": True,
            "enable_vae_slicing": False,
            "enable_model_cpu_offload": False,
            "enable_attention_slicing": False,
            "max_resolution": (512, 768),
            "max_frames": 161,
        },
    }
    
    def __init__(self, preset: str = "16gb"):
        """
        Initialize memory configuration.
        
        Args:
            preset: One of "mps_24gb", "16gb", "24gb", "32gb" 
        """
        if preset not in self.PRESETS:
            raise ValueError(f"Unknown preset: {preset}. Choose from {list(self.PRESETS.keys())}")
        
        config = self.PRESETS[preset]
        for key, value in config.items():
            setattr(self, key, value)
        
        self.preset = preset


def apply_memory_optimizations(pipe, config: MemoryConfig, verbose: bool = True):
    """
    Apply memory optimizations to a diffusion pipeline.
    Cross-platform: works on CUDA, MPS, and CPU.
    
    Args:
        pipe: The diffusion pipeline (OmnimatteZero or similar)
        config: Memory configuration to apply
        verbose: Print optimization info
    """
    device = get_device()
    device_type = get_device_type()
    
    if verbose:
        print(f"Applying memory optimizations for {config.preset} preset on {device_type}...")
    
    # 1. Enable VAE tiling (reduces VAE memory usage significantly)
    if config.enable_vae_tiling:
        pipe.vae.enable_tiling()
        if verbose:
            print("  ✓ VAE tiling enabled")
    
    # 2. Enable VAE slicing (for batch processing)
    if config.enable_vae_slicing and hasattr(pipe, 'enable_vae_slicing'):
        pipe.enable_vae_slicing()
        if verbose:
            print("  ✓ VAE slicing enabled")
    
    # 3. Enable attention slicing (critical for MPS, optional for CUDA)
    if config.enable_attention_slicing and hasattr(pipe, 'enable_attention_slicing'):
        pipe.enable_attention_slicing()
        if verbose:
            print("  ✓ Attention slicing enabled")
    
    # 4. Apply FP8 layerwise casting to transformer (CUDA only)
    if config.enable_layerwise_casting and is_cuda() and hasattr(pipe.transformer, 'enable_layerwise_casting'):
        try:
            pipe.transformer.enable_layerwise_casting(
                storage_dtype=config.storage_dtype,
                compute_dtype=config.compute_dtype
            )
            if verbose:
                print(f"  ✓ Layerwise casting enabled (storage: {config.storage_dtype}, compute: {config.compute_dtype})")
        except Exception as e:
            if verbose:
                print(f"  ⚠ Layerwise casting failed: {e}")
    
    # 5. Apply group offloading (CUDA with streams only)
    if config.enable_group_offload and is_cuda():
        try:
            from diffusers.hooks import apply_group_offloading
            
            onload_device = torch.device("cuda")
            offload_device = torch.device("cpu")
            
            # Apply to transformer with CUDA streams for speed
            if hasattr(pipe.transformer, 'enable_group_offload'):
                pipe.transformer.enable_group_offload(
                    onload_device=onload_device,
                    offload_device=offload_device,
                    offload_type=config.offload_type,
                    use_stream=config.use_stream
                )
            else:
                apply_group_offloading(
                    pipe.transformer,
                    onload_device=onload_device,
                    offload_type=config.offload_type,
                    use_stream=config.use_stream
                )
            
            # Apply to text encoder with block-level offloading
            if hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None:
                apply_group_offloading(
                    pipe.text_encoder,
                    onload_device=onload_device,
                    offload_type="block_level",
                    num_blocks_per_group=2
                )
            
            # Apply to VAE with leaf-level offloading
            if hasattr(pipe.vae, 'enable_group_offload'):
                pipe.vae.enable_group_offload(
                    onload_device=onload_device,
                    offload_device=offload_device,
                    offload_type="leaf_level"
                )
            else:
                apply_group_offloading(
                    pipe.vae,
                    onload_device=onload_device,
                    offload_type="leaf_level"
                )
            
            if verbose:
                print(f"  ✓ Group offloading enabled ({config.offload_type}, stream={config.use_stream})")
        except Exception as e:
            if verbose:
                print(f"  ⚠ Group offloading failed: {e}, falling back to model CPU offload")
            # Fallback to model CPU offload
            if config.enable_model_cpu_offload:
                pipe.enable_model_cpu_offload()
                if verbose:
                    print("  ✓ Model CPU offload enabled (fallback)")
    
    # 6. Enable model CPU offload if no group offloading
    elif config.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload()
        if verbose:
            print("  ✓ Model CPU offload enabled")
    
    # Clear memory
    clear_memory()
    
    if verbose:
        mem = get_memory_usage()
        print(f"  Memory after optimization: {mem:.2f} GB")
    
    return pipe


def apply_memory_optimizations_vae_only(vae, config: MemoryConfig, verbose: bool = True):
    """Apply optimizations to a standalone VAE."""
    device = get_device()
    
    if verbose:
        print(f"Applying VAE memory optimizations...")
    
    if config.enable_vae_tiling:
        vae.enable_tiling()
        if verbose:
            print("  ✓ VAE tiling enabled")
    
    if config.enable_group_offload and is_cuda():
        try:
            from diffusers.hooks import apply_group_offloading
            apply_group_offloading(
                vae,
                onload_device=torch.device("cuda"),
                offload_type="leaf_level"
            )
            if verbose:
                print("  ✓ VAE group offloading enabled")
        except Exception as e:
            if verbose:
                print(f"  ⚠ VAE group offloading failed: {e}")
    
    return vae


def estimate_memory_requirement(height: int, width: int, num_frames: int, batch_size: int = 1) -> float:
    """
    Estimate memory requirement for given video dimensions.
    Returns estimated memory in GB.
    """
    # LTX-Video specific calculations
    vae_spatial_compression = 32
    vae_temporal_compression = 8
    latent_channels = 128
    
    latent_h = height // vae_spatial_compression
    latent_w = width // vae_spatial_compression
    latent_frames = (num_frames - 1) // vae_temporal_compression + 1
    
    # Model weights: ~4-5GB for LTX-Video 2B in FP16
    model_memory = 5.0
    
    # Latent memory (with classifier-free guidance, 2x)
    latent_memory = 2 * batch_size * latent_channels * latent_frames * latent_h * latent_w * 2 / 1024**3
    
    # VAE memory for encoding/decoding
    vae_memory = batch_size * 3 * num_frames * height * width * 2 / 1024**3
    
    # Activation memory (rough estimate)
    activation_memory = latent_memory * 4
    
    total = model_memory + latent_memory + vae_memory + activation_memory
    return total


def get_recommended_settings(available_memory: float, device_type: str = None) -> dict:
    """
    Get recommended settings based on available memory.
    
    Args:
        available_memory: Available memory in GB
        device_type: Override device type detection ('cuda', 'mps', 'cpu')
    """
    if device_type is None:
        device_type = get_device_type()
    
    if device_type == "mps":
        # Apple Silicon unified memory — OS/apps use some of the pool
        effective_memory = available_memory * 0.65  # ~65% available for ML
        if effective_memory >= 16:
            return {
                "preset": "mps_24gb",
                "height": 480,
                "width": 704,
                "num_frames": 97,
                "num_inference_steps": 30,
            }
        else:
            return {
                "preset": "mps_24gb",
                "height": 384,
                "width": 576,
                "num_frames": 65,
                "num_inference_steps": 20,
                "warning": "Limited unified memory. Results may be degraded.",
            }
    
    # CUDA presets
    if available_memory >= 32:
        return {
            "preset": "32gb",
            "height": 512,
            "width": 768,
            "num_frames": 161,
            "num_inference_steps": 30,
        }
    elif available_memory >= 24:
        return {
            "preset": "24gb",
            "height": 512,
            "width": 768,
            "num_frames": 121,
            "num_inference_steps": 25,
        }
    elif available_memory >= 16:
        return {
            "preset": "16gb",
            "height": 480,
            "width": 704,
            "num_frames": 97,
            "num_inference_steps": 20,
        }
    else:
        return {
            "preset": "16gb",
            "height": 384,
            "width": 576,
            "num_frames": 65,
            "num_inference_steps": 15,
            "warning": "Very limited VRAM. Results may be degraded.",
        }


def round_to_vae_compatible(height: int, width: int, vae_ratio: int = 32) -> tuple:
    """Round dimensions to be compatible with VAE compression ratio."""
    height = height - (height % vae_ratio)
    width = width - (width % vae_ratio)
    return height, width


def round_frames_to_vae_compatible(num_frames: int, temporal_ratio: int = 8) -> int:
    """Round frame count to be compatible with VAE temporal compression."""
    # LTX-Video requires (k * 8 + 1) frames
    k = (num_frames - 1) // temporal_ratio
    return k * temporal_ratio + 1


# Auto-detect and set optimal configuration
def auto_configure() -> MemoryConfig:
    """Automatically configure based on available device and memory."""
    device_type = get_device_type()
    total_memory = get_total_memory()
    device_name = get_device_name()
    
    print(f"Detected device: {device_name}")
    print(f"Total memory: {total_memory:.1f} GB")
    print(f"Backend: {device_type}")
    
    if device_type == "mps":
        preset = "mps_24gb"
        print(f"Apple Silicon detected — using preset: {preset}")
    elif device_type == "cuda":
        if total_memory >= 32:
            preset = "32gb"
        elif total_memory >= 24:
            preset = "24gb"
        else:
            preset = "16gb"
        print(f"CUDA GPU detected — using preset: {preset}")
    else:
        preset = "16gb"
        print("No GPU found — using CPU with 16gb preset (will be very slow)")
    
    return MemoryConfig(preset)
