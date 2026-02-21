"""
Memory optimization utilities for OmnimatteZero.
Enables running on consumer GPUs (16GB VRAM) with various optimization techniques.
"""

import torch
import gc
from typing import Optional, Literal


class MemoryConfig:
    """Configuration for memory optimizations."""
    
    # Presets for different VRAM targets
    PRESETS = {
        # 16GB VRAM - Aggressive optimizations
        "16gb": {
            "enable_layerwise_casting": False,
            "storage_dtype": torch.float16,
            "compute_dtype": torch.float16,
            "enable_group_offload": True,
            "offload_type": "leaf_level",
            "use_stream": True,
            "enable_vae_tiling": True,
            "enable_vae_slicing": True,
            "enable_model_cpu_offload": True,
            "max_resolution": (480, 704),  # Slightly reduced
            "max_frames": 97,  # Reduced frame count
        },
        # 24GB VRAM - Moderate optimizations
        "24gb": {
            "enable_layerwise_casting": False,
            "storage_dtype": torch.float16,
            "compute_dtype": torch.float16,
            "enable_group_offload": True,
            "offload_type": "leaf_level",
            "use_stream": True,
            "enable_vae_tiling": True,
            "enable_vae_slicing": False,
            "enable_model_cpu_offload": False,
            "max_resolution": (512, 768),
            "max_frames": 121,
        },
        # 32GB+ VRAM - Minimal optimizations for quality
        "32gb": {
            "enable_layerwise_casting": False,
            "storage_dtype": torch.float16,
            "compute_dtype": torch.float16,
            "enable_group_offload": False,
            "offload_type": None,
            "use_stream": False,
            "enable_vae_tiling": True,
            "enable_vae_slicing": False,
            "enable_model_cpu_offload": False,
            "max_resolution": (512, 768),
            "max_frames": 161,
        },
    }
    
    def __init__(self, preset: str = "16gb"):
        """
        Initialize memory configuration.
        
        Args:
            preset: One of "16gb", "24gb", "32gb" 
        """
        if preset not in self.PRESETS:
            raise ValueError(f"Unknown preset: {preset}. Choose from {list(self.PRESETS.keys())}")
        
        config = self.PRESETS[preset]
        for key, value in config.items():
            setattr(self, key, value)
        
        self.preset = preset


def get_optimal_dtype():
    """Get the optimal dtype for the current GPU."""
    if torch.backends.mps.is_available():
        return torch.float16, torch.float16
    return torch.float32, torch.float32


def apply_memory_optimizations(pipe, config: MemoryConfig, verbose: bool = True):
    """
    Apply memory optimizations to a diffusion pipeline.
    
    Args:
        pipe: The diffusion pipeline (OmnimatteZero or similar)
        config: Memory configuration to apply
        verbose: Print optimization info
    """
    if verbose:
        print(f"Applying memory optimizations for {config.preset} preset...")
    
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
    
    # 3. Apply FP8 layerwise casting to transformer
    if config.enable_layerwise_casting and hasattr(pipe.transformer, 'enable_layerwise_casting'):
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
    
    # 4. Apply group offloading
    if config.enable_group_offload:
        try:
            from diffusers.hooks import apply_group_offloading
            
            onload_device = torch.device("mps")
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
    
    # 5. Enable model CPU offload if no group offloading
    elif config.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload()
        if verbose:
            print("  ✓ Model CPU offload enabled")
    
    # Clear memory
    clear_memory()
    
    if verbose:
        print(f"  Memory after optimization: {get_gpu_memory_usage():.2f} GB")
    
    return pipe


def apply_memory_optimizations_vae_only(vae, config: MemoryConfig, verbose: bool = True):
    """Apply optimizations to a standalone VAE."""
    if verbose:
        print(f"Applying VAE memory optimizations...")
    
    if config.enable_vae_tiling:
        vae.enable_tiling()
        if verbose:
            print("  ✓ VAE tiling enabled")
    
    if config.enable_group_offload:
        try:
            from diffusers.hooks import apply_group_offloading
            apply_group_offloading(
                vae,
                onload_device=torch.device("mps"),
                offload_type="leaf_level"
            )
            if verbose:
                print("  ✓ VAE group offloading enabled")
        except Exception as e:
            if verbose:
                print(f"  ⚠ VAE group offloading failed: {e}")
    
    return vae


def clear_memory():
    """Aggressively clear GPU and CPU memory."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        torch.mps.synchronize()


def get_gpu_memory_usage() -> float:
    """Get current GPU memory usage in GB."""
    if torch.backends.mps.is_available():
        return torch.mps.current_allocated_memory() / 1024**3
    return 0.0


def get_gpu_memory_reserved() -> float:
    """Get total reserved GPU memory in GB."""
    if torch.backends.mps.is_available():
        return 0.0 / 1024**3
    return 0.0


def print_memory_stats():
    """Print detailed memory statistics."""
    if torch.backends.mps.is_available():
        allocated = torch.mps.current_allocated_memory() / 1024**3
        reserved = 0.0 / 1024**3
        max_allocated = torch.mps.driver_allocated_memory() / 1024**3
        print(f"GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved, {max_allocated:.2f} GB peak")
    else:
        print("CUDA not available")


def estimate_memory_requirement(height: int, width: int, num_frames: int, batch_size: int = 1) -> float:
    """
    Estimate VRAM requirement for given video dimensions.
    
    Returns estimated VRAM in GB.
    """
    # LTX-Video specific calculations
    vae_spatial_compression = 32
    vae_temporal_compression = 8
    latent_channels = 128
    
    latent_h = height // vae_spatial_compression
    latent_w = width // vae_spatial_compression
    latent_frames = (num_frames - 1) // vae_temporal_compression + 1
    
    # Transformer memory (rough estimate)
    # Model weights: ~12GB for LTX-Video 0.9.7
    model_memory = 12.0
    
    # Latent memory (with classifier-free guidance, 2x)
    latent_memory = 2 * batch_size * latent_channels * latent_frames * latent_h * latent_w * 2 / 1024**3
    
    # VAE memory for encoding/decoding
    vae_memory = batch_size * 3 * num_frames * height * width * 2 / 1024**3
    
    # Activation memory (rough estimate, depends on batch size)
    activation_memory = latent_memory * 4
    
    total = model_memory + latent_memory + vae_memory + activation_memory
    return total


def get_recommended_settings(available_vram: float) -> dict:
    """
    Get recommended settings based on available VRAM.
    
    Args:
        available_vram: Available VRAM in GB
        
    Returns:
        Dictionary with recommended settings
    """
    if available_vram >= 32:
        return {
            "preset": "32gb",
            "height": 512,
            "width": 768,
            "num_frames": 161,
            "num_inference_steps": 30,
        }
    elif available_vram >= 24:
        return {
            "preset": "24gb",
            "height": 512,
            "width": 768,
            "num_frames": 121,
            "num_inference_steps": 25,
        }
    elif available_vram >= 16:
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


class MemoryTracker:
    """Context manager for tracking memory usage during operations."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_memory = 0
        self.peak_memory = 0
    
    def __enter__(self):
        if torch.backends.mps.is_available():
            pass
            self.start_memory = torch.mps.current_allocated_memory()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.backends.mps.is_available():
            end_memory = torch.mps.current_allocated_memory()
            self.peak_memory = torch.mps.driver_allocated_memory()
            
            print(f"{self.name}:")
            print(f"  Start: {self.start_memory / 1024**3:.2f} GB")
            print(f"  End: {end_memory / 1024**3:.2f} GB")
            print(f"  Peak: {self.peak_memory / 1024**3:.2f} GB")
            print(f"  Delta: {(end_memory - self.start_memory) / 1024**3:.2f} GB")


# Auto-detect and set optimal configuration
def auto_configure() -> MemoryConfig:
    """Automatically configure based on available GPU memory."""
    if torch.backends.mps.is_available():
        total_memory = (24 * 1024**3) / 1024**3
        print(f"Detected GPU: {"Apple Silicon (MPS)"}")
        print(f"Total VRAM: {total_memory:.1f} GB")
        
        if total_memory >= 32:
            preset = "32gb"
        elif total_memory >= 24:
            preset = "24gb"
        else:
            preset = "16gb"
        
        print(f"Using preset: {preset}")
        return MemoryConfig(preset)
    else:
        print("No CUDA device found, using CPU (will be very slow)")
        return MemoryConfig("16gb")
