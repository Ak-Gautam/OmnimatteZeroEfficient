"""
Memory optimization utilities for OmnimatteZero on Apple Silicon (MPS).

Key insight: Apple Silicon uses UNIFIED memory — RAM and VRAM share the same pool.
CPU↔GPU "offloading" does NOT free memory. Instead, we focus on:
  - Using the 2B distilled model (6.34GB vs 28.6GB for 13B)
  - FP16 precision (halves memory vs FP32)
  - VAE tiling (reduces peak memory during encode/decode)
  - Text encoder decoupling (load → encode → delete before loading transformer)
  - Aggressive garbage collection
"""

import torch
import gc
from typing import Optional


class MemoryConfig:
    """Configuration for memory optimizations on Apple Silicon."""

    # Presets tuned for SHARED memory with 2B distilled model
    PRESETS = {
        # For 16GB unified memory Macs — aggressive settings
        "16gb": {
            "enable_vae_tiling": True,
            "enable_vae_slicing": True,
            "max_resolution": (480, 704),
            "max_frames": 49,  # 6 * 8 + 1
            "default_inference_steps": 8,
        },
        # For 24GB unified memory Macs (M4 Pro, etc.)
        "24gb": {
            "enable_vae_tiling": True,
            "enable_vae_slicing": False,
            "max_resolution": (480, 704),
            "max_frames": 97,  # 12 * 8 + 1
            "default_inference_steps": 8,
        },
        # For 32GB+ unified memory Macs
        "32gb": {
            "enable_vae_tiling": True,
            "enable_vae_slicing": False,
            "max_resolution": (512, 768),
            "max_frames": 121,  # 15 * 8 + 1
            "default_inference_steps": 10,
        },
    }

    def __init__(self, preset: str = "24gb"):
        if preset not in self.PRESETS:
            raise ValueError(f"Unknown preset: {preset}. Choose from {list(self.PRESETS.keys())}")

        config = self.PRESETS[preset]
        for key, value in config.items():
            setattr(self, key, value)

        self.preset = preset


def apply_memory_optimizations(pipe, config: MemoryConfig, verbose: bool = True):
    """
    Apply memory optimizations to a diffusion pipeline on MPS.

    On Apple Silicon, the only optimizations that actually help are:
      1. VAE tiling — processes encode/decode in tiles, big memory savings
      2. VAE slicing — processes batches one at a time (marginal benefit here)

    We intentionally DO NOT use:
      - Group offloading (pointless on unified memory)
      - Model CPU offload (pointless on unified memory)
      - FP8/layerwise casting (not supported on MPS)
      - CUDA streams (don't exist on MPS)
    """
    if verbose:
        print(f"Applying MPS memory optimizations ({config.preset} preset)...")

    # VAE tiling — the key optimization for MPS
    if config.enable_vae_tiling:
        pipe.vae.enable_tiling()
        if verbose:
            print("  ✓ VAE tiling enabled")

    # VAE slicing — minor benefit for batch processing
    if config.enable_vae_slicing and hasattr(pipe, 'enable_vae_slicing'):
        pipe.enable_vae_slicing()
        if verbose:
            print("  ✓ VAE slicing enabled")

    clear_memory()

    if verbose:
        print(f"  Memory after optimization: {get_memory_usage():.2f} GB")

    return pipe


def clear_memory():
    """Aggressively clear memory."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        torch.mps.synchronize()


def get_memory_usage() -> float:
    """Get current MPS memory usage in GB."""
    if torch.backends.mps.is_available():
        return torch.mps.current_allocated_memory() / 1024**3
    return 0.0


def get_driver_memory() -> float:
    """Get driver-level allocated memory in GB (closer to actual system usage)."""
    if torch.backends.mps.is_available():
        return torch.mps.driver_allocated_memory() / 1024**3
    return 0.0


def print_memory_stats():
    """Print memory statistics."""
    if torch.backends.mps.is_available():
        allocated = torch.mps.current_allocated_memory() / 1024**3
        driver = torch.mps.driver_allocated_memory() / 1024**3
        print(f"MPS Memory: {allocated:.2f} GB allocated, {driver:.2f} GB driver total")
    else:
        print("MPS not available")


def round_to_vae_compatible(height: int, width: int, vae_ratio: int = 32) -> tuple:
    """Round dimensions to be compatible with VAE compression ratio."""
    height = height - (height % vae_ratio)
    width = width - (width % vae_ratio)
    return height, width


def round_frames_to_vae_compatible(num_frames: int, temporal_ratio: int = 8) -> int:
    """Round frame count to be compatible with VAE temporal compression.
    LTX-Video requires (k * 8 + 1) frames."""
    k = (num_frames - 1) // temporal_ratio
    if k < 1:
        k = 1
    return k * temporal_ratio + 1


class MemoryTracker:
    """Context manager for tracking memory usage during operations."""

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_memory = 0

    def __enter__(self):
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
            self.start_memory = torch.mps.current_allocated_memory()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
            end_memory = torch.mps.current_allocated_memory()
            driver_memory = torch.mps.driver_allocated_memory()

            print(f"{self.name}:")
            print(f"  Start: {self.start_memory / 1024**3:.2f} GB")
            print(f"  End: {end_memory / 1024**3:.2f} GB")
            print(f"  Driver total: {driver_memory / 1024**3:.2f} GB")
            print(f"  Delta: {(end_memory - self.start_memory) / 1024**3:.2f} GB")


def auto_configure() -> MemoryConfig:
    """Auto-configure for Apple Silicon. Defaults to 24gb preset."""
    if torch.backends.mps.is_available():
        print("Detected: Apple Silicon (MPS)")
        preset = "24gb"
        print(f"Using preset: {preset}")
        return MemoryConfig(preset)
    else:
        print("MPS not available, using 16gb preset (will be very slow on CPU)")
        return MemoryConfig("16gb")
