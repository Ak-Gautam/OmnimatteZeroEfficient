"""
Device utilities for cross-platform support (CUDA / MPS / CPU).
Provides unified APIs for device detection, memory management, dtype selection,
and pipeline loading from local checkpoints.
"""

import os
import torch
import gc
import sys
from typing import Optional, Tuple

# Default paths — relative to this file's directory
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT = os.path.join(_PROJECT_ROOT, "model_checkpoint", "ltx-video-2b-v0.9.5.safetensors")
DEFAULT_CHECKPOINT_DISTILLED = os.path.join(_PROJECT_ROOT, "model_checkpoint", "ltxv-2b-0.9.8-distilled.safetensors")
# T5 encoder from the official Lightricks repo on HuggingFace
T5_ENCODER_REPO = "Lightricks/LTX-Video"


def get_device() -> torch.device:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_type() -> str:
    """Get device type as string ('cuda', 'mps', or 'cpu')."""
    return str(get_device())


def is_cuda() -> bool:
    return torch.cuda.is_available()


def is_mps() -> bool:
    return torch.backends.mps.is_available() and not torch.cuda.is_available()


def is_cpu() -> bool:
    return not torch.cuda.is_available() and not torch.backends.mps.is_available()


def get_optimal_dtype() -> torch.dtype:
    """
    Get the optimal dtype for the current device.
    
    - CUDA with SM 8.9+ (Ada/Hopper): bfloat16 
    - CUDA with SM 8.0+ (Ampere): bfloat16
    - MPS (Apple Silicon): float16 (bfloat16 has limited support)
    - CPU: float32
    """
    if is_cuda():
        capability = torch.cuda.get_device_capability()
        if capability[0] >= 8:
            return torch.bfloat16
        return torch.float16
    elif is_mps():
        # MPS has limited bfloat16 support; float16 is more reliable
        return torch.float16
    return torch.float32


def get_compute_dtype() -> torch.dtype:
    """Get the compute dtype (used during forward pass)."""
    if is_cuda():
        return torch.bfloat16
    elif is_mps():
        return torch.float16
    return torch.float32


def get_generator(seed: int = 0, device: Optional[torch.device] = None) -> torch.Generator:
    """
    Create a torch.Generator on the appropriate device.
    
    Note: MPS generators should be created on CPU for compatibility.
    """
    if device is None:
        device = get_device()
    
    # MPS generators need to be on CPU
    if str(device) == "mps":
        gen_device = torch.device("cpu")
    else:
        gen_device = device
    
    return torch.Generator(device=gen_device).manual_seed(seed)


def clear_memory(force_gc_cycles: int = 3):
    """Aggressively clear GPU/MPS and CPU memory.
    
    On MPS (unified memory), proper cleanup order is critical:
    1. Delete Python references (via gc.collect multiple times)
    2. Synchronize to ensure all MPS ops complete
    3. Empty the MPS memory cache
    
    Args:
        force_gc_cycles: Number of gc.collect() calls (default 3 for thorough cleanup)
    """
    # Multiple GC cycles to handle reference chains and weak references
    for _ in range(force_gc_cycles):
        gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Sync BEFORE empty_cache
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()  # Sync BEFORE empty_cache - critical for MPS!
        torch.mps.empty_cache()
    
    # Final GC pass after cache clear
    gc.collect()


def synchronize():
    """Synchronize the current device."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()


def get_memory_usage() -> float:
    """Get current device memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    elif torch.backends.mps.is_available():
        try:
            return torch.mps.current_allocated_memory() / 1024**3
        except AttributeError:
            # Older PyTorch may not have this
            return 0.0
    return 0.0


def get_memory_reserved() -> float:
    """Get total reserved memory in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / 1024**3
    elif torch.backends.mps.is_available():
        try:
            return torch.mps.driver_allocated_memory() / 1024**3
        except AttributeError:
            return 0.0
    return 0.0


def get_total_memory() -> float:
    """Get total available device memory in GB."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / 1024**3
    elif torch.backends.mps.is_available():
        # Apple Silicon unified memory — estimate via system
        try:
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True
            )
            total_bytes = int(result.stdout.strip())
            return total_bytes / 1024**3
        except Exception:
            return 24.0  # fallback assumption
    return 0.0


def get_device_name() -> str:
    """Get the device name."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        # Try to get Apple chip name
        try:
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True
            )
            return result.stdout.strip()
        except Exception:
            return "Apple Silicon (MPS)"
    return "CPU"


def print_memory_stats():
    """Print detailed memory statistics for any device."""
    device_type = get_device_type()
    
    if device_type == "cuda":
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved, {max_allocated:.2f} GB peak")
    elif device_type == "mps":
        try:
            allocated = torch.mps.current_allocated_memory() / 1024**3
            driver = torch.mps.driver_allocated_memory() / 1024**3
            print(f"MPS Memory: {allocated:.2f} GB allocated, {driver:.2f} GB driver-reserved")
        except AttributeError:
            print("MPS: memory stats not available (update PyTorch for memory tracking)")
    else:
        print("Running on CPU (no GPU memory stats)")


def print_device_info():
    """Print device information summary."""
    device = get_device()
    dtype = get_optimal_dtype()
    name = get_device_name()
    total_mem = get_total_memory()
    
    print(f"Device: {device} ({name})")
    print(f"Total Memory: {total_mem:.1f} GB")
    print(f"Optimal dtype: {dtype}")
    
    if is_mps():
        print(f"Note: Using unified memory (shared between CPU/GPU/OS)")
        print(f"Effective available for ML: ~{total_mem * 0.65:.0f}-{total_mem * 0.75:.0f} GB")


class MemoryTracker:
    """Context manager for tracking memory usage during operations (cross-platform)."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_memory = 0
        self.peak_memory = 0
    
    def __enter__(self):
        if is_cuda():
            torch.cuda.reset_peak_memory_stats()
            self.start_memory = torch.cuda.memory_allocated()
        elif is_mps():
            try:
                self.start_memory = torch.mps.current_allocated_memory()
            except AttributeError:
                self.start_memory = 0
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if is_cuda():
            end_memory = torch.cuda.memory_allocated()
            self.peak_memory = torch.cuda.max_memory_allocated()
            print(f"{self.name}:")
            print(f"  Start: {self.start_memory / 1024**3:.2f} GB")
            print(f"  End: {end_memory / 1024**3:.2f} GB")
            print(f"  Peak: {self.peak_memory / 1024**3:.2f} GB")
            print(f"  Delta: {(end_memory - self.start_memory) / 1024**3:.2f} GB")
        elif is_mps():
            try:
                end_memory = torch.mps.current_allocated_memory()
                print(f"{self.name}:")
                print(f"  Start: {self.start_memory / 1024**3:.2f} GB")
                print(f"  End: {end_memory / 1024**3:.2f} GB")
                print(f"  Delta: {(end_memory - self.start_memory) / 1024**3:.2f} GB")
            except AttributeError:
                print(f"{self.name}: completed (memory tracking unavailable)")


def load_pipeline(
    checkpoint_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    pipeline_class=None,
    load_text_encoder: bool = True,
    force_vae_fp32: bool = True,
):
    """
    Load the OmnimatteZero pipeline from a local safetensors checkpoint.
    
    Loads the transformer + VAE from the local .safetensors file,
    and the T5 text encoder + tokenizer from Lightricks/LTX-Video on HuggingFace.
    
    Args:
        checkpoint_path: Path to .safetensors file. Defaults to model_checkpoint/ltx-video-2b-v0.9.5.safetensors
        cache_dir: HuggingFace cache directory for T5 encoder download
        dtype: Data type. Defaults to get_optimal_dtype()
        pipeline_class: Pipeline class to use. Defaults to OmnimatteZero
    
    Returns:
        Loaded pipeline instance
    """
    if pipeline_class is None:
        from OmnimatteZero import OmnimatteZero
        pipeline_class = OmnimatteZero
    
    if dtype is None:
        dtype = get_optimal_dtype()
    
    if checkpoint_path is None:
        checkpoint_path = DEFAULT_CHECKPOINT
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at: {checkpoint_path}\n"
            f"Please place your ltx-video-2b-v0.9.5.safetensors in the model_checkpoint/ directory."
        )
    
    print(f"Loading checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"  dtype: {dtype}")
    print(f"  size: {os.path.getsize(checkpoint_path) / 1024**3:.1f} GB")

    text_encoder = None
    tokenizer = None
    if load_text_encoder:
        # Load T5 text encoder and tokenizer from official Lightricks repo
        # These are cached after first download (~few GB)
        from transformers import T5EncoderModel, T5TokenizerFast

        print(f"Loading T5 encoder from {T5_ENCODER_REPO}...")
        text_encoder = T5EncoderModel.from_pretrained(
            T5_ENCODER_REPO,
            subfolder="text_encoder",
            torch_dtype=dtype,
            cache_dir=cache_dir,
        )
        tokenizer = T5TokenizerFast.from_pretrained(
            T5_ENCODER_REPO,
            subfolder="tokenizer",
            cache_dir=cache_dir,
        )
    else:
        print("Skipping T5 encoder/tokenizer load (expects prompt_embeds at runtime)")

    # Load pipeline from local checkpoint. If text encoder/tokenizer are None, the pipeline
    # can still run as long as prompt embeddings are provided.
    print("Loading pipeline from local checkpoint...")
    pipe = pipeline_class.from_single_file(
        checkpoint_path,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        torch_dtype=dtype,
    )

    # CRITICAL: Force VAE to float32 to avoid numerical instability/garbage output
    # LTX-Video VAE is sensitive and can overflow in FP16 on MPS/some GPUs
    if force_vae_fp32:
        print("Forcing VAE to float32 for stability...")
        pipe.vae = pipe.vae.to(dtype=torch.float32)
    
    print(f"Pipeline loaded successfully! (2B parameter model)")
    return pipe


def load_vae(
    checkpoint_path: Optional[str] = None,
    cache_dir: Optional[str] = None, 
    dtype: Optional[torch.dtype] = None,
):
    """
    Load just the VAE from the pipeline checkpoint.
    Useful for foreground composition where a custom VAE subclass is needed.
    
    Args:
        checkpoint_path: Path to .safetensors file
        cache_dir: HuggingFace cache directory
        dtype: Data type (ignored for VAE, always forced to float32)
    
    Returns:
        VAE model instance
    """
    from diffusers import AutoencoderKLLTXVideo
    
    if checkpoint_path is None:
        checkpoint_path = DEFAULT_CHECKPOINT
    
    # Load VAE from the same single-file checkpoint
    # Always load VAE in float32 for stability
    print("Loading VAE (forced to float32)...")
    vae = AutoencoderKLLTXVideo.from_single_file(
        checkpoint_path,
        torch_dtype=torch.float32,
    )
    return vae

