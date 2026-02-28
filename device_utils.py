"""
Device utilities for Apple Silicon (MPS) support.
Provides APIs for device detection, memory management, dtype selection,
and pipeline loading from local checkpoints.
"""

import os
import torch
import gc
from typing import Optional

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT = os.path.join(_PROJECT_ROOT, "model_checkpoint", "ltx-video-2b-v0.9.5.safetensors")
T5_ENCODER_REPO = "Lightricks/LTX-Video"


def get_device() -> torch.device:
    """Auto-detect the best available device (MPS preferred, CPU fallback)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_type() -> str:
    """Get device type as string ('mps' or 'cpu')."""
    return get_device().type


def is_mps() -> bool:
    return torch.backends.mps.is_available()


def get_optimal_dtype() -> torch.dtype:
    """Get the optimal dtype for the current device.

    MPS (Apple Silicon): float16 (bfloat16 has limited MPS support).
    CPU: float32.
    """
    if is_mps():
        return torch.float16
    return torch.float32


def get_generator(seed: int = 0, device: Optional[torch.device] = None) -> torch.Generator:
    """Create a torch.Generator. MPS generators are created on CPU for compatibility."""
    if device is None:
        device = get_device()

    # MPS generators must be on CPU
    gen_device = torch.device("cpu") if device.type == "mps" else device
    return torch.Generator(device=gen_device).manual_seed(seed)


def clear_memory(force_gc_cycles: int = 3):
    """Aggressively clear MPS and CPU memory.

    On MPS (unified memory), proper cleanup order is critical:
    1. Delete Python references (via gc.collect multiple times)
    2. Synchronize to ensure all MPS ops complete
    3. Empty the MPS memory cache
    """
    for _ in range(force_gc_cycles):
        gc.collect()

    if torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()

    gc.collect()


def synchronize():
    """Synchronize the current device."""
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def get_memory_usage() -> float:
    """Get current device memory usage in GB."""
    if torch.backends.mps.is_available():
        try:
            return torch.mps.current_allocated_memory() / 1024**3
        except AttributeError:
            return 0.0
    return 0.0


def get_memory_reserved() -> float:
    """Get total reserved memory in GB."""
    if torch.backends.mps.is_available():
        try:
            return torch.mps.driver_allocated_memory() / 1024**3
        except AttributeError:
            return 0.0
    return 0.0


def get_total_memory() -> float:
    """Get total system memory in GB (Apple Silicon unified memory)."""
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True
        )
        return int(result.stdout.strip()) / 1024**3
    except Exception:
        return 24.0  # fallback assumption


def get_device_name() -> str:
    """Get the device/chip name."""
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True
        )
        return result.stdout.strip()
    except Exception:
        return "Apple Silicon (MPS)"


def print_memory_stats():
    """Print memory statistics."""
    if is_mps():
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
    """Context manager for tracking memory usage during operations."""

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_memory = 0

    def __enter__(self):
        if is_mps():
            try:
                self.start_memory = torch.mps.current_allocated_memory()
            except AttributeError:
                self.start_memory = 0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if is_mps():
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
        from transformers import T5EncoderModel, T5TokenizerFast

        print(f"Loading T5 encoder from {T5_ENCODER_REPO}...")
        text_encoder = T5EncoderModel.from_pretrained(
            T5_ENCODER_REPO, subfolder="text_encoder",
            torch_dtype=dtype, cache_dir=cache_dir,
        )
        tokenizer = T5TokenizerFast.from_pretrained(
            T5_ENCODER_REPO, subfolder="tokenizer",
            cache_dir=cache_dir,
        )
    else:
        print("Skipping T5 encoder/tokenizer load (expects prompt_embeds at runtime)")

    print("Loading pipeline from local checkpoint...")
    pipe = pipeline_class.from_single_file(
        checkpoint_path,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        torch_dtype=dtype,
    )

    # CRITICAL: Force VAE to float32 to avoid numerical instability/garbage output.
    # LTX-Video VAE is sensitive and can overflow in FP16 on MPS.
    if force_vae_fp32:
        print("Forcing VAE to float32 for stability...")
        pipe.vae = pipe.vae.to(dtype=torch.float32)

    # Move pipeline to device directly. On unified memory (Apple Silicon),
    # this avoids the repeated 2x peak from enable_model_cpu_offload() transitions.
    device = get_device()
    pipe = pipe.to(device)

    print(f"Pipeline loaded successfully on {device}! (2B parameter model)")
    return pipe


def load_vae(
    checkpoint_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
):
    """Load just the VAE from the pipeline checkpoint (always float32 for stability)."""
    from diffusers import AutoencoderKLLTXVideo

    if checkpoint_path is None:
        checkpoint_path = DEFAULT_CHECKPOINT

    print("Loading VAE (forced to float32)...")
    vae = AutoencoderKLLTXVideo.from_single_file(
        checkpoint_path,
        torch_dtype=torch.float32,
    )
    return vae
