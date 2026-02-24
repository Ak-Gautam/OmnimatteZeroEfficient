
import torch
from device_utils import load_pipeline, get_device

def debug_pipeline():
    print("Loading pipeline...")
    pipe = load_pipeline()
    
    print("\n=== Scheduler Info ===")
    print(f"Scheduler Type: {type(pipe.scheduler)}")
    print(f"Scheduler Config: {pipe.scheduler.config}")
    
    print("\n=== VAE Info ===")
    print(f"VAE Type: {type(pipe.vae)}")
    print(f"VAE Dtype: {pipe.vae.dtype}")
    print(f"VAE Config: {pipe.vae.config}")

    print("\n=== Transformer Info ===")
    print(f"Transformer Dtype: {pipe.transformer.dtype}")

if __name__ == "__main__":
    debug_pipeline()
