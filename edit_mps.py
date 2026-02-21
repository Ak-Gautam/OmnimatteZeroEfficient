import os

def replace_in_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # General replacements
    content = content.replace('torch.float8_e4m3fn', 'torch.float16')
    content = content.replace('torch.bfloat16', 'torch.float16')
    content = content.replace('"enable_layerwise_casting": True', '"enable_layerwise_casting": False')
    content = content.replace('device="cuda"', 'device="mps"')
    content = content.replace("device='cuda'", "device='mps'")
    content = content.replace('onload_device=torch.device("cuda")', 'onload_device=torch.device("mps")')
    content = content.replace('onload_device = torch.device("cuda")', 'onload_device = torch.device("mps")')
    
    # memory_utils.py specific
    content = content.replace('torch.cuda.is_available()', 'torch.backends.mps.is_available()')
    content = content.replace('torch.cuda.empty_cache()', 'torch.mps.empty_cache()')
    content = content.replace('torch.cuda.synchronize()', 'torch.mps.synchronize()')
    content = content.replace('torch.cuda.memory_allocated()', 'torch.mps.current_allocated_memory()')
    content = content.replace('torch.cuda.max_memory_allocated()', 'torch.mps.driver_allocated_memory()')
    content = content.replace('torch.cuda.memory_reserved()', '0.0')
    content = content.replace('torch.cuda.get_device_properties(0).total_memory', '(24 * 1024**3)')
    content = content.replace('torch.cuda.get_device_name(0)', '"Apple Silicon (MPS)"')
    content = content.replace('torch.cuda.reset_peak_memory_stats()', 'pass')

    # object_removal_optimized.py specific imports
    content = content.replace('from OmnimatteZero import OmnimatteZero', 'from OmnimatteZero_mps import OmnimatteZero')
    content = content.replace('from memory_utils import (', 'from memory_utils_mps import (')

    with open(filepath, 'w') as f:
        f.write(content)

replace_in_file('memory_utils_mps.py')
replace_in_file('OmnimatteZero_mps.py')
replace_in_file('object_removal_optimized_mps.py')

print('Replacements complete.')
