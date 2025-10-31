import torch
import yaml
from model import HotModel

def measure_inference_vram(cfg_path='cfg.yaml'):
    """
    Run inference on a single tensor and measure max VRAM used.
    
    Args:
        cfg_path: Path to the config file
        
    Returns:
        float: Maximum VRAM used in GB
    """
    # Load config
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device(cfg['device'])
    
    # Reset VRAM stats
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    
    # Initialize model
    print("Loading model...")
    model = HotModel(**cfg['model']).to(device)
    model.eval()
    
    # Create a single dummy tensor (batch size 1, 3 channels, resize_to x resize_to)
    img_size = cfg['data']['resize_to']
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
    
    print(f"Running inference on tensor with shape: {dummy_input.shape}")
    
    # Run inference
    with torch.no_grad():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(dummy_input)
    
    # print(f"Output: {output}")
    
    # Get max VRAM used
    max_memory_bytes = torch.cuda.max_memory_allocated(device)
    max_memory_gb = max_memory_bytes / (1024 ** 3)
    
    print(f"\nMax VRAM used: {max_memory_gb:.4f} GB")
    print(f"Max VRAM used: {max_memory_bytes / (1024 ** 2):.2f} MB")
    
    return max_memory_gb

if __name__ == "__main__":
    max_vram_gb = measure_inference_vram()
    print(f"\nFinal result: {max_vram_gb:.4f} GB")

