import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import v2
from model import HotModel
from tqdm import tqdm
from dataset import make_transform


class SimpleImageDataset(Dataset):
    """Simple dataset that loads all images from a directory."""
    
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        
        # Get all image files
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
        self.image_files = [
            f for f in os.listdir(image_dir) 
            if f.lower().endswith(valid_extensions)
        ]
        self.image_files.sort()
        print(f"Found {len(self.image_files)} images in {image_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {e}")
        
        if self.transform:
            image = self.transform(image)
        
        return image, img_name


def main():
    # Configuration
    image_dir = "/root/exp_photos"
    checkpoint_path = "/root/hot_detector/checkpoint.pt"
    batch_size = 128
    output_json = "/root/hot_detector/inference_results.json"
    
    # Model configuration from cfg.yaml
    backbone = "facebook/dinov3-vit7b16-pretrain-lvd1689m"
    bb_dim = 4096
    freeze_bb = True
    
    device = torch.device("cuda")
    print(f"Using device: {device}")
    
    # Create dataset with transform
    transform = make_transform(512)
    dataset = SimpleImageDataset(image_dir, transform=transform)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    
    # Load model
    print("Loading model...")
    model = HotModel(backbone=backbone, bb_dim=bb_dim, freeze_bb=freeze_bb)
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    
    print("Running inference...")
    all_outputs = []
    all_filenames = []
    
    # Run inference in bf16
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            for images, filenames in tqdm(dataloader, desc="Processing batches"):
                images = images.to(device)
                outputs = model(images)
                
                # Convert to float32 for JSON serialization
                outputs = outputs.cpu().float().tolist()
                all_outputs.extend(outputs)
                all_filenames.extend(filenames)
    
    # Prepare results as a list of dictionaries
    results = [
        {"filename": filename, "score": score}
        for filename, score in zip(all_filenames, all_outputs)
    ]
    
    # Save to JSON
    print(f"Saving results to {output_json}")
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Done! Processed {len(results)} images.")
    print(f"Results saved to {output_json}")


if __name__ == "__main__":
    main()

