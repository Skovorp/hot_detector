import os
import pickle
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional, Callable, Tuple
import torchvision
from torchvision.transforms import v2

def make_transform(resize_size):
    to_tensor = v2.ToImage()
    if resize_size is not None:
        resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    if resize_size is not None:
        return v2.Compose([to_tensor, resize, to_float, normalize])
    else:
        return v2.Compose([to_tensor, to_float, normalize])
    
    
class PairDatasetPreprocessed(Dataset):
    def __init__(
        self, 
        partition: str, 
        resize_to: int,
        root_dir: str = "."
    ):
        if partition not in ['train', 'test']:
            raise ValueError("Partition must be either 'train' or 'test'")
        
        self.partition = partition
        self.root_dir = root_dir
        self.transform = make_transform(resize_to)
        
        csv_path = os.path.join(root_dir, f"_{partition}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        self.data = pd.read_csv(csv_path)
        
        required_columns = ['from_photo_path', 'to_photo_path', 'is_like']
        if not all(col in self.data.columns for col in required_columns):
            raise ValueError(f"CSV file must contain columns: {required_columns}")
        
        print(f"Loaded {len(self.data)} samples from {partition} partition")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:        
        from_img_path = os.path.join(self.root_dir, self.data.iloc[idx]['from_photo_path'].split('/')[-1])
        to_img_path = os.path.join(self.root_dir, self.data.iloc[idx]['to_photo_path'].split('/')[-1])
        target = self.data.iloc[idx]['is_like']
        
        try:
            from_image = Image.open(from_img_path).convert('RGB')
            from_image = self.transform(from_image)
        except Exception as e:
            raise RuntimeError(f"Error loading image {from_img_path}: {e}")

        try:
            to_image = Image.open(to_img_path).convert('RGB')
            to_image = self.transform(to_image)
        except Exception as e:
            raise RuntimeError(f"Error loading image {to_img_path}: {e}")

        target = torch.tensor(target, dtype=torch.float32)
        
        return from_image, to_image, target