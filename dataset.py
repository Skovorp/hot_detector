import os
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

class HotDetectorDataset(Dataset):
    """
    PyTorch Dataset for Hot Detector project.
    
    Args:
        partition (str): Either 'train' or 'test' to specify which CSV file to load
        root_dir (str): Root directory containing the CSV files and photos folder
        transform (Optional[Callable]): Optional transform to be applied on images
        target_transform (Optional[Callable]): Optional transform to be applied on targets
    """
    
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
        
        required_columns = ['photo_path', 'part_likes']
        if not all(col in self.data.columns for col in required_columns):
            raise ValueError(f"CSV file must contain columns: {required_columns}")
        
        print(f"Loaded {len(self.data)} samples from {partition} partition")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Image tensor and target tensor
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image path and target
        img_path = os.path.join(self.root_dir, self.data.iloc[idx]['photo_path'].split('/')[-1])
        target = self.data.iloc[idx]['part_likes']
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {e}")
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        target = torch.tensor(target, dtype=torch.float32)
        
        return image, target
    
    def get_stats(self) -> dict:
        """
        Get basic statistics about the dataset.
        
        Returns:
            dict: Dictionary containing dataset statistics
        """
        stats = {
            'partition': self.partition,
            'num_samples': len(self.data),
            'target_mean': self.data['part_likes'].mean(),
            'target_var': self.data['part_likes'].var(),
            'target_std': self.data['part_likes'].std(),
            'target_min': self.data['part_likes'].min(),
            'target_max': self.data['part_likes'].max()
        }
        return stats
