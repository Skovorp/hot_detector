import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from safetensors.torch import save_file
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from dataset import make_transform
from models import PreprocModel
from dotenv import load_dotenv

load_dotenv()

IMAGE_ROOT = Path("/root/used_images")
CACHE_ROOT = Path("/root/hot_detector/pairwise/image_cache")
BATCH_SIZE = 256
SKIP_LAYERS = 1
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
NUM_WORKERS = 8 # max(1, min(8, (os.cpu_count() or 1) // 2))
MAPPING_FILENAME = "partition_mapping.json"


class ImageFolderDataset(Dataset):
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.root_dir}")

        self.images: List[str] = sorted(
            str(path)
            for path in self.root_dir.rglob("*")
            if (
                path.is_file()
                and not path.name.startswith(".")
                and path.suffix.lower() in ALLOWED_EXTENSIONS
            )
        )
        if not self.images:
            raise RuntimeError(f"No images found under {self.root_dir}")

        self.transform = make_transform(512)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        image_path = self.images[idx]
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image)
        return tensor, image_path


def main() -> None:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    # Assert that the cache root directory is empty
    if any(CACHE_ROOT.iterdir()):
        raise RuntimeError(f"Cache directory {CACHE_ROOT} is not empty. Please clear it before proceeding.")

    dataset = ImageFolderDataset(IMAGE_ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PreprocModel(SKIP_LAYERS).to(device)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    partition_idx = 1
    stem_to_partition: Dict[str, str] = {}

    with torch.no_grad():
        for batch_images, batch_paths in tqdm(loader, desc="Caching images"):
            batch_images = batch_images.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16
            ):
                outputs = model(batch_images).detach().cpu()
            partition_path = CACHE_ROOT / f"partition_{partition_idx:05d}.safetensors"
            partition_data: Dict[str, torch.Tensor] = {}
            partition_path_str = str(partition_path)
            for tensor, img_path in zip(outputs, batch_paths):
                stem = Path(img_path).stem
                partition_data[stem] = tensor.contiguous().bfloat16()
                stem_to_partition[stem] = partition_path_str

            save_file(partition_data, partition_path_str)
            partition_idx += 1

    mapping_path = CACHE_ROOT / MAPPING_FILENAME
    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(stem_to_partition, f, indent=2, sort_keys=True)
    partitions_written = max(0, partition_idx - 1)
    print(f"Saved {len(stem_to_partition)} encodings across {partitions_written} partitions.")
    print(f"Partition mapping written to {mapping_path}")

if __name__ == "__main__":
    main()
