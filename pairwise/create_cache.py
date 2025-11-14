import os
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from safetensors.torch import save_file
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from dataset import make_transform
from models import PreprocModel


IMAGE_ROOT = Path("/root/photos/photos")
CACHE_ROOT = Path("/root/hot_detector/pairwise/image_cache")
BATCH_SIZE = 128
SKIP_LAYERS = 2
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
NUM_WORKERS = max(1, min(8, (os.cpu_count() or 1) // 2))


class ImageFolderDataset(Dataset):
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.root_dir}")

        self.images: List[str] = sorted(
            str(path)
            for path in self.root_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
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

    dataset = ImageFolderDataset(IMAGE_ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PreprocModel(SKIP_LAYERS).to(device)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    autocast_enabled = device.type == "cuda"
    autocast_dtype = torch.bfloat16 if autocast_enabled else torch.float32

    with torch.no_grad():
        for batch_images, batch_paths in tqdm(loader, desc="Caching images"):
            batch_images = batch_images.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                outputs = model(batch_images)

            outputs = outputs.detach().cpu()
            for tensor, img_path in zip(outputs, batch_paths):
                stem = Path(img_path).stem
                cache_path = CACHE_ROOT / f"{stem}.safetensors"
                if cache_path.exists():
                    continue
                save_file({"encoding": tensor.contiguous()}, str(cache_path))


if __name__ == "__main__":
    main()
