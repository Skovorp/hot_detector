import random
from pathlib import Path
from typing import Dict, Tuple
import json
import traceback

import pandas as pd
import torch
from safetensors.torch import load_file
from torch.utils.data import Dataset
from torchvision.transforms import v2
from safetensors.torch import safe_open


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
    """
    Dataset that serves cached encodings for image pairs and their binary label.

    The encodings must be produced in advance by running `create_cache.py`,
    which stores a `.safetensors` file per source image under ``cache_dir``.
    """

    REQUIRED_COLUMNS = ("from_photo_path", "to_photo_path", "is_like")

    def __init__(
        self,
        partition: str,
        cache_dir: str,
        num_items_epoch: int,
        csv_pattern: str = "_{partition}_pairs.csv",
        max_retries: int = 5,
    ):
        if partition not in ("train", "test"):
            raise ValueError("Partition must be either 'train' or 'test'")

        self.partition = partition
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.num_items_epoch = num_items_epoch

        if not self.cache_dir.exists():
            raise FileNotFoundError(f"Cache directory not found: {self.cache_dir}")

        csv_path = self.cache_dir / f"_{partition}_pairs.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        self.data = pd.read_csv(csv_path)
        self.mapping = json.load(open(self.cache_dir / 'partition_mapping.json'))

        self.open_partitions = {x: safe_open(x, framework="pt", device="cpu") for x in set(self.mapping.values())}
        available_stems = [y for x in self.open_partitions.values() for y in x.keys() ]

        missing_columns = [
            col for col in self.REQUIRED_COLUMNS if col not in self.data.columns
        ]
        if missing_columns:
            raise ValueError(
                f"CSV file must contain columns: {', '.join(self.REQUIRED_COLUMNS)}"
            )

        self.data["from_stem"] = self.data["from_photo_path"].apply(self._extract_stem)
        self.data["to_stem"] = self.data["to_photo_path"].apply(self._extract_stem)

        unique_stems = pd.unique(
            pd.concat([self.data["from_stem"], self.data["to_stem"]], ignore_index=True)
        )

        print("before length", len(self.data))
        self.data = self.data[(self.data["from_stem"].isin(available_stems)) & (self.data["to_stem"].isin(available_stems))]
        print("after  length", len(self.data))
        
        self.max_retries = max(1, int(max_retries))

        print(f"Loaded {len(self.data)} samples from {partition} partition")
        
        assert self.num_items_epoch > 0 and self.num_items_epoch <= len(self.data), (self.num_items_epoch, len(self.data))
        self.sampled_data = self.data
        if self.partition == 'train':
            self.resample_data()

    def __len__(self) -> int:
        return len(self.sampled_data)
    
    def resample_data(self, ):
        self.sampled_data = self.data.sample(n=self.num_items_epoch)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # attempts = 0
        # last_error: Exception | None = None

        # while attempts < self.max_retries:
        row = self.sampled_data.iloc[idx]
        #     try:
        from_encoding = self._load_encoding(row["from_stem"])
        to_encoding = self._load_encoding(row["to_stem"])
        target = torch.tensor(row["is_like"])
        return from_encoding, to_encoding, target
        #     except Exception as exc:
        #         attempts += 1
        #         last_error = exc
        #         print(f"Failed to load smth: {exc}.")
        #         traceback.print_exc()
        #         idx = random.randrange(len(self.data))
        #         continue

        # raise RuntimeError(
        #     f"Failed to fetch sample after {self.max_retries} attempts: {last_error}"
        # )

    def _extract_stem(self, path_str: str) -> str:
        """Return filename stem independent of incoming relative path."""
        return Path(path_str).stem

    def _load_encoding(self, stem: str) -> torch.Tensor:
        try:
            return self.open_partitions[self.mapping[stem]].get_tensor(stem)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load cached encoding for {stem}: {exc}"
            ) from exc
        
if __name__ == "__main__":
    ds = PairDatasetPreprocessed('train', '/root/hot_detector/pairwise/image_cache')
