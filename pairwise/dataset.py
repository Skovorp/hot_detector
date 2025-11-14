import random
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch
from safetensors.torch import load_file
from torch.utils.data import Dataset
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
    """
    Dataset that serves cached encodings for image pairs and their binary label.

    The encodings must be produced in advance by running `create_cache.py`,
    which stores a `.safetensors` file per source image under ``cache_dir``.
    """

    REQUIRED_COLUMNS = ("from_photo_path", "to_photo_path", "is_like")

    def __init__(
        self,
        partition: str,
        root_dir: str = ".",
        cache_dir: str | None = None,
        csv_pattern: str = "_{partition}_pairs.csv",
        max_retries: int = 5,
    ):
        if partition not in ("train", "test"):
            raise ValueError("Partition must be either 'train' or 'test'")

        self.partition = partition
        self.root_dir = Path(root_dir).expanduser().resolve()
        default_cache_dir = Path(__file__).resolve().parent / "image_cache"
        self.cache_dir = (
            Path(cache_dir).expanduser().resolve() if cache_dir else default_cache_dir
        )

        if not self.cache_dir.exists():
            raise FileNotFoundError(f"Cache directory not found: {self.cache_dir}")

        try:
            csv_filename = csv_pattern.format(partition=partition)
        except KeyError as exc:
            raise ValueError(
                "csv_pattern must be a format string accepting the 'partition' keyword"
            ) from exc

        csv_path = self.root_dir / csv_filename
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        self.data = pd.read_csv(csv_path)

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

        self._stem_to_path: Dict[str, Path] = {}
        missing_cache = []
        for stem in unique_stems:
            cache_path = self.cache_dir / f"{stem}.safetensors"
            if cache_path.exists():
                self._stem_to_path[stem] = cache_path
            else:
                missing_cache.append(stem)

        if missing_cache:
            sample = ", ".join(missing_cache[:5])
            print(
                f"Cached encodings missing for {len(missing_cache)} photos "
                f"(showing up to 5): {sample}"
            )
        self.data = self.data[(~self.data["from_stem"].isin(missing_cache)) & (~self.data["to_stem"].isin(missing_cache))]

        self.max_retries = max(1, int(max_retries))

        print(f"Loaded {len(self.data)} samples from {partition} partition")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.item()

        attempts = 0
        last_error: Exception | None = None

        while attempts < self.max_retries:
            row = self.data.iloc[idx]
            try:
                from_encoding = self._load_encoding(row["from_stem"])
                to_encoding = self._load_encoding(row["to_stem"])
                target = torch.tensor(row["is_like"], dtype=torch.float32)
                return from_encoding, to_encoding, target
            except Exception as exc:
                attempts += 1
                last_error = exc
                print(f"Failed to load smth: {exc}")
                idx = random.randrange(len(self.data))
                continue

        raise RuntimeError(
            f"Failed to fetch sample after {self.max_retries} attempts: {last_error}"
        )

    def _extract_stem(self, path_str: str) -> str:
        """Return filename stem independent of incoming relative path."""
        return Path(path_str).stem

    def _load_encoding(self, stem: str) -> torch.Tensor:
        cache_path = self._stem_to_path[stem]
        try:
            tensor = load_file(str(cache_path))["encoding"]
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load cached encoding for {stem}: {exc}"
            ) from exc
        if tensor.dtype != torch.float32:
            tensor = tensor.to(torch.float32)
        return tensor.contiguous()