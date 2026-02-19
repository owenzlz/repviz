"""Data loading utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def make_dinov2_transform(resize: int = 518):
    """Standard DINOv2/v3 evaluation transform (ImageNet normalization)."""
    import torchvision.transforms as T

    return T.Compose([
        T.Resize((resize, resize), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def inverse_normalize(tensor: torch.Tensor) -> np.ndarray:
    """Inverse ImageNet normalization → numpy image in [0, 1]."""
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img = tensor.cpu().float() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


class ImageFolderFlat(Dataset):
    """Simple dataset that loads all images from a directory."""

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    def __init__(self, root: str | Path, transform=None, max_images: int | None = None):
        self.root = Path(root)
        self.transform = transform
        self.paths = sorted([
            p for p in self.root.rglob("*")
            if p.suffix.lower() in self.EXTENSIONS
        ])
        if max_images:
            self.paths = self.paths[:max_images]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, str(self.paths[idx])


class URLImageDataset(Dataset):
    """Load images from URLs (for quick demos)."""

    DEMO_URLS = [
        "http://images.cocodataset.org/val2017/000000039769.jpg",  # cats
        "http://images.cocodataset.org/val2017/000000397133.jpg",  # bus
        "http://images.cocodataset.org/val2017/000000037777.jpg",  # giraffe
        "http://images.cocodataset.org/val2017/000000252219.jpg",  # kitchen
        "http://images.cocodataset.org/val2017/000000087038.jpg",  # pizza
    ]

    def __init__(self, urls: list[str] | None = None, transform=None):
        self.urls = urls or self.DEMO_URLS
        self.transform = transform
        self._cache: dict[str, Image.Image] = {}

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        url = self.urls[idx]
        if url not in self._cache:
            import requests
            from io import BytesIO
            response = requests.get(url, timeout=10)
            self._cache[url] = Image.open(BytesIO(response.content)).convert("RGB")

        img = self._cache[url]
        if self.transform:
            return self.transform(img), url
        return img, url


def compute_grid_size(num_patches: int, patch_size: int, image_size: int) -> tuple[int, int]:
    """Compute spatial grid dimensions from number of patches."""
    h = w = int(num_patches ** 0.5)
    assert h * w == num_patches, f"Non-square patch grid: {num_patches} patches"
    return h, w
