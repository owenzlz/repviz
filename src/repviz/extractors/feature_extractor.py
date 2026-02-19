"""Bulk feature extraction with caching."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..models.base import BackboneWrapper


class FeatureCache:
    """Cache extracted features to disk."""

    def __init__(self, cache_dir: str | Path = ".repviz_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, model_name: str, dataset_name: str, layer: str) -> str:
        raw = f"{model_name}_{dataset_name}_{layer}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, model_name: str, dataset_name: str, layer: str) -> torch.Tensor | None:
        path = self.cache_dir / f"{self._key(model_name, dataset_name, layer)}.pt"
        if path.exists():
            return torch.load(path, weights_only=True)
        return None

    def put(self, model_name: str, dataset_name: str, layer: str, tensor: torch.Tensor) -> None:
        path = self.cache_dir / f"{self._key(model_name, dataset_name, layer)}.pt"
        torch.save(tensor, path)


def extract_features(
    backbone: BackboneWrapper,
    dataloader: DataLoader,
    layers: list[int] | None = None,
    max_samples: int | None = None,
    show_progress: bool = True,
) -> dict[str, torch.Tensor]:
    """Extract features from all specified layers.

    Returns:
        Dict mapping layer name to feature tensor (N_samples, ...).
    """
    backbone.register_hooks(layers=layers)

    all_features: dict[str, list[torch.Tensor]] = {}
    all_cls: list[torch.Tensor] = []
    all_patches: list[torch.Tensor] = []
    n_collected = 0

    iterator = tqdm(dataloader, desc="Extracting features") if show_progress else dataloader

    for batch in iterator:
        if isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch

        output = backbone.extract(images)

        if output.cls_token is not None:
            all_cls.append(output.cls_token.cpu())
        if output.patch_tokens is not None:
            all_patches.append(output.patch_tokens.cpu())

        for name, feat in output.intermediate_features.items():
            if name not in all_features:
                all_features[name] = []
            all_features[name].append(feat.cpu())

        n_collected += images.shape[0]
        if max_samples and n_collected >= max_samples:
            break

    backbone.remove_hooks()

    # Concatenate
    result = {}
    if all_cls:
        result["cls"] = torch.cat(all_cls, dim=0)
    if all_patches:
        result["patches"] = torch.cat(all_patches, dim=0)
    for name, feats in all_features.items():
        result[name] = torch.cat(feats, dim=0)

    return result
