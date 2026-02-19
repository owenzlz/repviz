"""Neuron / channel-level analysis."""

from __future__ import annotations

import numpy as np
import torch


def dead_neuron_fraction(
    features: torch.Tensor,
    threshold: float = 1e-6,
) -> float:
    """Fraction of feature dimensions that are never activated.

    Args:
        features: (N, D) feature matrix.
        threshold: Activation threshold.

    Returns:
        Fraction of dead neurons in [0, 1].
    """
    feat = features.float()
    # A dimension is "dead" if its max activation across all samples is below threshold
    max_act = feat.abs().max(dim=0).values  # (D,)
    dead = (max_act < threshold).float().mean()
    return float(dead)


def channel_redundancy(
    features: torch.Tensor,
    max_samples: int = 2000,
) -> np.ndarray:
    """Compute correlation matrix between feature channels.

    High correlation = redundant channels.

    Args:
        features: (N, D) feature matrix.
        max_samples: Subsample for speed.

    Returns:
        (D, D) correlation matrix.
    """
    feat = features.float().cpu()
    if feat.shape[0] > max_samples:
        idx = torch.randperm(feat.shape[0])[:max_samples]
        feat = feat[idx]

    # Standardize
    feat = feat - feat.mean(dim=0, keepdim=True)
    std = feat.std(dim=0, keepdim=True).clamp(min=1e-8)
    feat = feat / std

    corr = (feat.T @ feat) / (feat.shape[0] - 1)
    return corr.numpy()


def channel_redundancy_summary(features: torch.Tensor, max_samples: int = 2000) -> dict[str, float]:
    """Summary statistics of channel redundancy.

    Returns:
        Dict with mean_abs_corr and fraction of highly correlated pairs (>0.9).
    """
    corr = channel_redundancy(features, max_samples)
    D = corr.shape[0]
    # Upper triangle only (exclude diagonal)
    mask = np.triu(np.ones((D, D), dtype=bool), k=1)
    upper = np.abs(corr[mask])
    return {
        "mean_abs_correlation": float(upper.mean()),
        "frac_high_corr_0.9": float((upper > 0.9).mean()),
        "frac_high_corr_0.8": float((upper > 0.8).mean()),
    }


def neuron_selectivity_index(
    features: torch.Tensor,
    labels: torch.Tensor,
) -> np.ndarray:
    """Class selectivity index per feature dimension.

    For each neuron, measures how selective it is to a single class
    vs responding uniformly. CSI = (mu_max - mu_mean) / (mu_max + |mu_mean|)

    Args:
        features: (N, D) feature matrix.
        labels: (N,) class labels.

    Returns:
        (D,) selectivity index per dimension.
    """
    feat = features.float().cpu()
    labels = labels.long().cpu()
    classes = labels.unique()

    class_means = []
    for c in classes:
        mask = labels == c
        class_means.append(feat[mask].mean(dim=0))
    class_means = torch.stack(class_means)  # (C, D)

    mu_max = class_means.max(dim=0).values  # (D,)
    mu_mean = class_means.mean(dim=0)  # (D,)

    denom = mu_max.abs() + mu_mean.abs() + 1e-10
    csi = (mu_max - mu_mean) / denom
    return csi.numpy()


def topk_activating_patches(
    features: torch.Tensor,
    dim_idx: int,
    k: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Find top-k most activating samples for a given feature dimension.

    Args:
        features: (N, D) feature matrix.
        dim_idx: Feature dimension index.
        k: Number of top activations.

    Returns:
        (k,) indices of top samples, (k,) activation values.
    """
    activations = features[:, dim_idx].float()
    k = min(k, activations.shape[0])
    values, indices = activations.topk(k)
    return indices.cpu().numpy(), values.cpu().numpy()
