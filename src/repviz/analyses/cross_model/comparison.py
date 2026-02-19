"""Cross-model comparison: CKA and mutual k-NN analysis."""

from __future__ import annotations

import numpy as np
import torch

from ..layerwise.cka import linear_cka
from ..layerwise.progression import mutual_knn


def cross_model_cka(
    features_a: torch.Tensor,
    features_b: torch.Tensor,
) -> float:
    """Compute linear CKA between features from two different models.

    Args:
        features_a: (N, D1) features from model A.
        features_b: (N, D2) features from model B.

    Returns:
        CKA score in [0, 1].
    """
    return linear_cka(features_a, features_b)


def cross_model_cka_matrix(
    model_features: dict[str, torch.Tensor],
    max_samples: int = 2000,
) -> tuple[np.ndarray, list[str]]:
    """Compute CKA matrix between all pairs of models.

    Args:
        model_features: Dict mapping model name to (N, D) features.
        max_samples: Subsample for efficiency.

    Returns:
        (M, M) CKA matrix and list of model names.
    """
    names = sorted(model_features.keys())
    M = len(names)

    feats = {}
    for name in names:
        f = model_features[name].float()
        if f.ndim > 2:
            f = f.reshape(f.shape[0], -1)
        if f.shape[0] > max_samples:
            idx = torch.randperm(f.shape[0])[:max_samples]
            f = f[idx]
        feats[name] = f

    matrix = np.zeros((M, M))
    for i in range(M):
        for j in range(i, M):
            n = min(feats[names[i]].shape[0], feats[names[j]].shape[0])
            score = linear_cka(feats[names[i]][:n], feats[names[j]][:n])
            matrix[i, j] = score
            matrix[j, i] = score

    return matrix, names


def cross_model_mutual_knn(
    features_a: torch.Tensor,
    features_b: torch.Tensor,
    k: int = 10,
    max_samples: int = 1000,
) -> float:
    """Mutual k-NN overlap between two model representations.

    Measures whether the two models agree on neighborhood structure
    (Platonic representation hypothesis).

    Args:
        features_a: (N, D1) features from model A.
        features_b: (N, D2) features from model B.
        k: Number of neighbors.
        max_samples: Subsample for speed.

    Returns:
        Mutual k-NN overlap in [0, 1].
    """
    n = min(features_a.shape[0], features_b.shape[0], max_samples)
    idx = torch.randperm(min(features_a.shape[0], features_b.shape[0]))[:n]
    return mutual_knn(features_a[idx], features_b[idx], k=k)
