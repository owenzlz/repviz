"""Layer-wise progression analysis: norms, rank, and similarity across layers."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def feature_norm_progression(
    layer_features: dict[str, torch.Tensor],
    token_type: str = "all",
) -> dict[str, float]:
    """Compute mean L2 norm of features at each layer.

    Args:
        layer_features: Dict mapping layer name to (N, seq_len, D) or (N, D) features.
        token_type: "cls" (first token), "patch" (tokens 1:), or "all".

    Returns:
        Dict mapping layer name to mean L2 norm.
    """
    results = {}
    for name in sorted(layer_features.keys()):
        feat = layer_features[name].float()
        if feat.ndim == 3:
            if token_type == "cls":
                feat = feat[:, 0]  # (N, D)
            elif token_type == "patch":
                feat = feat[:, 1:]  # (N, seq-1, D)
                feat = feat.reshape(-1, feat.shape[-1])
            else:
                feat = feat.reshape(-1, feat.shape[-1])
        elif feat.ndim > 3:
            feat = feat.reshape(-1, feat.shape[-1])
        norms = feat.norm(dim=-1)
        results[name] = float(norms.mean())
    return results


def effective_rank_progression(
    layer_features: dict[str, torch.Tensor],
    max_samples: int = 1000,
) -> dict[str, float]:
    """Compute effective rank at each layer.

    Args:
        layer_features: Dict mapping layer name to (N, ...) features.
        max_samples: Subsample for speed.

    Returns:
        Dict mapping layer name to effective rank.
    """
    results = {}
    for name in sorted(layer_features.keys()):
        feat = layer_features[name].float()
        if feat.ndim == 3:
            feat = feat.reshape(-1, feat.shape[-1])  # (B*seq, D)
        elif feat.ndim > 2:
            feat = feat.reshape(-1, feat.shape[-1])
        if feat.shape[0] > max_samples:
            idx = torch.randperm(feat.shape[0])[:max_samples]
            feat = feat[idx]

        # Center
        feat = feat - feat.mean(dim=0, keepdim=True)
        _, s, _ = torch.linalg.svd(feat, full_matrices=False)
        s = s.numpy()
        s = s / s.sum()
        s = s[s > 1e-10]
        entropy = -(s * np.log(s)).sum()
        results[name] = float(np.exp(entropy))
    return results


def procrustes_distance(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Procrustes distance between two feature matrices.

    Finds the optimal orthogonal alignment and returns residual distance.

    Args:
        X, Y: (N, D) feature matrices (same N required).

    Returns:
        Procrustes distance (lower = more similar).
    """
    X = X.float().cpu()
    Y = Y.float().cpu()
    # Center
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    # Normalize
    X = X / (X.norm() + 1e-10)
    Y = Y / (Y.norm() + 1e-10)
    # SVD of cross-covariance
    U, _, Vt = torch.linalg.svd(X.T @ Y, full_matrices=False)
    # Optimal rotation
    R = U @ Vt
    # Distance
    return float((X @ R - Y).norm() ** 2)


def mutual_knn(
    X: torch.Tensor,
    Y: torch.Tensor,
    k: int = 10,
) -> float:
    """Mutual k-NN overlap between two representations.

    For each point, check if its k-NN set is preserved between X and Y.

    Args:
        X, Y: (N, D) feature matrices.
        k: Number of neighbors.

    Returns:
        Fraction of mutual k-NN overlap in [0, 1].
    """
    X = X.float().cpu()
    Y = Y.float().cpu()
    N = X.shape[0]
    if N < k + 1:
        return float("nan")

    # k-NN in X space
    dists_x = torch.cdist(X, X)
    dists_x.fill_diagonal_(float("inf"))
    _, knn_x = dists_x.topk(k, largest=False, dim=1)  # (N, k)

    # k-NN in Y space
    dists_y = torch.cdist(Y, Y)
    dists_y.fill_diagonal_(float("inf"))
    _, knn_y = dists_y.topk(k, largest=False, dim=1)

    # Compute overlap
    overlap = 0.0
    for i in range(N):
        set_x = set(knn_x[i].tolist())
        set_y = set(knn_y[i].tolist())
        overlap += len(set_x & set_y) / k
    return float(overlap / N)


def consecutive_layer_similarity(
    layer_features: dict[str, torch.Tensor],
    method: str = "procrustes",
    max_samples: int = 500,
    k: int = 10,
) -> dict[str, float]:
    """Compute similarity between consecutive layers.

    Args:
        layer_features: Dict mapping layer name to (N, ...) features.
        method: "procrustes" or "mutual_knn".
        max_samples: Subsample for speed.

    Returns:
        Dict mapping "layer_i->layer_j" to similarity score.
    """
    names = sorted(layer_features.keys())
    results = {}

    for i in range(len(names) - 1):
        X = layer_features[names[i]].float()
        Y = layer_features[names[i + 1]].float()

        if X.ndim == 3:
            X = X.reshape(-1, X.shape[-1])
        elif X.ndim > 2:
            X = X.reshape(-1, X.shape[-1])
        if Y.ndim == 3:
            Y = Y.reshape(-1, Y.shape[-1])
        elif Y.ndim > 2:
            Y = Y.reshape(-1, Y.shape[-1])

        n = min(X.shape[0], Y.shape[0], max_samples)
        idx = torch.randperm(min(X.shape[0], Y.shape[0]))[:n]
        X, Y = X[idx], Y[idx]

        # Align dimensions if different
        d = min(X.shape[1], Y.shape[1])
        X, Y = X[:, :d], Y[:, :d]

        key = f"{names[i]}->{names[i+1]}"
        if method == "procrustes":
            results[key] = procrustes_distance(X, Y)
        else:
            results[key] = mutual_knn(X, Y, k=k)

    return results
