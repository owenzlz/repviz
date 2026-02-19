"""Centered Kernel Alignment (CKA) for comparing representations.

CKA (Kornblith et al. 2019) measures similarity between two representation
matrices, invariant to orthogonal transforms and isotropic scaling.
"""

from __future__ import annotations

import numpy as np
import torch


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute linear CKA between two feature matrices.

    Args:
        X: (N, D1) features from representation 1.
        Y: (N, D2) features from representation 2.

    Returns:
        CKA score in [0, 1]. 1 = identical representations.
    """
    X = X.float()
    Y = Y.float()

    # Center
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # Compute HSIC
    hsic_xy = (X.T @ Y).norm() ** 2
    hsic_xx = (X.T @ X).norm() ** 2
    hsic_yy = (Y.T @ Y).norm() ** 2

    return float(hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + 1e-10))


def rbf_cka(X: torch.Tensor, Y: torch.Tensor, sigma: float | None = None) -> float:
    """Compute RBF kernel CKA.

    More expressive than linear CKA but slower.
    """
    X = X.float()
    Y = Y.float()

    def rbf_kernel(Z: torch.Tensor, s: float) -> torch.Tensor:
        dists = torch.cdist(Z, Z, p=2)
        return torch.exp(-dists ** 2 / (2 * s ** 2))

    def centering(K: torch.Tensor) -> torch.Tensor:
        n = K.shape[0]
        H = torch.eye(n, device=K.device) - 1.0 / n
        return H @ K @ H

    if sigma is None:
        # Median heuristic
        dists_x = torch.cdist(X, X, p=2)
        sigma_x = dists_x.median().item()
        dists_y = torch.cdist(Y, Y, p=2)
        sigma_y = dists_y.median().item()
        sigma = (sigma_x + sigma_y) / 2
        if sigma < 1e-8:
            sigma = 1.0

    Kx = centering(rbf_kernel(X, sigma))
    Ky = centering(rbf_kernel(Y, sigma))

    hsic_xy = (Kx * Ky).sum()
    hsic_xx = (Kx * Kx).sum()
    hsic_yy = (Ky * Ky).sum()

    return float(hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + 1e-10))


def layerwise_cka_matrix(
    features: dict[str, torch.Tensor],
    max_samples: int = 2000,
    method: str = "linear",
) -> tuple[np.ndarray, list[str]]:
    """Compute CKA between all pairs of layer representations.

    Args:
        features: Dict mapping layer name to (N, ...) features.
        max_samples: Subsample for efficiency.
        method: "linear" or "rbf".

    Returns:
        (L, L) CKA matrix and list of layer names.
    """
    import re
    def _sort_key(name):
        nums = re.findall(r'\d+', name)
        return int(nums[-1]) if nums else name
    layer_names = sorted(features.keys(), key=_sort_key)
    L = len(layer_names)

    # Flatten to (N_total, D) and subsample
    flat_feats = {}
    for name in layer_names:
        f = features[name]
        if f.ndim == 3:
            # (B, tokens, D) -> (B*tokens, D)
            f = f.reshape(-1, f.shape[-1])
        elif f.ndim > 2:
            f = f.reshape(f.shape[0], -1)
        if f.shape[0] > max_samples:
            idx = torch.randperm(f.shape[0])[:max_samples]
            f = f[idx]
        flat_feats[name] = f

    cka_fn = linear_cka if method == "linear" else rbf_cka
    matrix = np.zeros((L, L))

    for i in range(L):
        for j in range(i, L):
            score = cka_fn(flat_feats[layer_names[i]], flat_feats[layer_names[j]])
            matrix[i, j] = score
            matrix[j, i] = score

    return matrix, layer_names
