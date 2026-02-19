"""Intrinsic dimensionality estimation.

Two-NN estimator (Facco et al. 2017) and MLE-based ID estimation.
"""

from __future__ import annotations

import numpy as np
import torch


def two_nn_id(features: torch.Tensor, fraction: float = 0.9) -> float:
    """Two-NN intrinsic dimensionality estimator (Facco et al. 2017).

    Uses the ratio of distances to 2nd and 1st nearest neighbors.

    Args:
        features: (N, D) feature matrix.
        fraction: Fraction of points to use (trim outliers).

    Returns:
        Estimated intrinsic dimensionality.
    """
    feat = features.float().cpu()
    N = feat.shape[0]
    if N < 10:
        return float("nan")

    # Compute pairwise distances
    dists = torch.cdist(feat, feat, p=2)  # (N, N)
    # Set self-distance to inf
    dists.fill_diagonal_(float("inf"))

    # Find 1st and 2nd nearest neighbors
    sorted_dists, _ = dists.sort(dim=1)
    r1 = sorted_dists[:, 0]  # distance to 1st NN
    r2 = sorted_dists[:, 1]  # distance to 2nd NN

    # Ratio mu = r2/r1
    valid = r1 > 1e-10
    mu = r2[valid] / r1[valid]
    mu = mu.numpy()

    if len(mu) < 5:
        return float("nan")

    # Sort and trim
    mu = np.sort(mu)
    n_use = max(int(len(mu) * fraction), 5)
    mu = mu[:n_use]

    # MLE: d = N / sum(log(mu))
    log_mu = np.log(mu)
    log_mu = log_mu[np.isfinite(log_mu)]
    if len(log_mu) == 0 or log_mu.sum() < 1e-10:
        return float("nan")

    d = len(log_mu) / log_mu.sum()
    return float(d)


def mle_id(
    features: torch.Tensor,
    k: int = 5,
) -> float:
    """MLE-based intrinsic dimensionality estimator (Levina & Bickel 2004).

    Args:
        features: (N, D) feature matrix.
        k: Number of nearest neighbors.

    Returns:
        Estimated intrinsic dimensionality.
    """
    feat = features.float().cpu()
    N = feat.shape[0]
    if N < k + 2:
        return float("nan")

    dists = torch.cdist(feat, feat, p=2)
    dists.fill_diagonal_(float("inf"))
    sorted_dists, _ = dists.sort(dim=1)

    # Use distances 1..k
    nn_dists = sorted_dists[:, :k]  # (N, k)
    # Filter out points with zero distances
    valid = nn_dists[:, 0] > 1e-10
    nn_dists = nn_dists[valid]

    if nn_dists.shape[0] < 5:
        return float("nan")

    # MLE estimate per point
    T_k = nn_dists[:, -1:]  # (N', 1) - distance to k-th neighbor
    log_ratios = torch.log(T_k / nn_dists[:, :-1])  # (N', k-1)
    m_hat = (k - 1) / log_ratios.sum(dim=1)  # (N',)

    # Average over points
    m_hat = m_hat.numpy()
    m_hat = m_hat[np.isfinite(m_hat) & (m_hat > 0)]
    if len(m_hat) == 0:
        return float("nan")

    return float(np.mean(m_hat))


def id_per_layer(
    layer_features: dict[str, torch.Tensor],
    method: str = "two_nn",
    max_samples: int = 1000,
    **kwargs,
) -> dict[str, float]:
    """Compute intrinsic dimensionality for each layer.

    Args:
        layer_features: Dict mapping layer name to (N, ...) features.
        method: "two_nn" or "mle".
        max_samples: Subsample for speed.

    Returns:
        Dict mapping layer name to ID estimate.
    """
    fn = two_nn_id if method == "two_nn" else mle_id
    results = {}
    for name in sorted(layer_features.keys()):
        feat = layer_features[name]
        if feat.ndim > 2:
            feat = feat.reshape(feat.shape[0], -1)
        if feat.shape[0] > max_samples:
            idx = torch.randperm(feat.shape[0])[:max_samples]
            feat = feat[idx]
        results[name] = fn(feat, **kwargs)
    return results
