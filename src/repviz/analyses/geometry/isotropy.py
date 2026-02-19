"""Feature isotropy and distribution analysis."""

from __future__ import annotations

import numpy as np
import torch


def average_cosine_similarity(features: torch.Tensor, n_pairs: int = 10000) -> float:
    """Estimate average pairwise cosine similarity by random sampling.

    Isotropic features → avg cosine sim ≈ 0.
    Collapsed features → avg cosine sim → 1.
    """
    N, D = features.shape
    features = torch.nn.functional.normalize(features.float(), dim=-1)

    idx_a = torch.randint(0, N, (n_pairs,))
    idx_b = torch.randint(0, N, (n_pairs,))
    # Avoid self-pairs
    mask = idx_a != idx_b
    idx_a, idx_b = idx_a[mask], idx_b[mask]

    cos_sim = (features[idx_a] * features[idx_b]).sum(dim=-1)
    return float(cos_sim.mean())


def isotropy_score(features: torch.Tensor) -> float:
    """Compute isotropy score (Mu et al. 2018).

    Based on eigenvalues of the covariance matrix.
    Score close to 1 = isotropic, close to 0 = anisotropic.
    """
    feat = features.float().cpu()
    feat = feat - feat.mean(dim=0, keepdim=True)
    cov = (feat.T @ feat) / (feat.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues.clamp(min=1e-10)

    # Isotropy = min(eigenvalue) / max(eigenvalue)
    return float(eigenvalues.min() / eigenvalues.max())


def feature_norms(features: torch.Tensor) -> np.ndarray:
    """Compute L2 norms of all feature vectors."""
    return features.float().norm(dim=-1).cpu().numpy()


def cosine_similarity_distribution(
    features: torch.Tensor,
    n_pairs: int = 50000,
) -> np.ndarray:
    """Sample pairwise cosine similarities for histogram plotting."""
    N, D = features.shape
    features = torch.nn.functional.normalize(features.float(), dim=-1)

    idx_a = torch.randint(0, N, (n_pairs,))
    idx_b = torch.randint(0, N, (n_pairs,))
    mask = idx_a != idx_b
    idx_a, idx_b = idx_a[mask], idx_b[mask]

    cos_sim = (features[idx_a] * features[idx_b]).sum(dim=-1)
    return cos_sim.cpu().numpy()
