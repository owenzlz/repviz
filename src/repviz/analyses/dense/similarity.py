"""Dense cosine similarity maps and patch-level correspondence."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def cosine_similarity_map(
    patch_features: torch.Tensor,
    query_idx: int,
    h: int,
    w: int,
) -> np.ndarray:
    """Compute cosine similarity between a query patch and all patches.

    Args:
        patch_features: (N_patches, D) features for one image.
        query_idx: Index of the query patch.
        h, w: Spatial grid dimensions.

    Returns:
        (H, W) similarity map in [-1, 1].
    """
    feats = F.normalize(patch_features.float(), dim=-1)
    query = feats[query_idx:query_idx + 1]  # (1, D)
    sim = (feats @ query.T).squeeze(-1)  # (N,)
    return sim.cpu().numpy().reshape(h, w)


def cross_image_similarity(
    features_a: torch.Tensor,
    features_b: torch.Tensor,
    query_idx: int,
    h: int,
    w: int,
) -> np.ndarray:
    """Cosine similarity between a query patch in image A and all patches in image B.

    Args:
        features_a: (N, D) patch features of image A.
        features_b: (N, D) patch features of image B.
        query_idx: Patch index in image A.
        h, w: Spatial grid of image B.

    Returns:
        (H, W) similarity map.
    """
    fa = F.normalize(features_a.float(), dim=-1)
    fb = F.normalize(features_b.float(), dim=-1)
    query = fa[query_idx:query_idx + 1]
    sim = (fb @ query.T).squeeze(-1)
    return sim.cpu().numpy().reshape(h, w)


def patch_correspondence(
    features_a: torch.Tensor,
    features_b: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Find nearest-neighbor patch correspondence between two images.

    Args:
        features_a: (N, D) patch features of image A.
        features_b: (M, D) patch features of image B.

    Returns:
        nn_indices: (N,) index in B for each patch in A.
        nn_scores: (N,) cosine similarity scores.
    """
    fa = F.normalize(features_a.float(), dim=-1)
    fb = F.normalize(features_b.float(), dim=-1)
    sim = fa @ fb.T  # (N, M)
    scores, indices = sim.max(dim=1)
    return indices.cpu().numpy(), scores.cpu().numpy()


def patch_clustering(
    patch_features: torch.Tensor,
    n_clusters: int,
    h: int,
    w: int,
) -> np.ndarray:
    """K-means clustering of patch features → segmentation mask.

    Args:
        patch_features: (N_patches, D) features.
        n_clusters: Number of clusters.
        h, w: Spatial grid.

    Returns:
        (H, W) integer cluster assignment map.
    """
    from sklearn.cluster import KMeans

    feats = patch_features.float().cpu().numpy()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(feats)
    return labels.reshape(h, w)


def self_similarity_matrix(
    patch_features: torch.Tensor,
) -> np.ndarray:
    """Compute full pairwise cosine similarity matrix of patches.

    Args:
        patch_features: (N, D) features.

    Returns:
        (N, N) similarity matrix.
    """
    feats = F.normalize(patch_features.float(), dim=-1)
    sim = (feats @ feats.T).cpu().numpy()
    return sim
