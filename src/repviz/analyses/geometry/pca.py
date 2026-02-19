"""PCA analysis and visualization of patch features.

The iconic DINOv2/v3 visualization: project patch features to top-3 PCA
components and map to RGB for a semantic colormap of the image.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.decomposition import PCA

from ...models.base import BackboneWrapper


def compute_pca(
    features: torch.Tensor,
    n_components: int = 3,
    whiten: bool = False,
) -> tuple[np.ndarray, PCA]:
    """Compute PCA on a feature matrix.

    Args:
        features: (N, D) feature matrix.
        n_components: Number of PCA components.

    Returns:
        (N, n_components) projected features and fitted PCA object.
    """
    feat_np = features.float().cpu().numpy()
    pca = PCA(n_components=n_components, whiten=whiten)
    projected = pca.fit_transform(feat_np)
    return projected, pca


def pca_feature_map(
    patch_features: torch.Tensor,
    h: int,
    w: int,
    n_components: int = 3,
    pca: PCA | None = None,
    normalize: bool = True,
) -> tuple[np.ndarray, PCA]:
    """Compute PCA RGB map from patch features of a single image.

    Args:
        patch_features: (N_patches, D) patch features for one image.
        h, w: Spatial dimensions of the patch grid.
        n_components: Number of PCA components (3 for RGB).
        pca: Pre-fitted PCA object (if None, fit on this image).
        normalize: Normalize to [0, 1] for visualization.

    Returns:
        (H, W, n_components) array and PCA object.
    """
    feat_np = patch_features.float().cpu().numpy()

    if pca is None:
        pca = PCA(n_components=n_components)
        projected = pca.fit_transform(feat_np)
    else:
        projected = pca.transform(feat_np)

    # Reshape to spatial
    feature_map = projected.reshape(h, w, n_components)

    if normalize:
        # Per-channel min-max normalization
        for c in range(n_components):
            vmin, vmax = feature_map[..., c].min(), feature_map[..., c].max()
            if vmax - vmin > 1e-8:
                feature_map[..., c] = (feature_map[..., c] - vmin) / (vmax - vmin)
            else:
                feature_map[..., c] = 0.5

    return feature_map, pca


def batch_pca_feature_maps(
    batch_patch_features: torch.Tensor,
    h: int,
    w: int,
    n_components: int = 3,
    shared_pca: bool = True,
) -> tuple[np.ndarray, PCA]:
    """Compute PCA RGB maps for a batch of images.

    Args:
        batch_patch_features: (B, N_patches, D) batch of patch features.
        h, w: Spatial grid dimensions.
        shared_pca: If True, fit PCA on all patches jointly (consistent colors).

    Returns:
        (B, H, W, n_components) array and PCA object.
    """
    B, N, D = batch_patch_features.shape
    assert N == h * w, f"Expected {h}*{w}={h*w} patches, got {N}"

    if shared_pca:
        # Fit PCA on all patches from all images
        all_feats = batch_patch_features.reshape(-1, D).float().cpu().numpy()
        pca = PCA(n_components=n_components)
        all_projected = pca.fit_transform(all_feats)
        all_projected = all_projected.reshape(B, h, w, n_components)

        # Normalize globally
        for c in range(n_components):
            vmin = all_projected[..., c].min()
            vmax = all_projected[..., c].max()
            if vmax - vmin > 1e-8:
                all_projected[..., c] = (all_projected[..., c] - vmin) / (vmax - vmin)
            else:
                all_projected[..., c] = 0.5

        return all_projected, pca
    else:
        # Fit PCA per image
        results = []
        pca = None
        for i in range(B):
            fm, pca = pca_feature_map(batch_patch_features[i], h, w, n_components)
            results.append(fm)
        return np.stack(results), pca


def singular_value_spectrum(
    features: torch.Tensor,
    normalize: bool = True,
) -> np.ndarray:
    """Compute singular value spectrum of feature matrix.

    Args:
        features: (N, D) feature matrix.
        normalize: If True, normalize so they sum to 1.

    Returns:
        Array of singular values.
    """
    # Center
    feat = features.float().cpu()
    feat = feat - feat.mean(dim=0, keepdim=True)

    _, s, _ = torch.linalg.svd(feat, full_matrices=False)
    s = s.numpy()

    if normalize:
        s = s / s.sum()

    return s


def effective_rank(features: torch.Tensor) -> float:
    """Compute effective rank: exp(entropy of normalized singular values).

    Higher = more dimensions are "used". Lower = more collapsed.
    """
    s = singular_value_spectrum(features, normalize=True)
    # Avoid log(0)
    s = s[s > 1e-10]
    entropy = -(s * np.log(s)).sum()
    return float(np.exp(entropy))
