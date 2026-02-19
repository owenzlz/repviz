"""Visualization utilities for feature analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure


def plot_pca_rgb(
    image: np.ndarray,
    pca_map: np.ndarray,
    title: str = "",
    figsize: tuple[int, int] = (10, 4),
    save_path: str | Path | None = None,
) -> Figure:
    """Plot original image alongside its PCA RGB feature map.

    Args:
        image: (H, W, 3) original image in [0, 1].
        pca_map: (h, w, 3) PCA feature map in [0, 1].
        title: Figure title.
        save_path: Optional path to save.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(pca_map)
    axes[1].set_title("PCA Features (RGB)")
    axes[1].axis("off")

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_pca_grid(
    images: np.ndarray,
    pca_maps: np.ndarray,
    titles: list[str] | None = None,
    figsize: tuple[int, int] = (16, 8),
    save_path: str | Path | None = None,
) -> Figure:
    """Plot grid of images with their PCA feature maps.

    Args:
        images: (B, H, W, 3) original images.
        pca_maps: (B, h, w, 3) PCA feature maps.
    """
    B = images.shape[0]
    fig, axes = plt.subplots(2, B, figsize=figsize)
    if B == 1:
        axes = axes.reshape(2, 1)

    for i in range(B):
        axes[0, i].imshow(images[i])
        axes[0, i].axis("off")
        if titles and i < len(titles):
            axes[0, i].set_title(titles[i], fontsize=10)

        axes[1, i].imshow(pca_maps[i])
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=12)
    axes[1, 0].set_ylabel("PCA RGB", fontsize=12)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_cosine_sim_map(
    image: np.ndarray,
    sim_map: np.ndarray,
    query_pos: tuple[int, int] | None = None,
    title: str = "Cosine Similarity",
    cmap: str = "RdBu_r",
    figsize: tuple[int, int] = (10, 4),
    save_path: str | Path | None = None,
) -> Figure:
    """Plot image with cosine similarity heatmap overlay.

    Args:
        image: (H, W, 3) original image.
        sim_map: (h, w) cosine similarity map.
        query_pos: (row, col) of query patch in grid coords (optional, draws marker).
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")
    if query_pos:
        # Scale query pos to image coords
        h_scale = image.shape[0] / sim_map.shape[0]
        w_scale = image.shape[1] / sim_map.shape[1]
        y = query_pos[0] * h_scale + h_scale / 2
        x = query_pos[1] * w_scale + w_scale / 2
        axes[0].plot(x, y, "rx", markersize=15, markeredgewidth=3)

    im = axes[1].imshow(sim_map, cmap=cmap, vmin=-1, vmax=1)
    axes[1].set_title(title)
    axes[1].axis("off")
    if query_pos:
        axes[1].plot(query_pos[1], query_pos[0], "rx", markersize=10, markeredgewidth=2)
    fig.colorbar(im, ax=axes[1], fraction=0.046)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_singular_values(
    singular_values: np.ndarray,
    effective_rank_val: float | None = None,
    title: str = "Singular Value Spectrum",
    log_scale: bool = True,
    figsize: tuple[int, int] = (8, 5),
    save_path: str | Path | None = None,
) -> Figure:
    """Plot singular value spectrum."""
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(singular_values, linewidth=2)
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Component Index", fontsize=12)
    ax.set_ylabel("Singular Value", fontsize=12)
    ax.set_title(title, fontsize=14)

    if effective_rank_val is not None:
        ax.axvline(x=effective_rank_val, color="red", linestyle="--", alpha=0.7,
                    label=f"Effective Rank ≈ {effective_rank_val:.1f}")
        ax.legend(fontsize=11)

    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_cka_matrix(
    cka_matrix: np.ndarray,
    layer_names: list[str],
    title: str = "Layer-wise CKA Similarity",
    figsize: tuple[int, int] = (8, 7),
    save_path: str | Path | None = None,
) -> Figure:
    """Plot CKA similarity matrix as heatmap."""
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cka_matrix,
        xticklabels=layer_names,
        yticklabels=layer_names,
        cmap="magma",
        vmin=0, vmax=1,
        square=True,
        ax=ax,
        cbar_kws={"label": "CKA Score"},
    )
    ax.set_title(title, fontsize=14)

    # Rotate labels
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_attention_map(
    image: np.ndarray,
    attention: np.ndarray,
    h: int,
    w: int,
    title: str = "Attention",
    figsize: tuple[int, int] = (10, 4),
    save_path: str | Path | None = None,
) -> Figure:
    """Plot attention map overlaid on image.

    Args:
        image: (H, W, 3) original image.
        attention: (N,) attention values for patches.
        h, w: Spatial grid.
    """
    attn_map = attention.reshape(h, w)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(attn_map, cmap="inferno")
    axes[1].set_title("Attention Map")
    axes[1].axis("off")

    # Overlay
    import cv2
    attn_resized = np.array(
        __import__("PIL").Image.fromarray(
            (attn_map * 255).astype(np.uint8)
        ).resize((image.shape[1], image.shape[0]), __import__("PIL").Image.BILINEAR)
    ).astype(np.float32) / 255.0

    overlay = image.copy().astype(np.float32)
    overlay = overlay * 0.4 + np.stack([attn_resized * 0.6] * 3, axis=-1)
    overlay = np.clip(overlay, 0, 1)

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_clustering_segmentation(
    image: np.ndarray,
    cluster_map: np.ndarray,
    n_clusters: int,
    title: str = "Patch Clustering",
    figsize: tuple[int, int] = (10, 4),
    save_path: str | Path | None = None,
) -> Figure:
    """Plot image with K-means patch clustering as segmentation."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    cmap = plt.cm.get_cmap("tab20", n_clusters)
    axes[1].imshow(cluster_map, cmap=cmap, interpolation="nearest")
    axes[1].set_title(f"K-means ({n_clusters} clusters)")
    axes[1].axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
