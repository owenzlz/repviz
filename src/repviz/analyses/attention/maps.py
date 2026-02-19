"""Attention map analysis: visualization, distance, entropy."""

from __future__ import annotations

import numpy as np
import torch


def attention_rollout(
    attention_maps: dict[str, torch.Tensor],
    head_fusion: str = "mean",
    discard_ratio: float = 0.0,
) -> np.ndarray:
    """Compute attention rollout across all layers.

    Recursively multiplies attention matrices from all layers to get
    the total attention from input tokens to the CLS token.

    Args:
        attention_maps: Dict of layer_name → (B, H, N, N) attention tensors.
        head_fusion: How to fuse heads ("mean", "max", "min").
        discard_ratio: Fraction of lowest attention to zero out per layer.

    Returns:
        (B, N) attention from CLS to all tokens (excluding CLS itself).
    """
    sorted_layers = sorted(attention_maps.keys())
    result = None

    for layer_name in sorted_layers:
        attn = attention_maps[layer_name].float()  # (B, H, N, N)

        # Fuse heads
        if head_fusion == "mean":
            attn = attn.mean(dim=1)  # (B, N, N)
        elif head_fusion == "max":
            attn = attn.max(dim=1).values
        elif head_fusion == "min":
            attn = attn.min(dim=1).values
        else:
            raise ValueError(f"Unknown head_fusion: {head_fusion}")

        # Optional: discard low attention
        if discard_ratio > 0:
            flat = attn.reshape(attn.shape[0], -1)
            k = int(flat.shape[1] * discard_ratio)
            if k > 0:
                threshold = flat.kthvalue(k, dim=1).values.unsqueeze(-1).unsqueeze(-1)
                attn = attn * (attn > threshold).float()
                # Re-normalize rows
                attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)

        # Add identity (residual connection)
        I = torch.eye(attn.shape[-1], device=attn.device).unsqueeze(0)
        attn = 0.5 * attn + 0.5 * I

        # Normalize
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)

        if result is None:
            result = attn
        else:
            result = result @ attn

    # Extract CLS attention (row 0, skip column 0 = CLS self-attention)
    cls_attn = result[:, 0, 1:]  # (B, N-1)
    return cls_attn.cpu().numpy()


def attention_distance(
    attention_map: torch.Tensor,
    h: int,
    w: int,
) -> np.ndarray:
    """Compute mean attention distance per head.

    For each head, compute the average spatial distance weighted by attention.
    This reveals receptive field size per head.

    Args:
        attention_map: (B, H, N, N) attention tensor for one layer.
            N includes CLS token; we use only patch-to-patch attention.
        h, w: Spatial grid dimensions.

    Returns:
        (H,) mean attention distance per head (averaged over batch).
    """
    B, num_heads, N, _ = attention_map.shape

    # Patch-to-patch attention only (skip CLS)
    attn = attention_map[:, :, 1:, 1:].float()  # (B, H, hw, hw)

    # Create position grid
    positions = torch.stack(torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    ), dim=-1).reshape(-1, 2)  # (hw, 2)

    # Pairwise distances
    dists = torch.cdist(positions, positions, p=2)  # (hw, hw)
    dists = dists.unsqueeze(0).unsqueeze(0)  # (1, 1, hw, hw)

    # Weighted mean distance
    mean_dist = (attn * dists.to(attn.device)).sum(dim=-1).mean(dim=(0, 2))  # (H,)
    return mean_dist.cpu().numpy()


def attention_entropy(attention_map: torch.Tensor) -> np.ndarray:
    """Compute entropy of attention distribution per head per layer.

    High entropy = diffuse attention. Low entropy = focused/sharp.

    Args:
        attention_map: (B, H, N, N) attention tensor.

    Returns:
        (H,) mean entropy per head (averaged over batch and query tokens).
    """
    attn = attention_map.float().clamp(min=1e-8)
    entropy = -(attn * attn.log()).sum(dim=-1)  # (B, H, N)
    return entropy.mean(dim=(0, 2)).cpu().numpy()  # (H,)


def head_similarity_matrix(
    attention_map: torch.Tensor,
) -> np.ndarray:
    """Compute cosine similarity between attention heads.

    Useful for identifying redundant or specialized heads.

    Args:
        attention_map: (B, H, N, N) attention tensor.

    Returns:
        (H, H) cosine similarity matrix.
    """
    B, H, N, _ = attention_map.shape
    # Flatten each head's attention pattern
    heads = attention_map.float().reshape(B, H, -1).mean(dim=0)  # (H, N*N)
    heads = torch.nn.functional.normalize(heads, dim=-1)
    sim = (heads @ heads.T).cpu().numpy()
    return sim
