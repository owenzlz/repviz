"""Comprehensive activation analysis for transformer representations."""

from __future__ import annotations

import numpy as np
import torch
from scipy import stats


def _sort_layers(keys: list[str]) -> list[str]:
    """Sort layer keys numerically (block_2 before block_10)."""
    def _key(k: str) -> int:
        parts = k.split("_")
        return int(parts[-1]) if parts[-1].isdigit() else 0
    return sorted(keys, key=_key)


def activation_statistics_per_layer(layer_features: dict[str, torch.Tensor]) -> dict[str, dict]:
    """For EACH layer, compute mean, std, min, max, fraction negative, kurtosis, skewness.

    Args:
        layer_features: dict of layer_name -> (B, seq_len, D) tensor.

    Returns:
        dict of layer_name -> stats_dict.
    """
    result = {}
    for name in _sort_layers(layer_features.keys()):
        feat = layer_features[name].float().reshape(-1).numpy()
        result[name] = {
            "mean": float(np.mean(feat)),
            "std": float(np.std(feat)),
            "min": float(np.min(feat)),
            "max": float(np.max(feat)),
            "frac_negative": float(np.mean(feat < 0)),
            "kurtosis": float(stats.kurtosis(feat, fisher=True)),
            "skewness": float(stats.skew(feat)),
        }
    return result


def massive_activation_analysis(
    layer_features: dict[str, torch.Tensor],
    threshold_std: float = 5.0,
    top_k: int = 20,
) -> dict[str, dict]:
    """For EACH layer, find neurons with massive activations.

    Args:
        layer_features: dict of layer_name -> (B, seq_len, D) tensor.
        threshold_std: Number of std devs to consider "massive".
        top_k: Number of top neurons to report.

    Returns:
        dict of layer_name -> massive_act_stats.
    """
    result = {}
    for name in _sort_layers(layer_features.keys()):
        feat = layer_features[name].float()  # (B, seq_len, D)
        # Flatten batch and seq dims
        flat = feat.reshape(-1, feat.shape[-1])  # (N, D)
        
        # Per-neuron stats
        neuron_mean = flat.mean(dim=0)  # (D,)
        neuron_max = flat.max(dim=0).values  # (D,)
        neuron_std = flat.std(dim=0)  # (D,)
        
        # Global stats
        global_mean = flat.mean().item()
        global_std = flat.std().item()
        threshold = global_mean + threshold_std * global_std
        
        # Neurons exceeding threshold
        max_vals = neuron_max.numpy()
        massive_mask = max_vals > threshold
        n_massive = int(massive_mask.sum())
        
        # Top-k neurons by max activation
        topk_indices = np.argsort(max_vals)[-top_k:][::-1]
        topk_values = max_vals[topk_indices]
        
        # Spikiness: ratio of max to mean per neuron
        neuron_mean_np = neuron_mean.numpy()
        spikiness = max_vals / (np.abs(neuron_mean_np) + 1e-8)
        
        # Gini coefficient of neuron activation magnitudes
        abs_means = np.abs(neuron_mean_np)
        sorted_abs = np.sort(abs_means)
        n = len(sorted_abs)
        index = np.arange(1, n + 1)
        gini = float((2 * np.sum(index * sorted_abs) / (n * np.sum(sorted_abs) + 1e-10)) - (n + 1) / n)
        
        result[name] = {
            "n_massive": n_massive,
            "frac_massive": n_massive / feat.shape[-1],
            "topk_indices": topk_indices.tolist(),
            "topk_values": topk_values.tolist(),
            "max_spikiness": float(np.max(spikiness)),
            "mean_spikiness": float(np.mean(spikiness)),
            "gini_coefficient": gini,
            "threshold": threshold,
        }
    return result


def per_neuron_activation_profile(
    layer_features: dict[str, torch.Tensor],
    top_k: int = 20,
    dead_threshold: float = 1e-6,
    saturated_percentile: float = 99.0,
) -> dict[str, dict]:
    """For EACH layer, compute per-neuron activation profiles.

    Args:
        layer_features: dict of layer_name -> (B, seq_len, D) tensor.
        top_k: Number of top neurons to highlight.
        dead_threshold: Threshold for dead neuron detection.
        saturated_percentile: Percentile for saturated neuron detection.

    Returns:
        dict with per-layer neuron profiles.
    """
    result = {}
    for name in _sort_layers(layer_features.keys()):
        feat = layer_features[name].float()
        flat = feat.reshape(-1, feat.shape[-1])  # (N, D)
        
        neuron_mean = flat.mean(dim=0).numpy()  # (D,)
        neuron_std = flat.std(dim=0).numpy()  # (D,)
        neuron_max = flat.max(dim=0).values.numpy()  # (D,)
        neuron_min = flat.min(dim=0).values.numpy()  # (D,)
        
        # Dead neurons: mean activation ≈ 0 and max ≈ 0
        dead_mask = np.abs(neuron_max) < dead_threshold
        n_dead = int(dead_mask.sum())
        
        # Saturated neurons: very high mean
        sat_threshold = np.percentile(np.abs(neuron_mean), saturated_percentile)
        saturated_mask = np.abs(neuron_mean) > sat_threshold
        n_saturated = int(saturated_mask.sum())
        
        # Massive activation neurons: very high max but relatively low mean
        max_threshold = np.percentile(neuron_max, 99)
        mean_threshold = np.percentile(np.abs(neuron_mean), 50)
        massive_mask = (neuron_max > max_threshold) & (np.abs(neuron_mean) < mean_threshold)
        n_massive = int(massive_mask.sum())
        
        result[name] = {
            "neuron_mean": neuron_mean,
            "neuron_std": neuron_std,
            "neuron_max": neuron_max,
            "neuron_min": neuron_min,
            "n_dead": n_dead,
            "n_saturated": n_saturated,
            "n_massive_act": n_massive,
            "dead_indices": np.where(dead_mask)[0].tolist(),
            "saturated_indices": np.where(saturated_mask)[0].tolist(),
            "massive_indices": np.where(massive_mask)[0].tolist(),
        }
    return result


def activation_distribution_per_layer(
    layer_features: dict[str, torch.Tensor],
    n_bins: int = 100,
    max_samples: int = 50000,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """For EACH layer, compute histogram of all activation values.

    Args:
        layer_features: dict of layer_name -> (B, seq_len, D) tensor.
        n_bins: Number of histogram bins.
        max_samples: Max samples per layer for speed.

    Returns:
        dict of layer_name -> (bin_edges, counts).
    """
    result = {}
    for name in _sort_layers(layer_features.keys()):
        feat = layer_features[name].float().reshape(-1).numpy()
        if len(feat) > max_samples:
            idx = np.random.choice(len(feat), max_samples, replace=False)
            feat = feat[idx]
        counts, bin_edges = np.histogram(feat, bins=n_bins, density=True)
        result[name] = (bin_edges, counts)
    return result


def layer_activation_heatmap(layer_features: dict[str, torch.Tensor]) -> np.ndarray:
    """Create a 2D array: rows = layers, cols = neurons, values = mean activation.

    Args:
        layer_features: dict of layer_name -> (B, seq_len, D) tensor.

    Returns:
        (n_layers, D) array of mean activations.
    """
    keys = _sort_layers(layer_features.keys())
    rows = []
    for name in keys:
        feat = layer_features[name].float()
        flat = feat.reshape(-1, feat.shape[-1])
        rows.append(flat.mean(dim=0).numpy())
    return np.stack(rows)


def token_activation_analysis(layer_features: dict[str, torch.Tensor]) -> dict[str, dict]:
    """Compare CLS token vs patch token activations per layer.

    Assumes seq_len = 1 (CLS) + N (patches), i.e., index 0 is CLS.

    Args:
        layer_features: dict of layer_name -> (B, seq_len, D) tensor.

    Returns:
        dict of layer_name -> {cls_mean, cls_std, patch_mean, patch_std, cls_only_neurons}.
    """
    result = {}
    for name in _sort_layers(layer_features.keys()):
        feat = layer_features[name].float()  # (B, seq_len, D)
        cls_feat = feat[:, 0, :]  # (B, D)
        patch_feat = feat[:, 1:, :]  # (B, N, D)
        
        cls_mean = cls_feat.mean(dim=0).numpy()  # (D,)
        cls_std = cls_feat.std(dim=0).numpy()  # (D,)
        patch_mean = patch_feat.reshape(-1, feat.shape[-1]).mean(dim=0).numpy()  # (D,)
        patch_std = patch_feat.reshape(-1, feat.shape[-1]).std(dim=0).numpy()  # (D,)
        
        # Neurons that primarily fire for CLS: high CLS activation, low patch activation
        cls_magnitude = np.abs(cls_mean)
        patch_magnitude = np.abs(patch_mean)
        ratio = cls_magnitude / (patch_magnitude + 1e-8)
        cls_dominant = int(np.sum(ratio > 5.0))
        
        result[name] = {
            "cls_mean": float(np.mean(cls_mean)),
            "cls_std": float(np.mean(cls_std)),
            "cls_mean_per_neuron": cls_mean,
            "patch_mean": float(np.mean(patch_mean)),
            "patch_std": float(np.mean(patch_std)),
            "patch_mean_per_neuron": patch_mean,
            "cls_dominant_neurons": cls_dominant,
            "cls_patch_ratio": ratio,
        }
    return result
