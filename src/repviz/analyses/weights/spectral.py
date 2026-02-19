"""Weight space analysis: distributions, rank, and spectral properties."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def weight_distributions(
    model: nn.Module,
    include_bias: bool = False,
) -> dict[str, np.ndarray]:
    """Extract weight distributions per layer.

    Args:
        model: PyTorch model.
        include_bias: Whether to include bias terms.

    Returns:
        Dict mapping parameter name to flattened weight values.
    """
    distributions = {}
    for name, param in model.named_parameters():
        if not include_bias and "bias" in name:
            continue
        if param.ndim >= 2:  # Only weight matrices
            distributions[name] = param.detach().float().cpu().numpy().flatten()
    return distributions


def weight_effective_rank(
    model: nn.Module,
) -> dict[str, float]:
    """Compute effective rank of each weight matrix.

    Returns:
        Dict mapping parameter name to effective rank.
    """
    results = {}
    for name, param in model.named_parameters():
        if param.ndim < 2:
            continue
        W = param.detach().float().cpu()
        # Reshape to 2D
        W = W.reshape(W.shape[0], -1)
        _, s, _ = torch.linalg.svd(W, full_matrices=False)
        s = s.numpy()
        s = s / s.sum()
        s = s[s > 1e-10]
        entropy = -(s * np.log(s)).sum()
        results[name] = float(np.exp(entropy))
    return results


def singular_value_spectrum(
    model: nn.Module,
    layer_names: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Compute singular value spectrum of weight matrices.

    Args:
        model: PyTorch model.
        layer_names: Specific parameter names to analyze (None = all 2D+ params).

    Returns:
        Dict mapping parameter name to singular values array.
    """
    results = {}
    for name, param in model.named_parameters():
        if param.ndim < 2:
            continue
        if layer_names is not None and name not in layer_names:
            continue
        W = param.detach().float().cpu().reshape(param.shape[0], -1)
        _, s, _ = torch.linalg.svd(W, full_matrices=False)
        results[name] = s.numpy()
    return results


def power_law_fit(
    singular_values: np.ndarray,
    fit_range: tuple[float, float] = (0.1, 0.9),
) -> tuple[float, float]:
    """Fit power law to singular value spectrum (Martin & Mahoney style).

    log(s_i) = -alpha * log(i) + c

    Args:
        singular_values: Array of singular values (descending).
        fit_range: Fraction range of indices to use for fitting.

    Returns:
        (alpha, r_squared) - power law exponent and R² of fit.
    """
    s = singular_values[singular_values > 1e-10]
    n = len(s)
    if n < 5:
        return float("nan"), float("nan")

    start = max(int(n * fit_range[0]), 1)
    end = int(n * fit_range[1])
    if end <= start:
        return float("nan"), float("nan")

    indices = np.arange(start, end) + 1  # 1-indexed
    log_i = np.log(indices)
    log_s = np.log(s[start:end])

    # Linear regression
    A = np.vstack([log_i, np.ones(len(log_i))]).T
    result = np.linalg.lstsq(A, log_s, rcond=None)
    slope, intercept = result[0]

    # R²
    predicted = slope * log_i + intercept
    ss_res = ((log_s - predicted) ** 2).sum()
    ss_tot = ((log_s - log_s.mean()) ** 2).sum()
    r_squared = 1 - ss_res / (ss_tot + 1e-10)

    return float(-slope), float(r_squared)


def weight_spectral_analysis(
    model: nn.Module,
) -> dict[str, dict]:
    """Full spectral analysis of all weight matrices.

    Returns:
        Dict mapping param name to {effective_rank, alpha, r_squared, top_sv, sv_ratio}.
    """
    results = {}
    spectra = singular_value_spectrum(model)
    ranks = weight_effective_rank(model)

    for name, svs in spectra.items():
        alpha, r2 = power_law_fit(svs)
        results[name] = {
            "effective_rank": ranks.get(name, float("nan")),
            "alpha": alpha,
            "r_squared": r2,
            "top_singular_value": float(svs[0]) if len(svs) > 0 else float("nan"),
            "sv_ratio": float(svs[0] / svs[-1]) if len(svs) > 1 and svs[-1] > 1e-10 else float("nan"),
            "num_singular_values": len(svs),
        }
    return results
