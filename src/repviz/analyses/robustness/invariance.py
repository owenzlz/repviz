"""Augmentation invariance and resolution sensitivity analysis."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ...models.base import BackboneWrapper


def cosine_invariance(
    features_original: torch.Tensor,
    features_augmented: torch.Tensor,
) -> float:
    """Mean pairwise cosine similarity between original and augmented features.

    Args:
        features_original: (N, D) features from original images.
        features_augmented: (N, D) features from augmented versions.

    Returns:
        Mean cosine similarity in [-1, 1].
    """
    a = F.normalize(features_original.float(), dim=-1)
    b = F.normalize(features_augmented.float(), dim=-1)
    return float((a * b).sum(dim=-1).mean())


def cka_invariance(
    features_original: torch.Tensor,
    features_augmented: torch.Tensor,
) -> float:
    """Linear CKA between original and augmented features."""
    from ..layerwise.cka import linear_cka
    return linear_cka(features_original, features_augmented)


def augmentation_invariance_suite(
    backbone: BackboneWrapper,
    images: torch.Tensor,
    image_size: int = 518,
) -> dict[str, float]:
    """Measure feature invariance under various augmentations.

    Args:
        backbone: Model wrapper.
        images: (B, 3, H, W) input images (already normalized).
        image_size: Image resolution.

    Returns:
        Dict mapping augmentation name to cosine similarity score.
    """
    import torchvision.transforms.functional as TF

    results = {}

    # Get original features
    out_orig = backbone.extract(images)
    feat_orig = out_orig.cls_token if out_orig.cls_token is not None else out_orig.patch_tokens.mean(dim=1)

    augmentations = {
        "hflip": lambda x: TF.hflip(x),
        "rotate_15": lambda x: TF.rotate(x, 15),
        "rotate_90": lambda x: TF.rotate(x, 90),
        "center_crop_0.8": lambda x: _center_crop_ratio(x, 0.8),
        "gaussian_blur": lambda x: TF.gaussian_blur(x, kernel_size=9),
        "color_jitter": lambda x: TF.adjust_brightness(TF.adjust_contrast(x, 1.3), 1.2),
        "grayscale": lambda x: TF.rgb_to_grayscale(x, num_output_channels=3),
    }

    for name, aug_fn in augmentations.items():
        try:
            aug_images = aug_fn(images)
            # Resize back if needed
            if aug_images.shape[-2:] != images.shape[-2:]:
                aug_images = F.interpolate(aug_images, size=images.shape[-2:], mode="bilinear", align_corners=False)
            out_aug = backbone.extract(aug_images)
            feat_aug = out_aug.cls_token if out_aug.cls_token is not None else out_aug.patch_tokens.mean(dim=1)
            results[name] = cosine_invariance(feat_orig, feat_aug)
        except Exception as e:
            results[name] = float("nan")

    return results


def _center_crop_ratio(x: torch.Tensor, ratio: float) -> torch.Tensor:
    """Center crop to a fraction of original size, then resize back."""
    _, _, H, W = x.shape
    h, w = int(H * ratio), int(W * ratio)
    top, left = (H - h) // 2, (W - w) // 2
    cropped = x[:, :, top:top + h, left:left + w]
    return F.interpolate(cropped, size=(H, W), mode="bilinear", align_corners=False)


def resolution_sensitivity(
    backbone: BackboneWrapper,
    images_pil: list[Image.Image],
    resolutions: list[int] | None = None,
    reference_resolution: int = 518,
) -> dict[int, float]:
    """Measure feature stability across resolutions.

    Args:
        backbone: Model wrapper.
        images_pil: List of PIL images.
        resolutions: List of resolutions to test.
        reference_resolution: Resolution to compare against.

    Returns:
        Dict mapping resolution to cosine similarity with reference.
    """
    import torchvision.transforms as T

    if resolutions is None:
        resolutions = [224, 336, 448, 518, 672]

    def _make_transform(size):
        return T.Compose([
            T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    ref_transform = _make_transform(reference_resolution)
    ref_batch = torch.stack([ref_transform(img) for img in images_pil])
    out_ref = backbone.extract(ref_batch)
    feat_ref = out_ref.cls_token if out_ref.cls_token is not None else out_ref.patch_tokens.mean(dim=1)

    results = {}
    for res in resolutions:
        try:
            transform = _make_transform(res)
            batch = torch.stack([transform(img) for img in images_pil])
            out = backbone.extract(batch)
            feat = out.cls_token if out.cls_token is not None else out.patch_tokens.mean(dim=1)
            results[res] = cosine_invariance(feat_ref, feat)
        except Exception:
            results[res] = float("nan")

    return results
