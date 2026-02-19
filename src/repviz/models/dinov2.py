"""DINOv2 backbone wrapper.

Works for both DINOv2 and DINOv3 models loaded via torch.hub,
since DINOv3 shares the same ViT architecture interface.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .base import BackboneWrapper, ModelOutput


# Available DINOv2 models via torch.hub
DINOV2_MODELS = {
    "dinov2_vits14": ("facebookresearch/dinov2", "dinov2_vits14"),
    "dinov2_vitb14": ("facebookresearch/dinov2", "dinov2_vitb14"),
    "dinov2_vitl14": ("facebookresearch/dinov2", "dinov2_vitl14"),
    "dinov2_vitg14": ("facebookresearch/dinov2", "dinov2_vitg14"),
    "dinov2_vits14_reg": ("facebookresearch/dinov2", "dinov2_vits14_reg"),
    "dinov2_vitb14_reg": ("facebookresearch/dinov2", "dinov2_vitb14_reg"),
    "dinov2_vitl14_reg": ("facebookresearch/dinov2", "dinov2_vitl14_reg"),
    "dinov2_vitg14_reg": ("facebookresearch/dinov2", "dinov2_vitg14_reg"),
}


class DINOv2Wrapper(BackboneWrapper):
    """Wrapper for DINOv2 ViT models."""

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        device: str | torch.device = "cpu",
        model: nn.Module | None = None,
    ):
        if model is None:
            if model_name in DINOV2_MODELS:
                repo, entry = DINOV2_MODELS[model_name]
                model = torch.hub.load(repo, entry)
            else:
                raise ValueError(
                    f"Unknown model: {model_name}. "
                    f"Available: {list(DINOV2_MODELS.keys())}. "
                    f"Or pass model= directly."
                )
        self.model_name = model_name
        super().__init__(model, device)

    @property
    def embed_dim(self) -> int:
        return self.model.embed_dim

    @property
    def num_layers(self) -> int:
        return len(self.model.blocks)

    @property
    def patch_size(self) -> int:
        return self.model.patch_embed.patch_size[0]

    @property
    def layer_modules(self) -> list[nn.Module]:
        return list(self.model.blocks)

    def get_layer_name(self, layer_idx: int) -> str:
        return f"block_{layer_idx}"

    def _register_attention_hooks(self, layers: list[int]) -> None:
        """Hook into attention modules to capture attention weights."""
        for idx in layers:
            block = self.model.blocks[idx]
            attn_module = block.attn
            name = f"attn_{idx}"

            def make_attn_hook(n: str):
                def hook_fn(mod, input, output):
                    # DINOv2 attention: output is (attn_output, attn_weights)
                    # We need to modify the forward to return attention weights.
                    # Instead, we hook into qkv and compute attention ourselves.
                    pass
                return hook_fn

            # For attention maps, we use a different strategy:
            # override the forward with attn output
            # This is done in the extract_with_attention method
            pass

    @torch.no_grad()
    def extract_with_attention(self, images: torch.Tensor) -> ModelOutput:
        """Extract features AND attention maps using DINOv2's built-in method."""
        images = images.to(self.device)

        # Use get_intermediate_layers if available (DINOv2 API)
        if hasattr(self.model, "get_intermediate_layers"):
            # Get all intermediate features
            intermediate = self.model.get_intermediate_layers(
                images,
                n=self.num_layers,
                reshape=False,
                return_class_token=True,
            )
            # intermediate is list of (patch_tokens, cls_token)
            features = {}
            for i, (patches, cls) in enumerate(intermediate):
                features[f"block_{i}"] = torch.cat([cls.unsqueeze(1), patches], dim=1)

            # Final output
            last_patches, last_cls = intermediate[-1]
            self._intermediate_features = features

            return ModelOutput(
                cls_token=last_cls,
                patch_tokens=last_patches,
                intermediate_features=dict(features),
            )

        # Fallback: just run forward with hooks
        return self.extract(images)

    def _parse_output(self, raw_output: Any) -> ModelOutput:
        """DINOv2 forward returns (B, 1+N, D) tensor or just cls token depending on call."""
        if isinstance(raw_output, torch.Tensor):
            if raw_output.ndim == 2:
                # CLS token only
                return ModelOutput(
                    cls_token=raw_output,
                    intermediate_features=dict(self._intermediate_features),
                    attention_maps=dict(self._attention_maps),
                    raw_output=raw_output,
                )
            elif raw_output.ndim == 3:
                # Full sequence: [CLS] + patches
                return ModelOutput(
                    cls_token=raw_output[:, 0],
                    patch_tokens=raw_output[:, 1:],
                    intermediate_features=dict(self._intermediate_features),
                    attention_maps=dict(self._attention_maps),
                    raw_output=raw_output,
                )
        return ModelOutput(raw_output=raw_output)

    @torch.no_grad()
    def extract_attention_maps(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract attention maps from all layers.

        Returns dict mapping layer name to attention tensor (B, H, N, N).
        """
        images = images.to(self.device)
        attn_maps = {}

        # Manually run forward, capturing attention at each block
        x = self.model.prepare_tokens_with_masks(images) if hasattr(self.model, "prepare_tokens_with_masks") else self.model.patch_embed(images)

        # Handle different DINOv2 versions
        if hasattr(self.model, "prepare_tokens_with_masks"):
            x = self.model.prepare_tokens_with_masks(images)
        else:
            x = self.model.patch_embed(images)
            # Add CLS token
            cls_tokens = self.model.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.model.pos_embed

        for i, block in enumerate(self.model.blocks):
            # Compute attention weights manually
            B, N, C = x.shape
            qkv = block.attn.qkv(block.norm1(x) if hasattr(block, "norm1") else x)
            qkv = qkv.reshape(B, N, 3, block.attn.num_heads, C // block.attn.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)

            scale = (C // block.attn.num_heads) ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            attn_maps[f"block_{i}"] = attn.detach().cpu()

            # Continue normal forward
            x = block(x)

        return attn_maps


def load_dinov2(
    model_name: str = "dinov2_vitb14",
    device: str | torch.device = "cpu",
) -> DINOv2Wrapper:
    """Convenience loader."""
    return DINOv2Wrapper(model_name=model_name, device=device)
