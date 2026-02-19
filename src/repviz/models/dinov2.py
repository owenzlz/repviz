"""DINOv2 backbone wrapper using HuggingFace Transformers.

Works for DINOv2 models. Compatible with Python 3.9+.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .base import BackboneWrapper, ModelOutput


# HuggingFace model names
HF_MODELS = {
    "dinov2_vits14": "facebook/dinov2-small",
    "dinov2_vitb14": "facebook/dinov2-base",
    "dinov2_vitl14": "facebook/dinov2-large",
    "dinov2_vitg14": "facebook/dinov2-giant",
    "dinov2_vits14_reg": "facebook/dinov2-small-imagenet1k-1-layer",
    "dinov2_vitb14_reg": "facebook/dinov2-base-imagenet1k-1-layer",
}


class DINOv2Wrapper(BackboneWrapper):
    """Wrapper for DINOv2 ViT models via HuggingFace."""

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        device: str = "cpu",
        model: Optional[nn.Module] = None,
    ):
        self.model_name = model_name

        if model is None:
            from transformers import AutoModel
            hf_name = HF_MODELS.get(model_name, model_name)
            model = AutoModel.from_pretrained(hf_name, attn_implementation="eager")

        self._embed_dim = model.config.hidden_size
        self._num_layers = model.config.num_hidden_layers
        self._patch_size = model.config.patch_size
        self._num_heads = model.config.num_attention_heads

        super().__init__(model, device)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def patch_size(self) -> int:
        return self._patch_size

    @property
    def layer_modules(self) -> list:
        return list(self.model.encoder.layer)

    def get_layer_name(self, layer_idx: int) -> str:
        return f"block_{layer_idx}"

    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> ModelOutput:
        """Extract features from images.

        Args:
            images: (B, 3, H, W) tensor, already preprocessed.
        """
        images = images.to(self.device)

        # Use output_hidden_states to get all intermediate features
        outputs = self.model(
            pixel_values=images,
            output_hidden_states=True,
            output_attentions=False,
        )

        last_hidden = outputs.last_hidden_state  # (B, 1+N, D)
        cls_token = last_hidden[:, 0]  # (B, D)
        patch_tokens = last_hidden[:, 1:]  # (B, N, D)

        # Collect intermediate features
        intermediate = {}
        if outputs.hidden_states is not None:
            for i, hs in enumerate(outputs.hidden_states):
                if i == 0:
                    continue  # skip embedding layer output
                intermediate[f"block_{i-1}"] = hs

        return ModelOutput(
            cls_token=cls_token,
            patch_tokens=patch_tokens,
            intermediate_features=intermediate,
        )

    @torch.no_grad()
    def extract_with_attention(self, images: torch.Tensor) -> ModelOutput:
        """Extract features AND attention maps."""
        images = images.to(self.device)

        outputs = self.model(
            pixel_values=images,
            output_hidden_states=True,
            output_attentions=True,
        )

        last_hidden = outputs.last_hidden_state
        cls_token = last_hidden[:, 0]
        patch_tokens = last_hidden[:, 1:]

        intermediate = {}
        if outputs.hidden_states is not None:
            for i, hs in enumerate(outputs.hidden_states):
                if i == 0:
                    continue
                intermediate[f"block_{i-1}"] = hs

        attn_maps = {}
        if outputs.attentions is not None:
            for i, attn in enumerate(outputs.attentions):
                attn_maps[f"block_{i}"] = attn.detach().cpu()

        return ModelOutput(
            cls_token=cls_token,
            patch_tokens=patch_tokens,
            intermediate_features=intermediate,
            attention_maps=attn_maps,
        )

    @torch.no_grad()
    def extract_attention_maps(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract attention maps from all layers."""
        output = self.extract_with_attention(images)
        return output.attention_maps

    def _parse_output(self, raw_output: Any) -> ModelOutput:
        # Not used — we override extract() directly
        return ModelOutput(raw_output=raw_output)


def load_dinov2(
    model_name: str = "dinov2_vits14",
    device: str = "cpu",
) -> DINOv2Wrapper:
    """Convenience loader."""
    return DINOv2Wrapper(model_name=model_name, device=device)
