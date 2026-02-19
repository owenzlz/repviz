"""Abstract backbone interface for model-agnostic analysis."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


@dataclass
class ModelOutput:
    """Standardized output from any backbone."""

    cls_token: torch.Tensor | None = None  # (B, D)
    patch_tokens: torch.Tensor | None = None  # (B, N, D)
    intermediate_features: dict[str, torch.Tensor] = field(default_factory=dict)
    attention_maps: dict[str, torch.Tensor] = field(default_factory=dict)
    raw_output: Any = None


class BackboneWrapper(ABC):
    """Abstract wrapper that standardizes any vision backbone."""

    def __init__(self, model: nn.Module, device: str | torch.device = "cpu"):
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._intermediate_features: dict[str, torch.Tensor] = {}
        self._attention_maps: dict[str, torch.Tensor] = {}

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        """Feature dimension."""
        ...

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """Number of transformer/backbone layers."""
        ...

    @property
    @abstractmethod
    def patch_size(self) -> int:
        """Spatial patch size."""
        ...

    @property
    @abstractmethod
    def layer_modules(self) -> list[nn.Module]:
        """List of layer modules (for hooking)."""
        ...

    @abstractmethod
    def get_layer_name(self, layer_idx: int) -> str:
        """Human-readable name for a layer."""
        ...

    def register_hooks(
        self,
        layers: list[int] | None = None,
        capture_attention: bool = False,
    ) -> None:
        """Register forward hooks on specified layers."""
        self.remove_hooks()
        self._intermediate_features.clear()
        self._attention_maps.clear()

        modules = self.layer_modules
        if layers is None:
            layers = list(range(len(modules)))

        for idx in layers:
            name = self.get_layer_name(idx)
            module = modules[idx]

            def make_hook(n: str):
                def hook_fn(mod, input, output):
                    if isinstance(output, tuple):
                        self._intermediate_features[n] = output[0].detach()
                    else:
                        self._intermediate_features[n] = output.detach()
                return hook_fn

            h = module.register_forward_hook(make_hook(name))
            self._hooks.append(h)

        if capture_attention:
            self._register_attention_hooks(layers)

    def _register_attention_hooks(self, layers: list[int]) -> None:
        """Override in subclass to capture attention maps."""
        pass

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> ModelOutput:
        """Run forward pass, return standardized output."""
        images = images.to(self.device)
        raw = self.model(images)
        return self._parse_output(raw)

    @abstractmethod
    def _parse_output(self, raw_output: Any) -> ModelOutput:
        """Parse raw model output into ModelOutput."""
        ...

    def __del__(self):
        if hasattr(self, "_hooks"):
            self.remove_hooks()
