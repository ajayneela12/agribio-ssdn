"""DINOv3 ViT backbone for AgriBio-SSDN.

This module intentionally exposes encoder features only. Prediction heads,
training logic, and losses belong in later project phases.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from transformers import AutoImageProcessor, AutoModel


DEFAULT_DINOV3_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"


class DINOv3Backbone(nn.Module):
    """Pretrained DINOv3 ViT encoder with global and patch-level features.

    Parameters are frozen by default for the initial baseline. Set
    ``freeze=False`` when a later experiment needs to fine-tune the encoder.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_DINOV3_MODEL,
        *,
        freeze: bool = True,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # from_pretrained resolves the checkpoint and its matching processor.
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)

        if freeze:
            self.freeze()

        # Evaluation is the safe default. Do not use no_grad here: callers may
        # explicitly fine-tune this backbone in a later phase.
        self.model.eval()

    @property
    def hidden_size(self) -> int:
        """Embedding width of every CLS, register, and patch token."""
        return int(self.model.config.hidden_size)

    @property
    def num_register_tokens(self) -> int:
        """Number of register tokens placed after the CLS token, if any."""
        return int(getattr(self.model.config, "num_register_tokens", 0))

    @property
    def patch_size(self) -> int | tuple[int, int]:
        """Image patch size declared by the loaded checkpoint configuration."""
        patch_size = self.model.config.patch_size
        return tuple(patch_size) if isinstance(patch_size, list) else patch_size

    def freeze(self) -> None:
        """Disable gradients for all encoder parameters."""
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def unfreeze(self) -> None:
        """Enable gradients for all encoder parameters explicitly."""
        for parameter in self.model.parameters():
            parameter.requires_grad = True

    def configuration(self) -> dict[str, Any]:
        """Return useful configuration values from the loaded checkpoint."""
        return {
            "model_name": self.model_name,
            "model_class": self.model.__class__.__name__,
            "processor_class": self.processor.__class__.__name__,
            "device": str(self.device),
            "hidden_size": self.hidden_size,
            "patch_size": self.patch_size,
            "num_register_tokens": self.num_register_tokens,
            "frozen": not any(parameter.requires_grad for parameter in self.model.parameters()),
        }

    def forward(self, pixel_values: Tensor, **model_kwargs: Any) -> dict[str, Tensor]:
        """Extract features from already processor-prepared image tensors.

        ``pixel_values`` must have shape ``(batch, channels, height, width)``.
        Token order is [CLS] [optional register tokens] [image patch tokens].
        """
        outputs = self.model(pixel_values=pixel_values, **model_kwargs)
        last_hidden_state = outputs.last_hidden_state

        # Use the model's global/CLS representation when available. Fallback
        # to the explicit CLS token for compatible outputs without a pooler.
        global_features = outputs.pooler_output
        if global_features is None:
            global_features = last_hidden_state[:, 0]

        # Exclude both the CLS token and any optional register tokens; only
        # image-patch tokens are returned in patch_features.
        patch_start = 1 + self.num_register_tokens
        if patch_start > last_hidden_state.shape[1]:
            raise RuntimeError(
                "DINOv3 output has fewer tokens than its configured CLS and register tokens."
            )
        patch_features = last_hidden_state[:, patch_start:]

        return {
            "global_features": global_features,
            "patch_features": patch_features,
            "last_hidden_state": last_hidden_state,
        }
