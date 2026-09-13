"""Simple DINOv3 baseline for multi-component pasture biomass regression."""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from src.backbone import DEFAULT_DINOV3_MODEL, DINOv3Backbone


class DINOv3Baseline(nn.Module):
    """DINOv3 global-feature regressor for the five biomass components.

    This intentionally uses only the backbone's global feature. Patch features,
    metadata, and all spatial-structural components are deferred to later work.
    """

    TARGET_NAMES = [
        "Dry_Clover_g",
        "Dry_Dead_g",
        "Dry_Green_g",
        "Dry_Total_g",
        "GDM_g",
    ]

    def __init__(
        self,
        model_name: str = DEFAULT_DINOV3_MODEL,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()

        # DINOv3Backbone manages checkpoint loading, processor access, device
        # selection, and the explicit frozen/unfrozen encoder state.
        self.backbone = DINOv3Backbone(model_name=model_name, freeze=freeze_backbone)

        # The input width is derived from the loaded checkpoint; it is not
        # hard-coded so compatible DINOv3 variants remain usable.
        self.regression_head = nn.Sequential(
            nn.Linear(self.backbone.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, len(self.TARGET_NAMES)),
        )

    def forward(self, pixel_values: Tensor, **backbone_kwargs: Any) -> Tensor:
        """Return five biomass predictions with shape ``(batch_size, 5)``."""
        # Only the global CLS/pooler feature is used in this initial baseline.
        global_features = self.backbone(pixel_values, **backbone_kwargs)["global_features"]
        return self.regression_head(global_features)

    def predict_dict(self, pixel_values: Tensor, **backbone_kwargs: Any) -> dict[str, Tensor]:
        """Return one batch-sized prediction tensor per target name.

        Each dictionary value has shape ``(batch_size,)``. This method keeps
        gradients intact; callers may choose inference/no_grad behavior.
        """
        predictions = self(pixel_values, **backbone_kwargs)
        return {
            target_name: predictions[:, index]
            for index, target_name in enumerate(self.TARGET_NAMES)
        }

    def configuration(self) -> dict[str, Any]:
        """Report the baseline architecture and checkpoint-derived settings."""
        backbone_configuration = self.backbone.configuration()
        return {
            "model_name": self.backbone.model_name,
            "hidden_size": self.backbone.hidden_size,
            "num_outputs": len(self.TARGET_NAMES),
            "target_names": self.TARGET_NAMES.copy(),
            "backbone_frozen": backbone_configuration["frozen"],
            "regression_head": "Linear(hidden_size, 128) -> ReLU -> Dropout(0.2) -> Linear(128, 5)",
        }
