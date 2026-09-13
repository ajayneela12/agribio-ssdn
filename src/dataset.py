"""RGB-only DINOv3 datasets and fixed-fold data-loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


TARGET_NAMES = [
    "Dry_Clover_g",
    "Dry_Dead_g",
    "Dry_Green_g",
    "Dry_Total_g",
    "GDM_g",
]


class BiomassDataset(Dataset[dict[str, Any]]):
    """One-row-per-image RGB dataset for the five biomass regression targets.

    The processor is passed in by the caller so this module never loads DINOv3
    weights. Its resize, rescale, and normalization are used directly; no
    augmentation or manual DINOv3 normalization is applied here.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        processor: Any,
        image_root: str | Path,
        target_names: Sequence[str] = TARGET_NAMES,
        training: bool = False,
    ) -> None:
        self.target_names = list(target_names)
        self.processor = processor
        self.image_root = Path(image_root)
        self.training = training

        required_columns = ["image_path", *self.target_names]
        missing_columns = [column for column in required_columns if column not in dataframe.columns]
        if missing_columns:
            raise ValueError(f"Dataset dataframe is missing required columns: {missing_columns}")
        if dataframe[required_columns].isna().any().any():
            raise ValueError("Dataset dataframe contains missing image paths or target values.")

        # Keep only the values required for the RGB baseline. Metadata is not a
        # model input in this phase.
        self.dataframe = dataframe[required_columns].reset_index(drop=True).copy()

    def __len__(self) -> int:
        return len(self.dataframe)

    def _resolve_image_path(self, image_path: str) -> Path:
        """Resolve a CSV-relative path safely below ``image_root``."""
        root = self.image_root.resolve()
        resolved = (root / image_path).resolve()
        if root not in resolved.parents and resolved != root:
            raise ValueError(f"image_path resolves outside image_root: {image_path}")
        return resolved

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.dataframe.iloc[index]
        image_path = str(row["image_path"])
        resolved_path = self._resolve_image_path(image_path)
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Image file does not exist: {resolved_path}")

        # Explicit RGB conversion prevents grayscale/RGBA source modes from
        # changing the backbone's expected three-channel input.
        with Image.open(resolved_path) as image:
            rgb_image = image.convert("RGB")
            processed = self.processor(images=rgb_image, return_tensors="pt")

        pixel_values: Tensor = processed["pixel_values"].squeeze(0)
        targets = torch.tensor(
            [float(row[target_name]) for target_name in self.target_names],
            dtype=torch.float32,
        )

        return {
            "pixel_values": pixel_values,
            "targets": targets,
            "image_path": image_path,
        }


def create_fold_dataframes(
    selected_fold: int,
    train_wide_path: str | Path = "data/processed/train_wide.csv",
    split_path: str | Path = "data/splits/group_kfold_5.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return fixed-fold train/validation frames with strict leakage checks."""
    train_wide = pd.read_csv(train_wide_path)
    splits = pd.read_csv(split_path)
    required_split_columns = {"image_path", "fold", "group_id"}
    missing_split_columns = required_split_columns.difference(splits.columns)
    if missing_split_columns:
        raise ValueError(f"Split file is missing required columns: {sorted(missing_split_columns)}")
    if train_wide["image_path"].duplicated().any() or splits["image_path"].duplicated().any():
        raise ValueError("train_wide and split image_path values must each be unique.")

    merged = train_wide.merge(
        splits[["image_path", "fold", "group_id"]],
        on="image_path",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(train_wide) or len(merged) != len(splits):
        raise ValueError("train_wide and split file do not contain the same image_path set.")
    if selected_fold not in set(merged["fold"].astype(int)):
        raise ValueError(f"Selected fold {selected_fold} is not present in the split file.")

    validation_frame = merged[merged["fold"].astype(int) == selected_fold].copy()
    training_frame = merged[merged["fold"].astype(int) != selected_fold].copy()
    train_images = set(training_frame["image_path"])
    validation_images = set(validation_frame["image_path"])
    train_groups = set(training_frame["group_id"])
    validation_groups = set(validation_frame["group_id"])
    if train_images.intersection(validation_images):
        raise ValueError("Image leakage detected between training and validation datasets.")
    if train_groups.intersection(validation_groups):
        raise ValueError("group_id leakage detected between training and validation datasets.")

    return training_frame, validation_frame


def create_fold_datasets(
    selected_fold: int,
    processor: Any,
    image_root: str | Path,
    train_wide_path: str | Path = "data/processed/train_wide.csv",
    split_path: str | Path = "data/splits/group_kfold_5.csv",
    target_names: Sequence[str] = TARGET_NAMES,
) -> tuple[BiomassDataset, BiomassDataset]:
    """Build datasets from the existing fixed fold assignment only."""
    training_frame, validation_frame = create_fold_dataframes(
        selected_fold=selected_fold,
        train_wide_path=train_wide_path,
        split_path=split_path,
    )
    root = Path(image_root)
    missing_paths = [
        image_path
        for image_path in pd.concat([training_frame["image_path"], validation_frame["image_path"]])
        if not (root / image_path).is_file()
    ]
    if missing_paths:
        raise FileNotFoundError(
            f"{len(missing_paths)} fold image paths do not exist under {root}: "
            f"{missing_paths[:5]}"
        )

    return (
        BiomassDataset(training_frame, processor, image_root, target_names, training=True),
        BiomassDataset(validation_frame, processor, image_root, target_names, training=False),
    )


def create_dataloaders(
    training_dataset: BiomassDataset,
    validation_dataset: BiomassDataset,
    training_batch_size: int = 4,
    validation_batch_size: int = 4,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Create CPU-friendly loaders with train shuffle and fixed validation order."""
    return (
        DataLoader(training_dataset, batch_size=training_batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=False, num_workers=num_workers),
    )


def validate_dataset_sample(dataset: BiomassDataset, index: int = 0) -> dict[str, Any]:
    """Load and validate one sample without iterating over a DataLoader."""
    if len(dataset) == 0:
        raise ValueError("Dataset is empty.")
    sample = dataset[index]
    pixel_values = sample["pixel_values"]
    targets = sample["targets"]
    if pixel_values.ndim != 3 or pixel_values.shape[0] != 3:
        raise ValueError(f"Expected RGB pixel_values with shape (3, H, W); got {tuple(pixel_values.shape)}")
    if targets.shape != (len(dataset.target_names),):
        raise ValueError(f"Expected target shape ({len(dataset.target_names)},); got {tuple(targets.shape)}")
    if targets.dtype != torch.float32:
        raise TypeError(f"Expected float32 targets; got {targets.dtype}")
    if not isinstance(sample["image_path"], str) or not sample["image_path"]:
        raise ValueError("Sample image_path is missing or invalid.")
    if not torch.isfinite(pixel_values).all() or not torch.isfinite(targets).all():
        raise ValueError("Sample contains non-finite pixel or target values.")
    expected_targets = torch.tensor(
        [float(dataset.dataframe.iloc[index][target_name]) for target_name in dataset.target_names],
        dtype=torch.float32,
    )
    if not torch.equal(targets, expected_targets):
        raise ValueError("Sample targets do not exactly match the source dataframe row.")
    return {
        "dataset_length": len(dataset),
        "pixel_values_shape": tuple(pixel_values.shape),
        "pixel_values_dtype": pixel_values.dtype,
        "targets_shape": tuple(targets.shape),
        "targets_dtype": targets.dtype,
        "image_path": sample["image_path"],
        "targets_match_dataframe": True,
    }
