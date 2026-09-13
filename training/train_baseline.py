"""Train the RGB-only DINOv3 multi-target baseline on a fixed grouped fold."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import TARGET_NAMES, create_dataloaders, create_fold_datasets
from src.model_baseline import DEFAULT_DINOV3_MODEL, DINOv3Baseline


def parse_args() -> argparse.Namespace:
    """Parse parameters without starting training while imported."""
    parser = argparse.ArgumentParser(description="Train the DINOv3 biomass baseline on one fixed fold.")
    parser.add_argument(
        "--fold",
        default="0",
        help="One fixed validation fold (0-4), or 'all' for sequential 5-fold training.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "checkpoints")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
        help="Directory for completed-fold metrics only.",
    )
    parser.add_argument("--train-wide-path", type=Path, default=PROJECT_ROOT / "data" / "processed" / "train_wide.csv")
    parser.add_argument("--split-path", type=Path, default=PROJECT_ROOT / "data" / "splits" / "group_kfold_5.csv")
    parser.add_argument("--image-root", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set reproducibility controls for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _weighted_mean(total: float, count: int) -> float:
    if count == 0:
        raise ValueError("Cannot calculate a mean from zero samples.")
    return total / count


def resolve_folds(fold_argument: str) -> list[int]:
    """Resolve the CLI fold selection without generating or changing splits."""
    if fold_argument == "all":
        return [0, 1, 2, 3, 4]
    try:
        fold = int(fold_argument)
    except ValueError as error:
        raise ValueError("--fold must be one of 0, 1, 2, 3, 4, or 'all'.") from error
    if fold not in {0, 1, 2, 3, 4}:
        raise ValueError("--fold must be one of 0, 1, 2, 3, 4, or 'all'.")
    return [fold]


def train_one_epoch(model: DINOv3Baseline, loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device) -> float:
    """Optimize the regression head and return mean loss across batches."""
    model.train()
    # Module.train() recurses into children. The frozen image encoder must stay
    # deterministic while the regression head keeps dropout/training behavior.
    model.backbone.eval()
    model.regression_head.train()
    total_training_loss, number_of_training_batches = 0.0, 0
    for batch in loader:
        pixel_values: Tensor = batch["pixel_values"].to(device)
        targets: Tensor = batch["targets"].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(pixel_values), targets)
        loss.backward()
        optimizer.step()
        total_training_loss += loss.item()
        number_of_training_batches += 1
    return _weighted_mean(total_training_loss, number_of_training_batches)


def calculate_regression_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    """Compute per-target metrics and pooled-element aggregate metrics.

    Aggregate MAE/RMSE pool all five target columns into one flattened vector.
    Aggregate R² is standard R² on that same pooled vector; it is not an
    average of the individual target R² values.
    """
    if predictions.shape != targets.shape or predictions.ndim != 2 or predictions.shape[1] != len(TARGET_NAMES):
        raise ValueError("Predictions and targets must both have shape (n_samples, 5).")

    def metric_set(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        errors = y_pred - y_true
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(np.square(errors))))
        total_sum_of_squares = float(np.sum(np.square(y_true - np.mean(y_true))))
        r2 = float("nan") if total_sum_of_squares == 0.0 else float(1.0 - np.sum(np.square(errors)) / total_sum_of_squares)
        return {"mae": mae, "rmse": rmse, "r2": r2}

    return {
        "per_target": {name: metric_set(targets[:, index], predictions[:, index]) for index, name in enumerate(TARGET_NAMES)},
        "aggregate_pooled": metric_set(targets.reshape(-1), predictions.reshape(-1)),
    }


def validate_one_epoch(model: DINOv3Baseline, loader: torch.utils.data.DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, dict[str, Any]]:
    """Validate without gradients and collect all fold predictions and targets."""
    model.eval()
    total_loss, total_samples = 0.0, 0
    prediction_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            pixel_values: Tensor = batch["pixel_values"].to(device)
            targets: Tensor = batch["targets"].to(device)
            predictions = model(pixel_values)
            loss = criterion(predictions, targets)
            total_loss += loss.item() * targets.shape[0]
            total_samples += targets.shape[0]
            prediction_batches.append(predictions.cpu().numpy())
            target_batches.append(targets.cpu().numpy())
    predictions_array = np.concatenate(prediction_batches, axis=0)
    targets_array = np.concatenate(target_batches, axis=0)
    return _weighted_mean(total_loss, total_samples), calculate_regression_metrics(predictions_array, targets_array)


def metrics_row(fold: int, best_epoch: int, train_loss: float, validation_loss: float, metrics: dict[str, Any]) -> dict[str, float | int]:
    """Flatten a completed fold's existing metrics for durable CSV reporting."""
    row: dict[str, float | int] = {
        "fold": fold,
        "best_epoch": best_epoch,
        "train_loss": train_loss,
        "validation_loss": validation_loss,
    }
    for target_name in TARGET_NAMES:
        target_metrics = metrics["per_target"][target_name]
        row[f"{target_name}_MAE"] = target_metrics["mae"]
        row[f"{target_name}_RMSE"] = target_metrics["rmse"]
        row[f"{target_name}_R2"] = target_metrics["r2"]
    pooled = metrics["aggregate_pooled"]
    row.update({"pooled_MAE": pooled["mae"], "pooled_RMSE": pooled["rmse"], "pooled_R2": pooled["r2"]})
    return row


def save_fold_metrics(row: dict[str, float | int], results_dir: Path) -> Path:
    """Persist one row only after its fold has completed and checkpointed."""
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "baseline_fold_metrics.csv"
    existing_rows: list[dict[str, str]] = []
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as handle:
            existing_rows = list(csv.DictReader(handle))
    fieldnames = list(row.keys())
    retained_rows = [existing for existing in existing_rows if int(existing["fold"]) != int(row["fold"])]
    retained_rows.append({key: str(value) for key, value in row.items()})
    retained_rows.sort(key=lambda item: int(item["fold"]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(retained_rows)
    return path


def train_fold(args: argparse.Namespace, fold: int) -> dict[str, float | int]:
    """Train one independent fixed fold and return its best completed metrics."""
    # Reapply the same seed before each fresh model construction so every fold
    # starts independently from the same reproducible initialization.
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DINOv3Baseline(model_name=DEFAULT_DINOV3_MODEL, freeze_backbone=True).to(device)
    train_dataset, validation_dataset = create_fold_datasets(
        selected_fold=fold, processor=model.backbone.processor, image_root=args.image_root,
        train_wide_path=args.train_wide_path, split_path=args.split_path, target_names=TARGET_NAMES,
    )
    train_loader, validation_loader = create_dataloaders(
        train_dataset, validation_dataset, training_batch_size=args.batch_size,
        validation_batch_size=args.batch_size, num_workers=args.num_workers,
    )
    backbone_trainable = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    if backbone_trainable != 0:
        raise RuntimeError("Baseline backbone must remain frozen.")
    trainable_head_parameters = [p for p in model.regression_head.parameters() if p.requires_grad]
    if not trainable_head_parameters:
        raise RuntimeError("Regression head has no trainable parameters.")
    criterion = nn.SmoothL1Loss()
    # This optimizer is deliberately created per fold; no optimizer state is shared.
    optimizer = torch.optim.AdamW(trainable_head_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    print(f"Device: {device} | Fold: {fold} | Train: {len(train_dataset)} | Validation: {len(validation_dataset)} | Frozen backbone trainable parameters: {backbone_trainable}")
    best_validation_loss = float("inf")
    best_train_loss: float | None = None
    best_epoch: int | None = None
    best_metrics: dict[str, Any] | None = None
    checkpoint_path = args.checkpoint_dir / f"dinov3_baseline_fold{fold}_best.pt"
    for epoch in range(1, args.epochs + 1):
        training_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        validation_loss, validation_metrics = validate_one_epoch(model, validation_loader, criterion, device)
        aggregate = validation_metrics["aggregate_pooled"]
        print(
            f"Fold {fold} | Epoch {epoch}/{args.epochs}\n"
            f"Train Loss: {training_loss:.6f}\n"
            f"Validation Loss: {validation_loss:.6f}\n"
            f"Validation MAE/RMSE/R² (pooled): {aggregate['mae']:.6f} / "
            f"{aggregate['rmse']:.6f} / {aggregate['r2']:.6f}"
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_train_loss, best_epoch, best_metrics = training_loss, epoch, validation_metrics
            args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "fold": fold, "epoch": epoch, "target_names": TARGET_NAMES,
                "model_name": DEFAULT_DINOV3_MODEL,
                "hyperparameters": {"epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "seed": args.seed, "num_workers": args.num_workers, "freeze_backbone": True},
                "train_loss": training_loss, "validation_loss": validation_loss,
                "validation_metrics": validation_metrics,
            }, checkpoint_path)
            print(f"Saved best checkpoint: {checkpoint_path}")
    if best_train_loss is None or best_epoch is None or best_metrics is None:
        raise RuntimeError(f"Fold {fold} did not complete an epoch and cannot be recorded.")
    return metrics_row(fold, best_epoch, best_train_loss, best_validation_loss, best_metrics)


def train(args: argparse.Namespace) -> None:
    """Train one fold or sequential independent folds from the fixed split file."""
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0 or args.weight_decay < 0 or args.num_workers < 0:
        raise ValueError("Invalid epochs, batch size, learning rate, weight decay, or worker count.")
    for fold in resolve_folds(args.fold):
        completed_row = train_fold(args, fold)
        result_path = save_fold_metrics(completed_row, args.results_dir)
        print(f"Saved completed fold metrics: {result_path}")


def main() -> None:
    """CLI entry point; training starts only when executed as a script."""
    train(parse_args())


if __name__ == "__main__":
    main()
