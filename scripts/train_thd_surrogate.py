"""Train and evaluate an isolated event-based THD surrogate experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from optimization.thd_surrogate import (
    EventTHDSurrogate,
    THD_SURROGATE_FEATURE_NAMES,
    THD_SURROGATE_TARGET_NAMES,
    encode_thd_surrogate_features,
    spearman_rank_correlation,
)


def _read_dataset(path: Path) -> np.ndarray:
    expected_header = (
        "modulation_index",
        "phase_angle_degrees",
        "cell_2_phase_degrees",
        "cell_3_phase_degrees",
        "event_total_thd_percent",
        "fundamental_phasor_error_percent",
    )
    with path.open("r", encoding="utf-8", newline="") as dataset_file:
        reader = csv.reader(dataset_file)
        header = next(reader, None)
        if tuple(header or ()) != expected_header:
            raise ValueError("Dataset columns do not match the event-THD schema.")
        rows = [
            [float(value) for value in row]
            for row in reader
            if row
        ]
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(expected_header):
        raise ValueError("Surrogate dataset must contain six numeric columns.")
    if values.shape[0] < 100:
        raise ValueError("At least 100 rows are required for a useful held-out split.")
    if not np.isfinite(values).all():
        raise ValueError("Surrogate dataset contains non-finite values.")
    if np.any(values[:, 0] < 0.55) or np.any(values[:, 0] > 0.95):
        raise ValueError("Dataset modulation indices must be in [0.55, 0.95].")
    if np.any(values[:, 4:] < 0.0):
        raise ValueError("Surrogate labels must be non-negative.")
    return values


def _metrics(expected: np.ndarray, predicted: np.ndarray) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for index, name in enumerate(THD_SURROGATE_TARGET_NAMES):
        truth = expected[:, index]
        estimate = predicted[:, index]
        residual = estimate - truth
        total_variance = float(np.sum(np.square(truth - truth.mean())))
        summary[name] = {
            "mae": float(np.mean(np.abs(residual))),
            "rmse": float(np.sqrt(np.mean(np.square(residual)))),
            "maximum_absolute_error": float(np.max(np.abs(residual))),
            "r_squared": (
                1.0 - float(np.sum(np.square(residual))) / total_variance
                if total_variance > 0.0
                else 0.0
            ),
            "spearman": spearman_rank_correlation(truth, estimate),
        }
    return summary


def train_thd_surrogate(
    *,
    dataset_path: Path,
    experiment_dir: Path,
    seed: int = 20261125,
    epochs: int = 160,
    batch_size: int = 256,
    patience: int = 18,
) -> dict[str, object]:
    """Train with a seeded random split and persist only to a new experiment."""
    if not 0 <= seed <= 2**31 - 1:
        raise ValueError("seed must be in [0, 2**31 - 1].")
    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("epochs, batch_size, and patience must be positive.")
    if experiment_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing THD surrogate experiment: {experiment_dir}"
        )
    values = _read_dataset(dataset_path)
    modulation_band = np.minimum(
        ((values[:, 0] - 0.55) / 0.40 * 40).astype(np.int64),
        39,
    )
    test_mask = modulation_band % 10 == 0
    validation_mask = modulation_band % 10 == 5
    train_mask = ~(test_mask | validation_mask)
    train_values = values[train_mask]
    validation_values = values[validation_mask]
    test_values = values[test_mask]
    if min(len(train_values), len(validation_values), len(test_values)) < 1:
        raise ValueError("Dataset is too small to create train/validation/test splits.")

    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

    train_features = encode_thd_surrogate_features(train_values[:, :4])
    validation_features = encode_thd_surrogate_features(validation_values[:, :4])
    test_features = encode_thd_surrogate_features(test_values[:, :4])
    train_targets = train_values[:, 4:5].astype(np.float32)
    validation_targets = validation_values[:, 4:5].astype(np.float32)
    test_targets = test_values[:, 4:5].astype(np.float32)
    target_mean = train_targets.mean(axis=0)
    target_scale = train_targets.std(axis=0)
    target_scale = np.where(target_scale > 1e-8, target_scale, 1.0)

    train_dataset = TensorDataset(
        torch.from_numpy(train_features),
        torch.from_numpy((train_targets - target_mean) / target_scale),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    validation_x = torch.from_numpy(validation_features)
    validation_y = torch.from_numpy(
        (validation_targets - target_mean) / target_scale
    )
    model = EventTHDSurrogate()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch_features, batch_targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_features)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.detach()))

        model.eval()
        with torch.inference_mode():
            validation_loss = float(criterion(model(validation_x), validation_y))
        history.append(
            {
                "epoch": float(epoch),
                "train_mse": float(np.mean(train_losses)),
                "validation_mse": validation_loss,
            }
        )
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: tensor.detach().clone()
                for name, tensor in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("THD surrogate training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        test_standardized = model(torch.from_numpy(test_features)).numpy()
    test_predictions = test_standardized * target_scale + target_mean
    test_metrics = _metrics(test_targets, test_predictions)
    if not np.isfinite(test_predictions).all():
        raise RuntimeError("THD surrogate produced non-finite held-out predictions.")

    dataset_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    experiment_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = experiment_dir / "thd_surrogate.pt"
    checkpoint = {
        "model_type": "EventTHDSurrogate",
        "model_state": best_state,
        "model_config": {
            "input_features": len(THD_SURROGATE_FEATURE_NAMES),
            "hidden_width": 128,
            "output_features": len(THD_SURROGATE_TARGET_NAMES),
        },
        "target_mean": target_mean.tolist(),
        "target_scale": target_scale.tolist(),
        "feature_names": list(THD_SURROGATE_FEATURE_NAMES),
        "target_names": list(THD_SURROGATE_TARGET_NAMES),
        "dataset_sha256": dataset_sha256,
        "seed": seed,
        "split_counts": {
            "training": len(train_values),
            "validation": len(validation_values),
            "testing": len(test_values),
        },
        "split_policy": (
            "Modulation-group holdout over 40 equal-width modulation bands: "
            "bands where index modulo 10 is 0 are test, bands where it is 5 "
            "are validation, and all other bands are training."
        ),
    }
    torch.save(checkpoint, checkpoint_path)
    summary: dict[str, object] = {
        "task": "predict exact event-based total voltage THD; feasibility remains exact-evaluator-only",
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": dataset_sha256,
        "seed": seed,
        "split_counts": checkpoint["split_counts"],
        "split_policy": checkpoint["split_policy"],
        "best_epoch": best_epoch,
        "best_validation_mse_standardized": best_loss,
        "test_metrics": test_metrics,
        "history": history,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    }
    (experiment_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("experiments/thd_surrogate_v1/dataset.csv"),
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("experiments/thd_surrogate_v1/model_run"),
    )
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=18)
    arguments = parser.parse_args()
    summary = train_thd_surrogate(
        dataset_path=arguments.dataset,
        experiment_dir=arguments.experiment_dir,
        seed=arguments.seed,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        patience=arguments.patience,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
