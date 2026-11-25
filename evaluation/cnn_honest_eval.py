"""Read-only, whole-test CNN versus analytical and trivial-baseline evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluation.cnn_test_evaluator import evaluate_cnn_test_set
from evaluation.metrics import calculate_metrics
from evaluation.scientific_validation import (
    analytical_reference_waveforms,
    assert_checkpoint_compatible,
    raw_operating_conditions,
)
from models.model_factory import create_model
from preprocessing.data_loader import load_dataset
from preprocessing.preprocess import PreparedDataset, preprocess_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    PROJECT_ROOT / "data" / "raw" / "Seven_Level_Inverter_Project_extracted"
)
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "colab_runs"
    / "baseline_20261120_080836"
    / "checkpoints"
    / "best.pt"
)
REPORT_DIRECTORY = PROJECT_ROOT / "evaluation" / "reports"
JSON_OUTPUT = REPORT_DIRECTORY / "cnn_eval.json"
PLOT_OUTPUT = REPORT_DIRECTORY / "cnn_comparison.png"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_disjoint_split_indices(prepared: PreparedDataset) -> dict[str, int]:
    """Require disjoint split index sets and the project's expected 7000/1500/1500 split."""
    indices = {
        name: prepared.indices(name)
        for name in ("training", "validation", "testing")
    }
    counts = {name: int(values.size) for name, values in indices.items()}
    if counts != {"training": 7000, "validation": 1500, "testing": 1500}:
        raise ValueError(f"Unexpected split counts: {counts}")
    for left_name, left in indices.items():
        for right_name, right in indices.items():
            if left_name == right_name:
                continue
            if np.intersect1d(left, right).size:
                raise RuntimeError(
                    f"Split leakage: {left_name} and {right_name} overlap."
                )
    return counts


def per_record_metrics(
    targets_v: np.ndarray,
    predictions_v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-record MAE, MSE, RMSE and R2 arrays."""
    targets = np.asarray(targets_v, dtype=np.float64)
    predictions = np.asarray(predictions_v, dtype=np.float64)
    if targets.shape != predictions.shape or targets.ndim != 2:
        raise ValueError("Targets and predictions must be equal (record, point) arrays.")
    if not np.isfinite(targets).all() or not np.isfinite(predictions).all():
        raise ValueError("Targets and predictions must contain finite values.")

    errors = predictions - targets
    mae = np.mean(np.abs(errors), axis=1, dtype=np.float64)
    mse = np.mean(np.square(errors), axis=1, dtype=np.float64)
    rmse = np.sqrt(mse)
    r2 = np.empty(targets.shape[0], dtype=np.float64)
    r2.fill(np.nan)
    for index, target in enumerate(targets):
        total_variance = float(np.sum(np.square(target - np.mean(target))))
        r2[index] = (
            float(1.0 - np.sum(np.square(errors[index])) / total_variance)
            if total_variance > 0.0
            else np.nan
        )
    return mae, mse, rmse, r2


def aggregate_metrics(
    targets_v: np.ndarray,
    predictions_v: np.ndarray,
) -> dict[str, Any]:
    """Aggregate one predictor over all test records and then report per-record means."""
    targets = np.asarray(targets_v, dtype=np.float64)
    predictions = np.asarray(predictions_v, dtype=np.float64)
    pooled = calculate_metrics(targets, predictions)
    record_mae, record_mse, record_rmse, record_r2 = per_record_metrics(
        targets, predictions
    )
    valid_r2 = record_r2[np.isfinite(record_r2)]
    return {
        "aggregate_units": "volts",
        "pooled_point_metrics": {
            "mae": float(pooled["mae"]),
            "mse": float(pooled["mse"]),
            "rmse": float(pooled["rmse"]),
            "r2": float(pooled["r_squared"]),
        },
        "per_record_mean": {
            "mae": float(np.mean(record_mae)),
            "mse": float(np.mean(record_mse)),
            "rmse": float(np.mean(record_rmse)),
            "r2": float(np.mean(valid_r2)) if valid_r2.size else None,
        },
    }


def trivial_training_mean_baseline(
    prepared: PreparedDataset,
    testing_targets_v: np.ndarray,
) -> np.ndarray:
    """Training-pointwise mean target waveform, broadcast over test records."""
    training_indices = prepared.indices("training")
    training_targets_v = prepared.stats.denormalize_target(
        prepared.target_voltage[training_indices]
    )
    mean_waveform = np.mean(
        np.asarray(training_targets_v, dtype=np.float64),
        axis=0,
        dtype=np.float64,
    )
    return np.broadcast_to(mean_waveform, np.asarray(testing_targets_v).shape).copy()


def run_honest_evaluation(
    *,
    dataset_path: Path = DEFAULT_DATASET,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    json_output: Path = JSON_OUTPUT,
    plot_output: Path = PLOT_OUTPUT,
    batch_size: int = 64,
    device: str = "cpu",
) -> dict[str, Any]:
    """Evaluate existing checkpoint and baselines only on the explicit test split."""
    dataset_path = dataset_path.resolve()
    checkpoint_path = checkpoint_path.resolve()
    json_output = json_output.resolve()
    plot_output = plot_output.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if json_output.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {json_output}")
    if plot_output.exists():
        raise FileExistsError(f"Refusing to overwrite existing plot: {plot_output}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    tables = load_dataset(dataset_path)
    prepared = preprocess_dataset(tables)
    split_counts = verify_disjoint_split_indices(prepared)
    checkpoint_sha_before = _sha256(checkpoint_path)
    dataset_sha_before = {
        path.name: _sha256(path) for path in sorted(tables.root.glob("*.csv"))
    }

    device_obj = torch.device(device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device_obj,
        weights_only=True,
    )
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError("Invalid checkpoint: model_state is missing.")
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint metadata is missing.")
    normalization = assert_checkpoint_compatible(checkpoint, prepared)
    model_config = metadata.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("Checkpoint model_config is missing.")

    model = create_model(model_config).to(device_obj)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    test_indices = prepared.indices("testing")
    cnn_result = evaluate_cnn_test_set(
        model,
        prepared,
        normalization=normalization,
        device=device_obj,
        batch_size=batch_size,
    )
    if not np.array_equal(
        cnn_result.sample_ids,
        prepared.sample_ids[test_indices],
    ):
        raise RuntimeError("CNN test inference IDs do not match the testing split.")

    raw_conditions = raw_operating_conditions(prepared)
    test_conditions = raw_conditions[test_indices]
    analytical_predictions = analytical_reference_waveforms(
        test_conditions,
        prepared.operating_feature_names,
    )
    trivial_predictions = trivial_training_mean_baseline(
        prepared,
        cnn_result.targets_v,
    )

    predictors = {
        "cnn": cnn_result.predictions_v,
        "analytical_reference": analytical_predictions,
        "trivial_training_mean": trivial_predictions,
    }
    results = {
        name: aggregate_metrics(cnn_result.targets_v, predictions)
        for name, predictions in predictors.items()
    }
    cnn_rmse = results["cnn"]["pooled_point_metrics"]["rmse"]
    analytical_rmse = results["analytical_reference"]["pooled_point_metrics"]["rmse"]
    rmse_delta = cnn_rmse - analytical_rmse
    if abs(rmse_delta) <= 1e-12:
        comparison = "equal"
    elif rmse_delta < 0.0:
        comparison = "CNN beats analytical reference"
    else:
        comparison = "CNN does not beat analytical reference"

    checkpoint_sha_after = _sha256(checkpoint_path)
    dataset_sha_after = {
        path.name: _sha256(path) for path in sorted(tables.root.glob("*.csv"))
    }
    if checkpoint_sha_after != checkpoint_sha_before:
        raise RuntimeError("Checkpoint changed during read-only evaluation.")
    if dataset_sha_after != dataset_sha_before:
        raise RuntimeError("Dataset CSV changed during read-only evaluation.")

    report: dict[str, Any] = {
        "evaluation": "held-out CNN versus analytical reference and trivial baseline",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha_before,
        "checkpoint_modified": False,
        "dataset_path": str(tables.root.resolve()),
        "dataset_csv_sha256": dataset_sha_before,
        "dataset_modified": False,
        "split_counts": split_counts,
        "test_record_count": 1500,
        "test_sample_ids": [int(value) for value in cnn_result.sample_ids],
        "waveform_points_per_record": 400,
        "units": "volts",
        "ground_truth": "PreparedDataset Target_Voltage denormalized with training-fitted target statistics",
        "cnn_prediction": {
            "description": "Existing checkpoint prediction of normalized target voltage",
            "raw_model_output": "normalized waveform",
            "physical_output": "denormalized voltage waveform",
            "shape": [1500, 400],
        },
        "analytical_reference": {
            "description": "300*m*sin(2*pi*50*n/20000 + phaseAngle)",
            "units": "volts",
            "shape": [1500, 400],
        },
        "trivial_baseline": {
            "description": "Pointwise mean of denormalized training Target_Voltage waveforms",
            "units": "volts",
            "shape": [1500, 400],
            "fit_data": "training split only",
        },
        "normalization": {
            "statistics_fit": "training split only",
            "cnn_output_denormalization": "prediction * target_voltage_std + target_voltage_mean",
            "ground_truth_denormalization": "target * target_voltage_std + target_voltage_mean",
        },
        "metrics": results,
        "analytical_comparison": {
            "cnn_pooled_rmse_minus_analytical_pooled_rmse": float(rmse_delta),
            "conclusion": comparison,
        },
        "method": {
            "batch_size": batch_size,
            "device": str(device_obj),
            "pooling": "Every 600,000 test waveform points are flattened for pooled metrics.",
            "per_sample_aggregation": "Per-record MAE/MSE/RMSE are averaged across 1,500 test records.",
            "r2": "Pooled R2 is one metric over all points; per-record means exclude undefined constant-target R2.",
            "no_retraining": True,
            "test_split_used_for_tuning": False,
        },
    }

    json_output.parent.mkdir(parents=True, exist_ok=True)
    plot_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    first_index = 0
    time_ms = np.arange(400, dtype=np.float64) / 20000.0 * 1000.0
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(
        time_ms,
        cnn_result.targets_v[first_index],
        color="#e5e7eb",
        linewidth=1.8,
        label="Test target (dataset)",
    )
    axes[0].plot(
        time_ms,
        cnn_result.predictions_v[first_index],
        color="#22d3ee",
        linewidth=1.4,
        label="CNN prediction",
    )
    axes[0].plot(
        time_ms,
        analytical_predictions[first_index],
        color="#f59e0b",
        linewidth=1.4,
        linestyle="--",
        label="Analytical reference",
    )
    axes[0].plot(
        time_ms,
        trivial_predictions[first_index],
        color="#a78bfa",
        linewidth=1.2,
        linestyle=":",
        label="Training-mean baseline",
    )
    axes[0].set_ylabel("Voltage (V)")
    axes[0].set_title("Test Sample: CNN, analytical and trivial predictions")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.2)

    metrics_names = ("cnn", "analytical_reference", "trivial_training_mean")
    metrics_labels = ("CNN", "Analytical", "Training mean")
    axes[1].bar(
        metrics_labels,
        [results[name]["pooled_point_metrics"]["rmse"] for name in metrics_names],
        color=("#22d3ee", "#f59e0b", "#a78bfa"),
    )
    axes[1].set_ylabel("Pooled RMSE (V)")
    axes[1].set_title(comparison)
    axes[1].grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(plot_output, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return report


def _render_documentation() -> str:
    return """# CNN Documentation

## Task and prediction

The existing `WaveformResidualCNN` predicts a **400-point target/reference voltage waveform**.
It does not predict an output measurement, switching gate, optimized inverter harmonic, or controller action.

- Inputs: normalized input voltage `(batch,400)`, normalized load current `(batch,400)`, and normalized operating conditions `(batch,4)` in feature order `modulationIndex`, `loadResistance`, `loadInductance`, `phaseAngle`.
- Model output: normalized target voltage `(batch,400)`.
- Physical output: the normalized waveform multiplied by the training-fitted `target_voltage_std` and then added to `target_voltage_mean`.
- Ground truth: `Target_Voltage.csv`, denormalized from the same target normalization.

## Architecture

The model is a residual 1D CNN with:

- Separate 32/64-channel voltage and load-current residual branches.
- A 4-to-32 condition MLP with linear/GELU/linear/GELU operations.
- 64-channel fusion residual blocks.
- A one-channel 1x1 convolution returning 400 waveform points.
- 201,601 parameters in the existing checkpoint configuration.

## Normalization

Normalization statistics are fitted on training IDs only:

- Input voltage mean/std.
- Load-current mean/std.
- Four per-feature operating-condition mean/std.
- Target voltage mean/std.

The exact formulas are affine standardization and its affine inverse. No test-record normalization statistics are used.

## Loss and training

The existing configuration uses MAE loss with derivative weight 0.0.
The training seed is 42. AdamW and the validation-based early stopping/checkpoint selection are already defined in `training/trainer.py`.

The test split is not accepted by `Trainer.fit` and is not used for training or checkpoint selection.

## Leakage verification

The split manifest contains 7,000 training, 1,500 validation and 1,500 testing records.
The implementation uses disjoint `prepared.indices(...)` index sets. Normalization means/stds are masked to training IDs.

## Predictor comparison

All predictors are converted to volts and evaluated against the same 1,500 test `Target_Voltage` waveforms:

1. Existing CNN: denormalized checkpoint prediction.
2. Analytical reference: `300*m*sin(2*pi*50*n/20000 + phaseAngle)`.
3. Trivial baseline: pointwise mean of denormalized training target waveforms, broadcast to each test record.

Reported metrics are waveform MAE, MSE, RMSE and R². Pooled metrics flatten all 600,000 test points. Per-record metrics are also computed and averaged. No retraining or test-split tuning is performed.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--json-output", type=Path, default=JSON_OUTPUT)
    parser.add_argument("--plot-output", type=Path, default=PLOT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = run_honest_evaluation(
        dataset_path=args.dataset,
        checkpoint_path=args.checkpoint,
        json_output=args.json_output,
        plot_output=args.plot_output,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps(report["analytical_comparison"], indent=2))


if __name__ == "__main__":
    main()
