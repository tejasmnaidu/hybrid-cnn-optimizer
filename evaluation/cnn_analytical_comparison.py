"""Compare the saved CNN with the analytical target and a training-mean baseline."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import platform
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

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
from project_paths import resolve_project_path
from training.dataset import WaveformDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "raw" / "Seven_Level_Inverter_Project_extracted"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "colab_runs"
    / "baseline_20261120_080836"
    / "checkpoints"
    / "best.pt"
)
SPLIT_NAMES = ("training", "validation", "testing")
APPROACH_NAMES = ("cnn", "analytical", "training_mean")


@dataclass(frozen=True)
class SplitPredictions:
    sample_ids: np.ndarray
    targets_v: np.ndarray
    cnn_v: np.ndarray
    analytical_v: np.ndarray
    training_mean_v: np.ndarray
    conditions: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _predict_split(
    model: torch.nn.Module,
    prepared: PreparedDataset,
    *,
    split: str,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = WaveformDataset(prepared, split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    sample_ids: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            output = model(
                batch["input_voltage"].to(device),
                batch["load_current"].to(device),
                batch["operating_conditions"].to(device),
            )
            target = batch["target_voltage"]
            if not isinstance(output, torch.Tensor) or output.shape != target.shape:
                shape = tuple(output.shape) if isinstance(output, torch.Tensor) else type(output).__name__
                raise ValueError(
                    f"{split} model output {shape} does not match target shape {tuple(target.shape)}."
                )
            if not torch.isfinite(output).all():
                raise ValueError(f"{split} model output contains non-finite values.")
            predictions.append(output.detach().cpu().numpy())
            targets.append(target.numpy())
            sample_ids.append(batch["sample_id"].numpy())
    return (
        np.concatenate(sample_ids).astype(np.int64, copy=False),
        np.concatenate(targets, axis=0),
        np.concatenate(predictions, axis=0),
    )


def _raw_targets(prepared: PreparedDataset, indices: np.ndarray) -> np.ndarray:
    return np.asarray(
        prepared.stats.denormalize_target(prepared.target_voltage[indices]),
        dtype=np.float64,
    )


def _split_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    pooled = calculate_metrics(targets, predictions)
    errors = predictions - targets
    record_mae = np.mean(np.abs(errors), axis=1, dtype=np.float64)
    record_rmse = np.sqrt(np.mean(np.square(errors), axis=1, dtype=np.float64))
    record_variance = np.sum(
        np.square(targets - np.mean(targets, axis=1, keepdims=True)),
        axis=1,
        dtype=np.float64,
    )
    record_sse = np.sum(np.square(errors), axis=1, dtype=np.float64)
    defined_r2 = record_variance > 0.0
    record_r2 = 1.0 - record_sse[defined_r2] / record_variance[defined_r2]
    return {
        "pooled_point_metrics": pooled,
        "record_count": int(targets.shape[0]),
        "waveform_points_per_record": int(targets.shape[1]),
        "mean_per_record_mae_v": float(np.mean(record_mae)),
        "median_per_record_mae_v": float(np.median(record_mae)),
        "mean_per_record_rmse_v": float(np.mean(record_rmse)),
        "median_per_record_rmse_v": float(np.median(record_rmse)),
        "mean_per_record_r_squared": float(np.mean(record_r2)) if record_r2.size else None,
    }


def _records_for_split(
    model: torch.nn.Module,
    prepared: PreparedDataset,
    *,
    split: str,
    device: torch.device,
    batch_size: int,
    training_mean_v: np.ndarray,
    conditions: np.ndarray,
) -> SplitPredictions:
    indices = prepared.indices(split)
    if split == "testing":
        result = evaluate_cnn_test_set(
            model,
            prepared,
            normalization=prepared.stats,
            device=device,
            batch_size=batch_size,
        )
        sample_ids = result.sample_ids
        targets_v = result.targets_v
        cnn_v = result.predictions_v
    else:
        sample_ids, targets_normalized, cnn_normalized = _predict_split(
            model,
            prepared,
            split=split,
            device=device,
            batch_size=batch_size,
        )
        targets_v = np.asarray(
            prepared.stats.denormalize_target(targets_normalized), dtype=np.float64
        )
        cnn_v = np.asarray(
            prepared.stats.denormalize_target(cnn_normalized), dtype=np.float64
        )
    if not np.array_equal(sample_ids, prepared.sample_ids[indices]):
        raise RuntimeError(f"{split} prediction records do not match the declared split.")
    if not np.isfinite(targets_v).all() or not np.isfinite(cnn_v).all():
        raise ValueError(f"{split} denormalized waveforms contain non-finite values.")
    split_conditions = conditions[indices]
    analytical_v = analytical_reference_waveforms(
        split_conditions,
        prepared.operating_feature_names,
    )
    if not np.isfinite(analytical_v).all():
        raise ValueError(f"{split} analytical reference contains non-finite values.")
    return SplitPredictions(
        sample_ids=sample_ids,
        targets_v=targets_v,
        cnn_v=cnn_v,
        analytical_v=analytical_v,
        training_mean_v=np.broadcast_to(
            training_mean_v, targets_v.shape
        ).copy(),
        conditions=split_conditions,
    )


def _condition_groups(
    *,
    split: str,
    records: SplitPredictions,
    train_conditions: np.ndarray,
    feature_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add_group(
        feature: str,
        group: str,
        mask: np.ndarray,
        lower: float | None = None,
        upper: float | None = None,
    ) -> None:
        count = int(np.count_nonzero(mask))
        if not count:
            return
        for approach in APPROACH_NAMES:
            predictions = getattr(records, f"{approach}_v")
            metrics = calculate_metrics(records.targets_v[mask], predictions[mask])
            rows.append(
                {
                    "split": split,
                    "feature": feature,
                    "group": group,
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "record_count": count,
                    "approach": approach,
                    "mae_v": metrics["mae"],
                    "rmse_v": metrics["rmse"],
                    "r_squared": metrics["r_squared"],
                    "correlation": metrics["correlation"],
                    "maximum_absolute_error_v": metrics["maximum_absolute_error"],
                }
            )

    modulation_column = feature_names.index("modulationIndex")
    train_modulation = train_conditions[:, modulation_column]
    split_modulation = records.conditions[:, modulation_column]
    boundaries = np.unique(np.quantile(train_modulation, np.linspace(0.0, 1.0, 11)))
    for index, (lower, upper) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        if index == len(boundaries) - 2:
            mask = (split_modulation >= lower) & (split_modulation <= upper)
        else:
            mask = (split_modulation >= lower) & (split_modulation < upper)
        add_group(
            "modulationIndex",
            f"training_decile_{index + 1}",
            mask,
            float(lower),
            float(upper),
        )

    for feature in ("loadResistance", "loadInductance"):
        column = feature_names.index(feature)
        for value in np.unique(train_conditions[:, column]):
            mask = np.isclose(records.conditions[:, column], value, rtol=0.0, atol=1e-6)
            add_group(feature, f"{value:g}", mask, float(value), float(value))

    phase_column = feature_names.index("phaseAngle")
    phase_degrees = np.mod(np.degrees(records.conditions[:, phase_column]), 360.0)
    phase_bins = np.minimum(np.floor(phase_degrees / 45.0).astype(np.int64), 7)
    for index in range(8):
        add_group(
            "phaseAngle",
            f"{index * 45}-{(index + 1) * 45} degrees",
            phase_bins == index,
            float(index * 45),
            float((index + 1) * 45),
        )
    return rows


def _per_record_rows(
    split: str,
    records: SplitPredictions,
    feature_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for approach in APPROACH_NAMES:
        predictions = getattr(records, f"{approach}_v")
        errors = predictions - records.targets_v
        for index, sample_id in enumerate(records.sample_ids):
            target = records.targets_v[index]
            error = errors[index]
            variance = float(np.sum(np.square(target - np.mean(target))))
            rows.append(
                {
                    "split": split,
                    "sample_id": int(sample_id),
                    "approach": approach,
                    **{
                        name: float(records.conditions[index, column])
                        for column, name in enumerate(feature_names)
                    },
                    "mae_v": float(np.mean(np.abs(error))),
                    "rmse_v": float(np.sqrt(np.mean(np.square(error)))),
                    "r_squared": (
                        float(1.0 - np.sum(np.square(error)) / variance)
                        if variance > 0.0
                        else None
                    ),
                    "maximum_absolute_error_v": float(np.max(np.abs(error))),
                }
            )
    return rows


def _representative_indices(records: SplitPredictions, resistance_column: int) -> list[int]:
    selected: list[int] = []
    for resistance in (10.0, 20.0, 30.0, 40.0):
        matches = np.flatnonzero(
            np.isclose(records.conditions[:, resistance_column], resistance, rtol=0.0, atol=1e-5)
        )
        if not matches.size:
            raise ValueError(
                f"No records for {resistance:g} ohm in the split; cannot make fixed-condition plots."
            )
        selected.append(int(matches[np.argmin(records.sample_ids[matches])]))
    return selected


def _write_waveform_plot(
    path: Path,
    split: str,
    records: SplitPredictions,
    selected_indices: list[int],
) -> list[dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time_ms = np.arange(records.targets_v.shape[1], dtype=np.float64) / 20_000.0 * 1000.0
    figure, axes = plt.subplots(
        len(selected_indices), 2, figsize=(15, 3.2 * len(selected_indices)), squeeze=False
    )
    selected_records: list[dict[str, Any]] = []
    for row, index in enumerate(selected_indices):
        target = records.targets_v[index]
        cnn_error = records.cnn_v[index] - target
        analytical_error = records.analytical_v[index] - target
        conditions = records.conditions[index]
        axes[row, 0].plot(time_ms, target, color="#e5e7eb", linewidth=1.6, label="Target")
        axes[row, 0].plot(time_ms, records.cnn_v[index], color="#22d3ee", linewidth=1.2, label="CNN")
        axes[row, 0].plot(
            time_ms,
            records.analytical_v[index],
            color="#f59e0b",
            linewidth=1.1,
            linestyle="--",
            label="Analytical",
        )
        axes[row, 0].set_title(
            f"Sample {int(records.sample_ids[index])} · "
            f"m={conditions[0]:.3f}, R={conditions[1]:.0f} Ω, "
            f"L={conditions[2]:.2f} H, phase={math.degrees(conditions[3]):.1f}°"
        )
        axes[row, 0].set_ylabel("Voltage (V)")
        axes[row, 0].grid(alpha=0.22)
        if row == 0:
            axes[row, 0].legend(ncol=3, loc="upper right")

        axes[row, 1].plot(time_ms, cnn_error, color="#22d3ee", linewidth=1.2, label="CNN − target")
        axes[row, 1].plot(
            time_ms,
            analytical_error,
            color="#f59e0b",
            linewidth=1.1,
            linestyle="--",
            label="Analytical − target",
        )
        axes[row, 1].axhline(0.0, color="#9ca3af", linewidth=0.8)
        axes[row, 1].set_title("Pointwise prediction error")
        axes[row, 1].set_ylabel("Error (V)")
        axes[row, 1].grid(alpha=0.22)
        if row == 0:
            axes[row, 1].legend(ncol=2, loc="upper right")
        selected_records.append(
            {
                "sample_id": int(records.sample_ids[index]),
                "selection_rule": "lowest SampleID within each load-resistance category; not selected by prediction error",
                "modulation_index": float(conditions[0]),
                "load_resistance_ohm": float(conditions[1]),
                "load_inductance_h": float(conditions[2]),
                "phase_angle_degrees": float(math.degrees(conditions[3])),
            }
        )
    axes[-1, 0].set_xlabel("Time (ms)")
    axes[-1, 1].set_xlabel("Time (ms)")
    figure.suptitle(f"{split.title()} examples: reference, CNN, analytical estimate", y=1.002)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return selected_records


def _write_condition_plot(path: Path, split: str, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    features = ("modulationIndex", "loadResistance", "loadInductance", "phaseAngle")
    names = {
        "modulationIndex": "Modulation (training deciles)",
        "loadResistance": "Load resistance (Ω)",
        "loadInductance": "Load inductance (H)",
        "phaseAngle": "Phase angle (45° bins)",
    }
    colors = {"cnn": "#22d3ee", "analytical": "#f59e0b", "training_mean": "#9ca3af"}
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    for axis, feature in zip(axes.flat, features):
        group_rows = [row for row in rows if row["feature"] == feature]
        groups = list(dict.fromkeys(row["group"] for row in group_rows))
        positions = np.arange(len(groups), dtype=np.float64)
        for approach in APPROACH_NAMES:
            by_group = {
                row["group"]: row["mae_v"]
                for row in group_rows
                if row["approach"] == approach
            }
            values = [by_group.get(group, np.nan) for group in groups]
            axis.plot(
                positions,
                values,
                marker="o",
                markersize=3,
                linewidth=1.3,
                color=colors[approach],
                label=approach.replace("_", " ").title(),
            )
        axis.set_title(names[feature])
        axis.set_ylabel("Pooled point MAE (V)")
        axis.set_xticks(positions, groups, rotation=35, ha="right")
        axis.grid(alpha=0.22)
    axes.flat[0].legend()
    figure.suptitle(f"{split.title()} MAE by operating-condition group")
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _format_metrics_table(summary: dict[str, Any]) -> str:
    lines = [
        "| Split | Approach | MAE (V) | RMSE (V) | R² | Correlation | Max absolute error (V) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for split in SPLIT_NAMES:
        for approach in APPROACH_NAMES:
            metrics = summary["metrics"][split][approach]["pooled_point_metrics"]
            lines.append(
                "| {split} | {approach} | {mae:.9g} | {rmse:.9g} | {r2} | "
                "{corr} | {maxerr:.9g} |".format(
                    split=split,
                    approach={
                        "cnn": "CNN",
                        "analytical": "Analytical",
                        "training_mean": "Training mean",
                    }[approach],
                    mae=metrics["mae"],
                    rmse=metrics["rmse"],
                    r2=format(metrics["r_squared"], ".16g"),
                    corr=format(metrics["correlation"], ".16g"),
                    maxerr=metrics["maximum_absolute_error"],
                )
            )
    return "\n".join(lines)


def _format_condition_table(condition_rows: list[dict[str, Any]], split: str) -> str:
    selected = [
        row
        for row in condition_rows
        if row["split"] == split and row["approach"] in {"cnn", "analytical"}
    ]
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in selected:
        grouped.setdefault((row["feature"], row["group"]), {})[row["approach"]] = row
    lines = [
        "| Condition feature | Group | Records | CNN MAE / RMSE (V) | Analytical MAE / RMSE (V) |",
        "|---|---|---:|---:|---:|",
    ]
    for (feature, group), approaches in grouped.items():
        cnn = approaches["cnn"]
        analytical = approaches["analytical"]
        lines.append(
            f"| {feature} | {group} | {cnn['record_count']} | "
            f"{cnn['mae_v']:.7g} / {cnn['rmse_v']:.7g} | "
            f"{analytical['mae_v']:.7g} / {analytical['rmse_v']:.7g} |"
        )
    return "\n".join(lines)


def _render_report(
    summary: dict[str, Any],
    output_dir: Path,
    report_path: Path,
    condition_rows: list[dict[str, Any]],
) -> None:
    validation = summary["metrics"]["validation"]
    validation_winner = min(
        APPROACH_NAMES,
        key=lambda approach: validation[approach]["pooled_point_metrics"]["rmse"],
    )
    test_metrics = summary["metrics"]["testing"]
    cnn_test = test_metrics["cnn"]["pooled_point_metrics"]
    analytical_test = test_metrics["analytical"]["pooled_point_metrics"]
    gap = cnn_test["rmse"] - analytical_test["rmse"]
    if gap > 0:
        conclusion = (
            f"The analytical reference is more accurate on the final test split: its pooled "
            f"RMSE is lower by {gap:.9g} V. The CNN does not provide measurable prediction "
            "accuracy value over the known analytical target equation in this dataset."
        )
    else:
        conclusion = (
            f"The CNN pooled RMSE is lower than the analytical reference by {-gap:.9g} V "
            "on the final test split. This is a waveform prediction comparison only; it "
            "does not establish inverter harmonic reduction or physical performance."
        )
    output_relative = output_dir.relative_to(PROJECT_ROOT).as_posix()
    test_count = summary["split_counts"]["testing"]
    report = f"""# CNN vs Analytical Reference Evaluation

**Run timestamp:** {summary["created_at_local"]}  
**Dataset:** `{summary["dataset_path"]}`  
**Checkpoint:** `{summary["checkpoint_path"]}`  
**Inference device:** `{summary["device"]}`; batch size `{summary["batch_size"]}`  
**Reproducible outputs:** [`{output_relative}/`](./{output_relative}/)

## Result in brief

Validation pooled RMSE selected **{validation_winner.replace("_", " ")}** as the best of the evaluated fixed predictors. No model weights, hyperparameters, or analytical equation were selected or tuned using the testing split. The test split is reported once as the final held-out comparison.

{conclusion}

## Inspection and provenance

- The saved model is `WaveformResidualCNN`, a separate 1D residual branch for input voltage and load current, with a condition encoder fused into residual convolution blocks. The checkpoint was loaded strictly against its stored model configuration.
- Checkpoint architecture: 400-point waveforms; two residual branches with 32/64 channels; four operating-condition inputs encoded to width 32; 64-channel fusion blocks; kernel size 5. The selected checkpoint is from epoch {summary["checkpoint_metadata"]["epoch_one_based"]}.
- Checkpoint parameter count: **{summary["model_parameter_count"]:,}**.
- Dataset tables are aligned by `SampleID`; the stored input, target, and operating-condition normalization was checked against statistics refit from training records only.
- Dataset split sizes: **{summary["split_counts"]["training"]:,} training / {summary["split_counts"]["validation"]:,} validation / {test_count:,} testing**.
- CNN inference ran in evaluation mode under `torch.inference_mode()`; checkpoint weights were not modified and no training or fine-tuning ran.
- Checkpoint SHA-256: `{summary["checkpoint_sha256"]}`.
- Source dataset CSV SHA-256 values are recorded in [`summary.json`](./{output_relative}/summary.json).
- The project contains no original MATLAB/source dataset generator file. The analytical reference used by the existing scientific validation code is:

  `Vref[n] = 3 × 100 V × m × sin(2π × 50 Hz × n/20,000 Hz + phaseAngleRadians)`, for `n=0..399`.

  This estimate uses modulation index and phase angle only. It is compared with the stored `Target_Voltage.csv`; it does not use target values to construct its prediction.

## Metrics

Pooled metrics flatten all waveform points within each split. The per-record means and medians are reported in `summary.json`; per-record metrics for all three predictors are in [`per_record_metrics.csv`](./{output_relative}/per_record_metrics.csv). R² and correlation shown below are computed across pooled points, not averaged recordwise.

{_format_metrics_table(summary)}

### Validation-based comparison

Validation records were used to compare the fixed CNN, analytical equation, and a training-mean waveform baseline. No model tuning was done. The training-mean waveform is the pointwise mean target over the training split only. The validation winner is chosen by pooled RMSE; held-out test results are not used in that choice.

### Final held-out test comparison

The complete testing split contains **{test_count:,} records × {summary["waveform_points"]} points = {test_count * summary["waveform_points"]:,} points**. Only this section's test metrics are final held-out reporting; test rows were not used for development, model selection, normalization fitting, or training.

## Operating-condition comparisons

See [`condition_metrics.csv`](./{output_relative}/condition_metrics.csv) and the plots below. Modulation decile boundaries are fit to training values. Load resistance and inductance groups are the categories observed in training. Phase angle uses fixed 45-degree bins. Group metrics are descriptive: they do not justify claims about unobserved values or untested joint operating-condition combinations.

### Validation condition metrics

MAE/RMSE entries are pooled over all waveform points in the listed records.

{_format_condition_table(condition_rows, "validation")}

### Held-out test condition metrics

MAE/RMSE entries are pooled over all waveform points in the listed records.

{_format_condition_table(condition_rows, "testing")}

### Validation waveforms and errors

![Validation waveform comparisons](./{output_relative}/plots/validation_waveforms.png)

### Test waveforms and errors

![Final held-out test waveform comparisons](./{output_relative}/plots/test_waveforms.png)

### Validation MAE by operating condition

![Validation condition comparison](./{output_relative}/plots/validation_condition_mae.png)

### Test MAE by operating condition

![Test condition comparison](./{output_relative}/plots/test_condition_mae.png)

Displayed example records are selected deterministically as the lowest SampleID within each load-resistance category, not by their prediction errors. The test plots are for final reporting only and were not used to select or modify the model.

## Interpretation and limitations

{conclusion}

- The target data are nearly identical to a deterministic analytical sinusoid under the stated conditions, so the CNN is approximating a calculation already available from the operating point.
- The analytical formula assumes a 300 V total DC-link amplitude, 50 Hz fundamental, and the dataset's phase convention. Its near-zero numerical residual does not independently establish that the dataset's original MATLAB generation script used this exact code; that source script is absent from the project folder.
- Aggregate metrics do not establish causal benefit, extrapolation beyond the training support, robustness to measurement noise, or real-time runtime advantage.
- CNN waveform-prediction accuracy is not evidence of lower inverter output THD. The CNN does not predict switching gates.
- This is dataset/software validation only; it is not hardware or Simulink validation.

## Reproduction

From the project root, run:

```powershell
& .\\.venv\\Scripts\\python.exe -B -m evaluation.cnn_analytical_comparison --output-dir results\\cnn_vs_analytical_<new-run-name>
```

The output directory must not already exist. The script records package versions, hashes, split counts, normalization source, metrics, and selected plot-record IDs in `summary.json`.
"""
    report_path.write_text(report, encoding="utf-8")


def run_comparison(
    *,
    dataset_path: Path = DEFAULT_DATASET,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    output_dir: Path,
    report_path: Path = PROJECT_ROOT / "CNN_VS_ANALYTICAL_EVALUATION.md",
    batch_size: int = 64,
    device: str | None = None,
) -> Path:
    dataset_path = resolve_project_path(dataset_path).resolve()
    checkpoint_path = resolve_project_path(checkpoint_path).resolve()
    output_dir = resolve_project_path(output_dir).resolve()
    report_path = resolve_project_path(report_path).resolve()
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    dataset_tables = load_dataset(dataset_path)
    prepared = preprocess_dataset(dataset_tables)
    split_counts = {
        split: int(prepared.indices(split).size)
        for split in SPLIT_NAMES
    }
    if any(count == 0 for count in split_counts.values()):
        raise ValueError(f"All data splits must be populated: {split_counts}")

    device_obj = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint_hash_before = _sha256(checkpoint_path)
    source_files = sorted(dataset_tables.root.glob("*.csv"))
    dataset_hashes_before = {path.name: _sha256(path) for path in source_files}
    checkpoint = torch.load(checkpoint_path, map_location=device_obj, weights_only=True)
    normalization = assert_checkpoint_compatible(checkpoint, prepared)
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("model_config"), dict):
        raise ValueError("Checkpoint does not contain its model configuration.")
    model = create_model(metadata["model_config"]).to(device_obj)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    model_parameter_count = model.parameter_count()
    raw_conditions = raw_operating_conditions(prepared)
    training_indices = prepared.indices("training")
    training_targets_v = _raw_targets(prepared, training_indices)
    training_mean_v = np.mean(training_targets_v, axis=0, dtype=np.float64)
    train_conditions = raw_conditions[training_indices]

    predictions: dict[str, SplitPredictions] = {}
    for split in SPLIT_NAMES:
        predictions[split] = _records_for_split(
            model,
            prepared,
            split=split,
            device=device_obj,
            batch_size=batch_size,
            training_mean_v=training_mean_v,
            conditions=raw_conditions,
        )

    metrics: dict[str, dict[str, Any]] = {}
    condition_rows: list[dict[str, Any]] = []
    per_record_rows: list[dict[str, Any]] = []
    for split, records in predictions.items():
        metrics[split] = {
            approach: _split_metrics(
                records.targets_v,
                getattr(records, f"{approach}_v"),
            )
            for approach in APPROACH_NAMES
        }
        condition_rows.extend(
            _condition_groups(
                split=split,
                records=records,
                train_conditions=train_conditions,
                feature_names=prepared.operating_feature_names,
            )
        )
        per_record_rows.extend(
            _per_record_rows(split, records, prepared.operating_feature_names)
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir()
    representative_ids: dict[str, list[dict[str, Any]]] = {}
    resistance_column = prepared.operating_feature_names.index("loadResistance")
    for split in ("validation", "testing"):
        indices = _representative_indices(predictions[split], resistance_column)
        plot_name = "test" if split == "testing" else split
        representative_ids[split] = _write_waveform_plot(
            plots_dir / f"{plot_name}_waveforms.png",
            split,
            predictions[split],
            indices,
        )
        split_rows = [row for row in condition_rows if row["split"] == split]
        _write_condition_plot(
            plots_dir / f"{plot_name}_condition_mae.png",
            split,
            split_rows,
        )
    _write_csv(output_dir / "per_record_metrics.csv", per_record_rows)
    _write_csv(output_dir / "condition_metrics.csv", condition_rows)

    checkpoint_hash_after = _sha256(checkpoint_path)
    dataset_hashes_after = {path.name: _sha256(path) for path in source_files}
    if checkpoint_hash_after != checkpoint_hash_before:
        raise RuntimeError("Checkpoint hash changed during inference; weights must remain untouched.")
    if dataset_hashes_after != dataset_hashes_before:
        raise RuntimeError("Dataset CSV hashes changed during inference; original data must remain untouched.")

    summary = {
        "evaluation": "CNN versus analytical target waveform; inference only",
        "created_at_local": datetime.now().astimezone().isoformat(),
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "dataset_path": str(dataset_tables.root.resolve()),
        "checkpoint_path": str(checkpoint_path),
        "dataset_csv_sha256": dataset_hashes_before,
        "checkpoint_sha256": checkpoint_hash_before,
        "checkpoint_metadata": {
            "model_config": metadata["model_config"],
            "normalization": asdict(normalization),
            "epoch_one_based": (
                int(checkpoint["epoch"]) + 1
                if isinstance(checkpoint.get("epoch"), (int, float))
                else None
            ),
            "parameter_count": model_parameter_count,
            "loaded_strictly": True,
            "normalization_matches_training_split": True,
        },
        "model_parameter_count": model_parameter_count,
        "dataset_split_counts": split_counts,
        "split_counts": split_counts,
        "waveform_points": int(prepared.target_voltage.shape[1]),
        "batch_size": batch_size,
        "device": str(device_obj),
        "normalization_statistics_fit_on": "training split only",
        "evaluation_policy": {
            "training": "in-sample descriptive measurement; also used to fit the pointwise-mean baseline",
            "validation": "fixed-predictor comparison; pooled RMSE determines the reported comparison winner",
            "testing": "complete held-out final report only; not used for predictor selection or tuning",
            "checkpoint_weights_modified": False,
            "training_or_fine_tuning_run": False,
        },
        "approaches": {
            "cnn": "Existing strictly loaded WaveformResidualCNN checkpoint.",
            "analytical": "300*m*sin(2*pi*50*t + phaseAngleRadians), t=n/20000, n=0..399.",
            "training_mean": "Pointwise mean of the denormalized training target waveforms only.",
        },
        "analytical_formula_source": (
            "Reused analytical equation from evaluation.scientific_validation; the original MATLAB/data generator is not present."
        ),
        "metrics": metrics,
        "validation_winner_by_pooled_rmse": min(
            APPROACH_NAMES,
            key=lambda name: metrics["validation"][name]["pooled_point_metrics"]["rmse"],
        ),
        "condition_group_definition": (
            "Modulation deciles from training-only quantiles; R/L categories from training; "
            "phase bins fixed at 45 degrees. Same bins applied to every split."
        ),
        "representative_plot_records": representative_ids,
        "artifacts": {
            "per_record_metrics": "per_record_metrics.csv",
            "condition_metrics": "condition_metrics.csv",
            "validation_waveforms": "plots/validation_waveforms.png",
            "test_waveforms": "plots/test_waveforms.png",
            "validation_condition_mae": "plots/validation_condition_mae.png",
            "test_condition_mae": "plots/test_condition_mae.png",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _render_report(summary, output_dir, report_path, condition_rows)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "results"
        / f"cnn_vs_analytical_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "CNN_VS_ANALYTICAL_EVALUATION.md",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    output_dir = run_comparison(
        dataset_path=args.dataset,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        report_path=args.report,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(f"CNN versus analytical comparison outputs: {output_dir}")


if __name__ == "__main__":
    main()


__all__ = ["analytical_reference_waveforms", "run_comparison"]
