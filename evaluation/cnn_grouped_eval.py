"""Inference-only CNN versus analytical metrics on condition-grouped test records."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

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
MINIMUM_GROUP_RECORDS = 20
RMSE_EQUALITY_TOLERANCE_V = 1e-12


def training_modulation_edges(
    training_modulation: np.ndarray,
    *,
    band_count: int = 10,
) -> np.ndarray:
    """Create reproducible modulation bands using training-only quantiles."""
    values = np.asarray(training_modulation, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Training modulation values must be non-empty and finite.")
    if band_count < 1:
        raise ValueError("band_count must be positive.")
    edges = np.unique(
        np.quantile(values, np.linspace(0.0, 1.0, band_count + 1))
    )
    if edges.size < 2:
        raise ValueError("Training modulation values do not define multiple bands.")
    return edges


def modulation_band_indices(
    modulation: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    """Assign values to half-open bands, including the final right edge."""
    values = np.asarray(modulation, dtype=np.float64).reshape(-1)
    boundaries = np.asarray(edges, dtype=np.float64).reshape(-1)
    if boundaries.size < 2 or not np.isfinite(boundaries).all():
        raise ValueError("At least two finite modulation band edges are required.")
    if np.any(np.diff(boundaries) <= 0):
        raise ValueError("Modulation band edges must be strictly increasing.")
    if not np.isfinite(values).all():
        raise ValueError("Modulation values must be finite.")
    return np.searchsorted(boundaries[1:-1], values, side="right").astype(np.int64)


def _metric_summary(
    targets_v: np.ndarray,
    predictions_v: np.ndarray,
    *,
    record_count: int,
    minimum_records: int,
) -> dict[str, Any]:
    metrics = calculate_metrics(targets_v, predictions_v)
    return {
        "record_count": int(record_count),
        "minimum_record_count": int(minimum_records),
        "below_minimum_record_count": bool(record_count < minimum_records),
        "mae_v": float(metrics["mae"]),
        "rmse_v": float(metrics["rmse"]),
        "r_squared": float(metrics["r_squared"]),
        "correlation": float(metrics["correlation"]),
    }


def compare_group(
    targets_v: np.ndarray,
    cnn_predictions_v: np.ndarray,
    analytical_predictions_v: np.ndarray,
    *,
    minimum_records: int = MINIMUM_GROUP_RECORDS,
    rmse_equality_tolerance_v: float = RMSE_EQUALITY_TOLERANCE_V,
) -> dict[str, Any]:
    """Summarize both predictors on paired records and classify by pooled RMSE."""
    targets = np.asarray(targets_v, dtype=np.float64)
    cnn = np.asarray(cnn_predictions_v, dtype=np.float64)
    analytical = np.asarray(analytical_predictions_v, dtype=np.float64)
    if targets.ndim != 2 or targets.shape != cnn.shape or targets.shape != analytical.shape:
        raise ValueError("Targets and predictions must have identical (records, points) shapes.")
    if targets.shape[0] == 0 or targets.shape[1] == 0:
        raise ValueError("A comparison group must contain records and waveform points.")
    if minimum_records < 1:
        raise ValueError("minimum_records must be positive.")
    if not np.isfinite(rmse_equality_tolerance_v) or rmse_equality_tolerance_v < 0:
        raise ValueError("RMSE equality tolerance must be finite and non-negative.")

    record_count = int(targets.shape[0])
    cnn_metrics = _metric_summary(
        targets,
        cnn,
        record_count=record_count,
        minimum_records=minimum_records,
    )
    analytical_metrics = _metric_summary(
        targets,
        analytical,
        record_count=record_count,
        minimum_records=minimum_records,
    )
    rmse_delta = cnn_metrics["rmse_v"] - analytical_metrics["rmse_v"]
    if abs(rmse_delta) <= rmse_equality_tolerance_v:
        comparison = "equal"
    elif rmse_delta < 0.0:
        comparison = "CNN better"
    else:
        comparison = "CNN worse"
    return {
        "record_count": record_count,
        "minimum_record_count": int(minimum_records),
        "below_minimum_record_count": record_count < minimum_records,
        "cnn": cnn_metrics,
        "analytical": analytical_metrics,
        "cnn_minus_analytical_rmse_v": float(rmse_delta),
        "rmse_equality_tolerance_v": float(rmse_equality_tolerance_v),
        "comparison_by_pooled_rmse": comparison,
    }


def grouped_test_metrics(
    *,
    training_conditions: np.ndarray,
    test_conditions: np.ndarray,
    targets_v: np.ndarray,
    cnn_predictions_v: np.ndarray,
    analytical_predictions_v: np.ndarray,
    feature_names: tuple[str, ...],
    minimum_records: int = MINIMUM_GROUP_RECORDS,
    modulation_band_count: int = 10,
) -> dict[str, Any]:
    """Compute test-only condition groups using boundaries/categories from train."""
    training = np.asarray(training_conditions, dtype=np.float64)
    test = np.asarray(test_conditions, dtype=np.float64)
    names = tuple(feature_names)
    if training.ndim != 2 or test.ndim != 2:
        raise ValueError("Operating conditions must be two-dimensional.")
    if training.shape[1] != len(names) or test.shape[1] != len(names):
        raise ValueError("Condition feature dimensions must match feature_names.")
    if not np.isfinite(training).all() or not np.isfinite(test).all():
        raise ValueError("Operating conditions must contain only finite values.")
    targets = np.asarray(targets_v)
    if targets.shape[0] != test.shape[0]:
        raise ValueError("Test waveform and condition record counts do not match.")

    required_features = ("modulationIndex", "loadResistance", "loadInductance")
    missing = [name for name in required_features if name not in names]
    if missing:
        raise ValueError(f"Required condition features are missing: {missing}.")
    columns = {name: names.index(name) for name in required_features}
    edges = training_modulation_edges(
        training[:, columns["modulationIndex"]],
        band_count=modulation_band_count,
    )
    training_bands = modulation_band_indices(
        training[:, columns["modulationIndex"]], edges
    )
    test_bands = modulation_band_indices(test[:, columns["modulationIndex"]], edges)
    groups: list[dict[str, Any]] = []

    def append_group(
        category: str,
        label: str,
        mask: np.ndarray,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        count = int(np.count_nonzero(mask))
        if count == 0:
            return
        comparison = compare_group(
            targets[mask],
            np.asarray(cnn_predictions_v)[mask],
            np.asarray(analytical_predictions_v)[mask],
            minimum_records=minimum_records,
        )
        groups.append(
            {
                "category": category,
                "group": label,
                **(details or {}),
                **comparison,
            }
        )

    modulation_edges = [
        {
            "band": index + 1,
            "lower_inclusive": float(lower),
            "upper_exclusive": (
                float(upper) if index < len(edges) - 2 else None
            ),
            "upper_inclusive": (
                None if index < len(edges) - 2 else float(upper)
            ),
        }
        for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:]))
    ]
    for band in range(edges.size - 1):
        mask = test_bands == band
        append_group(
            "modulation_band",
            f"training_quantile_band_{band + 1:02d}",
            mask,
            details=modulation_edges[band],
        )

    training_resistance = np.unique(training[:, columns["loadResistance"]])
    training_inductance = np.unique(training[:, columns["loadInductance"]])
    for feature in ("loadResistance", "loadInductance"):
        column = columns[feature]
        for value in np.unique(training[:, column]):
            mask = np.isclose(test[:, column], value, rtol=0.0, atol=1e-8)
            append_group(
                feature,
                f"{value:g}",
                mask,
                details={"value": float(value)},
            )

    train_joint = set(
        zip(
            training_bands.tolist(),
            training[:, columns["loadResistance"]].tolist(),
            training[:, columns["loadInductance"]].tolist(),
        )
    )
    test_joint_keys = list(
        zip(
            test_bands.tolist(),
            test[:, columns["loadResistance"]].tolist(),
            test[:, columns["loadInductance"]].tolist(),
        )
    )
    heldout_joint_groups: list[dict[str, Any]] = []
    for key in sorted(set(test_joint_keys) - train_joint):
        mask = np.asarray([candidate == key for candidate in test_joint_keys])
        band, resistance, inductance = key
        comparison = compare_group(
            targets[mask],
            np.asarray(cnn_predictions_v)[mask],
            np.asarray(analytical_predictions_v)[mask],
            minimum_records=minimum_records,
        )
        heldout_joint_groups.append(
            {
                "group": (
                    f"band_{band + 1:02d}_R_{resistance:g}_L_{inductance:g}"
                ),
                "modulation_band": int(band + 1),
                "load_resistance_ohm": float(resistance),
                "load_inductance_h": float(inductance),
                **comparison,
            }
        )
    groups.extend(
        {**entry, "category": "held_out_joint_combination"}
        for entry in heldout_joint_groups
    )
    return {
        "minimum_group_records": int(minimum_records),
        "modulation_band_definition": (
            f"{modulation_band_count} quantile bands derived from training modulation "
            "values only; duplicate quantile edges are removed."
        ),
        "modulation_band_edges": modulation_edges,
        "joint_combination_definition": (
            "A training-derived modulation band × resistance × inductance combination "
            "absent from training records."
        ),
        "held_out_joint_combination_count": len(heldout_joint_groups),
        "held_out_joint_combination_record_count": int(
            sum(group["record_count"] for group in heldout_joint_groups)
        ),
        "groups": groups,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_grouped_test_evaluation(
    *,
    dataset_path: Path = DEFAULT_DATASET,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    output_path: Path,
    report_path: Path,
    batch_size: int = 64,
    minimum_records: int = MINIMUM_GROUP_RECORDS,
    device: str = "cpu",
) -> dict[str, Any]:
    """Load existing data/checkpoint, infer test split once, and write fresh outputs."""
    dataset_path = dataset_path.resolve()
    checkpoint_path = checkpoint_path.resolve()
    output_path = output_path.resolve()
    report_path = report_path.resolve()
    if output_path.exists() or report_path.exists():
        raise FileExistsError("Refusing to overwrite an existing grouped evaluation output.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if minimum_records < 1:
        raise ValueError("minimum_records must be positive.")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    tables = load_dataset(dataset_path)
    prepared = preprocess_dataset(tables)
    checkpoint_digest_before = _sha256(checkpoint_path)
    csv_paths = sorted(tables.root.glob("*.csv"))
    dataset_digests_before = {path.name: _sha256(path) for path in csv_paths}
    device_obj = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device_obj, weights_only=True)
    normalization = assert_checkpoint_compatible(checkpoint, prepared)
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("model_config"), dict):
        raise ValueError("Checkpoint does not contain its model configuration.")
    model = create_model(metadata["model_config"]).to(device_obj)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    test_indices = prepared.indices("testing")
    train_indices = prepared.indices("training")
    inference = evaluate_cnn_test_set(
        model,
        prepared,
        normalization=normalization,
        device=device_obj,
        batch_size=batch_size,
    )
    expected_ids = prepared.sample_ids[test_indices]
    if not np.array_equal(inference.sample_ids, expected_ids):
        raise RuntimeError("CNN predictions are not aligned to the declared test split.")

    operating = np.asarray(
        prepared.operating_conditions.astype(np.float64)
        * np.asarray(prepared.stats.operating_std, dtype=np.float64)
        + np.asarray(prepared.stats.operating_mean, dtype=np.float64),
        dtype=np.float64,
    )
    test_conditions = operating[test_indices]
    analytical = analytical_reference_waveforms(
        test_conditions,
        prepared.operating_feature_names,
    )
    grouped = grouped_test_metrics(
        training_conditions=operating[train_indices],
        test_conditions=test_conditions,
        targets_v=inference.targets_v,
        cnn_predictions_v=inference.predictions_v,
        analytical_predictions_v=analytical,
        feature_names=prepared.operating_feature_names,
        minimum_records=minimum_records,
    )
    checkpoint_digest_after = _sha256(checkpoint_path)
    dataset_digests_after = {path.name: _sha256(path) for path in csv_paths}
    if checkpoint_digest_after != checkpoint_digest_before:
        raise RuntimeError("Checkpoint changed during inference; evaluation must be read-only.")
    if dataset_digests_after != dataset_digests_before:
        raise RuntimeError("Dataset files changed during evaluation; evaluation must be read-only.")

    report: dict[str, Any] = {
        "evaluation": "condition-grouped held-out test evaluation; inference only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(tables.root.resolve()),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_digest_before,
        "checkpoint_loaded_strictly": True,
        "checkpoint_modified": False,
        "training_or_fine_tuning_run": False,
        "dataset_csv_sha256": dataset_digests_before,
        "dataset_modified": False,
        "split_policy": (
            "CNN and analytical metrics use the same complete testing records. "
            "Training split is used only to define modulation quantiles and observed "
            "condition combinations. Testing records are not used for tuning."
        ),
        "split_record_counts": {
            "training": int(train_indices.size),
            "testing": int(test_indices.size),
        },
        "test_sample_ids": [int(sample_id) for sample_id in inference.sample_ids],
        "waveform_points_per_record": int(inference.targets_v.shape[1]),
        "batch_size": batch_size,
        "device": str(device_obj),
        "comparison_metric": "pooled-point RMSE on each group's identical target records",
        "comparison_tolerance_v": RMSE_EQUALITY_TOLERANCE_V,
        "groups": grouped,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown = render_grouped_report(report)
    report_path.write_text(markdown, encoding="utf-8")
    return report


def render_grouped_report(report: dict[str, Any]) -> str:
    groups = report["groups"]
    rows = []
    for group in groups["groups"]:
        cnn = group["cnn"]
        analytical = group["analytical"]
        low_count = " **LOW N**" if group["below_minimum_record_count"] else ""
        rows.append(
            "| {category} | {name} | {n}{low} | {cmae:.6g} | {crmse:.6g} | "
            "{cr2} | {ccorr} | {amae:.6g} | {armse:.6g} | {ar2} | {acorr} | {result} |".format(
                category=group["category"],
                name=group["group"],
                n=group["record_count"],
                low=low_count,
                cmae=cnn["mae_v"],
                crmse=cnn["rmse_v"],
                cr2="undefined" if cnn["r_squared"] is None else f"{cnn['r_squared']:.6g}",
                ccorr="undefined" if cnn["correlation"] is None else f"{cnn['correlation']:.6g}",
                amae=analytical["mae_v"],
                armse=analytical["rmse_v"],
                ar2=(
                    "undefined"
                    if analytical["r_squared"] is None
                    else f"{analytical['r_squared']:.6g}"
                ),
                acorr=(
                    "undefined"
                    if analytical["correlation"] is None
                    else f"{analytical['correlation']:.6g}"
                ),
                result=group["comparison_by_pooled_rmse"],
            )
        )
    table = "\n".join(rows) if rows else "| — | No evaluable groups | — | — | — | — | — | — | — | — | — | — |"
    heldout_count = groups["held_out_joint_combination_count"]
    heldout_summary = (
        f"{heldout_count} unseen training-band × R × L combinations "
        f"({groups['held_out_joint_combination_record_count']} test records)."
        if heldout_count
        else "No held-out training-band × R × L combinations occurred in the test split."
    )
    return f"""# Condition-Grouped CNN Test Evaluation

**Created (UTC):** {report['created_at_utc']}  
**Status:** inference-only; checkpoint and dataset were not modified.  
**Testing records:** {report['split_record_counts']['testing']}; **waveform points/record:** {report['waveform_points_per_record']}  
**Checkpoint SHA-256:** `{report['checkpoint_sha256']}`

## Method

The existing checkpoint was loaded strictly and evaluated once on the complete held-out testing split. It was not retrained or tuned. CNN and analytical-reference metrics below are computed against the same target records. Group boundaries for modulation are training-only quantiles; R/L categories and the training-joint-combination set are derived from training data. The test split is used only for evaluation.

Groups with fewer than **{groups['minimum_group_records']} records** are marked **LOW N**; their estimates are descriptive and less stable. Better/equal/worse is determined from pooled RMSE using an equality tolerance of {report['comparison_tolerance_v']:.1e} V. Metrics (MAE, RMSE, R², correlation) pool waveform points in each group; R² or correlation may be undefined for constant data.

## Results

| Group type | Group | Records | CNN MAE (V) | CNN RMSE (V) | CNN R² | CNN corr. | Analytical MAE (V) | Analytical RMSE (V) | Analytical R² | Analytical corr. | CNN vs analytical |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
{table}

## Held-out joint combinations

{heldout_summary}

## Interpretation

These are prediction-error comparisons against the dataset target, not inverter-output THD comparisons, switch-control validation, hardware measurements, or evidence of physical performance. No optimizer or THD calculations are part of this evaluation. Any conclusion is computed from grouped metric values above; a worse CNN result is reported as **no improvement** over the analytical reference for that group.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "cnn_grouped_test_eval_20261124.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "docs" / "CNN_GROUPED_TEST_EVALUATION.md",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--minimum-records", type=int, default=MINIMUM_GROUP_RECORDS)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = run_grouped_test_evaluation(
        dataset_path=args.dataset,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        report_path=args.report,
        batch_size=args.batch_size,
        minimum_records=args.minimum_records,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "report": str(args.report.resolve()),
                "test_records": report["split_record_counts"]["testing"],
                "held_out_joint_combinations": report["groups"][
                    "held_out_joint_combination_count"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
