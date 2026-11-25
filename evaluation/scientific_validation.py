"""Reproducible, inference-only validation for the saved inverter CNN."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import hashlib
import platform
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluation.cnn_test_evaluator import evaluate_cnn_test_set
from evaluation.event_based_thd import evaluate_event_based_thd
from evaluation.live_simulation import (
    operating_condition_support_warnings,
    run_live_simulation,
)
from evaluation.metrics import calculate_metrics, mae, rmse, r_squared
from models.model_factory import create_model
from multilevel.inverter_simulator import (
    FUNDAMENTAL_FREQUENCY_HZ,
    SAMPLING_FREQUENCY_HZ,
    SAMPLES_PER_CYCLE,
)
from preprocessing.data_loader import load_dataset
from preprocessing.preprocess import NormalizationStats, preprocess_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "raw" / "Seven_Level_Inverter_Project_extracted"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "colab_runs"
    / "baseline_20261120_080836"
    / "checkpoints"
    / "best.pt"
)
PSO_RESULT_DIRECTORY = PROJECT_ROOT / "results" / "circular_phase_pso_20261120_183109"
_EXPECTED_SPLIT_COUNTS = {"training": 7000, "validation": 1500, "testing": 1500}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_operating_conditions(prepared) -> np.ndarray:
    means = np.asarray(prepared.stats.operating_mean, dtype=np.float64)
    scales = np.asarray(prepared.stats.operating_std, dtype=np.float64)
    return prepared.operating_conditions.astype(np.float64) * scales + means


def _exact_train_test_input_duplicates(prepared) -> int:
    training = prepared.indices("training")
    testing = prepared.indices("testing")
    inputs = np.concatenate(
        (
            prepared.input_voltage,
            prepared.load_current,
            prepared.operating_conditions,
        ),
        axis=1,
    )
    training_hashes = {
        hashlib.sha256(np.ascontiguousarray(inputs[index]).view(np.uint8)).digest()
        for index in training
    }
    return sum(
        hashlib.sha256(np.ascontiguousarray(inputs[index]).view(np.uint8)).digest()
        in training_hashes
        for index in testing
    )


def assert_checkpoint_compatible(checkpoint: dict[str, Any], prepared) -> NormalizationStats:
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint metadata is missing.")
    normalization = metadata.get("normalization")
    model_config = metadata.get("model_config")
    feature_names = metadata.get("operating_feature_names")
    if not isinstance(normalization, dict) or not isinstance(model_config, dict):
        raise ValueError("Checkpoint model or normalization metadata is missing.")
    if list(feature_names or ()) != list(prepared.operating_feature_names):
        raise ValueError("Checkpoint operating feature names do not match the dataset.")
    stats = NormalizationStats(**normalization)
    current_stats = asdict(prepared.stats)
    saved_stats = asdict(stats)
    for key, saved_value in saved_stats.items():
        current_value = current_stats[key]
        if not np.allclose(saved_value, current_value, rtol=1e-5, atol=1e-7):
            raise ValueError(f"Checkpoint normalization field {key!r} does not match the dataset.")
    return stats


def analytical_reference_waveforms(
    operating_conditions: np.ndarray,
    feature_names: tuple[str, ...],
) -> np.ndarray:
    """Generate the project analytical target from modulation and phase."""
    conditions = np.asarray(operating_conditions, dtype=np.float64)
    if conditions.ndim != 2 or conditions.shape[1] != len(feature_names):
        raise ValueError("operating_conditions must align with feature_names.")
    if not np.isfinite(conditions).all():
        raise ValueError("operating_conditions must contain only finite values.")
    try:
        modulation_column = feature_names.index("modulationIndex")
        phase_column = feature_names.index("phaseAngle")
    except ValueError as exc:
        raise ValueError("Required modulationIndex and phaseAngle features are missing.") from exc

    time_s = np.arange(SAMPLES_PER_CYCLE, dtype=np.float64) / SAMPLING_FREQUENCY_HZ
    modulation = conditions[:, modulation_column]
    phase = conditions[:, phase_column]
    return (
        3.0
        * 100.0
        * modulation[:, None]
        * np.sin(
            2.0 * math.pi * FUNDAMENTAL_FREQUENCY_HZ * time_s[None, :]
            + phase[:, None]
        )
    )


def _record_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "mae_v": mae(targets, predictions),
        "rmse_v": rmse(targets, predictions),
        "r_squared": r_squared(targets, predictions),
        "maximum_absolute_error_v": float(np.max(np.abs(predictions - targets))),
    }


def _distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    percentiles = np.percentile(values, [0, 25, 50, 75, 90, 95, 99, 100])
    return {
        "minimum": float(percentiles[0]),
        "p25": float(percentiles[1]),
        "median": float(percentiles[2]),
        "mean": float(np.mean(values)),
        "p75": float(percentiles[3]),
        "p90": float(percentiles[4]),
        "p95": float(percentiles[5]),
        "p99": float(percentiles[6]),
        "maximum": float(percentiles[7]),
    }


def _condition_breakdowns(
    *,
    train_conditions: np.ndarray,
    test_conditions: np.ndarray,
    feature_names: tuple[str, ...],
    targets_v: np.ndarray,
    predictions_v: np.ndarray,
) -> list[dict[str, Any]]:
    error_by_record_mae = np.mean(np.abs(predictions_v - targets_v), axis=1)
    error_by_record_rmse = np.sqrt(np.mean(np.square(predictions_v - targets_v), axis=1))
    rows: list[dict[str, Any]] = []

    def append_group(feature: str, group: str, mask: np.ndarray, lower=None, upper=None) -> None:
        count = int(np.count_nonzero(mask))
        if count == 0:
            return
        metric_values = calculate_metrics(targets_v[mask], predictions_v[mask])
        rows.append(
            {
                "feature": feature,
                "group": group,
                "lower_bound": lower,
                "upper_bound": upper,
                "record_count": count,
                "pooled_point_mae_v": metric_values["mae"],
                "pooled_point_rmse_v": metric_values["rmse"],
                "global_r_squared": metric_values["r_squared"],
                "mean_record_mae_v": float(np.mean(error_by_record_mae[mask])),
                "median_record_mae_v": float(np.median(error_by_record_mae[mask])),
                "mean_record_rmse_v": float(np.mean(error_by_record_rmse[mask])),
                "median_record_rmse_v": float(np.median(error_by_record_rmse[mask])),
            }
        )

    modulation_index = feature_names.index("modulationIndex")
    train_modulation = train_conditions[:, modulation_index]
    test_modulation = test_conditions[:, modulation_index]
    boundaries = np.unique(np.quantile(train_modulation, np.linspace(0.0, 1.0, 11)))
    for bin_index, (lower, upper) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        if bin_index == len(boundaries) - 2:
            mask = (test_modulation >= lower) & (test_modulation <= upper)
        else:
            mask = (test_modulation >= lower) & (test_modulation < upper)
        append_group(
            "modulationIndex",
            f"training_decile_{bin_index + 1}",
            mask,
            float(lower),
            float(upper),
        )

    for feature in ("loadResistance", "loadInductance"):
        column = feature_names.index(feature)
        for value in np.unique(train_conditions[:, column]):
            mask = np.isclose(test_conditions[:, column], value, rtol=0.0, atol=1e-6)
            append_group(feature, f"{value:g}", mask, float(value), float(value))

    phase_column = feature_names.index("phaseAngle")
    phase_degrees = np.mod(np.degrees(test_conditions[:, phase_column]), 360.0)
    phase_bins = np.floor(phase_degrees / 45.0).astype(np.int64)
    phase_bins = np.minimum(phase_bins, 7)
    for bin_index in range(8):
        append_group(
            "phaseAngle",
            f"{bin_index * 45}–{(bin_index + 1) * 45} degrees",
            phase_bins == bin_index,
            float(bin_index * 45),
            float((bin_index + 1) * 45),
        )
    return rows


def _baseline_comparison(
    *,
    prepared,
    test_indices: np.ndarray,
    targets_v: np.ndarray,
    test_conditions: np.ndarray,
) -> dict[str, Any]:
    train_indices = prepared.indices("training")
    train_targets_v = np.asarray(
        prepared.stats.denormalize_target(prepared.target_voltage[train_indices]),
        dtype=np.float64,
    )
    test_count, points = targets_v.shape
    mean_training_waveform = np.mean(train_targets_v, axis=0, dtype=np.float64)
    mean_predictions = np.broadcast_to(mean_training_waveform, (test_count, points))

    analytical_predictions = analytical_reference_waveforms(
        test_conditions,
        prepared.operating_feature_names,
    )

    def summarize(predictions: np.ndarray) -> dict[str, Any]:
        micro = calculate_metrics(targets_v, predictions)
        per_record_mae = np.mean(np.abs(predictions - targets_v), axis=1)
        per_record_rmse = np.sqrt(np.mean(np.square(predictions - targets_v), axis=1))
        return {
            "pooled_point_metrics": micro,
            "per_record_mae_v": _distribution(per_record_mae),
            "per_record_rmse_v": _distribution(per_record_rmse),
            "definition": (
                "One prediction waveform per record; metrics use only training-derived "
                "mean target for the mean baseline, or known m/phase/frequency for the "
                "analytical baseline."
            ),
        }

    return {
        "mean_training_target_waveform": summarize(mean_predictions),
        "analytical_reference_from_operating_conditions": summarize(analytical_predictions),
        "analytical_baseline_formula": (
            "300*m*sin(2*pi*50*t + phaseAngle), t=n/20000, n=0..399"
        ),
        "training_mean_waveform_fit_on_training_records_only": int(train_indices.size),
    }


def _pso_validation_audit(prepared, validation_indices: np.ndarray, operating: np.ndarray) -> dict[str, Any]:
    result_path = PSO_RESULT_DIRECTORY / "final_results.json"
    run_summary_path = PSO_RESULT_DIRECTORY / "run_summary.csv"
    validation_file = PSO_RESULT_DIRECTORY / "validation_comparison.csv"
    if not all(path.is_file() for path in (result_path, run_summary_path, validation_file)):
        return {"status": "missing_saved_experiment_artifacts"}

    saved = json.loads(result_path.read_text(encoding="utf-8"))
    run_rows = list(csv.DictReader(run_summary_path.open(encoding="utf-8", newline="")))
    validation_rows = list(csv.DictReader(validation_file.open(encoding="utf-8", newline="")))
    ids = prepared.sample_ids[validation_indices]
    feature_names = prepared.operating_feature_names
    modulation = operating[validation_indices, feature_names.index("modulationIndex")]
    phase = operating[validation_indices, feature_names.index("phaseAngle")]
    baseline_values = np.empty(validation_indices.size, dtype=np.float64)
    selected_values = np.empty(validation_indices.size, dtype=np.float64)
    selected_offsets = (
        0.0,
        float(saved["selected_phases_degrees"][0]),
        float(saved["selected_phases_degrees"][1]),
    )
    for index, (m_value, phase_value) in enumerate(zip(modulation, phase)):
        baseline_values[index] = evaluate_event_based_thd(
            float(m_value),
            float(phase_value),
            carrier_phase_offsets_degrees=(0.0, 60.0, 120.0),
        ).total_thd_percent
        selected_values[index] = evaluate_event_based_thd(
            float(m_value),
            float(phase_value),
            carrier_phase_offsets_degrees=selected_offsets,
        ).total_thd_percent

    differences = selected_values - baseline_values
    rng = np.random.default_rng(20261120)
    bootstrap_means = np.empty(10_000, dtype=np.float64)
    for bootstrap_index in range(bootstrap_means.size):
        sample = rng.integers(0, differences.size, size=differences.size)
        bootstrap_means[bootstrap_index] = float(np.mean(differences[sample]))
    ci_low, ci_high = np.percentile(bootstrap_means, [2.5, 97.5])
    return {
        "status": "recomputed_from_saved_candidate_on_validation_only",
        "saved_experiment_result": saved,
        "saved_validation_comparison": validation_rows,
        "run_count": len(run_rows),
        "all_runs_feasible": all(row.get("feasible", "").lower() == "true" for row in run_rows),
        "validation_paired_record_count": int(differences.size),
        "validation_baseline_mean_thd_percent_recomputed": float(np.mean(baseline_values)),
        "validation_selected_mean_thd_percent_recomputed": float(np.mean(selected_values)),
        "validation_selected_minus_baseline_mean_pp": float(np.mean(differences)),
        "validation_paired_delta_median_pp": float(np.median(differences)),
        "validation_paired_delta_p95_pp": float(np.percentile(differences, 95)),
        "validation_paired_delta_p05_pp": float(np.percentile(differences, 5)),
        "validation_paired_bootstrap_95_percent_ci_pp": [float(ci_low), float(ci_high)],
        "bootstrap_replicates": int(bootstrap_means.size),
        "bootstrap_seed": 20261120,
        "training_subset_mean_thd_percent": saved["training_subset_mean_thd_percent"],
        "training_subset_baseline_mean_thd_percent": saved[
            "training_subset_baseline_mean_thd_percent"
        ],
        "training_subset_improvement_pp": (
            saved["training_subset_baseline_mean_thd_percent"]
            - saved["training_subset_mean_thd_percent"]
        ),
        "test_split_used_for_pso": False,
    }


def run_validation(
    *,
    dataset_path: Path = DEFAULT_DATASET,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    output_parent: Path = PROJECT_ROOT / "results",
    batch_size: int = 64,
) -> Path:
    """Run the full held-out evaluation and write new, non-overwriting outputs."""
    dataset_path = dataset_path.resolve()
    checkpoint_path = checkpoint_path.resolve()
    output_parent = output_parent.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    tables = load_dataset(dataset_path)
    prepared = preprocess_dataset(tables)
    observed_counts = {
        split: int(np.count_nonzero(prepared.splits == split))
        for split in _EXPECTED_SPLIT_COUNTS
    }
    if observed_counts != _EXPECTED_SPLIT_COUNTS:
        raise ValueError(f"Dataset split sizes differ from the expected project counts: {observed_counts}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    stats = assert_checkpoint_compatible(checkpoint, prepared)
    model_config = checkpoint["metadata"]["model_config"]
    model = create_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    test_indices = prepared.indices("testing")
    test_evaluation = evaluate_cnn_test_set(
        model,
        prepared,
        normalization=stats,
        device=device,
        batch_size=batch_size,
    )
    if not np.array_equal(test_evaluation.sample_ids, prepared.sample_ids[test_indices]):
        raise RuntimeError("Held-out evaluation sample IDs do not match the explicit testing split.")

    operating = raw_operating_conditions(prepared)
    test_conditions = operating[test_indices]
    condition_breakdowns = _condition_breakdowns(
        train_conditions=operating[prepared.indices("training")],
        test_conditions=test_conditions,
        feature_names=prepared.operating_feature_names,
        targets_v=test_evaluation.targets_v,
        predictions_v=test_evaluation.predictions_v,
    )
    baselines = _baseline_comparison(
        prepared=prepared,
        test_indices=test_indices,
        targets_v=test_evaluation.targets_v,
        test_conditions=test_conditions,
    )

    per_record = [
        {
            **asdict(record),
            **{
                feature: float(test_conditions[row_index, feature_index])
                for feature_index, feature in enumerate(prepared.operating_feature_names)
            },
        }
        for row_index, record in enumerate(test_evaluation.per_record_metrics)
    ]
    mae_values = np.asarray([record["mae"] for record in per_record], dtype=np.float64)
    rmse_values = np.asarray([record["rmse"] for record in per_record], dtype=np.float64)
    worst_records = sorted(per_record, key=lambda record: record["rmse"], reverse=True)[:10]

    training_conditions = operating[prepared.indices("training")]
    feature_support: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(prepared.operating_feature_names):
        values = training_conditions[:, index]
        feature_support[name] = {
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "p01": float(np.percentile(values, 1)),
            "median": float(np.median(values)),
            "p99": float(np.percentile(values, 99)),
            "unique_count": int(np.unique(np.round(values, 8)).size),
            "observed_values": (
                np.unique(np.round(values, 8)).tolist()
                if name in {"loadResistance", "loadInductance"}
                else None
            ),
        }

    live_conditions = (
        (0.75, 30.0, 0.03, 0.0),
        (0.55, 10.0, 0.01, 90.0),
        (0.95, 40.0, 0.05, 270.0),
        (0.80, 20.0, 0.02, 180.0),
    )
    live_results: list[dict[str, Any]] = []
    training_conditions = operating[prepared.indices("training")]
    for modulation, resistance, inductance, phase_degrees in live_conditions:
        live = run_live_simulation(
            model,
            stats,
            modulation_index=modulation,
            load_resistance_ohm=resistance,
            load_inductance_h=inductance,
            phase_angle_degrees=phase_degrees,
            device=device,
            carrier_phase_offsets_degrees=(0.0, 60.0, 120.0),
        )
        live_results.append(
            {
                "modulation_index": modulation,
                "load_resistance_ohm": resistance,
                "load_inductance_h": inductance,
                "phase_angle_degrees": phase_degrees,
                "marginal_training_support_warnings": list(
                    operating_condition_support_warnings(
                        np.asarray(
                            [
                                modulation,
                                resistance,
                                inductance,
                                math.radians(phase_degrees),
                            ],
                            dtype=np.float64,
                        ),
                        training_conditions,
                        prepared.operating_feature_names,
                    )
                ),
                "cnn_target_metrics": live.cnn_metrics,
                "parseval_full_spectrum_thd_percent": live.harmonic_result.total_thd_percent,
                "finite_band_thd_2_to_100_percent": live.harmonic_result.thd_percent,
                "sampled_20khz_thd_2_to_100_percent": live.harmonic_result.sampled_thd_percent,
                "fundamental_peak_amplitude_v": live.harmonic_result.fundamental_peak_amplitude_v,
                "target_fundamental_peak_amplitude_v": live.harmonic_result.target_fundamental_peak_amplitude_v,
                "fundamental_phasor_tracking_error_percent": (
                    live.harmonic_result.fundamental_phasor_error_percent
                ),
                "target_waveform_mae_v": float(live.cnn_metrics["mae"]),
            }
        )

    history = checkpoint.get("history", [])
    checkpoint_epoch = int(checkpoint["epoch"])
    best_embedded_history = min(history, key=lambda row: row["validation_loss"]) if history else None
    history_path = checkpoint_path.parent.parent / "results" / "training_history.csv"
    external_history = None
    if history_path.is_file():
        import pandas as pd

        history_frame = pd.read_csv(history_path)
        row = history_frame.loc[history_frame["validation_loss"].idxmin()]
        external_history = {
            "rows": int(len(history_frame)),
            "best_epoch": int(row["epoch"]),
            "best_validation_loss": float(row["validation_loss"]),
            "last_epoch": int(history_frame["epoch"].iloc[-1]),
        }

    pso = _pso_validation_audit(prepared, prepared.indices("validation"), operating)
    source_files = sorted(dataset_path.glob("*.csv"))
    artifact_directory = output_parent / (
        "scientific_validation_" + datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    )
    artifact_directory.mkdir(parents=True, exist_ok=False)

    summary = {
        "audit_name": "Scientific validation audit: trained inverter CNN",
        "created_at_local": datetime.now().astimezone().isoformat(),
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "evaluation_policy": {
            "cnn_weights_modified": False,
            "training_or_fine_tuning_run": False,
            "test_used_for_model_selection_or_tuning": False,
            "records_evaluated": "Only rows labelled testing in Sample_Splits.csv for held-out CNN metrics.",
            "test_record_count": test_evaluation.record_count,
            "inference_batch_size": test_evaluation.batch_size,
            "device": str(device),
        },
        "provenance": {
            "dataset_path": str(dataset_path),
            "dataset_csv_sha256": {file.name: _sha256(file) for file in source_files},
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "checkpoint_epoch_zero_based": checkpoint_epoch,
            "checkpoint_epoch_one_based": checkpoint_epoch + 1,
            "checkpoint_best_validation_loss": float(checkpoint["best_validation_loss"]),
            "checkpoint_embedded_history_rows": len(history),
            "checkpoint_embedded_history_best_row": best_embedded_history,
            "external_history": external_history,
            "train_config": checkpoint.get("train_config"),
            "model_config": model_config,
            "model_parameter_count": model.parameter_count(),
            "state_dict_loaded_strictly": True,
            "normalization_matches_current_training_split": True,
            "operating_feature_names": list(prepared.operating_feature_names),
            "dataset_hash_stored_in_checkpoint": False,
            "dataset_identity_assessment": (
                "The run path, saved metadata and current training-only normalization agree, "
                "but the checkpoint contains no dataset path, archive hash or CSV hashes; "
                "the exact training archive cannot be cryptographically established."
            ),
        },
        "dataset": {
            "sample_count": int(len(prepared.sample_ids)),
            "split_counts": observed_counts,
            "disjoint_sample_ids": True,
            "exact_training_testing_input_record_duplicates": _exact_train_test_input_duplicates(
                prepared
            ),
            "input_duplicate_definition": (
                "Exact equality of the prepared input-voltage, load-current and normalized "
                "operating-condition rows across training and testing."
            ),
            "training_operating_condition_support": feature_support,
        },
        "checkpoint_selection": {
            "selection_metric": "Minimum validation loss (MAE objective in normalized target space).",
            "checkpoint_epoch_zero_based": checkpoint_epoch,
            "checkpoint_epoch_one_based": checkpoint_epoch + 1,
            "best_checkpoint_validation_loss": float(checkpoint["best_validation_loss"]),
            "trainer_fit_accepts_training_and_validation_only": True,
            "test_loader_not_accepted_by_trainer_fit": True,
        },
        "test_evaluation": {
            "records_evaluated": test_evaluation.record_count,
            "waveform_points_per_record": test_evaluation.waveform_points,
            "pooled_point_metrics": asdict(test_evaluation.micro_metrics),
            "mean_per_record_mae_v": float(np.mean(mae_values)),
            "median_per_record_mae_v": float(np.median(mae_values)),
            "mean_per_record_rmse_v": float(np.mean(rmse_values)),
            "median_per_record_rmse_v": float(np.median(rmse_values)),
            "per_record_mae_distribution_v": _distribution(mae_values),
            "per_record_rmse_distribution_v": _distribution(rmse_values),
            "pooled_point_prediction_error_distribution_v": asdict(
                test_evaluation.error_distribution
            ),
            "worst_10_records_by_rmse": worst_records,
            "metric_aggregation_definitions": {
                "pooled_point_metrics": (
                    "Computed over all 1,500 x 400 = 600,000 actual and predicted voltage points."
                ),
                "mean_median_per_record": (
                    "Computed from each record's own 400-point metric; not the pooled-point metric."
                ),
                "global_r_squared": (
                    "1 - pooled sum squared errors / pooled sum squared deviations from pooled target mean."
                ),
                "global_correlation": "Pearson correlation across all concatenated waveform points.",
            },
        },
        "baseline_comparison": baselines,
        "generalization": {
            "condition_breakdowns": condition_breakdowns,
            "conditioning_note": (
                "Modulation bins are deciles defined from training values; R/L are the "
                "training-observed categories; phase uses fixed circular 45-degree bins. "
                "These are descriptive test-set slices, not tuning."
            ),
            "live_demo_operating_points": live_results,
            "live_input_distribution_assessment": (
                "Compare each candidate input to training marginal support in the app. "
                "Marginal support checks do not prove the joint operating point is in-distribution."
            ),
        },
        "pso_audit": pso,
        "outputs": {
            "per_record_csv": "cnn_test_per_record.csv",
            "condition_breakdown_csv": "cnn_test_condition_breakdown.csv",
            "summary_json": "scientific_validation_summary.json",
        },
    }

    csv_path = artifact_directory / "cnn_test_per_record.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(per_record[0]))
        writer.writeheader()
        writer.writerows(per_record)
    group_path = artifact_directory / "cnn_test_condition_breakdown.csv"
    with group_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(condition_breakdowns[0]))
        writer.writeheader()
        writer.writerows(condition_breakdowns)
    summary_path = artifact_directory / "scientific_validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return artifact_directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-parent", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    output_directory = run_validation(
        dataset_path=args.dataset,
        checkpoint_path=args.checkpoint,
        output_parent=args.output_parent,
        batch_size=args.batch_size,
    )
    print(f"Scientific validation outputs: {output_directory}")


if __name__ == "__main__":
    main()


__all__ = [
    "analytical_reference_waveforms",
    "assert_checkpoint_compatible",
    "raw_operating_conditions",
    "run_validation",
]
