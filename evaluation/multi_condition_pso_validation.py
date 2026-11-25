"""Validate a training-selected carrier-phase PSO candidate across validation conditions."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
from typing import Any, Callable, Iterable

import numpy as np

from evaluation.event_based_thd import EventBasedHarmonicResult, evaluate_event_based_thd
from optimization.circular_phase_pso import BASELINE_PHASES_DEGREES
from optimization.phase_pso_experiment import load_split_labels


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "raw" / "Seven_Level_Inverter_Project_extracted"
DEFAULT_SOURCE_EXPERIMENT = PROJECT_ROOT / "results" / "circular_phase_pso_20261120_183109"
_UNCHANGED_THD_TOLERANCE_PP = 1e-10


@dataclass(frozen=True)
class ValidationOperatingCondition:
    sample_id: int
    modulation_index: float
    load_resistance_ohm: float
    load_inductance_h: float
    phase_angle_rad: float


Evaluator = Callable[..., EventBasedHarmonicResult]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no CSV header.")
        return list(reader.fieldnames), list(reader)


def _column_map(fieldnames: Iterable[str], path: Path) -> dict[str, str]:
    names = tuple(fieldnames)
    result = {name.casefold(): name for name in names}
    if len(result) != len(names):
        raise ValueError(f"{path} contains duplicate case-insensitive column names.")
    return result


def _load_condition_records(
    path: Path,
    split_labels: dict[int, str],
    *,
    split: str,
    sample_ids: set[int] | None = None,
) -> tuple[ValidationOperatingCondition, ...]:
    if split not in {"training", "validation"}:
        raise ValueError("Only training and validation operating records may be loaded.")
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        columns = _column_map(reader.fieldnames or (), path)
        required = (
            "sampleid",
            "modulationindex",
            "loadresistance",
            "loadinductance",
            "phaseangle",
        )
        if any(name not in columns for name in required):
            raise ValueError(f"{path} is missing one of the required columns: {required}.")
        records: list[ValidationOperatingCondition] = []
        seen: set[int] = set()
        for row in reader:
            sample_id = int(row[columns["sampleid"]])
            if split_labels.get(sample_id) != split:
                continue
            if sample_ids is not None and sample_id not in sample_ids:
                continue
            if sample_id in seen:
                raise ValueError(f"Duplicate SampleID {sample_id} in {path} for {split}.")
            seen.add(sample_id)
            record = ValidationOperatingCondition(
                sample_id=sample_id,
                modulation_index=float(row[columns["modulationindex"]]),
                load_resistance_ohm=float(row[columns["loadresistance"]]),
                load_inductance_h=float(row[columns["loadinductance"]]),
                phase_angle_rad=float(row[columns["phaseangle"]]),
            )
            if not all(
                math.isfinite(value)
                for value in (
                    record.modulation_index,
                    record.load_resistance_ohm,
                    record.load_inductance_h,
                    record.phase_angle_rad,
                )
            ):
                raise ValueError(f"Non-finite operating condition at SampleID {sample_id}.")
            records.append(record)
    expected = (
        sample_ids
        if sample_ids is not None
        else {sample_id for sample_id, label in split_labels.items() if label == split}
    )
    if seen != expected:
        raise ValueError(
            f"{path} {split} ID mismatch: missing={len(expected - seen)}, "
            f"unexpected={len(seen - expected)}."
        )
    return tuple(sorted(records, key=lambda record: record.sample_id))


def _load_training_selected_ids(path: Path, split_labels: dict[int, str]) -> set[int]:
    fieldnames, rows = _read_csv_rows(path)
    columns = _column_map(fieldnames, path)
    if "sampleid" not in columns:
        raise ValueError(f"{path} must contain SampleID.")
    ids = {int(row[columns["sampleid"]]) for row in rows}
    if len(ids) != len(rows):
        raise ValueError(f"{path} contains duplicate training SampleIDs.")
    non_training = [sample_id for sample_id in ids if split_labels.get(sample_id) != "training"]
    if non_training:
        raise ValueError(
            f"Saved PSO subset contains non-training SampleIDs: {non_training[:5]}."
        )
    if not ids:
        raise ValueError("The saved PSO training subset is empty.")
    return ids


def _load_source_experiment(experiment_dir: Path) -> dict[str, Any]:
    required = (
        "experiment_config.json",
        "final_results.json",
        "run_summary.csv",
        "training_subset.csv",
        "convergence_history.csv",
    )
    missing = [name for name in required if not (experiment_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Saved training PSO experiment is incomplete; missing: {missing}"
        )
    config = json.loads((experiment_dir / "experiment_config.json").read_text(encoding="utf-8"))
    final = json.loads((experiment_dir / "final_results.json").read_text(encoding="utf-8"))
    if tuple(config.get("baseline_carrier_offsets_degrees", ())) != BASELINE_PHASES_DEGREES:
        raise ValueError("Saved PSO experiment does not use the requested conventional baseline.")
    if config.get("validation_and_test_policy", "").find("test records never evaluated") < 0:
        raise ValueError("Saved PSO experiment does not document held-out test exclusion.")
    if final.get("test_records_evaluated") != 0:
        raise ValueError("Saved training PSO experiment reports that test records were evaluated.")
    if not final.get("all_runs_completed"):
        raise ValueError("Saved training PSO experiment did not complete all runs.")

    fieldnames, run_rows = _read_csv_rows(experiment_dir / "run_summary.csv")
    columns = _column_map(fieldnames, experiment_dir / "run_summary.csv")
    seed_column = columns.get("seed")
    if seed_column is None:
        raise ValueError("PSO run summary is missing its seed column.")
    seed_rows = [row for row in run_rows if int(row[seed_column]) == int(final["selected_seed"])]
    if len(seed_rows) != 1:
        raise ValueError("Selected PSO seed must identify exactly one completed training run.")
    selected_row = seed_rows[0]
    for key, final_value in (
        ("phase_cell2_deg", final["selected_phases_degrees"][0]),
        ("phase_cell3_deg", final["selected_phases_degrees"][1]),
    ):
        if not math.isclose(float(selected_row[columns[key.casefold()]]), float(final_value), abs_tol=1e-12):
            raise ValueError("Selected phase shifts do not match the saved training run.")
    return {
        "config": config,
        "final": final,
        "runs": run_rows,
        "selected_run": selected_row,
        "settings": config.get("pso_config", {}),
    }


def _evaluate_pair(
    records: tuple[ValidationOperatingCondition, ...],
    candidate_phases: tuple[float, float],
    *,
    evaluator: Evaluator = evaluate_event_based_thd,
) -> list[dict[str, Any]]:
    candidate_offsets = (0.0, *candidate_phases)
    rows: list[dict[str, Any]] = []
    for record in records:
        baseline = evaluator(
            record.modulation_index,
            record.phase_angle_rad,
            carrier_phase_offsets_degrees=BASELINE_PHASES_DEGREES,
        )
        candidate = evaluator(
            record.modulation_index,
            record.phase_angle_rad,
            carrier_phase_offsets_degrees=candidate_offsets,
        )
        for label, result in (("baseline", baseline), ("candidate", candidate)):
            numeric_values = (
                result.total_thd_percent,
                result.fundamental_peak_amplitude_v,
                result.target_fundamental_peak_amplitude_v,
                result.fundamental_phasor_error_percent,
            )
            if not np.isfinite(numeric_values).all():
                raise ValueError(
                    f"{label} evaluator returned non-finite metrics at "
                    f"SampleID {record.sample_id}."
                )
            if result.harmonic_orders.size != result.harmonic_peak_amplitudes_v.size:
                raise ValueError("Event evaluator harmonic orders and amplitudes do not align.")
            if not np.array_equal(result.harmonic_orders, np.arange(1, 101)):
                raise ValueError("Expected event harmonic amplitudes for orders 1 through 100.")
        if not math.isclose(
            baseline.target_fundamental_peak_amplitude_v,
            candidate.target_fundamental_peak_amplitude_v,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError("Baseline and candidate target fundamental amplitudes differ.")
        baseline_amplitude_error = (
            abs(
                baseline.fundamental_peak_amplitude_v
                - baseline.target_fundamental_peak_amplitude_v
            )
            / baseline.target_fundamental_peak_amplitude_v
            * 100.0
        )
        candidate_amplitude_error = (
            abs(
                candidate.fundamental_peak_amplitude_v
                - candidate.target_fundamental_peak_amplitude_v
            )
            / candidate.target_fundamental_peak_amplitude_v
            * 100.0
        )
        row: dict[str, Any] = {
            "sample_id": record.sample_id,
            "modulation_index": record.modulation_index,
            "load_resistance_ohm": record.load_resistance_ohm,
            "load_inductance_h": record.load_inductance_h,
            "phase_angle_rad": record.phase_angle_rad,
            "phase_angle_degrees": math.degrees(record.phase_angle_rad) % 360.0,
            "baseline_phases_degrees": list(BASELINE_PHASES_DEGREES),
            "candidate_phases_degrees": list(candidate_offsets),
            "baseline_full_spectrum_parseval_thd_percent": baseline.total_thd_percent,
            "candidate_full_spectrum_parseval_thd_percent": candidate.total_thd_percent,
            "thd_change_percentage_points": (
                candidate.total_thd_percent - baseline.total_thd_percent
            ),
            "relative_thd_change_percent": (
                (candidate.total_thd_percent / baseline.total_thd_percent - 1.0) * 100.0
                if baseline.total_thd_percent > 0.0
                else None
            ),
            "baseline_fundamental_peak_v": baseline.fundamental_peak_amplitude_v,
            "candidate_fundamental_peak_v": candidate.fundamental_peak_amplitude_v,
            "target_fundamental_peak_v": baseline.target_fundamental_peak_amplitude_v,
            "baseline_fundamental_amplitude_error_percent": baseline_amplitude_error,
            "candidate_fundamental_amplitude_error_percent": candidate_amplitude_error,
            "baseline_fundamental_phasor_tracking_error_percent": (
                baseline.fundamental_phasor_error_percent
            ),
            "candidate_fundamental_phasor_tracking_error_percent": (
                candidate.fundamental_phasor_error_percent
            ),
            "baseline_fundamental_phase_rad": baseline.fundamental_phase_rad,
            "candidate_fundamental_phase_rad": candidate.fundamental_phase_rad,
        }
        for order, amplitude in zip(
            baseline.harmonic_orders[1:],
            baseline.harmonic_peak_amplitudes_v[1:],
        ):
            row[f"baseline_harmonic_{int(order)}_peak_v"] = float(amplitude)
        for order, amplitude in zip(
            candidate.harmonic_orders[1:],
            candidate.harmonic_peak_amplitudes_v[1:],
        ):
            row[f"candidate_harmonic_{int(order)}_peak_v"] = float(amplitude)
        rows.append(row)
    return rows


def _statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = np.asarray(
        [row["baseline_full_spectrum_parseval_thd_percent"] for row in rows],
        dtype=np.float64,
    )
    candidate = np.asarray(
        [row["candidate_full_spectrum_parseval_thd_percent"] for row in rows],
        dtype=np.float64,
    )
    delta = candidate - baseline
    relative = np.asarray(
        [row["relative_thd_change_percent"] for row in rows],
        dtype=np.float64,
    )
    threshold = _UNCHANGED_THD_TOLERANCE_PP
    tracking_baseline = np.asarray(
        [row["baseline_fundamental_phasor_tracking_error_percent"] for row in rows]
    )
    tracking_candidate = np.asarray(
        [row["candidate_fundamental_phasor_tracking_error_percent"] for row in rows]
    )
    amplitude_baseline = np.asarray(
        [row["baseline_fundamental_amplitude_error_percent"] for row in rows]
    )
    amplitude_candidate = np.asarray(
        [row["candidate_fundamental_amplitude_error_percent"] for row in rows]
    )
    if not rows:
        raise ValueError("No validation operating conditions were evaluated.")
    return {
        "validation_condition_count": len(rows),
        "baseline_mean_full_spectrum_parseval_thd_percent": float(np.mean(baseline)),
        "candidate_mean_full_spectrum_parseval_thd_percent": float(np.mean(candidate)),
        "mean_thd_change_percentage_points": float(np.mean(delta)),
        "median_thd_change_percentage_points": float(np.median(delta)),
        "minimum_thd_change_percentage_points": float(np.min(delta)),
        "maximum_thd_change_percentage_points": float(np.max(delta)),
        "standard_deviation_thd_change_percentage_points_sample": float(np.std(delta, ddof=1))
        if len(delta) > 1
        else 0.0,
        "mean_relative_thd_change_percent": float(np.mean(relative)),
        "median_relative_thd_change_percent": float(np.median(relative)),
        "improved_condition_count": int(np.count_nonzero(delta < -threshold)),
        "unchanged_condition_count": int(np.count_nonzero(np.abs(delta) <= threshold)),
        "worsened_condition_count": int(np.count_nonzero(delta > threshold)),
        "unchanged_tolerance_percentage_points": threshold,
        "baseline_mean_fundamental_peak_v": float(
            np.mean([row["baseline_fundamental_peak_v"] for row in rows])
        ),
        "candidate_mean_fundamental_peak_v": float(
            np.mean([row["candidate_fundamental_peak_v"] for row in rows])
        ),
        "target_mean_fundamental_peak_v": float(
            np.mean([row["target_fundamental_peak_v"] for row in rows])
        ),
        "baseline_mean_fundamental_phasor_tracking_error_percent": float(
            np.mean(tracking_baseline)
        ),
        "candidate_mean_fundamental_phasor_tracking_error_percent": float(
            np.mean(tracking_candidate)
        ),
        "baseline_max_fundamental_phasor_tracking_error_percent": float(
            np.max(tracking_baseline)
        ),
        "candidate_max_fundamental_phasor_tracking_error_percent": float(
            np.max(tracking_candidate)
        ),
        "baseline_mean_fundamental_amplitude_error_percent": float(
            np.mean(amplitude_baseline)
        ),
        "candidate_mean_fundamental_amplitude_error_percent": float(
            np.mean(amplitude_candidate)
        ),
    }


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline = np.asarray(
        [row["baseline_full_spectrum_parseval_thd_percent"] for row in rows],
        dtype=np.float64,
    )
    candidate = np.asarray(
        [row["candidate_full_spectrum_parseval_thd_percent"] for row in rows],
        dtype=np.float64,
    )
    delta = candidate - baseline
    low = float(min(np.min(baseline), np.min(candidate)))
    high = float(max(np.max(baseline), np.max(candidate)))
    padding = max((high - low) * 0.05, 1e-9)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(baseline, candidate, s=12, alpha=0.55, color="#087eaa")
    axes[0].plot(
        [low - padding, high + padding],
        [low - padding, high + padding],
        linestyle="--",
        color="#777777",
        linewidth=1,
        label="No change",
    )
    axes[0].set_xlim(low - padding, high + padding)
    axes[0].set_ylim(low - padding, high + padding)
    axes[0].set_xlabel("Baseline full-spectrum Parseval THD (%)")
    axes[0].set_ylabel("Candidate full-spectrum Parseval THD (%)")
    axes[0].set_title("Paired validation conditions")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].hist(delta, bins=40, color="#087eaa", edgecolor="white", alpha=0.9)
    axes[1].axvline(0.0, color="#222222", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Candidate − baseline THD (percentage points)")
    axes[1].set_ylabel("Validation condition count")
    axes[1].set_title("Paired THD change distribution")
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle("Training-selected PSO phase candidate vs conventional carrier shifts")
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _aggregate_by_feature(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        value = row[key]
        groups.setdefault(f"{float(value):g}", []).append(row)
    aggregates: list[dict[str, Any]] = []
    for label, records in sorted(groups.items(), key=lambda item: float(item[0])):
        changes = np.asarray(
            [row["thd_change_percentage_points"] for row in records],
            dtype=np.float64,
        )
        aggregates.append(
            {
                "value": float(label),
                "condition_count": len(records),
                "baseline_mean_thd_percent": float(
                    np.mean(
                        [
                            row["baseline_full_spectrum_parseval_thd_percent"]
                            for row in records
                        ]
                    )
                ),
                "candidate_mean_thd_percent": float(
                    np.mean(
                        [
                            row["candidate_full_spectrum_parseval_thd_percent"]
                            for row in records
                        ]
                    )
                ),
                "mean_thd_change_percentage_points": float(np.mean(changes)),
            }
        )
    return aggregates


def _render_report(
    *,
    report_path: Path,
    output_directory: Path,
    source_experiment: Path,
    phase_results: dict[str, Any],
    summary: dict[str, Any],
    condition_rows: list[dict[str, Any]],
) -> None:
    relative_dir = output_directory.relative_to(PROJECT_ROOT).as_posix()
    phases = phase_results["candidate_carrier_offsets_degrees"]
    config = phase_results["pso_configuration"]
    stats = summary["statistics"]
    if stats["mean_thd_change_percentage_points"] < -_UNCHANGED_THD_TOLERANCE_PP:
        conclusion = (
            "On this fixed validation set, the training-selected candidate reduced mean "
            "full-spectrum Parseval THD. This supports a measured validation-set reduction "
            "for the tested conditions only; it is not evidence of hardware or general "
            "out-of-distribution performance."
        )
    elif stats["mean_thd_change_percentage_points"] > _UNCHANGED_THD_TOLERANCE_PP:
        conclusion = (
            "The training-selected candidate worsened mean full-spectrum Parseval THD on "
            "the fixed validation set. This experiment does not support a harmonic-reduction claim."
        )
    else:
        conclusion = (
            "The training-selected candidate made no material change in mean full-spectrum "
            "Parseval THD on the fixed validation set. This experiment does not support a "
            "harmonic-reduction claim."
        )
    metric_lines = [
        "| Metric | Value |",
        "|---|---:|",
        f"| Validation operating conditions | {stats['validation_condition_count']:,} |",
        f"| Mean baseline full-spectrum Parseval THD | {stats['baseline_mean_full_spectrum_parseval_thd_percent']:.9f}% |",
        f"| Mean candidate full-spectrum Parseval THD | {stats['candidate_mean_full_spectrum_parseval_thd_percent']:.9f}% |",
        f"| Mean candidate − baseline change | {stats['mean_thd_change_percentage_points']:+.9g} percentage points |",
        f"| Mean relative THD change | {stats['mean_relative_thd_change_percent']:+.9g}% |",
        f"| Median paired THD change | {stats['median_thd_change_percentage_points']:+.9g} percentage points |",
        f"| Minimum / maximum paired THD change | {stats['minimum_thd_change_percentage_points']:+.9g} / {stats['maximum_thd_change_percentage_points']:+.9g} percentage points |",
        f"| Sample standard deviation of paired change | {stats['standard_deviation_thd_change_percentage_points_sample']:.9g} percentage points |",
        f"| Improved / unchanged / worsened conditions | {stats['improved_condition_count']} / {stats['unchanged_condition_count']} / {stats['worsened_condition_count']} |",
        f"| Mean baseline / candidate fundamental peak | {stats['baseline_mean_fundamental_peak_v']:.9f} / {stats['candidate_mean_fundamental_peak_v']:.9f} V |",
        f"| Mean target fundamental peak | {stats['target_mean_fundamental_peak_v']:.9f} V |",
        f"| Mean baseline / candidate phasor tracking error | {stats['baseline_mean_fundamental_phasor_tracking_error_percent']:.9g}% / {stats['candidate_mean_fundamental_phasor_tracking_error_percent']:.9g}% |",
        f"| Mean baseline / candidate fundamental amplitude error | {stats['baseline_mean_fundamental_amplitude_error_percent']:.9g}% / {stats['candidate_mean_fundamental_amplitude_error_percent']:.9g}% |",
        f"| Maximum candidate phasor tracking error | {stats['candidate_max_fundamental_phasor_tracking_error_percent']:.9g}% (limit {config['tracking_tolerance_percent']}%) |",
    ]
    group_table = [
        "| Group | Conditions | Mean baseline THD (%) | Mean candidate THD (%) | Mean change (percentage points) |",
        "|---|---:|---:|---:|---:|",
    ]
    for feature_name, label, unit in (
        ("modulation_index_decile", "Modulation decile", ""),
        ("load_resistance", "Load resistance", " Ω"),
        ("load_inductance", "Load inductance", " H"),
        ("phase_angle_45_degree_bin", "Reference phase", ""),
    ):
        for group in summary["condition_groups"][feature_name]:
            if feature_name == "modulation_index_decile":
                units_value = (
                    f"{group['lower_bound']:.3f}–{group['upper_bound']:.3f}"
                )
            elif feature_name == "phase_angle_45_degree_bin":
                units_value = str(group["group"])
            else:
                units_value = f"{group['value']:g}{unit}"
            group_table.append(
                f"| {label} {units_value} | {group['condition_count']} | "
                f"{group['baseline_mean_thd_percent']:.9f} | "
                f"{group['candidate_mean_thd_percent']:.9f} | "
                f"{group['mean_thd_change_percentage_points']:+.9g} |"
            )
    sample_records = condition_rows[:8]
    condition_table = [
        "| SampleID | m | R (Ω) | L (H) | Phase (°) | Baseline THD (%) | Candidate THD (%) | Δ (pp) | Relative Δ (%) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sample_records:
        condition_table.append(
            f"| {row['sample_id']} | {row['modulation_index']:.5f} | "
            f"{row['load_resistance_ohm']:.5g} | {row['load_inductance_h']:.5g} | "
            f"{row['phase_angle_degrees']:.4f} | "
            f"{row['baseline_full_spectrum_parseval_thd_percent']:.9f} | "
            f"{row['candidate_full_spectrum_parseval_thd_percent']:.9f} | "
            f"{row['thd_change_percentage_points']:+.9g} | "
            f"{row['relative_thd_change_percent']:+.9g} |"
        )
    report = f"""# Multi-Condition PSO Validation

**Run:** {summary["created_at_local"]}  
**Validation conditions:** {stats["validation_condition_count"]:,} (the complete labeled validation split)  
**Source training-only PSO experiment:** `{source_experiment}`  
**Reproducible outputs:** [`{relative_dir}/`](./{relative_dir}/)

## Objective and scientific limitation

This experiment compares the conventional carrier phases `[0°, 60°, 120°]` with one globally selected training-only circular-PSO candidate on a fixed validation population. It does not tune on validation or use the held-out testing split.

The existing event-based voltage-THD objective takes modulation index, reference phase, and carrier phase offsets. The ideal output-voltage switching model is independent of load resistance and inductance; R and L therefore do not enter this objective. They are retained in each validation row and their categories are described, but this experiment does **not** establish load-dependent voltage-THD optimization. The existing candidate selection stratifies training records by modulation-index quantile and reference-phase bin, not by R/L; its selected 30 records nevertheless contain all observed R and L categories.

## Methodology and split policy

- Baseline: cell carriers `[0°, 60°, 120°]`.
- Candidate: `[0°, {phases[1]:.12g}°, {phases[2]:.12g}°]`, fixed from the completed training-only experiment before this independent validation evaluation.
- Validation: every one of the {stats["validation_condition_count"]:,} rows labeled `validation` in `Sample_Splits.csv`, each evaluated with both phase configurations under identical modulation and phase. R/L are recorded without influencing the electrical THD computation.
- Training: used only through the saved 30-record stratified training subset and previously completed PSO study to select the global candidate. No PSO run was fit or selected against validation values in this validation run.
- Testing: no test operating-condition values or waveforms were parsed or evaluated. The saved source PSO metadata reports zero testing records evaluated.
- CNN, dataset values, and saved experiment artifacts were read-only. This validation writes a new uniquely named results folder.

## PSO configuration and provenance

- Seed runs: {len(phase_results["run_seeds"])} seeds, `{phase_results["run_seeds"][0]}` through `{phase_results["run_seeds"][-1]}`; selected seed **{phase_results["selected_seed"]}**.
- Swarm: {config["particles"]} particles × {config["iterations"]} iterations per seed; inertia {config["inertia"]}, cognitive/social {config["cognitive"]}/{config["social"]}, max velocity {config["maximum_velocity_degrees"]}°.
- Variables: cell-2 and cell-3 carrier offsets, modulo 360°, canonicalized by sorting; cell 1 remains fixed at 0°.
- Training subset: {phase_results["training_subset_count"]} training records, chosen by the saved 5 modulation quantiles × 6 circular phase bins. Baseline is retained as an incumbent.
- Historical optimization accounting: {phase_results["objective_requests"]:,} objective requests, {phase_results["unique_candidate_evaluations"]:,} unique candidate evaluations, {phase_results["record_evaluations"]:,} training-record evaluations, and {phase_results["total_runtime_seconds"]:.3f} seconds summed over all saved seeded runs.
- Candidate's training-subset mean THD: {phase_results["training_subset_candidate_mean_thd_percent"]:.9f}% vs baseline {phase_results["training_subset_baseline_mean_thd_percent"]:.9f}% (training change {phase_results["training_subset_change_percentage_points"]:+.9g} pp).
- Hard fundamental phasor tracking limit: {config["tracking_tolerance_percent"]}%.

The evaluation uses `evaluate_event_based_thd` identically for baseline and candidate. Its **full-spectrum Parseval THD** is based on continuous event intervals and total AC energy after removing the DC mean and fundamental. It is not the event evaluator's finite harmonic-band THD or its sampled 20 kHz THD. Individual event Fourier amplitudes for orders 2–100 are included in `per_condition_metrics.csv`; the full-spectrum THD statistic is not limited to those listed harmonics.

## Validation summary

Percentage-point change is `candidate THD (%) − baseline THD (%)`. Relative percentage change is `(candidate / baseline − 1) × 100%`; negative values indicate lower candidate THD. “Unchanged” uses an absolute tolerance of {stats["unchanged_tolerance_percentage_points"]:.1e} percentage points. Standard deviation is sample standard deviation (`ddof=1`).

{chr(10).join(metric_lines)}

### Condition-group breakdown

The objective does not depend on these load categories; the table documents validation coverage rather than asserting load sensitivity.

{chr(10).join(group_table)}

### Per-condition examples

The complete 1,500 paired rows are in [`per_condition_metrics.csv`](./{relative_dir}/per_condition_metrics.csv). First eight rows by SampleID:

{chr(10).join(condition_table)}

![Paired baseline and candidate THD across validation conditions](./{relative_dir}/baseline_vs_optimized_thd.png)

## Fundamental tracking

The source PSO candidate was accepted under a hard maximum phasor-error constraint of {config["tracking_tolerance_percent"]}%. On the full validation set, its maximum phasor tracking error was **{stats["candidate_max_fundamental_phasor_tracking_error_percent"]:.9g}%**. Thus it {("satisfies" if stats["candidate_max_fundamental_phasor_tracking_error_percent"] <= config["tracking_tolerance_percent"] else "does not satisfy")} that constraint in these validation cases. The CSV also records baseline and candidate fundamental peak amplitude and amplitude-only error for every condition. Fundamental tracking does not by itself establish harmonic improvement.

## Scientific interpretation

{conclusion}

This result applies to this ideal software model, the fixed dataset validation conditions, and the project's event-based Fourier definition. It does not establish hardware performance, MATLAB/Simulink agreement, load-dependent THD effects, statistical independence of nearby/synthetic operating records, or generalization beyond the observed conditions. The dataset may contain synthetic records; condition-level counts should not be interpreted as independent physical experiments.
"""
    report_path.write_text(report, encoding="utf-8")


def run_multi_condition_validation(
    *,
    dataset_dir: Path = DEFAULT_DATASET,
    source_experiment_dir: Path = DEFAULT_SOURCE_EXPERIMENT,
    output_dir: Path,
    report_path: Path = PROJECT_ROOT / "PSO_MULTI_CONDITION_VALIDATION.md",
    evaluator: Evaluator = evaluate_event_based_thd,
) -> Path:
    dataset_dir = dataset_dir.resolve()
    source_experiment_dir = source_experiment_dir.resolve()
    output_dir = output_dir.resolve()
    report_path = report_path.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite results directory: {output_dir}")
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {report_path}")
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    source = _load_source_experiment(source_experiment_dir)
    dataset_splits_path = dataset_dir / "Sample_Splits.csv"
    conditions_path = dataset_dir / "Operating_Conditions.csv"
    split_labels = load_split_labels(dataset_splits_path)
    training_ids = _load_training_selected_ids(
        source_experiment_dir / "training_subset.csv",
        split_labels,
    )
    training_records = _load_condition_records(
        conditions_path,
        split_labels,
        split="training",
        sample_ids=training_ids,
    )
    validation_records = _load_condition_records(
        conditions_path,
        split_labels,
        split="validation",
    )
    expected_validation_count = sum(label == "validation" for label in split_labels.values())
    if len(validation_records) != expected_validation_count:
        raise ValueError(
            f"Loaded {len(validation_records)} validation conditions; "
            f"split table declares {expected_validation_count}."
        )
    if any(record.sample_id in training_ids for record in validation_records):
        raise RuntimeError("Training PSO subset intersects the validation split.")

    final = source["final"]
    candidate_phases = tuple(float(value) for value in final["selected_phases_degrees"])
    if len(candidate_phases) != 2 or not all(np.isfinite(candidate_phases)):
        raise ValueError("Saved PSO experiment does not contain two finite phase shifts.")
    config = source["settings"]
    if config.get("tracking_tolerance_percent") != 0.05:
        raise ValueError("Saved PSO experiment differs from the predeclared 0.05% tracking limit.")

    condition_rows = _evaluate_pair(
        validation_records,
        candidate_phases,
        evaluator=evaluator,
    )
    statistics = _statistics(condition_rows)
    candidate_phase_offsets = (0.0, *candidate_phases)
    training_load_resistance_counts: dict[str, int] = {}
    training_load_inductance_counts: dict[str, int] = {}
    for record in training_records:
        r_label = f"{record.load_resistance_ohm:g}"
        l_label = f"{record.load_inductance_h:g}"
        training_load_resistance_counts[r_label] = training_load_resistance_counts.get(r_label, 0) + 1
        training_load_inductance_counts[l_label] = training_load_inductance_counts.get(l_label, 0) + 1

    output_dir.mkdir(parents=True, exist_ok=False)
    for source_name in ("run_summary.csv", "convergence_history.csv", "training_subset.csv"):
        shutil.copy2(source_experiment_dir / source_name, output_dir / source_name)
    _write_rows_csv(output_dir / "per_condition_metrics.csv", condition_rows)
    _write_plot(output_dir / "baseline_vs_optimized_thd.png", condition_rows)

    run_rows = source["runs"]
    run_columns = _column_map(run_rows[0].keys(), source_experiment_dir / "run_summary.csv")
    phase_results = {
        "baseline_carrier_offsets_degrees": list(BASELINE_PHASES_DEGREES),
        "candidate_carrier_offsets_degrees": list(candidate_phase_offsets),
        "selected_seed": int(final["selected_seed"]),
        "run_seeds": [int(row[run_columns["seed"]]) for row in run_rows],
        "training_subset_count": len(training_records),
        "training_subset_sample_ids": sorted(training_ids),
        "training_subset_load_resistance_counts": training_load_resistance_counts,
        "training_subset_load_inductance_counts": training_load_inductance_counts,
        "training_subset_baseline_mean_thd_percent": float(
            final["training_subset_baseline_mean_thd_percent"]
        ),
        "training_subset_candidate_mean_thd_percent": float(
            final["training_subset_mean_thd_percent"]
        ),
        "training_subset_change_percentage_points": float(
            final["training_subset_mean_thd_percent"]
            - final["training_subset_baseline_mean_thd_percent"]
        ),
        "pso_configuration": config,
        "objective_requests": sum(int(row[run_columns["objective_requests"]]) for row in run_rows),
        "unique_candidate_evaluations": sum(
            int(row[run_columns["unique_candidate_evaluations"]]) for row in run_rows
        ),
        "record_evaluations": sum(int(row[run_columns["record_evaluations"]]) for row in run_rows),
        "total_runtime_seconds": sum(float(row[run_columns["runtime_seconds"]]) for row in run_rows),
        "optimization_used_test_records": False,
        "validation_used_for_optimization_or_candidate_selection": False,
    }
    (output_dir / "phase_shift_results.json").write_text(
        json.dumps(phase_results, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    condition_groups = {
        "load_resistance": _aggregate_by_feature(condition_rows, "load_resistance_ohm"),
        "load_inductance": _aggregate_by_feature(condition_rows, "load_inductance_h"),
        "modulation_index_decile": [],
        "phase_angle_45_degree_bin": [],
    }
    # Training quantiles define the modulation bins; apply the same cuts to validation.
    training_points_for_bins = _load_condition_records(
        conditions_path,
        split_labels,
        split="training",
    )
    modulation_values = np.asarray(
        [record.modulation_index for record in training_points_for_bins],
        dtype=np.float64,
    )
    bin_edges = np.unique(np.quantile(modulation_values, np.linspace(0.0, 1.0, 11)))
    for index, (lower, upper) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        selected = [
            row
            for row, record in zip(condition_rows, validation_records)
            if (
                lower <= record.modulation_index <= upper
                if index == len(bin_edges) - 2
                else lower <= record.modulation_index < upper
            )
        ]
        if selected:
            condition_groups["modulation_index_decile"].append(
                {
                    "group": index + 1,
                    "lower_bound": float(lower),
                    "upper_bound": float(upper),
                    "condition_count": len(selected),
                    "baseline_mean_thd_percent": float(
                        np.mean(
                            [
                                row["baseline_full_spectrum_parseval_thd_percent"]
                                for row in selected
                            ]
                        )
                    ),
                    "candidate_mean_thd_percent": float(
                        np.mean(
                            [
                                row["candidate_full_spectrum_parseval_thd_percent"]
                                for row in selected
                            ]
                        )
                    ),
                    "mean_thd_change_percentage_points": float(
                        np.mean([row["thd_change_percentage_points"] for row in selected])
                    ),
                }
            )
    phase_bins = np.floor(
        np.mod(
            np.degrees([record.phase_angle_rad for record in validation_records]),
            360.0,
        )
        / 45.0
    ).astype(np.int64)
    for index in range(8):
        selected = [
            row for row, phase_bin in zip(condition_rows, phase_bins) if phase_bin == index
        ]
        if selected:
            condition_groups["phase_angle_45_degree_bin"].append(
                {
                    "group": f"{index * 45}-{(index + 1) * 45} degrees",
                    "condition_count": len(selected),
                    "baseline_mean_thd_percent": float(
                        np.mean(
                            [
                                row["baseline_full_spectrum_parseval_thd_percent"]
                                for row in selected
                            ]
                        )
                    ),
                    "candidate_mean_thd_percent": float(
                        np.mean(
                            [
                                row["candidate_full_spectrum_parseval_thd_percent"]
                                for row in selected
                            ]
                        )
                    ),
                    "mean_thd_change_percentage_points": float(
                        np.mean([row["thd_change_percentage_points"] for row in selected])
                    ),
                }
            )

    dataset_hashes = {
        name: _sha256(dataset_dir / name)
        for name in ("Operating_Conditions.csv", "Sample_Splits.csv")
    }
    source_hashes = {
        name: _sha256(source_experiment_dir / name)
        for name in (
            "experiment_config.json",
            "final_results.json",
            "run_summary.csv",
            "convergence_history.csv",
            "training_subset.csv",
        )
    }
    metadata = {
        "created_at_local": datetime.now().astimezone().isoformat(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "dataset_directory": str(dataset_dir),
        "dataset_input_sha256": dataset_hashes,
        "source_training_experiment_directory": str(source_experiment_dir),
        "source_training_experiment_sha256": source_hashes,
        "validation_record_count": len(validation_records),
        "training_subset_record_count": len(training_records),
        "validation_sample_ids": [record.sample_id for record in validation_records],
        "training_subset_sample_ids": sorted(training_ids),
        "test_split_policy": (
            "No test operating-condition values, waveforms, or metrics loaded for optimization or validation."
        ),
        "validation_used_for_candidate_selection": False,
        "objective_definition": (
            "Existing event-based full-spectrum Parseval THD; exact same evaluator/settings "
            "for baseline and training-selected candidate."
        ),
        "load_parameter_limitation": (
            "R and L are recorded for coverage, but the existing ideal output-voltage THD objective "
            "does not depend on load resistance or inductance."
        ),
        "unchanged_tolerance_percentage_points": _UNCHANGED_THD_TOLERANCE_PP,
    }
    summary = {
        "created_at_local": metadata["created_at_local"],
        "validation_operating_condition_count": len(validation_records),
        "statistics": statistics,
        "phase_shift_results": phase_results,
        "condition_groups": condition_groups,
        "evaluation_definition": {
            "thd_method": "Event-based continuous-time Fourier interval integration with Parseval full-spectrum THD.",
            "harmonic_amplitudes_available": "Orders 2 through 100 in per_condition_metrics.csv; THD total uses total AC energy after removing mean and fundamental.",
            "percentage_point_change": "candidate_thd_percent - baseline_thd_percent",
            "relative_percent_change": "(candidate_thd_percent / baseline_thd_percent - 1) * 100",
            "tracking_constraint_percent": config["tracking_tolerance_percent"],
            "candidate_fundamental_tracking_satisfied": (
                statistics["candidate_max_fundamental_phasor_tracking_error_percent"]
                <= config["tracking_tolerance_percent"]
            ),
        },
        "held_out_test_records_evaluated": 0,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (output_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _render_report(
        report_path=report_path,
        output_directory=output_dir,
        source_experiment=source_experiment_dir,
        phase_results=phase_results,
        summary=summary,
        condition_rows=condition_rows,
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--source-experiment", type=Path, default=DEFAULT_SOURCE_EXPERIMENT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "results"
        / f"multi_condition_pso_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "PSO_MULTI_CONDITION_VALIDATION.md",
    )
    args = parser.parse_args()
    output = run_multi_condition_validation(
        dataset_dir=args.dataset,
        source_experiment_dir=args.source_experiment,
        output_dir=args.output_dir,
        report_path=args.report,
    )
    print(f"Multi-condition PSO validation outputs: {output}")


if __name__ == "__main__":
    main()


__all__ = ["run_multi_condition_validation"]
