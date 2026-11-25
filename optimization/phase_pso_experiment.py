"""Reproducible training-subset experiment for circular carrier-phase PSO."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import json
import math
from pathlib import Path
import time
from typing import Iterable

import numpy as np

from .circular_phase_pso import (
    BASELINE_PHASES_DEGREES,
    TRACKING_TOLERANCE_PERCENT,
    CandidateEvaluation,
    CircularPSOConfig,
    CircularPhasePSO,
    OperatingPoint,
    PhaseOptimizationProblem,
    canonicalize_candidate,
)


def load_split_labels(path: str | Path) -> dict[int, str]:
    """Load IDs and split labels only; no waveform or operating data are read."""
    split_labels: dict[int, str] = {}
    with Path(path).open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        columns = {name.casefold(): name for name in reader.fieldnames or ()}
        id_column = columns.get("sampleid")
        split_column = columns.get("splitlabel")
        if id_column is None or split_column is None:
            raise ValueError(f"{path} must contain sampleID and splitLabel.")
        for row in reader:
            sample_id = int(row[id_column])
            split = row[split_column].strip().lower()
            if split not in {"training", "validation", "testing"}:
                raise ValueError(f"Unsupported split label {split!r} for SampleID {sample_id}.")
            if sample_id in split_labels:
                raise ValueError(f"Duplicate SampleID {sample_id} in {path}.")
            split_labels[sample_id] = split
    return split_labels


def load_operating_points(
    path: str | Path,
    split_labels: dict[int, str],
    *,
    split: str,
) -> tuple[OperatingPoint, ...]:
    """Read operating-condition values only for the requested split."""
    if split not in {"training", "validation"}:
        raise ValueError("This experiment loader intentionally supports training/validation only.")
    points: list[OperatingPoint] = []
    seen: set[int] = set()
    with Path(path).open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        columns = {name.casefold(): name for name in reader.fieldnames or ()}
        required = ("sampleid", "modulationindex", "phaseangle")
        if any(column not in columns for column in required):
            raise ValueError(f"{path} is missing one of the required columns: {required}.")
        for row in reader:
            sample_id = int(row[columns["sampleid"]])
            if split_labels.get(sample_id) != split:
                continue
            if sample_id in seen:
                raise ValueError(f"Duplicate {split} SampleID {sample_id} in {path}.")
            seen.add(sample_id)
            points.append(
                OperatingPoint(
                    sample_id=sample_id,
                    modulation_index=float(row[columns["modulationindex"]]),
                    phase_angle_rad=float(row[columns["phaseangle"]]),
                )
            )
    expected_ids = {sample_id for sample_id, label in split_labels.items() if label == split}
    if seen != expected_ids:
        raise ValueError(
            f"{path} {split} ID mismatch: missing={len(expected_ids - seen)}, "
            f"unexpected={len(seen - expected_ids)}."
        )
    return tuple(sorted(points, key=lambda point: point.sample_id))


def select_stratified_training_subset(
    training_points: Iterable[OperatingPoint],
    *,
    modulation_bins: int = 5,
    phase_bins: int = 6,
) -> tuple[OperatingPoint, ...]:
    """Select one deterministic representative from each m-quantile/phase stratum."""
    points = tuple(sorted(training_points, key=lambda point: point.sample_id))
    if modulation_bins < 1 or phase_bins < 1:
        raise ValueError("Stratification bin counts must be positive.")
    expected = modulation_bins * phase_bins
    if len(points) < expected:
        raise ValueError(f"Need at least {expected} training records, found {len(points)}.")
    modulation = np.asarray([point.modulation_index for point in points], dtype=np.float64)
    phases = np.mod(
        np.asarray([point.phase_angle_rad for point in points], dtype=np.float64),
        2.0 * math.pi,
    )
    quantile_edges = np.quantile(
        modulation,
        np.linspace(0.0, 1.0, modulation_bins + 1),
    )
    modulation_strata = np.searchsorted(quantile_edges[1:-1], modulation, side="right")
    phase_width = 2.0 * math.pi / phase_bins
    phase_strata = np.floor(phases / phase_width).astype(np.int64)
    phase_strata = np.minimum(phase_strata, phase_bins - 1)

    selected: list[OperatingPoint] = []
    for modulation_stratum in range(modulation_bins):
        group_mask = modulation_strata == modulation_stratum
        group_median = float(np.median(modulation[group_mask]))
        group_span = float(np.ptp(modulation[group_mask]))
        modulation_scale = group_span if group_span > 0 else max(abs(group_median), 1.0)
        for phase_stratum in range(phase_bins):
            member_indices = np.flatnonzero(
                group_mask & (phase_strata == phase_stratum)
            )
            if member_indices.size == 0:
                raise ValueError(
                    "Training data do not populate every configured modulation/phase stratum."
                )
            phase_center = (phase_stratum + 0.5) * phase_width
            phase_errors = np.abs(
                (phases[member_indices] - phase_center + math.pi) % (2.0 * math.pi)
                - math.pi
            )
            scores = (
                np.abs(modulation[member_indices] - group_median) / modulation_scale
                + phase_errors / phase_width
            )
            best_local = min(
                range(member_indices.size),
                key=lambda index: (
                    float(scores[index]),
                    points[int(member_indices[index])].sample_id,
                ),
            )
            selected.append(points[int(member_indices[best_local])])
    selected.sort(key=lambda point: point.sample_id)
    if len({point.sample_id for point in selected}) != expected:
        raise RuntimeError("Stratified subset selection returned duplicate SampleIDs.")
    return tuple(selected)


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _candidate_row(evaluation: CandidateEvaluation) -> dict[str, object]:
    return {
        "phase_cell2_deg": evaluation.canonical_phases_degrees[0],
        "phase_cell3_deg": evaluation.canonical_phases_degrees[1],
        "mean_full_spectrum_thd_percent": evaluation.mean_total_thd_percent,
        "max_tracking_error_percent": evaluation.maximum_tracking_error_percent,
        "feasible": evaluation.feasible,
        "worst_tracking_sample_id": evaluation.worst_tracking_sample_id,
        "record_count": evaluation.record_count,
    }


def _create_output_directory(results_root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        explicit.mkdir(parents=True, exist_ok=False)
        return explicit
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output = results_root / f"circular_phase_pso_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    return output


def run_experiment(
    project_root: str | Path,
    *,
    config: CircularPSOConfig | None = None,
    output_directory: str | Path | None = None,
) -> Path:
    """Run seeded training-subset PSO, then evaluate baseline/winner on validation."""
    root = Path(project_root).resolve()
    dataset_root = root / "data" / "raw" / "Seven_Level_Inverter_Project_extracted"
    split_labels = load_split_labels(dataset_root / "Sample_Splits.csv")
    training_points = load_operating_points(
        dataset_root / "Operating_Conditions.csv",
        split_labels,
        split="training",
    )
    subset = select_stratified_training_subset(training_points)
    settings = config or CircularPSOConfig()
    if not math.isclose(
        settings.tracking_tolerance_percent,
        TRACKING_TOLERANCE_PERCENT,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("The study's predeclared tracking constraint must remain 0.05%.")
    output_path = _create_output_directory(
        root / "results",
        Path(output_directory).resolve() if output_directory is not None else None,
    )

    _write_csv(
        output_path / "training_subset.csv",
        ["SampleID", "modulation_index", "reference_phase_rad", "selection_stratum"],
        (
            {
                "SampleID": point.sample_id,
                "modulation_index": point.modulation_index,
                "reference_phase_rad": point.phase_angle_rad,
                "selection_stratum": "5 modulation-index quantiles x 6 equal-width circular phase bins",
            }
            for point in subset
        ),
    )
    metadata = {
        "created_at_local": datetime.now().astimezone().isoformat(),
        "candidate_variables": ["cell2_phase_degrees", "cell3_phase_degrees"],
        "fixed_cell1_phase_degrees": 0.0,
        "phase_period_degrees": 360.0,
        "canonicalization": "wrap modulo 360 and sort cell2/cell3 offsets",
        "baseline_carrier_offsets_degrees": list(BASELINE_PHASES_DEGREES),
        "objective": "mean Parseval full-spectrum THD over fixed 30-record training subset",
        "hard_tracking_constraint_percent": settings.tracking_tolerance_percent,
        "selection": "one nearest-centroid record from each of 5 modulation quantiles x 6 circular phase bins",
        "training_records_available": len(training_points),
        "training_subset_sample_ids": [point.sample_id for point in subset],
        "validation_and_test_policy": "validation used once after all runs for baseline and selected winner; test records never evaluated",
        "pso_config": {
            **asdict(settings),
            "seeds": list(settings.seeds),
        },
    }
    (output_path / "experiment_config.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    problem = PhaseOptimizationProblem(
        subset,
        tracking_tolerance_percent=settings.tracking_tolerance_percent,
    )
    baseline = problem.evaluate(BASELINE_PHASES_DEGREES[1:])
    if not baseline.feasible:
        raise RuntimeError("Verified baseline is infeasible on the selected optimization subset.")
    optimizer = CircularPhasePSO(settings)
    run_summaries: list[dict[str, object]] = []
    convergence_rows: list[dict[str, object]] = []
    run_results = []
    for run_number, seed in enumerate(settings.seeds, start=1):
        result = optimizer.optimize(problem, seed=seed)
        if not result.best.feasible:
            raise RuntimeError(f"Run {seed} returned an infeasible best candidate.")
        if result.best.mean_total_thd_percent > baseline.mean_total_thd_percent + 1e-12:
            raise RuntimeError(f"Run {seed} returned a worse result than its baseline incumbent.")
        run_results.append(result)
        run_summaries.append(
            {
                "run": run_number,
                "seed": seed,
                **_candidate_row(result.best),
                "baseline_mean_full_spectrum_thd_percent": baseline.mean_total_thd_percent,
                "objective_requests": result.objective_requests,
                "unique_candidate_evaluations": result.unique_candidate_evaluations,
                "record_evaluations": result.record_evaluations,
                "cache_hits": result.cache_hits,
                "runtime_seconds": result.runtime_seconds,
                "final_particles_feasible": result.final_particles_feasible,
                "iterations_completed": settings.iterations,
            }
        )
        for iteration, (best, feasible_count) in enumerate(
            zip(result.convergence_history, result.feasibility_history)
        ):
            convergence_rows.append(
                {
                    "run": run_number,
                    "seed": seed,
                    "iteration": iteration,
                    **_candidate_row(best),
                    "feasible_particles_this_iteration": feasible_count,
                }
            )
        _write_csv(
            output_path / "run_summary.csv",
            list(run_summaries[0]),
            run_summaries,
        )
        _write_csv(
            output_path / "convergence_history.csv",
            list(convergence_rows[0]),
            convergence_rows,
        )
        print(
            f"Run {run_number}/{len(settings.seeds)} seed={seed}: "
            f"phases={result.best.canonical_phases_degrees}, "
            f"THD={result.best.mean_total_thd_percent:.6f}%, "
            f"runtime={result.runtime_seconds:.2f}s",
            flush=True,
        )

    best_run = min(
        run_results,
        key=lambda result: (
            result.best.mean_total_thd_percent,
            result.best.maximum_tracking_error_percent,
            result.seed,
        ),
    )
    selected_candidate = best_run.best
    validation_points = load_operating_points(
        dataset_root / "Operating_Conditions.csv",
        split_labels,
        split="validation",
    )
    validation_problem = PhaseOptimizationProblem(
        validation_points,
        tracking_tolerance_percent=settings.tracking_tolerance_percent,
    )
    validation_evaluations = {
        "baseline": validation_problem.evaluate(BASELINE_PHASES_DEGREES[1:]),
        "selected": validation_problem.evaluate(selected_candidate.canonical_phases_degrees),
    }
    validation_rows = []
    for label, evaluation in validation_evaluations.items():
        validation_rows.append({"candidate": label, **_candidate_row(evaluation)})
    _write_csv(
        output_path / "validation_comparison.csv",
        list(validation_rows[0]),
        validation_rows,
    )
    validation_baseline = validation_evaluations["baseline"]
    validation_selected = validation_evaluations["selected"]
    final_report = {
        "all_runs_completed": len(run_results) == len(settings.seeds),
        "runs_completed": len(run_results),
        "runs_requested": len(settings.seeds),
        "selected_seed": best_run.seed,
        "selected_phases_degrees": list(selected_candidate.canonical_phases_degrees),
        "training_subset_mean_thd_percent": selected_candidate.mean_total_thd_percent,
        "training_subset_baseline_mean_thd_percent": baseline.mean_total_thd_percent,
        "training_subset_relative_thd_change_percent": (
            selected_candidate.mean_total_thd_percent / baseline.mean_total_thd_percent - 1.0
        )
        * 100.0,
        "training_subset_max_tracking_error_percent": selected_candidate.maximum_tracking_error_percent,
        "validation_record_count": len(validation_points),
        "validation_baseline_mean_thd_percent": validation_baseline.mean_total_thd_percent,
        "validation_selected_mean_thd_percent": validation_selected.mean_total_thd_percent,
        "validation_relative_thd_change_percent": (
            validation_selected.mean_total_thd_percent
            / validation_baseline.mean_total_thd_percent
            - 1.0
        )
        * 100.0,
        "validation_baseline_max_tracking_error_percent": validation_baseline.maximum_tracking_error_percent,
        "validation_selected_max_tracking_error_percent": validation_selected.maximum_tracking_error_percent,
        "validation_selected_feasible": validation_selected.feasible,
        "test_records_evaluated": 0,
    }
    (output_path / "final_results.json").write_text(
        json.dumps(final_report, indent=2),
        encoding="utf-8",
    )
    print(f"Experiment outputs: {output_path}", flush=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--particles", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=20261120)
    parser.add_argument("--runs", type=int, default=20)
    arguments = parser.parse_args()
    if arguments.runs < 1:
        parser.error("--runs must be positive.")
    config = CircularPSOConfig(
        particles=arguments.particles,
        iterations=arguments.iterations,
        seeds=tuple(range(arguments.seed_start, arguments.seed_start + arguments.runs)),
    )
    run_experiment(
        arguments.project_root,
        config=config,
        output_directory=arguments.output_directory,
    )


if __name__ == "__main__":
    main()
