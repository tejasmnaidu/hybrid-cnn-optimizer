"""36x36 real-simulator ground truth and seeded optimizer comparison."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from evaluation.event_based_thd import evaluate_event_based_thd
from evaluation.live_phase_optimization import optimize_live_operating_point
from optimization.circular_phase_pso import (
    BASELINE_PHASES_DEGREES,
    candidate_to_carrier_offsets,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIRECTORY = PROJECT_ROOT / "results"
CONVERGENCE_OUTPUT = RESULTS_DIRECTORY / "optimizer_convergence.csv"
GRID_OUTPUT = RESULTS_DIRECTORY / "optimizer_grid_search.csv"
JSON_OUTPUT = RESULTS_DIRECTORY / "optimizer_validation.json"

OPERATING_POINT = {
    "modulation_index": 0.75,
    "dc_voltage_per_cell_v": 100.0,
    "load_resistance_ohm": 30.0,
    "load_inductance_h": 0.03,
    "phase_angle_degrees": 0.0,
}
SEEDS = (20261124, 20261125, 20261126)
GRID_PHASES = tuple(float(value) for value in np.linspace(0.0, 350.0, 36))
PARTICLES = 8
ITERATIONS = 8


def real_event_objective(second_degrees: float, third_degrees: float) -> float:
    """Evaluate event-based Parseval total THD at fixed live conditions."""
    result = evaluate_event_based_thd(
        OPERATING_POINT["modulation_index"],
        np.deg2rad(OPERATING_POINT["phase_angle_degrees"]),
        carrier_phase_offsets_degrees=(
            0.0,
            float(second_degrees),
            float(third_degrees),
        ),
        dc_voltage_per_cell_v=OPERATING_POINT["dc_voltage_per_cell_v"],
    )
    return float(result.total_thd_percent)


def symmetric_phase_gap(candidate: tuple[float, float], optimum: tuple[float, float]) -> float:
    """Return shortest circular pair distance under cell-2/cell-3 permutation."""
    candidate_sorted = tuple(sorted(candidate))
    optimum_sorted = tuple(sorted(optimum))
    direct = float(
        np.hypot(
            np.sin(np.deg2rad(candidate_sorted[0] - optimum_sorted[0])), 
            np.sin(np.deg2rad(candidate_sorted[1] - optimum_sorted[1])),
        )
    )
    swapped = float(
        np.hypot(
            np.sin(np.deg2rad(candidate_sorted[0] - optimum_sorted[1])), 
            np.sin(np.deg2rad(candidate_sorted[1] - optimum_sorted[0])),
        )
    )
    return min(direct, swapped) * 180.0 / np.pi


def run_grid_search() -> tuple[list[dict[str, float]], dict[str, float]]:
    rows: list[dict[str, float]] = []
    best_value = float("inf")
    best_second = 0.0
    best_third = 0.0
    for second in GRID_PHASES:
        for third in GRID_PHASES:
            value = real_event_objective(second, third)
            rows.append(
                {
                    "cell_1_offset_degrees": 0.0,
                    "cell_2_offset_degrees": second,
                    "cell_3_offset_degrees": third,
                    "event_parseval_total_thd_percent": value,
                }
            )
            if value < best_value:
                best_value = value
                best_second = second
                best_third = third
    return rows, {
        "thd_percent": best_value,
        "cell_2_degrees": best_second,
        "cell_3_degrees": best_third,
    }


def run_optimizer_comparison() -> tuple[
    dict[str, list[dict[str, object]]],
    list[dict[str, float | int | str]],
]:
    convergence_rows: list[dict[str, float | int | str]] = []
    seed_runs: dict[str, list[dict[str, object]]] = {}
    for optimizer_name in ("PSO", "POA", "PFO", "HYBRID"):
        runs: list[dict[str, object]] = []
        for seed in SEEDS:
            optimization = optimize_live_operating_point(
                **OPERATING_POINT,
                seed=seed,
                particles=PARTICLES,
                iterations=ITERATIONS,
                optimizer_name=optimizer_name,
            )
            result = optimization.run_result
            phase_pair = result.best.canonical_phases_degrees
            offsets = candidate_to_carrier_offsets(phase_pair)
            final_thd = real_event_objective(offsets[1], offsets[2])
            for iteration, evaluation in enumerate(result.convergence_history):
                candidate_offsets = candidate_to_carrier_offsets(
                    evaluation.canonical_phases_degrees
                )
                convergence_rows.append(
                    {
                        "optimizer": optimizer_name,
                        "seed": seed,
                        "iteration": iteration,
                        "best_event_parseval_total_thd_percent": (
                            float(evaluation.mean_total_thd_percent)
                        ),
                        "cell_1_offset_degrees": candidate_offsets[0],
                        "cell_2_offset_degrees": candidate_offsets[1],
                        "cell_3_offset_degrees": candidate_offsets[2],
                    }
                )
            runs.append(
                {
                    "seed": seed,
                    "best_candidate": [
                        float(offsets[0]),
                        float(offsets[1]),
                        float(offsets[2]),
                    ],
                    "candidate_canonical_phases": [
                        float(phase_pair[0]),
                        float(phase_pair[1]),
                    ],
                    "optimizer_reported_thd_percent": float(
                        result.best.mean_total_thd_percent
                    ),
                    "independent_resimulated_thd_percent": final_thd,
                    "objective_requests": int(result.objective_requests),
                    "unique_candidate_evaluations": int(
                        result.unique_candidate_evaluations
                    ),
                }
            )
        seed_runs[optimizer_name] = runs
    return seed_runs, convergence_rows


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing validation output: {path}")
    with path.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    grid_rows, grid_best = run_grid_search()
    seed_runs, convergence_rows = run_optimizer_comparison()
    for optimizer_name, runs in seed_runs.items():
        for run in runs:
            run["grid_thd_gap_percentage_points"] = (
                float(run["independent_resimulated_thd_percent"])
                - grid_best["thd_percent"]
            )
            run["phase_gap_degrees"] = symmetric_phase_gap(
                tuple(float(value) for value in run["candidate_canonical_phases"]),
                (grid_best["cell_2_degrees"], grid_best["cell_3_degrees"]),
            )
    write_csv(
        GRID_OUTPUT,
        (
            "cell_1_offset_degrees",
            "cell_2_offset_degrees",
            "cell_3_offset_degrees",
            "event_parseval_total_thd_percent",
        ),
        grid_rows,
    )
    write_csv(
        CONVERGENCE_OUTPUT,
        (
            "optimizer",
            "seed",
            "iteration",
            "best_event_parseval_total_thd_percent",
            "cell_1_offset_degrees",
            "cell_2_offset_degrees",
            "cell_3_offset_degrees",
        ),
        convergence_rows,
    )
    report = {
        "validation": "real-simulator 36x36 carrier-phase ground truth",
        "operating_point": OPERATING_POINT,
        "search_space": {
            "cell_1_offset_degrees": 0.0,
            "cell_2_offset_degrees": "0..350 degrees, 36 points, step 10",
            "cell_3_offset_degrees": "0..350 degrees, 36 points, step 10",
            "combinations": 1296,
            "objective": "event-based Parseval total_thd_percent",
        },
        "grid_best": grid_best,
        "seeds": list(SEEDS),
        "particle_or_population_size": PARTICLES,
        "iterations": ITERATIONS,
        "optimizers": seed_runs,
        "validation_statement": (
            "An optimizer is validated on this landscape only if independent "
            "real-simulator re-simulation approaches the grid optimum across seeds."
        ),
    }
    if JSON_OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {JSON_OUTPUT}")
    JSON_OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
