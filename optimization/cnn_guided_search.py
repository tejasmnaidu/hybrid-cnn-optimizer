"""CNN-ranked candidate screening for existing hybrid inverter optimization."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from evaluation.event_based_thd import evaluate_event_based_thd
from multilevel.inverter_simulator import (
    CELL_COUNT,
    InverterSimulation,
    simulate_inverter,
)
from preprocessing.preprocess import NormalizationStats
from optimization.circular_phase_pso import (
    BASELINE_PHASES_DEGREES,
    OperatingPoint,
    PhaseOptimizationProblem,
    TRACKING_TOLERANCE_PERCENT,
    canonicalize_candidate,
)
from optimization.nature_optimizers import (
    HybridPOAPFO,
    HybridPOAPFOConfig,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIRECTORY = PROJECT_ROOT / "results"
CORRELATION_OUTPUT = RESULTS_DIRECTORY / "cnn_guided_correlation.json"
COMPARISON_OUTPUT = RESULTS_DIRECTORY / "cnn_guided_comparison.csv"
CORRELATION_THRESHOLD = 0.70
COMPARISON_CANDIDATE_BUDGET = 64
COMPARISON_TOP_K = 8
GRID_SUCCESS_TOLERANCE_PERCENT = 0.1
SAMPLE_FREQUENCY_HZ = 20_000.0
FUNDAMENTAL_FREQUENCY_HZ = 50.0
MAX_HARMONIC = 100


def spearman_correlation(first: np.ndarray, second: np.ndarray) -> float:
    """Compute exact Spearman rank correlation, preserving tied ranks."""
    left = np.asarray(first, dtype=np.float64).reshape(-1)
    right = np.asarray(second, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or left.size < 2:
        raise ValueError("Spearman arrays must be identical, non-empty, and contain pairs.")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Spearman arrays must contain finite values.")

    def ranks(values: np.ndarray) -> np.ndarray:
        unique = np.unique(values)
        result = np.empty(values.size, dtype=np.float64)
        start = 0
        for value in unique:
            positions = np.flatnonzero(values == value)
            result[positions] = (start + 1.0 + start + positions.size) / 2.0
            start += positions.size
        return result

    left_ranks = ranks(left)
    right_ranks = ranks(right)
    left_ranks -= np.mean(left_ranks)
    right_ranks -= np.mean(right_ranks)
    denominator = float(
        np.sqrt(np.sum(left_ranks * left_ranks) * np.sum(right_ranks * right_ranks))
    )
    if denominator == 0.0:
        return 0.0
    return float(np.sum(left_ranks * right_ranks) / denominator)


def cnn_candidate_score(
    model: nn.Module,
    normalization: NormalizationStats,
    *,
    modulation_index: float,
    dc_voltage_per_cell_v: float,
    load_resistance_ohm: float,
    load_inductance_h: float,
    phase_angle_degrees: float,
    carrier_offsets_degrees: tuple[float, float, float],
    device: torch.device | str = "cpu",
) -> float:
    """CNN RMSE against candidate analytical reference; direct simulator-backed."""
    simulation = simulate_inverter(
        modulation_index,
        load_resistance_ohm,
        load_inductance_h,
        np.deg2rad(phase_angle_degrees),
        carrier_phase_offsets_degrees=carrier_offsets_degrees,
        dc_voltage_per_cell_v=dc_voltage_per_cell_v,
    )
    return _cnn_score_for_simulation(
        model,
        normalization,
        simulation,
        modulation_index=modulation_index,
        dc_voltage_per_cell_v=dc_voltage_per_cell_v,
        load_resistance_ohm=load_resistance_ohm,
        load_inductance_h=load_inductance_h,
        phase_angle_degrees=phase_angle_degrees,
        device=device,
    )


def _cnn_score_for_simulation(
    model: nn.Module,
    normalization: NormalizationStats,
    simulation: InverterSimulation,
    *,
    modulation_index: float,
    dc_voltage_per_cell_v: float,
    load_resistance_ohm: float,
    load_inductance_h: float,
    phase_angle_degrees: float,
    device: torch.device | str,
) -> float:
    raw_conditions = np.asarray(
        [[
            modulation_index,
            load_resistance_ohm,
            load_inductance_h,
            np.deg2rad(phase_angle_degrees),
        ]],
        dtype=np.float64,
    )
    normalized_voltage, normalized_current, normalized_conditions = (
        normalization.normalize_inputs(
            simulation.output_voltage[np.newaxis, :],
            simulation.load_current[np.newaxis, :],
            raw_conditions,
        )
    )
    device_obj = torch.device(device)
    was_training = model.training
    model.to(device_obj)
    model.eval()
    try:
        with torch.inference_mode():
            prediction = model(
                torch.as_tensor(normalized_voltage, dtype=torch.float32, device=device_obj),
                torch.as_tensor(normalized_current, dtype=torch.float32, device=device_obj),
                torch.as_tensor(normalized_conditions, dtype=torch.float32, device=device_obj),
            )
    finally:
        model.train(was_training)
    cnn_prediction = normalization.denormalize_target(
        prediction.detach().cpu().numpy()
    )[0]
    target_reference = (
        simulation.reference * CELL_COUNT * dc_voltage_per_cell_v
    )
    return float(
        np.sqrt(
            np.mean(
                np.square(
                    cnn_prediction
                    - target_reference
                )
            )
        )
    )


def measure_correlation(
    model: nn.Module,
    normalization: NormalizationStats,
    candidates: np.ndarray,
    *,
    operating_point: dict[str, float],
    seed: int,
    device: torch.device | str = "cpu",
) -> dict[str, object]:
    """Generate 500 random candidates once and correlate CNN score with event THD."""
    candidate_array = np.asarray(candidates, dtype=np.float64)
    if candidate_array.ndim != 2 or candidate_array.shape[1] != 2:
        raise ValueError("Candidates must have shape (N,2).")
    candidate_count = int(candidate_array.shape[0])
    if candidate_count < 2:
        raise ValueError("At least two candidates are required.")
    if not np.isfinite(candidate_array).all():
        raise ValueError("Candidates must contain finite values.")
    scores: list[float] = []
    thds: list[float] = []
    voltage_thds: list[float] = []
    current_thds: list[float] = []
    rows: list[dict[str, float]] = []

    for offset_index in range(candidate_count):
        cell_two = float(candidate_array[offset_index, 0])
        cell_three = float(candidate_array[offset_index, 1])
        offsets = (0.0, cell_two, cell_three)
        simulation = simulate_inverter(
            float(operating_point["modulation_index"]),
            float(operating_point["load_resistance_ohm"]),
            float(operating_point["load_inductance_h"]),
            np.deg2rad(float(operating_point["phase_angle_degrees"])),
            carrier_phase_offsets_degrees=offsets,
            dc_voltage_per_cell_v=float(operating_point["dc_voltage_per_cell_v"]),
        )
        score = _cnn_score_for_simulation(
            model,
            normalization,
            simulation,
            modulation_index=float(operating_point["modulation_index"]),
            dc_voltage_per_cell_v=float(operating_point["dc_voltage_per_cell_v"]),
            load_resistance_ohm=float(operating_point["load_resistance_ohm"]),
            load_inductance_h=float(operating_point["load_inductance_h"]),
            phase_angle_degrees=float(operating_point["phase_angle_degrees"]),
            device=device,
        )
        event = evaluate_event_based_thd(
            float(operating_point["modulation_index"]),
            np.deg2rad(float(operating_point["phase_angle_degrees"])),
            carrier_phase_offsets_degrees=offsets,
            dc_voltage_per_cell_v=float(operating_point["dc_voltage_per_cell_v"]),
        )
        from evaluation.thd_analysis import calculate_thd

        voltage_thd = float(
            calculate_thd(
                simulation.output_voltage,
                sampling_frequency_hz=SAMPLE_FREQUENCY_HZ,
                fundamental_frequency_hz=FUNDAMENTAL_FREQUENCY_HZ,
                max_harmonic=MAX_HARMONIC,
            )
        )
        current_thd = float(
            calculate_thd(
                simulation.load_current,
                sampling_frequency_hz=SAMPLE_FREQUENCY_HZ,
                fundamental_frequency_hz=FUNDAMENTAL_FREQUENCY_HZ,
                max_harmonic=MAX_HARMONIC,
            )
        )
        scores.append(score)
        thds.append(float(event.total_thd_percent))
        voltage_thds.append(voltage_thd)
        current_thds.append(current_thd)
        rows.append(
            {
                "cell_2_offset_degrees": cell_two,
                "cell_3_offset_degrees": cell_three,
                "cnn_score_rmse_v": score,
                "event_parseval_total_thd_percent": float(event.total_thd_percent),
                "sampled_voltage_thd_percent": voltage_thd,
                "sampled_current_thd_percent": current_thd,
            }
        )
    correlation = spearman_correlation(np.asarray(scores), np.asarray(thds))
    return {
        "candidate_count": candidate_count,
        "seed": int(seed),
        "simulator_waveform_generations": candidate_count,
        "full_event_thd_evaluations": candidate_count,
        "spearman_cnn_score_vs_event_parseval_thd": correlation,
        "usable_threshold": CORRELATION_THRESHOLD,
        "guidance_usable": bool(correlation >= CORRELATION_THRESHOLD),
        "cnn_score_direction": "lower CNN RMSE is predicted to rank lower simulator THD",
        "rows": rows,
    }


def screen_and_rank_candidates(
    candidates: np.ndarray,
    cnn_scores: np.ndarray,
    *,
    top_k: int,
) -> np.ndarray:
    """Return top-K candidate indices using ascending CNN score."""
    candidate_array = np.asarray(candidates, dtype=np.float64)
    score_array = np.asarray(cnn_scores, dtype=np.float64)
    if candidate_array.ndim != 2 or candidate_array.shape[1] != 2:
        raise ValueError("Candidates must have shape (N,2).")
    if candidate_array.shape[0] != score_array.size:
        raise ValueError("Candidate and CNN score counts must match.")
    if not np.isfinite(candidate_array).all() or not np.isfinite(score_array).all():
        raise ValueError("Candidates and scores must be finite.")
    if top_k < 1 or top_k > candidate_array.shape[0]:
        raise ValueError("top_k must be in [1, candidate count].")
    return np.argsort(score_array, kind="stable")[:top_k]


def write_correlation_report(report: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {path}")
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def write_comparison_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = (
        "method",
        "seed",
        "candidate_budget",
        "unique_candidate_evaluations",
        "optimizer_objective_requests",
        "final_event_parseval_total_thd_percent",
        "final_sampled_voltage_thd_percent",
        "final_sampled_current_thd_percent",
        "simulator_waveform_generations",
        "full_thd_evaluations",
        "runtime_seconds",
        "success_within_grid_optimum_percent",
        "success",
        "selected_cell_2_offset_degrees",
        "selected_cell_3_offset_degrees",
        "event_thd_gap_from_grid_percentage_points",
        "phase_space_gap_degrees",
        "feasible",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {path}")
    with path.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sampled_thd(signal: np.ndarray) -> float:
    from evaluation.thd_analysis import calculate_thd

    return float(
        calculate_thd(
            signal,
            sampling_frequency_hz=SAMPLE_FREQUENCY_HZ,
            fundamental_frequency_hz=FUNDAMENTAL_FREQUENCY_HZ,
            max_harmonic=MAX_HARMONIC,
        )
    )


def _phase_space_gap_degrees(
    candidate: tuple[float, float],
    grid_best: tuple[float, float],
) -> float:
    differences = [
        abs((candidate[index] - grid_best[index] + 180.0) % 360.0 - 180.0)
        for index in range(2)
    ]
    return float(np.sqrt(np.sum(np.square(differences))))


def _final_metrics(
    candidate: tuple[float, float],
    *,
    operating_point: dict[str, float],
) -> tuple[float, float, float, float]:
    offsets = (0.0, *candidate)
    simulation = simulate_inverter(
        float(operating_point["modulation_index"]),
        float(operating_point["load_resistance_ohm"]),
        float(operating_point["load_inductance_h"]),
        np.deg2rad(float(operating_point["phase_angle_degrees"])),
        carrier_phase_offsets_degrees=offsets,
        dc_voltage_per_cell_v=float(operating_point["dc_voltage_per_cell_v"]),
    )
    if not np.isfinite(simulation.output_voltage).all() or not np.isfinite(
        simulation.load_current
    ).all():
        raise RuntimeError("Final candidate simulation produced non-finite waveforms.")
    event = evaluate_event_based_thd(
        float(operating_point["modulation_index"]),
        np.deg2rad(float(operating_point["phase_angle_degrees"])),
        carrier_phase_offsets_degrees=offsets,
        dc_voltage_per_cell_v=float(operating_point["dc_voltage_per_cell_v"]),
    )
    return (
        float(event.total_thd_percent),
        _sampled_thd(simulation.output_voltage),
        _sampled_thd(simulation.load_current),
        float(event.fundamental_phasor_error_percent),
    )


def run_standalone_hybrid(
    *,
    seed: int,
    operating_point: dict[str, float],
    candidate_budget: int = COMPARISON_CANDIDATE_BUDGET,
) -> dict[str, object]:
    """Run existing Hybrid POA-PFO and independently re-evaluate its result."""
    event_evaluation_count = 0

    def counted_event_evaluation(
        modulation_index: float,
        phase_angle_rad: float,
        *,
        carrier_phase_offsets_degrees: tuple[float, ...],
    ):
        nonlocal event_evaluation_count
        event_evaluation_count += 1
        return evaluate_event_based_thd(
            modulation_index,
            phase_angle_rad,
            carrier_phase_offsets_degrees=carrier_phase_offsets_degrees,
            dc_voltage_per_cell_v=float(operating_point["dc_voltage_per_cell_v"]),
        )

    point = OperatingPoint(
        sample_id=1,
        modulation_index=float(operating_point["modulation_index"]),
        phase_angle_rad=np.deg2rad(float(operating_point["phase_angle_degrees"])),
    )
    problem = PhaseOptimizationProblem((point,), evaluator=counted_event_evaluation)
    started = time.perf_counter()
    result = HybridPOAPFO(
        HybridPOAPFOConfig(
            population_size=8,
            iterations=8,
            seed=seed,
            evaluation_budget=candidate_budget,
        )
    ).optimize(problem)
    runtime = time.perf_counter() - started
    candidate = canonicalize_candidate(result.best.canonical_phases_degrees)
    if not result.best.feasible:
        raise RuntimeError("Standalone Hybrid POA-PFO returned an infeasible candidate.")
    final_thd, voltage_thd, current_thd, final_tracking_error = _final_metrics(
        candidate,
        operating_point=operating_point,
    )
    if final_tracking_error > TRACKING_TOLERANCE_PERCENT:
        raise RuntimeError("Standalone final candidate failed independent feasibility check.")
    event_evaluation_count += 1
    return {
        "candidate": candidate,
        "event_thd": final_thd,
        "voltage_thd": voltage_thd,
        "current_thd": current_thd,
        "waveform_generations": 1,
        "full_thd_evaluations": event_evaluation_count,
        "unique_candidate_evaluations": result.unique_candidate_evaluations,
        "optimizer_objective_requests": result.objective_requests,
        "runtime_seconds": runtime,
    }


def _guided_candidate_pool(
    model: nn.Module,
    normalization: NormalizationStats,
    *,
    seed: int,
    operating_point: dict[str, float],
    candidate_budget: int,
    device: torch.device | str,
) -> dict[tuple[float, float], float]:
    """Use alternating POA/PFO proposals, with CNN score as proxy fitness."""
    rng = np.random.default_rng(seed)
    population = np.empty((8, 2), dtype=np.float64)
    population[0] = BASELINE_PHASES_DEGREES[1:]
    population[1:] = rng.uniform(0.0, 360.0, size=(7, 2))
    population = np.asarray(
        [canonicalize_candidate(position) for position in population],
        dtype=np.float64,
    )
    scores: dict[tuple[float, float], float] = {}

    def score(position: np.ndarray) -> float | None:
        key = canonicalize_candidate(position)
        if key in scores:
            return scores[key]
        if len(scores) >= candidate_budget:
            return None
        value = cnn_candidate_score(
            model,
            normalization,
            modulation_index=float(operating_point["modulation_index"]),
            dc_voltage_per_cell_v=float(operating_point["dc_voltage_per_cell_v"]),
            load_resistance_ohm=float(operating_point["load_resistance_ohm"]),
            load_inductance_h=float(operating_point["load_inductance_h"]),
            phase_angle_degrees=float(operating_point["phase_angle_degrees"]),
            carrier_offsets_degrees=(0.0, *key),
            device=device,
        )
        if not math.isfinite(value):
            raise RuntimeError(f"CNN candidate score is not finite for {key}.")
        scores[key] = value
        return value

    population_scores: list[float] = []
    for member in population:
        value = score(member)
        if value is None:
            break
        population_scores.append(value)

    iteration_count = max(1, (candidate_budget + len(population) - 1) // len(population))
    generation = 1
    while len(scores) < candidate_budget and generation <= candidate_budget * 4:
        mechanism = "poa" if generation % 2 == 1 else "pfo"
        for index in range(len(population)):
            if len(scores) >= candidate_budget:
                break
            position = population[index]
            current_score = score(position)
            if current_score is None:
                break
            if mechanism == "poa":
                prey = rng.uniform(0.0, 360.0, size=2)
                prey_score = score(prey)
                if prey_score is None:
                    break
                factor = int(rng.integers(1, 3))
                if prey_score < current_score:
                    proposal = position + rng.random(2) * (prey - factor * position)
                else:
                    proposal = position + rng.random(2) * (position - prey)
            else:
                better = [
                    population[other]
                    for other, other_score in enumerate(population_scores)
                    if other != index and other_score < current_score
                ]
                if better:
                    target = better[int(rng.integers(0, len(better)))]
                    factors = rng.integers(1, 3, size=2)
                    proposal = position + rng.random(2) * (target - factors * position)
                else:
                    proposal = position.copy()
            proposal = np.mod(proposal, 360.0)
            proposal_score = score(proposal)
            if proposal_score is not None and proposal_score < current_score:
                population[index] = canonicalize_candidate(proposal)
                population_scores[index] = proposal_score
            if mechanism == "poa" and len(scores) < candidate_budget:
                local_scale = 0.2 * max(
                    0.0, 1.0 - generation / iteration_count
                )
                local = np.mod(
                    population[index]
                    + local_scale * (2.0 * rng.random(2) - 1.0) * population[index],
                    360.0,
                )
                local_score = score(local)
                if local_score is not None and local_score < population_scores[index]:
                    population[index] = canonicalize_candidate(local)
                    population_scores[index] = local_score
            elif mechanism == "pfo" and len(scores) < candidate_budget:
                defense = np.mod(
                    population[index]
                    + (1.0 - 2.0 * rng.random(2)) * (360.0 / generation),
                    360.0,
                )
                defense_score = score(defense)
                if (
                    defense_score is not None
                    and defense_score < population_scores[index]
                ):
                    population[index] = canonicalize_candidate(defense)
                    population_scores[index] = defense_score
        generation += 1
    if len(scores) < candidate_budget:
        raise RuntimeError(
            f"Guided proposal loop scored {len(scores)} unique candidates, "
            f"below budget {candidate_budget}."
        )
    return scores


def run_cnn_guided_hybrid(
    model: nn.Module,
    normalization: NormalizationStats,
    *,
    seed: int,
    operating_point: dict[str, float],
    candidate_budget: int = COMPARISON_CANDIDATE_BUDGET,
    top_k: int = COMPARISON_TOP_K,
    device: torch.device | str = "cpu",
) -> dict[str, object]:
    """CNN-rank hybrid proposals, fully evaluate top-K, and re-simulate winner."""
    if candidate_budget < 8:
        raise ValueError("candidate_budget must be at least the hybrid population size.")
    if not 1 <= top_k <= candidate_budget:
        raise ValueError("top_k must be in [1, candidate_budget].")
    started = time.perf_counter()
    scores = _guided_candidate_pool(
        model,
        normalization,
        seed=seed,
        operating_point=operating_point,
        candidate_budget=candidate_budget,
        device=device,
    )
    candidates = np.asarray(list(scores), dtype=np.float64)
    cnn_scores = np.asarray(list(scores.values()), dtype=np.float64)
    ranked = screen_and_rank_candidates(candidates, cnn_scores, top_k=top_k)
    baseline = canonicalize_candidate(BASELINE_PHASES_DEGREES[1:])
    finalists = [tuple(candidates[index]) for index in ranked]
    if baseline not in finalists:
        finalists[-1] = baseline

    full_thd_evaluations = 0
    evaluated = []
    for candidate in finalists:
        event = evaluate_event_based_thd(
            float(operating_point["modulation_index"]),
            np.deg2rad(float(operating_point["phase_angle_degrees"])),
            carrier_phase_offsets_degrees=(0.0, *candidate),
            dc_voltage_per_cell_v=float(operating_point["dc_voltage_per_cell_v"]),
        )
        full_thd_evaluations += 1
        evaluated.append((candidate, event))
    feasible = [
        item
        for item in evaluated
        if item[1].fundamental_phasor_error_percent
        <= TRACKING_TOLERANCE_PERCENT
    ]
    if not feasible:
        raise RuntimeError("CNN-ranked finalists contain no electrically feasible candidate.")
    candidate, _ = min(feasible, key=lambda item: item[1].total_thd_percent)
    final_thd, voltage_thd, current_thd, final_tracking_error = _final_metrics(
        candidate,
        operating_point=operating_point,
    )
    if final_tracking_error > TRACKING_TOLERANCE_PERCENT:
        raise RuntimeError("Guided final candidate failed independent feasibility check.")
    full_thd_evaluations += 1
    return {
        "candidate": candidate,
        "event_thd": final_thd,
        "voltage_thd": voltage_thd,
        "current_thd": current_thd,
        "waveform_generations": len(scores) + 1,
        "full_thd_evaluations": full_thd_evaluations,
        "unique_candidate_evaluations": len(scores),
        "optimizer_objective_requests": "",
        "runtime_seconds": time.perf_counter() - started,
    }


def run_seeded_comparison(
    model: nn.Module,
    normalization: NormalizationStats,
    *,
    seeds: tuple[int, ...],
    operating_point: dict[str, float],
    grid_best_candidate: tuple[float, float],
    grid_best_thd_percent: float,
    candidate_budget: int = COMPARISON_CANDIDATE_BUDGET,
    top_k: int = COMPARISON_TOP_K,
    device: torch.device | str = "cpu",
) -> list[dict[str, object]]:
    """Compare existing hybrid and CNN-guided hybrid across fixed seeds."""
    rows: list[dict[str, object]] = []
    for seed in seeds:
        results = (
            (
                "Hybrid POA-PFO",
                run_standalone_hybrid(
                    seed=seed,
                    operating_point=operating_point,
                    candidate_budget=candidate_budget,
                ),
            ),
            (
                "CNN-guided Hybrid POA-PFO",
                run_cnn_guided_hybrid(
                    model,
                    normalization,
                    seed=seed,
                    operating_point=operating_point,
                    candidate_budget=candidate_budget,
                    top_k=top_k,
                    device=device,
                ),
            ),
        )
        for method, result in results:
            candidate = result["candidate"]
            thd = float(result["event_thd"])
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "candidate_budget": candidate_budget,
                    "unique_candidate_evaluations": result[
                        "unique_candidate_evaluations"
                    ],
                    "optimizer_objective_requests": result[
                        "optimizer_objective_requests"
                    ],
                    "final_event_parseval_total_thd_percent": thd,
                    "final_sampled_voltage_thd_percent": result["voltage_thd"],
                    "final_sampled_current_thd_percent": result["current_thd"],
                    "simulator_waveform_generations": result["waveform_generations"],
                    "full_thd_evaluations": result["full_thd_evaluations"],
                    "runtime_seconds": result["runtime_seconds"],
                    "success_within_grid_optimum_percent": GRID_SUCCESS_TOLERANCE_PERCENT,
                    "success": bool(
                        thd
                        <= grid_best_thd_percent
                        * (1.0 + GRID_SUCCESS_TOLERANCE_PERCENT / 100.0)
                    ),
                    "selected_cell_2_offset_degrees": candidate[0],
                    "selected_cell_3_offset_degrees": candidate[1],
                    "event_thd_gap_from_grid_percentage_points": (
                        thd - grid_best_thd_percent
                    ),
                    "phase_space_gap_degrees": _phase_space_gap_degrees(
                        candidate, grid_best_candidate
                    ),
                    "feasible": True,
                }
            )
    return rows
