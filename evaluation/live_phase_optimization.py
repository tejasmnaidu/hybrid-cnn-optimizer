"""Short, reproducible training-subset phase optimization for live demos."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from evaluation.event_based_thd import evaluate_event_based_thd
from evaluation.thd_analysis import calculate_thd
from optimization.circular_phase_pso import (
    BASELINE_PHASES_DEGREES,
    TRACKING_TOLERANCE_PERCENT,
    CandidateEvaluation,
    CircularPSOConfig,
    CircularPSORunResult,
    CircularPhasePSO,
    OperatingPoint,
    PhaseOptimizationProblem,
)
from optimization.nature_optimizers import (
    HybridPOAPFO,
    HybridPOAPFOConfig,
    NatureRunResult,
    PFOConfig,
    PFOOptimizer,
    POAConfig,
    POAOptimizer,
)
from optimization.phase_pso_experiment import select_stratified_training_subset
from multilevel.inverter_simulator import simulate_inverter
from preprocessing.preprocess import PreparedDataset


@dataclass(frozen=True)
class LivePhaseOptimization:
    baseline_training_evaluation: CandidateEvaluation
    run_result: CircularPSORunResult
    training_sample_ids: tuple[int, ...]
    particles: int
    iterations: int


@dataclass(frozen=True)
class LiveConditionedPhaseOptimization:
    baseline_evaluation: CandidateEvaluation
    run_result: CircularPSORunResult | NatureRunResult
    optimizer_name: str
    modulation_index: float
    dc_voltage_per_cell_v: float
    load_resistance_ohm: float
    load_inductance_h: float
    phase_angle_degrees: float
    particles: int
    iterations: int


@dataclass(frozen=True)
class LiveInverterCandidateMetrics:
    total_thd_percent: float
    sampled_thd_percent: float
    fundamental_phasor_error_percent: float


def optimize_live_operating_point(
    *,
    modulation_index: float,
    dc_voltage_per_cell_v: float,
    load_resistance_ohm: float,
    load_inductance_h: float,
    phase_angle_degrees: float,
    seed: int,
    particles: int = 8,
    iterations: int = 8,
    optimizer_name: str = "PSO",
) -> LiveConditionedPhaseOptimization:
    """Optimize carrier phases by simulating each candidate at live inputs."""
    if not 0 <= seed <= 2**31 - 1:
        raise ValueError("seed must be an integer in [0, 2**31 - 1].")
    if not 2 <= particles <= 12:
        raise ValueError("particles must be between 2 and 12 for a live run.")
    if not 1 <= iterations <= 12:
        raise ValueError("iterations must be between 1 and 12 for a live run.")
    try:
        modulation, dc_voltage, resistance, inductance, phase_degrees = (
            float(modulation_index),
            float(dc_voltage_per_cell_v),
            float(load_resistance_ohm),
            float(load_inductance_h),
            float(phase_angle_degrees),
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Live optimization conditions must be finite numeric scalars.") from exc
    if not all(
        math.isfinite(value)
        for value in (modulation, dc_voltage, resistance, inductance, phase_degrees)
    ):
        raise ValueError("Live optimization conditions must be finite numeric scalars.")
    if not 0.55 <= modulation <= 0.95:
        raise ValueError("modulation_index must be in [0.55, 0.95].")
    if dc_voltage <= 0.0:
        raise ValueError("dc_voltage_per_cell_v must be positive.")
    if resistance not in (10.0, 20.0, 30.0, 40.0):
        raise ValueError("load_resistance_ohm must be a supported live-load value.")
    if inductance not in (0.01, 0.02, 0.03, 0.05):
        raise ValueError("load_inductance_h must be a supported live-load value.")
    if not 0.0 <= phase_degrees <= 360.0:
        raise ValueError("phase_angle_degrees must be in [0, 360].")

    phase_radians = math.radians(phase_degrees)
    operating_point = OperatingPoint(
        sample_id=1,
        modulation_index=modulation,
        phase_angle_rad=phase_radians,
    )

    def evaluate_live_candidate(
        candidate_modulation: float,
        candidate_phase_radians: float,
        *,
        carrier_phase_offsets_degrees: tuple[float, ...],
    ):
        simulation = simulate_inverter(
            candidate_modulation,
            resistance,
            inductance,
            candidate_phase_radians,
            carrier_phase_offsets_degrees=carrier_phase_offsets_degrees,
            dc_voltage_per_cell_v=dc_voltage,
        )
        if not np.isfinite(simulation.output_voltage).all() or not np.isfinite(
            simulation.load_current
        ).all():
            raise RuntimeError("Live candidate simulation produced non-finite waveforms.")
        event_metrics = evaluate_event_based_thd(
            candidate_modulation,
            candidate_phase_radians,
            carrier_phase_offsets_degrees=carrier_phase_offsets_degrees,
            dc_voltage_per_cell_v=dc_voltage,
        )
        return LiveInverterCandidateMetrics(
            total_thd_percent=event_metrics.total_thd_percent,
            sampled_thd_percent=float(calculate_thd(simulation.output_voltage)),
            fundamental_phasor_error_percent=(
                event_metrics.fundamental_phasor_error_percent
            ),
        )

    problem = PhaseOptimizationProblem(
        (operating_point,),
        tracking_tolerance_percent=TRACKING_TOLERANCE_PERCENT,
        evaluator=evaluate_live_candidate,
    )
    baseline = problem.evaluate(BASELINE_PHASES_DEGREES[1:])
    if not baseline.feasible:
        raise RuntimeError("Conventional phase shifts fail the live tracking constraint.")
    normalized_optimizer_name = optimizer_name.strip().upper()
    if normalized_optimizer_name == "PSO":
        config = CircularPSOConfig(
            particles=particles,
            iterations=iterations,
            seeds=(seed,),
            tracking_tolerance_percent=TRACKING_TOLERANCE_PERCENT,
        )
        run_result = CircularPhasePSO(config).optimize(problem, seed=seed)
    elif normalized_optimizer_name == "POA":
        run_result = POAOptimizer(
            POAConfig(population_size=particles, iterations=iterations, seed=seed)
        ).optimize(problem)
    elif normalized_optimizer_name == "PFO":
        run_result = PFOOptimizer(
            PFOConfig(population_size=particles, iterations=iterations, seed=seed)
        ).optimize(problem)
    elif normalized_optimizer_name == "HYBRID":
        run_result = HybridPOAPFO(
            HybridPOAPFOConfig(population_size=particles, iterations=iterations, seed=seed)
        ).optimize(problem)
    else:
        raise ValueError("optimizer_name must be PSO, POA, PFO, or HYBRID.")
    if not run_result.best.feasible:
        raise RuntimeError("Live PSO returned a candidate that violates tracking.")
    if not math.isfinite(run_result.runtime_seconds):
        raise RuntimeError("Live PSO reported a non-finite runtime.")
    return LiveConditionedPhaseOptimization(
        baseline_evaluation=baseline,
        run_result=run_result,
        modulation_index=modulation,
        dc_voltage_per_cell_v=dc_voltage,
        load_resistance_ohm=resistance,
        load_inductance_h=inductance,
        phase_angle_degrees=phase_degrees,
        particles=particles,
        iterations=iterations,
        optimizer_name=normalized_optimizer_name,
    )


def optimize_training_phase_subset(
    prepared: PreparedDataset,
    *,
    seed: int,
    particles: int = 8,
    iterations: int = 8,
) -> LivePhaseOptimization:
    """Optimize on the fixed 30-point training subset; never read validation/test."""
    if not 0 <= seed <= 2**31 - 1:
        raise ValueError("seed must be an integer in [0, 2**31 - 1].")
    if not 2 <= particles <= 12:
        raise ValueError("particles must be between 2 and 12 for a live run.")
    if not 1 <= iterations <= 12:
        raise ValueError("iterations must be between 1 and 12 for a live run.")
    if "modulationIndex" not in prepared.operating_feature_names:
        raise ValueError("Prepared dataset is missing the modulationIndex feature.")
    if "phaseAngle" not in prepared.operating_feature_names:
        raise ValueError("Prepared dataset is missing the phaseAngle feature.")

    training_indices = prepared.indices("training")
    if training_indices.size == 0:
        raise ValueError("The prepared dataset contains no training records.")
    raw_conditions = (
        prepared.operating_conditions[training_indices].astype(np.float64)
        * np.asarray(prepared.stats.operating_std, dtype=np.float64)
        + np.asarray(prepared.stats.operating_mean, dtype=np.float64)
    )
    modulation_index = prepared.operating_feature_names.index("modulationIndex")
    phase_index = prepared.operating_feature_names.index("phaseAngle")
    training_points = tuple(
        OperatingPoint(
            sample_id=int(prepared.sample_ids[row_index]),
            modulation_index=float(raw_conditions[position, modulation_index]),
            phase_angle_rad=float(raw_conditions[position, phase_index]),
        )
        for position, row_index in enumerate(training_indices)
    )
    subset = select_stratified_training_subset(training_points)
    if len(subset) != 30:
        raise RuntimeError(f"Expected a 30-record optimization subset, got {len(subset)}.")
    if any(
        str(prepared.splits[row_index]).strip().lower() != "training"
        for row_index in training_indices
    ):
        raise RuntimeError("Training subset construction encountered a non-training record.")

    config = CircularPSOConfig(
        particles=particles,
        iterations=iterations,
        seeds=(seed,),
        tracking_tolerance_percent=TRACKING_TOLERANCE_PERCENT,
    )
    problem = PhaseOptimizationProblem(
        subset,
        tracking_tolerance_percent=config.tracking_tolerance_percent,
    )
    baseline = problem.evaluate(BASELINE_PHASES_DEGREES[1:])
    if not baseline.feasible:
        raise RuntimeError(
            "The baseline phase configuration is infeasible on the selected "
            "training subset; refusing to run optimization."
        )
    result = CircularPhasePSO(config).optimize(problem, seed=seed)
    if not result.best.feasible:
        raise RuntimeError("The optimizer returned a tracking-infeasible candidate.")
    if result.best.mean_total_thd_percent > baseline.mean_total_thd_percent + 1e-12:
        raise RuntimeError("The optimizer returned a worse result than its baseline incumbent.")
    if not math.isfinite(result.runtime_seconds):
        raise RuntimeError("The optimizer reported a non-finite runtime.")
    return LivePhaseOptimization(
        baseline_training_evaluation=baseline,
        run_result=result,
        training_sample_ids=tuple(point.sample_id for point in subset),
        particles=particles,
        iterations=iterations,
    )
