"""Simulator-verified carrier-phase recommendations (not a controller)."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np

from evaluation.event_based_thd import evaluate_event_based_thd
from evaluation.thd_analysis import calculate_thd
from multilevel.inverter_simulator import (
    CELL_COUNT,
    FUNDAMENTAL_FREQUENCY_HZ,
    SAMPLES_PER_CYCLE,
    SAMPLING_FREQUENCY_HZ,
    InverterSimulation,
    simulate_inverter,
)
from optimization.circular_phase_pso import (
    BASELINE_PHASES_DEGREES,
    TRACKING_TOLERANCE_PERCENT,
    CandidateEvaluation,
    OperatingPoint,
    PhaseOptimizationProblem,
    canonicalize_candidate,
)
from optimization.nature_optimizers import (
    HybridPOAPFO,
    HybridPOAPFOConfig,
)


THDObjective = Literal["voltage", "current"]
_SWITCH_COUNT_PER_CELL = 4
_EVENT_CURRENT_MAX_HARMONIC = 100


@dataclass(frozen=True)
class SwitchingTHDMetrics:
    event_voltage_total_thd_percent: float
    event_voltage_harmonics_2_100_thd_percent: float
    sampled_voltage_thd_percent: float
    event_current_harmonics_2_100_thd_percent: float
    sampled_current_thd_percent: float


@dataclass(frozen=True)
class SwitchingRecommendation:
    objective: THDObjective
    carrier_phase_offsets_degrees: tuple[float, float, float]
    gate_states: np.ndarray
    baseline_thd_percent: float
    simulator_verified_thd_percent: float
    percent_change: float
    percent_reduction: float
    baseline_metrics: SwitchingTHDMetrics
    verified_metrics: SwitchingTHDMetrics
    seed: int
    unique_candidate_evaluations: int
    objective_requests: int
    runtime_seconds: float
    used_baseline_fallback: bool
    message: str
    predicted_thd_percent: None = None


@dataclass(frozen=True)
class _ObjectiveMetrics:
    total_thd_percent: float
    fundamental_phasor_error_percent: float


def _validate_scalar(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number.")
    return result


def _objective_thd(metrics: SwitchingTHDMetrics, objective: THDObjective) -> float:
    if objective == "voltage":
        return metrics.event_voltage_total_thd_percent
    return metrics.event_current_harmonics_2_100_thd_percent


def validate_gate_states(
    simulation: InverterSimulation,
    modulation_index: float,
) -> None:
    """Reject invalid modulation, cell states, gate dimensions, or shoot-through."""
    modulation = _validate_scalar(modulation_index, "modulation_index")
    if not 0.0 <= modulation <= 1.0:
        raise ValueError("modulation_index must be in [0.0, 1.0]; it is never clamped.")
    states = np.asarray(simulation.cell_states)
    gates = np.asarray(simulation.switching_gates)
    if states.shape != (CELL_COUNT, SAMPLES_PER_CYCLE):
        raise RuntimeError(
            "Simulator cell_states must have shape (3, 400), "
            f"got {states.shape}."
        )
    if not np.isin(states, (-1, 0, 1)).all():
        raise RuntimeError("Simulator returned a cell state outside -1, 0, +1.")
    expected_gate_shape = (
        CELL_COUNT,
        _SWITCH_COUNT_PER_CELL,
        SAMPLES_PER_CYCLE,
    )
    if gates.shape != expected_gate_shape:
        raise RuntimeError(
            "Simulator switching_gates must have shape (3, 4, 400), "
            f"got {gates.shape}."
        )
    if not np.isin(gates, (0, 1)).all():
        raise RuntimeError("Simulator gate states must be binary.")
    shoot_through = (
        (gates[:, 0, :] == 1) & (gates[:, 1, :] == 1)
    ) | (
        (gates[:, 2, :] == 1) & (gates[:, 3, :] == 1)
    )
    if np.any(shoot_through):
        raise RuntimeError("Simulator gate states contain shoot-through.")
    reconstructed_states = (
        gates[:, 0, :].astype(np.int8) * gates[:, 3, :].astype(np.int8)
        - gates[:, 1, :].astype(np.int8) * gates[:, 2, :].astype(np.int8)
    )
    if not np.array_equal(reconstructed_states, states):
        raise RuntimeError("Simulator gates do not reconstruct the reported cell states.")


def _event_current_thd_percent(
    event_result,
    *,
    resistance_ohm: float,
    inductance_h: float,
) -> float:
    orders = np.asarray(event_result.harmonic_orders, dtype=np.int64)
    voltage_amplitudes = np.asarray(
        event_result.harmonic_peak_amplitudes_v,
        dtype=np.float64,
    )
    expected_orders = np.arange(1, _EVENT_CURRENT_MAX_HARMONIC + 1)
    if not np.array_equal(orders, expected_orders):
        raise RuntimeError("Event THD must provide voltage harmonics 1 through 100.")
    if voltage_amplitudes.shape != orders.shape or not np.isfinite(
        voltage_amplitudes
    ).all():
        raise RuntimeError("Event THD returned invalid harmonic amplitudes.")
    angular_frequency = 2.0 * np.pi * FUNDAMENTAL_FREQUENCY_HZ * orders
    impedance = np.hypot(resistance_ohm, angular_frequency * inductance_h)
    current_amplitudes = voltage_amplitudes / impedance
    fundamental = float(current_amplitudes[0])
    if fundamental <= np.finfo(np.float64).eps:
        raise ValueError("Current THD is undefined when the fundamental is zero.")
    return float(
        np.sqrt(np.sum(np.square(current_amplitudes[1:])))
        / fundamental
        * 100.0
    )


def _simulate_and_measure(
    *,
    modulation_index: float,
    reference_phase_degrees: float,
    dc_voltage_v: float,
    resistance_ohm: float,
    inductance_h: float,
    phases: tuple[float, float],
) -> tuple[InverterSimulation, SwitchingTHDMetrics]:
    offsets = (0.0, *canonicalize_candidate(phases))
    phase_radians = math.radians(reference_phase_degrees)
    simulation = simulate_inverter(
        modulation_index,
        resistance_ohm,
        inductance_h,
        phase_radians,
        carrier_phase_offsets_degrees=offsets,
        dc_voltage_per_cell_v=dc_voltage_v,
    )
    validate_gate_states(simulation, modulation_index)
    event = evaluate_event_based_thd(
        modulation_index,
        phase_radians,
        carrier_phase_offsets_degrees=offsets,
        dc_voltage_per_cell_v=dc_voltage_v,
    )
    metrics = SwitchingTHDMetrics(
        event_voltage_total_thd_percent=float(event.total_thd_percent),
        event_voltage_harmonics_2_100_thd_percent=float(event.thd_percent),
        sampled_voltage_thd_percent=float(
            calculate_thd(
                simulation.output_voltage,
                sampling_frequency_hz=SAMPLING_FREQUENCY_HZ,
                fundamental_frequency_hz=FUNDAMENTAL_FREQUENCY_HZ,
                max_harmonic=100,
            )
        ),
        event_current_harmonics_2_100_thd_percent=_event_current_thd_percent(
            event,
            resistance_ohm=resistance_ohm,
            inductance_h=inductance_h,
        ),
        sampled_current_thd_percent=float(
            calculate_thd(
                simulation.load_current,
                sampling_frequency_hz=SAMPLING_FREQUENCY_HZ,
                fundamental_frequency_hz=FUNDAMENTAL_FREQUENCY_HZ,
                max_harmonic=100,
            )
        ),
    )
    return simulation, metrics


def recommend_switching(
    *,
    modulation_index: float,
    reference_phase_degrees: float,
    dc_voltage_v: float,
    resistance_ohm: float,
    inductance_h: float,
    objective: THDObjective = "voltage",
    seed: int = 42,
    population_size: int = 8,
    iterations: int = 8,
) -> SwitchingRecommendation:
    """Recommend safe carrier offsets using hybrid POA-PFO and verify by simulation.

    The optimizer selects phase offsets only. The existing inverter simulator,
    not an AI model, generates binary gate states and produces final verified
    THD. This function is not a hardware controller or protection system.
    """
    modulation = _validate_scalar(modulation_index, "modulation_index")
    if not 0.0 <= modulation <= 1.0:
        raise ValueError("modulation_index must be in [0.0, 1.0]; it is never clamped.")
    if modulation == 0.0:
        raise ValueError("THD is undefined at zero modulation because the fundamental is zero.")
    reference_phase = _validate_scalar(
        reference_phase_degrees,
        "reference_phase_degrees",
    )
    if not 0.0 <= reference_phase <= 360.0:
        raise ValueError("reference_phase_degrees must be in [0, 360].")
    dc_voltage = _validate_scalar(dc_voltage_v, "dc_voltage_v")
    resistance = _validate_scalar(resistance_ohm, "resistance_ohm")
    inductance = _validate_scalar(inductance_h, "inductance_h")
    if dc_voltage <= 0.0 or resistance <= 0.0 or inductance <= 0.0:
        raise ValueError("DC voltage, resistance, and inductance must be positive.")
    if objective not in ("voltage", "current"):
        raise ValueError("objective must be 'voltage' or 'current'.")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 2**31 - 1:
        raise ValueError("seed must be an integer in [0, 2**31 - 1].")
    if (
        not isinstance(population_size, int)
        or isinstance(population_size, bool)
        or not isinstance(iterations, int)
        or isinstance(iterations, bool)
        or population_size < 2
        or iterations < 1
    ):
        raise ValueError("population_size must be at least 2 and iterations at least 1.")

    baseline_phases = canonicalize_candidate(BASELINE_PHASES_DEGREES[1:])
    baseline_simulation, baseline_metrics = _simulate_and_measure(
        modulation_index=modulation,
        reference_phase_degrees=reference_phase,
        dc_voltage_v=dc_voltage,
        resistance_ohm=resistance,
        inductance_h=inductance,
        phases=baseline_phases,
    )
    baseline_objective = _objective_thd(baseline_metrics, objective)
    operating_point = OperatingPoint(
        sample_id=1,
        modulation_index=modulation,
        phase_angle_rad=math.radians(reference_phase),
    )

    def evaluate_candidate(
        candidate_modulation: float,
        phase_angle_rad: float,
        *,
        carrier_phase_offsets_degrees: tuple[float, ...],
    ) -> _ObjectiveMetrics:
        event = evaluate_event_based_thd(
            candidate_modulation,
            phase_angle_rad,
            carrier_phase_offsets_degrees=carrier_phase_offsets_degrees,
            dc_voltage_per_cell_v=dc_voltage,
        )
        if objective == "voltage":
            objective_value = float(event.total_thd_percent)
        else:
            objective_value = _event_current_thd_percent(
                event,
                resistance_ohm=resistance,
                inductance_h=inductance,
            )
        return _ObjectiveMetrics(
            total_thd_percent=objective_value,
            fundamental_phasor_error_percent=float(
                event.fundamental_phasor_error_percent
            ),
        )

    problem = PhaseOptimizationProblem(
        (operating_point,),
        tracking_tolerance_percent=TRACKING_TOLERANCE_PERCENT,
        evaluator=evaluate_candidate,
    )
    fallback = False
    failure_reason = ""
    unique_evaluations = 0
    objective_requests = 0
    runtime_seconds = 0.0
    try:
        baseline_evaluation = problem.evaluate(baseline_phases)
        if not baseline_evaluation.feasible:
            raise RuntimeError(
                "The conventional baseline violates the fundamental tracking constraint."
            )
        run = HybridPOAPFO(
            HybridPOAPFOConfig(
                population_size=population_size,
                iterations=iterations,
                seed=seed,
            )
        ).optimize(problem)
        unique_evaluations = int(run.unique_candidate_evaluations)
        objective_requests = int(run.objective_requests)
        runtime_seconds = float(run.runtime_seconds)
        candidate_phases = canonicalize_candidate(
            run.best.canonical_phases_degrees
        )
        if not run.best.feasible:
            raise RuntimeError("Hybrid POA-PFO returned an infeasible phase candidate.")
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        fallback = True
        failure_reason = f"Optimization failed ({type(exc).__name__}: {exc}). "
        candidate_phases = baseline_phases

    try:
        if candidate_phases == baseline_phases:
            final_simulation, final_metrics = baseline_simulation, baseline_metrics
        else:
            final_simulation, final_metrics = _simulate_and_measure(
                modulation_index=modulation,
                reference_phase_degrees=reference_phase,
                dc_voltage_v=dc_voltage,
                resistance_ohm=resistance,
                inductance_h=inductance,
                phases=candidate_phases,
            )
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        fallback = True
        failure_reason = (
            f"Simulator verification failed ({type(exc).__name__}: {exc}). "
        )
        candidate_phases = baseline_phases
        final_simulation, final_metrics = baseline_simulation, baseline_metrics
    verified_objective = _objective_thd(final_metrics, objective)
    if verified_objective > baseline_objective + 1e-12:
        fallback = True
        failure_reason = "Verified objective was worse than baseline. "
        candidate_phases = baseline_phases
        final_simulation, final_metrics = baseline_simulation, baseline_metrics
        verified_objective = _objective_thd(final_metrics, objective)

    percent_change = (
        (verified_objective - baseline_objective) / baseline_objective * 100.0
        if baseline_objective > 0.0
        else 0.0
    )
    percent_reduction = -percent_change
    improved = verified_objective < baseline_objective - 1e-12
    message = (
        "Simulation-verified improvement found for this operating point."
        if improved
        else "No improvement found; the baseline carrier phases are recommended."
    )
    if failure_reason:
        message = failure_reason + message

    gate_states = np.array(final_simulation.switching_gates, copy=True)
    gate_states.setflags(write=False)
    return SwitchingRecommendation(
        objective=objective,
        carrier_phase_offsets_degrees=(0.0, *candidate_phases),
        gate_states=gate_states,
        baseline_thd_percent=baseline_objective,
        simulator_verified_thd_percent=verified_objective,
        percent_change=percent_change,
        percent_reduction=percent_reduction,
        baseline_metrics=baseline_metrics,
        verified_metrics=final_metrics,
        seed=seed,
        unique_candidate_evaluations=unique_evaluations,
        objective_requests=objective_requests,
        runtime_seconds=runtime_seconds,
        used_baseline_fallback=fallback or not improved,
        message=message,
    )


__all__ = [
    "SwitchingRecommendation",
    "SwitchingTHDMetrics",
    "recommend_switching",
    "validate_gate_states",
]
