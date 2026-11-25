"""Phase-aware circular PSO for carrier offsets in the ideal inverter."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable, Protocol, Sequence

import numpy as np

from evaluation.event_based_thd import EventBasedHarmonicResult, evaluate_event_based_thd


ANGLE_PERIOD_DEGREES = 360.0
BASELINE_PHASES_DEGREES = (0.0, 60.0, 120.0)
TRACKING_TOLERANCE_PERCENT = 0.05


def wrap_phase(angle_degrees: float) -> float:
    """Wrap one finite phase into [0, 360)."""
    value = float(angle_degrees)
    if not math.isfinite(value):
        raise ValueError("Phase angle must be finite.")
    wrapped = value % ANGLE_PERIOD_DEGREES
    return 0.0 if wrapped == ANGLE_PERIOD_DEGREES else wrapped


def signed_circular_difference(target_degrees: float, current_degrees: float) -> float:
    """Return shortest signed target-current phase difference in [-180, 180)."""
    target = wrap_phase(target_degrees)
    current = wrap_phase(current_degrees)
    return (target - current + 180.0) % ANGLE_PERIOD_DEGREES - 180.0


def advance_circular_position(
    position: Sequence[float],
    velocity: Sequence[float],
) -> np.ndarray:
    """Advance two phase coordinates and wrap each position into [0, 360)."""
    position_array = np.asarray(position, dtype=np.float64)
    velocity_array = np.asarray(velocity, dtype=np.float64)
    if position_array.shape != (2,) or velocity_array.shape != (2,):
        raise ValueError("Phase position and velocity must each contain two coordinates.")
    if not np.isfinite(position_array).all() or not np.isfinite(velocity_array).all():
        raise ValueError("Phase position and velocity must be finite.")
    return np.mod(position_array + velocity_array, ANGLE_PERIOD_DEGREES)


def canonicalize_candidate(candidate: Sequence[float]) -> tuple[float, float]:
    """Canonicalize interchangeable cell-2/cell-3 phases as a sorted pair."""
    values = np.asarray(candidate, dtype=np.float64)
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError("A candidate must contain exactly two finite phase angles.")
    return tuple(sorted((wrap_phase(float(values[0])), wrap_phase(float(values[1])))))


def candidate_to_carrier_offsets(candidate: Sequence[float]) -> tuple[float, float, float]:
    """Return canonical carrier offsets while keeping cell 1 fixed at zero."""
    second, third = canonicalize_candidate(candidate)
    return 0.0, second, third


def _matched_circular_displacement(
    target: Sequence[float],
    current: Sequence[float],
) -> np.ndarray:
    """Shortest pair displacement with interchangeable cells matched optimally."""
    target_pair = np.asarray(target, dtype=np.float64)
    current_pair = np.asarray(current, dtype=np.float64)
    if target_pair.shape != (2,) or current_pair.shape != (2,):
        raise ValueError("Circular phase positions must each contain two coordinates.")
    direct = np.array(
        [
            signed_circular_difference(target_pair[0], current_pair[0]),
            signed_circular_difference(target_pair[1], current_pair[1]),
        ],
        dtype=np.float64,
    )
    swapped = np.array(
        [
            signed_circular_difference(target_pair[1], current_pair[0]),
            signed_circular_difference(target_pair[0], current_pair[1]),
        ],
        dtype=np.float64,
    )
    return swapped if float(np.dot(swapped, swapped)) < float(np.dot(direct, direct)) else direct


def circular_velocity_update(
    current: Sequence[float],
    velocity: Sequence[float],
    personal_best: Sequence[float],
    global_best: Sequence[float],
    *,
    inertia: float,
    cognitive: float,
    social: float,
    random_personal: Sequence[float],
    random_global: Sequence[float],
) -> np.ndarray:
    """Apply the PSO velocity equation using shortest symmetry-aware arc deltas."""
    current_array = np.asarray(current, dtype=np.float64)
    velocity_array = np.asarray(velocity, dtype=np.float64)
    personal_random = np.asarray(random_personal, dtype=np.float64)
    global_random = np.asarray(random_global, dtype=np.float64)
    if any(array.shape != (2,) for array in (current_array, velocity_array, personal_random, global_random)):
        raise ValueError("Current phase, velocity, and random vectors must each have two values.")
    if not all(np.isfinite(array).all() for array in (current_array, velocity_array, personal_random, global_random)):
        raise ValueError("Circular PSO vectors must be finite.")
    if np.any((personal_random < 0.0) | (personal_random > 1.0)) or np.any(
        (global_random < 0.0) | (global_random > 1.0)
    ):
        raise ValueError("PSO random coefficients must be in [0, 1].")
    if not all(math.isfinite(value) and value >= 0.0 for value in (inertia, cognitive, social)):
        raise ValueError("PSO coefficients must be finite and non-negative.")
    return (
        inertia * velocity_array
        + cognitive
        * personal_random
        * _matched_circular_displacement(personal_best, current_array)
        + social
        * global_random
        * _matched_circular_displacement(global_best, current_array)
    )


@dataclass(frozen=True)
class OperatingPoint:
    sample_id: int
    modulation_index: float
    phase_angle_rad: float

    def __post_init__(self) -> None:
        if self.sample_id < 1:
            raise ValueError("sample_id must be a positive integer.")
        if not math.isfinite(self.modulation_index) or not 0.0 < self.modulation_index <= 1.0:
            raise ValueError("modulation_index must be finite and in (0, 1].")
        if not math.isfinite(self.phase_angle_rad):
            raise ValueError("phase_angle_rad must be finite.")


@dataclass(frozen=True)
class CandidateEvaluation:
    canonical_phases_degrees: tuple[float, float]
    mean_total_thd_percent: float
    maximum_tracking_error_percent: float
    feasible: bool
    worst_tracking_sample_id: int
    record_count: int

    def rank(self, tracking_tolerance_percent: float) -> tuple[float, float, float]:
        if self.feasible:
            return (0.0, self.mean_total_thd_percent, self.maximum_tracking_error_percent)
        violation = max(
            self.maximum_tracking_error_percent - tracking_tolerance_percent,
            0.0,
        )
        return (1.0, violation, self.mean_total_thd_percent)


class CandidateEvaluationError(RuntimeError):
    """A candidate could not be evaluated to valid finite electrical metrics."""


class PhaseEvaluation(Protocol):
    total_thd_percent: float
    fundamental_phasor_error_percent: float


Evaluator = Callable[..., PhaseEvaluation]


class PhaseOptimizationProblem:
    """Evaluate the hard-constrained mean Parseval THD on fixed operating points."""

    def __init__(
        self,
        operating_points: Sequence[OperatingPoint],
        *,
        tracking_tolerance_percent: float = TRACKING_TOLERANCE_PERCENT,
        evaluator: Evaluator = evaluate_event_based_thd,
    ) -> None:
        if not operating_points:
            raise ValueError("At least one operating point is required.")
        if not math.isfinite(tracking_tolerance_percent) or tracking_tolerance_percent <= 0.0:
            raise ValueError("tracking_tolerance_percent must be finite and positive.")
        self.operating_points = tuple(operating_points)
        self.tracking_tolerance_percent = float(tracking_tolerance_percent)
        self.evaluator = evaluator

    def evaluate(self, candidate: Sequence[float]) -> CandidateEvaluation:
        canonical = canonicalize_candidate(candidate)
        offsets = (0.0, *canonical)
        thd_values: list[float] = []
        tracking_values: list[tuple[int, float]] = []
        for point in self.operating_points:
            try:
                result = self.evaluator(
                    point.modulation_index,
                    point.phase_angle_rad,
                    carrier_phase_offsets_degrees=offsets,
                )
                thd = float(result.total_thd_percent)
                tracking = float(result.fundamental_phasor_error_percent)
            except Exception as exc:
                raise CandidateEvaluationError(
                    f"Evaluator failed for candidate {canonical} at SampleID "
                    f"{point.sample_id}: {type(exc).__name__}: {exc}"
                ) from exc
            if not math.isfinite(thd) or thd < 0.0:
                raise CandidateEvaluationError(
                    f"Evaluator returned invalid Parseval THD {thd!r} for "
                    f"candidate {canonical}, SampleID {point.sample_id}."
                )
            if not math.isfinite(tracking) or tracking < 0.0:
                raise CandidateEvaluationError(
                    f"Evaluator returned invalid phasor error {tracking!r} for "
                    f"candidate {canonical}, SampleID {point.sample_id}."
                )
            thd_values.append(thd)
            tracking_values.append((point.sample_id, tracking))
        worst_sample_id, maximum_tracking = max(tracking_values, key=lambda item: item[1])
        return CandidateEvaluation(
            canonical_phases_degrees=canonical,
            mean_total_thd_percent=float(np.mean(thd_values, dtype=np.float64)),
            maximum_tracking_error_percent=maximum_tracking,
            feasible=maximum_tracking <= self.tracking_tolerance_percent,
            worst_tracking_sample_id=worst_sample_id,
            record_count=len(self.operating_points),
        )


@dataclass(frozen=True)
class CircularPSOConfig:
    particles: int = 20
    iterations: int = 50
    seeds: tuple[int, ...] = tuple(range(20261120, 20261140))
    inertia: float = 0.7298
    cognitive: float = 1.49618
    social: float = 1.49618
    maximum_velocity_degrees: float = 72.0
    tracking_tolerance_percent: float = TRACKING_TOLERANCE_PERCENT

    def __post_init__(self) -> None:
        if self.particles < 2 or self.iterations < 1:
            raise ValueError("Circular PSO requires at least two particles and one iteration.")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("PSO seeds must be a non-empty sequence of unique integers.")
        if min(self.inertia, self.cognitive, self.social) < 0.0:
            raise ValueError("PSO coefficients must be non-negative.")
        if not math.isfinite(self.maximum_velocity_degrees) or self.maximum_velocity_degrees <= 0.0:
            raise ValueError("maximum_velocity_degrees must be finite and positive.")
        if not math.isfinite(self.tracking_tolerance_percent) or self.tracking_tolerance_percent <= 0.0:
            raise ValueError("tracking_tolerance_percent must be finite and positive.")


@dataclass(frozen=True)
class CircularPSORunResult:
    seed: int
    best: CandidateEvaluation
    convergence_history: tuple[CandidateEvaluation, ...]
    feasibility_history: tuple[int, ...]
    objective_requests: int
    unique_candidate_evaluations: int
    record_evaluations: int
    cache_hits: int
    runtime_seconds: float
    final_particles_feasible: int


class CircularPhasePSO:
    """Global-best PSO over a two-angle torus with cell-swap symmetry."""

    def __init__(self, config: CircularPSOConfig | None = None) -> None:
        self.config = config or CircularPSOConfig()

    def optimize(
        self,
        problem: PhaseOptimizationProblem,
        *,
        seed: int,
    ) -> CircularPSORunResult:
        if seed not in self.config.seeds:
            raise ValueError(f"Seed {seed} is not present in the configured seed list.")
        if not math.isclose(
            problem.tracking_tolerance_percent,
            self.config.tracking_tolerance_percent,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("PSO and objective tracking tolerances must match exactly.")
        started = time.perf_counter()
        rng = np.random.default_rng(seed)
        config = self.config
        positions = rng.uniform(0.0, ANGLE_PERIOD_DEGREES, size=(config.particles, 2))
        positions[0] = canonicalize_candidate(BASELINE_PHASES_DEGREES[1:])
        velocities = rng.uniform(
            -config.maximum_velocity_degrees,
            config.maximum_velocity_degrees,
            size=(config.particles, 2),
        )
        velocities[0] = 0.0

        cache: dict[tuple[float, float], CandidateEvaluation] = {}
        objective_requests = 0
        unique_evaluations = 0

        def score(position: np.ndarray) -> CandidateEvaluation:
            nonlocal objective_requests, unique_evaluations
            objective_requests += 1
            key = canonicalize_candidate(position)
            if key not in cache:
                cache[key] = problem.evaluate(key)
                unique_evaluations += 1
            return cache[key]

        scores = [score(position) for position in positions]
        personal_positions = positions.copy()
        personal_scores = list(scores)
        best_index = min(
            range(config.particles),
            key=lambda index: personal_scores[index].rank(problem.tracking_tolerance_percent),
        )
        global_position = personal_positions[best_index].copy()
        global_score = personal_scores[best_index]
        if not global_score.feasible:
            raise RuntimeError("Baseline incumbent was not feasible on the optimization subset.")

        history = [global_score]
        feasibility_history = [sum(result.feasible for result in scores)]
        for _ in range(config.iterations):
            for index in range(config.particles):
                velocities[index] = circular_velocity_update(
                    positions[index],
                    velocities[index],
                    personal_positions[index],
                    global_position,
                    inertia=config.inertia,
                    cognitive=config.cognitive,
                    social=config.social,
                    random_personal=rng.random(2),
                    random_global=rng.random(2),
                )
                velocities[index] = np.clip(
                    velocities[index],
                    -config.maximum_velocity_degrees,
                    config.maximum_velocity_degrees,
                )
                positions[index] = advance_circular_position(
                    positions[index],
                    velocities[index],
                )

            scores = [score(position) for position in positions]
            for index, result in enumerate(scores):
                if result.rank(problem.tracking_tolerance_percent) < personal_scores[index].rank(
                    problem.tracking_tolerance_percent
                ):
                    personal_scores[index] = result
                    personal_positions[index] = positions[index].copy()
                if result.rank(problem.tracking_tolerance_percent) < global_score.rank(
                    problem.tracking_tolerance_percent
                ):
                    global_score = result
                    global_position = positions[index].copy()
            history.append(global_score)
            feasibility_history.append(sum(result.feasible for result in scores))

        runtime = time.perf_counter() - started
        return CircularPSORunResult(
            seed=seed,
            best=global_score,
            convergence_history=tuple(history),
            feasibility_history=tuple(feasibility_history),
            objective_requests=objective_requests,
            unique_candidate_evaluations=unique_evaluations,
            record_evaluations=unique_evaluations * len(problem.operating_points),
            cache_hits=objective_requests - unique_evaluations,
            runtime_seconds=runtime,
            final_particles_feasible=feasibility_history[-1],
        )


__all__ = [
    "ANGLE_PERIOD_DEGREES",
    "BASELINE_PHASES_DEGREES",
    "TRACKING_TOLERANCE_PERCENT",
    "advance_circular_position",
    "CandidateEvaluation",
    "CandidateEvaluationError",
    "CircularPSOConfig",
    "CircularPSORunResult",
    "CircularPhasePSO",
    "OperatingPoint",
    "PhaseOptimizationProblem",
    "candidate_to_carrier_offsets",
    "canonicalize_candidate",
    "circular_velocity_update",
    "signed_circular_difference",
    "wrap_phase",
]
