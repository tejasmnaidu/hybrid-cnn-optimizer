"""Weighted objective over measured electrical outcomes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .electrical_evaluator import ElectricalEvaluator, EvaluationResult
from .switching_constraints import SwitchingConstraintSpec


@dataclass(frozen=True)
class ObjectiveWeights:
    harmonic_distortion: float = 1.0
    tracking_error: float = 1.0
    switching_transitions: float = 0.0
    invalid_output_level: float = 1e6
    constraint_violation: float = 1e6

    def __post_init__(self) -> None:
        if any(value < 0 for value in self.__dict__.values()):
            raise ValueError("Objective weights and penalties must be non-negative.")


class HybridObjective:
    """Score only measurements supplied by an electrical evaluator.

    Weighted terms are user-configurable; penalty magnitudes are not derived
    from undocumented inverter physics and should be selected for the study.
    """

    def __init__(
        self,
        evaluator: ElectricalEvaluator,
        *,
        weights: ObjectiveWeights | None = None,
        constraints: SwitchingConstraintSpec | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.weights = weights or ObjectiveWeights()
        self.constraints = constraints or SwitchingConstraintSpec()

    def score_result(self, result: EvaluationResult) -> float:
        measurements = (
            result.harmonic_distortion_percent,
            result.tracking_error,
            float(result.switching_transitions),
        )
        if not np.isfinite(measurements).all() or min(measurements) < 0:
            raise ValueError("Electrical evaluator returned invalid or negative metrics.")
        allowed = np.asarray(self.constraints.allowed_output_levels)
        invalid_level = not result.output_levels or any(
            not np.isclose(level, allowed, atol=1e-6).any() for level in result.output_levels
        )
        too_many_transitions = (
            self.constraints.maximum_transitions is not None
            and result.switching_transitions > self.constraints.maximum_transitions
        )
        score = (
            self.weights.harmonic_distortion * result.harmonic_distortion_percent
            + self.weights.tracking_error * result.tracking_error
            + self.weights.switching_transitions * result.switching_transitions
        )
        if invalid_level:
            score += self.weights.invalid_output_level
        if not result.constraints_satisfied or too_many_transitions:
            score += self.weights.constraint_violation
        return float(score)

    def __call__(self, candidate: np.ndarray) -> float:
        result = self.evaluator.evaluate_candidate(np.asarray(candidate, dtype=np.float64))
        return self.score_result(result)
