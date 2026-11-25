"""Adapter interface for externally verified electrical/simulation evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EvaluationResult:
    """Measured metrics returned by a simulator or trusted electrical model."""

    harmonic_distortion_percent: float
    tracking_error: float
    switching_transitions: int
    output_levels: tuple[float, ...]
    constraints_satisfied: bool
    details: dict[str, Any] | None = None


class ElectricalEvaluator(ABC):
    """Implement this adapter to map candidate controls to measured outcomes.

    The dataset does not document how its flattened gate channels map to the
    three cell states, switch devices, or a candidate switching schedule. No
    electrical relationship or gate ordering is assumed here.
    """

    @abstractmethod
    def evaluate_candidate(self, candidate: np.ndarray) -> EvaluationResult:
        """Run the verified simulation/electrical model for one candidate."""
