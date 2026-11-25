"""Common optimization interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np


Objective = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class OptimizationResult:
    position: np.ndarray
    objective_value: float
    evaluations: int
    iterations: int


class Optimizer(ABC):
    @abstractmethod
    def optimize(self, objective: Objective) -> OptimizationResult:
        """Minimize objective over the configured search domain."""
