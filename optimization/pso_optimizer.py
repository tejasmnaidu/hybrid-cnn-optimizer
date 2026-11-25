"""Bounded global-best particle swarm optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .optimizer_base import Objective, OptimizationResult, Optimizer


@dataclass(frozen=True)
class PSOConfig:
    bounds: tuple[tuple[float, float], ...]
    particles: int = 30
    iterations: int = 100
    inertia: float = 0.7
    cognitive: float = 1.5
    social: float = 1.5
    seed: int = 42
    velocity_fraction: float = 0.2

    def __post_init__(self) -> None:
        if not self.bounds or any(low >= high for low, high in self.bounds):
            raise ValueError("Every PSO bound must be a (low, high) pair with low < high.")
        if self.particles < 2 or self.iterations < 1:
            raise ValueError("PSO requires at least two particles and one iteration.")
        if min(self.inertia, self.cognitive, self.social) < 0:
            raise ValueError("PSO coefficients must be non-negative.")
        if self.velocity_fraction <= 0:
            raise ValueError("velocity_fraction must be positive.")


@dataclass(frozen=True)
class PSOResult(OptimizationResult):
    history: tuple[float, ...] = ()


class PSOOptimizer(Optimizer):
    def __init__(
        self,
        config: PSOConfig,
        *,
        feasible: Callable[[np.ndarray], bool] | None = None,
    ) -> None:
        self.config = config
        self.feasible = feasible

    def optimize(self, objective: Objective) -> PSOResult:
        rng = np.random.default_rng(self.config.seed)
        bounds = np.asarray(self.config.bounds, dtype=np.float64)
        low, high = bounds[:, 0], bounds[:, 1]
        span = high - low
        positions = rng.uniform(low, high, size=(self.config.particles, len(bounds)))
        velocities = rng.uniform(-span, span, size=positions.shape) * self.config.velocity_fraction
        scores = np.full(self.config.particles, np.inf)
        evaluations = 0

        def evaluate(position: np.ndarray) -> float:
            nonlocal evaluations
            evaluations += 1
            if self.feasible is not None and not self.feasible(position.copy()):
                return float("inf")
            value = float(objective(position.copy()))
            if not np.isfinite(value):
                return float("inf")
            return value

        for index in range(self.config.particles):
            scores[index] = evaluate(positions[index])
        personal_positions = positions.copy()
        personal_scores = scores.copy()
        best_index = int(np.argmin(scores))
        global_position = positions[best_index].copy()
        global_score = float(scores[best_index])
        history = [global_score]

        for _ in range(self.config.iterations):
            r_personal = rng.random(positions.shape)
            r_global = rng.random(positions.shape)
            velocities = (
                self.config.inertia * velocities
                + self.config.cognitive * r_personal * (personal_positions - positions)
                + self.config.social * r_global * (global_position - positions)
            )
            positions = np.clip(positions + velocities, low, high)
            for index in range(self.config.particles):
                score = evaluate(positions[index])
                if score < personal_scores[index]:
                    personal_scores[index] = score
                    personal_positions[index] = positions[index]
                if score < global_score:
                    global_score = score
                    global_position = positions[index].copy()
            history.append(global_score)
        if not np.isfinite(global_score):
            raise RuntimeError("PSO did not find a feasible candidate with a finite objective.")
        return PSOResult(
            position=global_position,
            objective_value=global_score,
            evaluations=evaluations,
            iterations=self.config.iterations,
            history=tuple(history),
        )
