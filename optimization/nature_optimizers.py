"""Genuine Pelican and Pufferfish optimization variants for circular phases.

Pelican Optimization Algorithm (POA), Trojovsky and Dehghani (2022):
  Phase 1:
    xP1 = x + rand*(prey - I*x) if F(prey) < F(x), otherwise
          x + rand*(x - prey), with I in {1, 2}.
  Phase 2:
    xP2 = x + R*(1-t/T)*(2*rand-1)*x, R=0.2 by default.

Pufferfish Optimization Algorithm (PFO), Darvishpoor et al. (2024):
  Phase 1:
    CPi = {Xk: Fk < Fi, k != i}; choose SPi randomly from CPi.
    xP1 = x + rand*(SPi - I*x), I in {1, 2}.
    If CPi is empty, no attack proposal is made.
  Phase 2:
    xP2 = x + (1-2*rand)*(ub-lb)/t.

All proposed points are canonicalized and wrapped on the 360-degree torus.
Candidates are evaluated through PhaseOptimizationProblem, so hard tracking
feasibility and candidate ranking remain unchanged. No fixed training record
is introduced into the live operating-point objective.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Sequence

import numpy as np

from .circular_phase_pso import (
    CandidateEvaluation,
    PhaseOptimizationProblem,
    canonicalize_candidate,
)


ANGLE_PERIOD_DEGREES = 360.0


@dataclass(frozen=True)
class NatureOptimizerConfig:
    population_size: int = 8
    iterations: int = 8
    seed: int = 42
    evaluation_budget: int | None = None

    def __post_init__(self) -> None:
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2.")
        if self.iterations < 1:
            raise ValueError("iterations must be at least 1.")
        if not 0 <= self.seed <= 2**31 - 1:
            raise ValueError("seed must be in [0, 2**31 - 1].")
        if self.evaluation_budget is not None:
            if self.evaluation_budget < self.population_size:
                raise ValueError("evaluation_budget must cover the initial population.")


@dataclass(frozen=True)
class NatureRunResult:
    seed: int
    best: CandidateEvaluation
    convergence_history: tuple[CandidateEvaluation, ...]
    feasibility_history: tuple[int, ...]
    objective_requests: int
    unique_candidate_evaluations: int
    cache_hits: int
    runtime_seconds: float


POAConfig = NatureOptimizerConfig
PFOConfig = NatureOptimizerConfig
POAResult = NatureRunResult
PFOResult = NatureRunResult
HybridPOAPFOConfig = NatureOptimizerConfig
HybridPOAPFORunResult = NatureRunResult


def _wrap(candidate: np.ndarray) -> np.ndarray:
    return np.mod(np.asarray(candidate, dtype=np.float64), ANGLE_PERIOD_DEGREES)


def _canonical(candidate: Sequence[float]) -> tuple[float, float]:
    return canonicalize_candidate(candidate)


class _NatureBase:
    def __init__(self, config: NatureOptimizerConfig) -> None:
        self.config = config

    def _run(
        self,
        problem: PhaseOptimizationProblem,
        *,
        mechanism: str,
    ) -> NatureRunResult:
        started = time.perf_counter()
        rng = np.random.default_rng(self.config.seed)
        evaluation_budget = (
            self.config.evaluation_budget
            if self.config.evaluation_budget is not None
            else self.config.population_size * (self.config.iterations + 1)
        )
        population = rng.uniform(
            0.0,
            ANGLE_PERIOD_DEGREES,
            size=(self.config.population_size, 2),
        )
        population = np.asarray([_canonical(row) for row in population], dtype=np.float64)
        population[0] = (60.0, 120.0)

        cache: dict[tuple[float, float], CandidateEvaluation] = {}
        objective_requests = 0
        baseline_key = (60.0, 120.0)
        cache[baseline_key] = problem.evaluate(baseline_key)
        objective_requests += 1

        def score(raw_position: np.ndarray) -> CandidateEvaluation:
            nonlocal objective_requests
            objective_requests += 1
            position = _wrap(raw_position)
            key = _canonical(position)
            if key == baseline_key:
                return cache[baseline_key]
            if key not in cache:
                if objective_requests > evaluation_budget:
                    objective_requests -= 1
                    return cache[baseline_key]
                cache[key] = problem.evaluate(key)
            return cache[key]

        scores = [score(row) for row in population]
        personal_positions = population.copy()
        personal_scores = list(scores)
        baseline = cache[baseline_key]
        if not baseline.feasible:
            raise RuntimeError("The baseline incumbent is infeasible.")

        best_index = min(
            range(len(scores)),
            key=lambda index: personal_scores[index].rank(problem.tracking_tolerance_percent),
        )
        best_position = personal_positions[best_index].copy()
        best_score = personal_scores[best_index]

        history = [best_score]
        feasibility_history = [sum(result.feasible for result in scores)]

        for iteration in range(1, self.config.iterations + 1):
            proposed_positions: list[np.ndarray] = []
            for index, position in enumerate(population):
                if mechanism == "hybrid" and iteration % 2 == 1:
                    selected_mechanism = "poa"
                elif mechanism == "hybrid":
                    selected_mechanism = "pfo"
                else:
                    selected_mechanism = mechanism

                if selected_mechanism == "poa":
                    prey = rng.uniform(0.0, ANGLE_PERIOD_DEGREES, size=2)
                    if score(prey).mean_total_thd_percent < scores[index].mean_total_thd_percent:
                        i_factor = rng.integers(1, 3)
                        proposal = position + rng.random(2) * (
                            prey - i_factor * position
                        )
                    else:
                        proposal = position + rng.random(2) * (position - prey)
                    proposal = _wrap(proposal)

                    proposed = score(proposal)
                    if proposed.rank(problem.tracking_tolerance_percent) < scores[index].rank(
                        problem.tracking_tolerance_percent
                    ):
                        position = proposal
                        scores[index] = proposed

                    local_scale = 0.2 * (1.0 - iteration / self.config.iterations)
                    local = position + local_scale * (2.0 * rng.random(2) - 1.0) * position
                    local = _wrap(local)
                    local_score = score(local)
                    if local_score.rank(problem.tracking_tolerance_percent) < scores[index].rank(
                        problem.tracking_tolerance_percent
                    ):
                        position = local
                        scores[index] = local_score

                else:
                    better = [
                        personal_positions[k]
                        for k in range(len(population))
                        if personal_scores[k].rank(problem.tracking_tolerance_percent)
                        < scores[index].rank(problem.tracking_tolerance_percent)
                        and k != index
                    ]
                    if better:
                        target = better[rng.integers(0, len(better))]
                        i_factor = rng.integers(1, 3, size=2)
                        proposal = position + rng.random(2) * (
                            target - i_factor * position
                        )
                    else:
                        proposal = position.copy()
                    proposal = _wrap(proposal)
                    proposed = score(proposal)
                    if proposed.rank(problem.tracking_tolerance_percent) < scores[index].rank(
                        problem.tracking_tolerance_percent
                    ):
                        position = proposal
                        scores[index] = proposed

                    # PFO defense: shrinking predator escape away from the defended pufferfish.
                    defense = position + (
                        (1.0 - 2.0 * rng.random(2))
                        * (ANGLE_PERIOD_DEGREES / iteration)
                    )
                    defense = _wrap(defense)
                    defense_score = score(defense)
                    if defense_score.rank(problem.tracking_tolerance_percent) < scores[index].rank(
                        problem.tracking_tolerance_percent
                    ):
                        position = defense
                        scores[index] = defense_score

                proposed_positions.append(position)
                if scores[index].rank(problem.tracking_tolerance_percent) < personal_scores[index].rank(
                    problem.tracking_tolerance_percent
                ):
                    personal_scores[index] = scores[index]
                    personal_positions[index] = position.copy()

            population = np.asarray(proposed_positions, dtype=np.float64)
            current_best = min(
                range(len(population)),
                key=lambda index: scores[index].rank(problem.tracking_tolerance_percent),
            )
            if scores[current_best].rank(problem.tracking_tolerance_percent) < best_score.rank(
                problem.tracking_tolerance_percent
            ):
                best_score = scores[current_best]
                best_position = population[current_best].copy()
            history.append(best_score)
            feasibility_history.append(sum(result.feasible for result in scores))

        runtime = time.perf_counter() - started
        return NatureRunResult(
            seed=self.config.seed,
            best=best_score,
            convergence_history=tuple(history),
            feasibility_history=tuple(feasibility_history),
            objective_requests=objective_requests,
            unique_candidate_evaluations=len(cache),
            cache_hits=objective_requests - len(cache),
            runtime_seconds=runtime,
        )


class POAOptimizer(_NatureBase):
    """Pelican Optimization Algorithm: prey-diving exploration plus surface-gliding exploitation."""

    def optimize(self, problem: PhaseOptimizationProblem) -> POAResult:
        return self._run(problem, mechanism="poa")


class PFOOptimizer(_NatureBase):
    """Pufferfish Optimization Algorithm: better-member attack plus shrinking defense escape."""

    def optimize(self, problem: PhaseOptimizationProblem) -> PFOResult:
        return self._run(problem, mechanism="pfo")


class HybridPOAPFO(_NatureBase):
    """Alternates published POA and PFO updates on odd/even iterations with a shared pool."""

    def optimize(self, problem: PhaseOptimizationProblem) -> HybridPOAPFORunResult:
        return self._run(problem, mechanism="hybrid")


__all__ = [
    "NatureOptimizerConfig",
    "NatureRunResult",
    "POAConfig",
    "POAOptimizer",
    "POAResult",
    "PFOConfig",
    "PFOOptimizer",
    "PFOResult",
    "HybridPOAPFOConfig",
    "HybridPOAPFORunResult",
    "HybridPOAPFO",
]
