from __future__ import annotations

import unittest
from types import SimpleNamespace

from optimization.circular_phase_pso import (
    OperatingPoint,
    PhaseOptimizationProblem,
)
from optimization.nature_optimizers import (
    HybridPOAPFO,
    HybridPOAPFOConfig,
    PFOConfig,
    PFOOptimizer,
    POAConfig,
    POAOptimizer,
)


def _result(thd: float, tracking: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(
        total_thd_percent=thd,
        fundamental_phasor_error_percent=tracking,
    )


class NatureOptimizerTests(unittest.TestCase):
    def _problem(self) -> PhaseOptimizationProblem:
        def evaluator(modulation: float, phase: float, *, carrier_phase_offsets_degrees):
            del modulation, phase
            second, third = carrier_phase_offsets_degrees[1:]
            thd = 10.0 + abs(second - 60.0) / 10.0 + abs(third - 120.0) / 10.0
            if second > 300.0:
                return _result(thd, 0.051)
            return _result(thd, 0.01)

        return PhaseOptimizationProblem(
            (OperatingPoint(1, 0.75, 0.0),),
            evaluator=evaluator,
        )

    def test_standalone_and_hybrid_are_reproducible_and_bounded(self) -> None:
        problem = self._problem()
        optimizers = (
            POAOptimizer(POAConfig(population_size=6, iterations=5, seed=1234)),
            PFOOptimizer(PFOConfig(population_size=6, iterations=5, seed=1234)),
            HybridPOAPFO(HybridPOAPFOConfig(population_size=6, iterations=5, seed=1234)),
        )
        for optimizer in optimizers:
            with self.subTest(name=type(optimizer).__name__):
                first = optimizer.optimize(problem)
                second = optimizer.optimize(problem)
                self.assertEqual(first.best.canonical_phases_degrees, second.best.canonical_phases_degrees)
                self.assertEqual(first.best.mean_total_thd_percent, second.best.mean_total_thd_percent)
                self.assertEqual(
                    [entry.canonical_phases_degrees for entry in first.convergence_history],
                    [entry.canonical_phases_degrees for entry in second.convergence_history],
                )
                second_phase, third_phase = first.best.canonical_phases_degrees
                self.assertTrue(0.0 <= second_phase < 360.0)
                self.assertTrue(0.0 <= third_phase < 360.0)

    def test_feasibility_history_and_convergence_are_tracked(self) -> None:
        problem = self._problem()
        result = HybridPOAPFO(
            HybridPOAPFOConfig(population_size=4, iterations=4, seed=20261120)
        ).optimize(problem)
        self.assertTrue(result.best.feasible)
        self.assertEqual(len(result.convergence_history), 5)
        self.assertEqual(len(result.feasibility_history), 5)
        self.assertGreaterEqual(result.unique_candidate_evaluations, 1)
        self.assertGreaterEqual(result.objective_requests, result.unique_candidate_evaluations)
        reevaluated = problem.evaluate(result.best.canonical_phases_degrees)
        self.assertEqual(reevaluated.canonical_phases_degrees, result.best.canonical_phases_degrees)
        self.assertEqual(reevaluated.mean_total_thd_percent, result.best.mean_total_thd_percent)

    def test_population_and_iteration_bounds_are_enforced(self) -> None:
        with self.assertRaises(ValueError):
            POAConfig(population_size=1)
        with self.assertRaises(ValueError):
            PFOConfig(iterations=0)
        with self.assertRaises(ValueError):
            HybridPOAPFOConfig(seed=-1)
        with self.assertRaises(ValueError):
            POAConfig(population_size=4, evaluation_budget=3)

    def test_distinct_candidates_are_evaluated_and_keep_baseline_incumbent(self) -> None:
        calls: list[tuple[float, float]] = []

        def evaluator(modulation: float, phase: float, *, carrier_phase_offsets_degrees):
            del modulation, phase
            second, third = carrier_phase_offsets_degrees[1:]
            calls.append((second, third))
            thd = 20.0 + abs(second - 60.0) / 10.0 + abs(third - 120.0) / 10.0
            return _result(thd, 0.01)

        problem = PhaseOptimizationProblem(
            (OperatingPoint(1, 0.75, 0.0),),
            evaluator=evaluator,
        )
        for optimizer in (
            POAOptimizer(POAConfig(population_size=5, iterations=4, seed=77)),
            PFOOptimizer(PFOConfig(population_size=5, iterations=4, seed=77)),
            HybridPOAPFO(HybridPOAPFOConfig(population_size=5, iterations=4, seed=77)),
        ):
            calls.clear()
            result = optimizer.optimize(problem)
            self.assertGreater(len(set(calls)), 1)
            reevaluated = problem.evaluate(result.best.canonical_phases_degrees)
            self.assertEqual(reevaluated.mean_total_thd_percent, result.best.mean_total_thd_percent)
