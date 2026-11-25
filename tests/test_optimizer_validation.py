from __future__ import annotations

import unittest

import numpy as np

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
from optimization.pso_optimizer import PSOConfig, PSOOptimizer


class OptimizerValidationTests(unittest.TestCase):
    def test_real_interface_landscape_is_not_flat(self) -> None:
        def evaluator(_modulation, _phase, *, carrier_phase_offsets_degrees):
            second, third = carrier_phase_offsets_degrees[1:]
            thd = 20.0 + abs(second - 60.0) / 10.0 + abs(third - 120.0) / 10.0
            return type(
                "Result",
                (),
                {
                    "total_thd_percent": thd,
                    "fundamental_phasor_error_percent": 0.01,
                },
            )()

        problem = PhaseOptimizationProblem(
            (OperatingPoint(1, 0.75, 0.0),),
            evaluator=evaluator,
        )
        baseline = problem.evaluate((60.0, 120.0))
        alternate = problem.evaluate((50.0, 110.0))
        self.assertNotAlmostEqual(
            baseline.mean_total_thd_percent,
            alternate.mean_total_thd_percent,
        )

    def test_nature_optimizers_improve_sphere_toy_function(self) -> None:
        def sphere(modulation, phase, *, carrier_phase_offsets_degrees):
            del modulation, phase
            second, third = carrier_phase_offsets_degrees[1:]
            value = ((second - 60.0) ** 2 + (third - 120.0) ** 2) / 100.0
            return type(
                "Result",
                (),
                {
                    "total_thd_percent": value,
                    "fundamental_phasor_error_percent": 0.01,
                },
            )()

        problem = PhaseOptimizationProblem(
            (OperatingPoint(1, 0.75, 0.0),),
            evaluator=sphere,
        )
        baseline = problem.evaluate((60.0, 120.0))
        for optimizer in (
            POAOptimizer(POAConfig(population_size=10, iterations=20, seed=1234)),
            PFOOptimizer(PFOConfig(population_size=10, iterations=20, seed=1234)),
            HybridPOAPFO(
                HybridPOAPFOConfig(population_size=10, iterations=20, seed=1234)
            ),
        ):
            result = optimizer.optimize(problem)
            reevaluated = problem.evaluate(result.best.canonical_phases_degrees)
            self.assertLessEqual(
                reevaluated.mean_total_thd_percent,
                baseline.mean_total_thd_percent + 1e-12,
            )
            np.testing.assert_allclose(
                reevaluated.canonical_phases_degrees,
                (60.0, 120.0),
                rtol=0.0,
                atol=1e-6,
            )

    def test_nature_optimizers_improve_rastrigin_toy_function(self) -> None:
        def rastrigin(modulation, phase, *, carrier_phase_offsets_degrees):
            del modulation, phase
            second, third = carrier_phase_offsets_degrees[1:]
            x = (second - 70.0) / 10.0
            y = (third - 130.0) / 10.0
            value = 20.0 + x * x - 10.0 * np.cos(2.0 * np.pi * x) + y * y - 10.0 * np.cos(2.0 * np.pi * y)
            return type(
                "Result",
                (),
                {
                    "total_thd_percent": value,
                    "fundamental_phasor_error_percent": 0.01,
                },
            )()

        problem = PhaseOptimizationProblem(
            (OperatingPoint(1, 0.75, 0.0),),
            evaluator=rastrigin,
        )
        baseline = problem.evaluate((60.0, 120.0))
        for optimizer in (
            POAOptimizer(POAConfig(population_size=20, iterations=60, seed=4321)),
            PFOOptimizer(PFOConfig(population_size=20, iterations=60, seed=4321)),
            HybridPOAPFO(
                HybridPOAPFOConfig(population_size=20, iterations=60, seed=4321)
            ),
        ):
            result = optimizer.optimize(problem)
            reevaluated = problem.evaluate(result.best.canonical_phases_degrees)
            self.assertLess(
                reevaluated.mean_total_thd_percent,
                baseline.mean_total_thd_percent - 1e-12,
            )

    def test_generic_pso_toy_function_respects_bounds_and_reproducibility(self) -> None:
        def sphere(position):
            return float(np.sum(np.square(position)))

        config = PSOConfig(
            bounds=((0.0, 10.0), (0.0, 10.0)),
            particles=12,
            iterations=40,
            seed=1234,
        )
        optimizer = PSOOptimizer(config)
        first = optimizer.optimize(sphere)
        second = optimizer.optimize(sphere)
        self.assertAlmostEqual(first.objective_value, second.objective_value)
        np.testing.assert_array_equal(first.position, second.position)
        self.assertTrue(np.all(first.position >= 0.0))
        self.assertTrue(np.all(first.position <= 10.0))
        self.assertLess(first.objective_value, 0.1)

    def test_cache_matches_fresh_evaluation_and_repeated_candidates(self) -> None:
        calls = []

        def evaluator(_modulation, _phase, *, carrier_phase_offsets_degrees):
            phases = tuple(float(value) for value in carrier_phase_offsets_degrees)
            calls.append(phases)
            value = (phases[1] - 60.0) ** 2 + (phases[2] - 120.0) ** 2
            return type(
                "Result",
                (),
                {
                    "total_thd_percent": value,
                    "fundamental_phasor_error_percent": 0.01,
                },
            )()

        problem = PhaseOptimizationProblem(
            (OperatingPoint(1, 0.75, 0.0),),
            evaluator=evaluator,
        )
        first = problem.evaluate((60.0, 120.0))
        second = problem.evaluate((60.0, 120.0))
        fresh = problem.evaluate((60.0, 120.0))
        self.assertEqual(first.mean_total_thd_percent, 0.0)
        self.assertEqual(second.mean_total_thd_percent, fresh.mean_total_thd_percent)
        self.assertEqual(len(calls), 3)


if __name__ == "__main__":
    unittest.main()
