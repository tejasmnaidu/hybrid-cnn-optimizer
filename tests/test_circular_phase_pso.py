from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from optimization.circular_phase_pso import (
    BASELINE_PHASES_DEGREES,
    CandidateEvaluationError,
    CircularPSOConfig,
    CircularPhasePSO,
    OperatingPoint,
    PhaseOptimizationProblem,
    advance_circular_position,
    candidate_to_carrier_offsets,
    canonicalize_candidate,
    circular_velocity_update,
    signed_circular_difference,
    wrap_phase,
)
from optimization.phase_pso_experiment import (
    load_operating_points,
    load_split_labels,
    select_stratified_training_subset,
)


def _result(
    total_thd_percent: float,
    tracking_error_percent: float = 0.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        total_thd_percent=total_thd_percent,
        fundamental_phasor_error_percent=tracking_error_percent,
    )


class CircularPhasePSOTests(unittest.TestCase):
    def test_shortest_circular_distance_across_zero(self) -> None:
        self.assertAlmostEqual(signed_circular_difference(1.0, 359.0), 2.0)
        self.assertAlmostEqual(signed_circular_difference(359.0, 1.0), -2.0)
        self.assertEqual(signed_circular_difference(180.0, 0.0), -180.0)

    def test_velocity_uses_shortest_path_angular_difference(self) -> None:
        updated = circular_velocity_update(
            (359.0, 90.0),
            (0.0, 0.0),
            (1.0, 90.0),
            (359.0, 90.0),
            inertia=0.0,
            cognitive=1.0,
            social=1.0,
            random_personal=(1.0, 1.0),
            random_global=(0.0, 0.0),
        )
        np.testing.assert_allclose(updated, (2.0, 0.0))

    def test_positions_wrap_modulo_without_clipping(self) -> None:
        self.assertAlmostEqual(wrap_phase(361.0), 1.0)
        self.assertAlmostEqual(wrap_phase(-1.0), 359.0)
        self.assertEqual(wrap_phase(720.0), 0.0)
        np.testing.assert_array_equal(
            advance_circular_position((359.0, 1.0), (2.0, -2.0)),
            np.asarray((1.0, 359.0)),
        )

    def test_cell_two_three_permutation_equivalence_is_canonicalized(self) -> None:
        self.assertEqual(canonicalize_candidate((30.0, 210.0)), (30.0, 210.0))
        self.assertEqual(canonicalize_candidate((210.0, 30.0)), (30.0, 210.0))
        self.assertEqual(canonicalize_candidate((390.0, -150.0)), (30.0, 210.0))

    def test_first_cell_phase_is_fixed_at_zero(self) -> None:
        self.assertEqual(candidate_to_carrier_offsets((120.0, 60.0)), (0.0, 60.0, 120.0))
        self.assertEqual(BASELINE_PHASES_DEGREES, (0.0, 60.0, 120.0))

    def test_objective_averages_parseval_thd_across_multiple_records(self) -> None:
        calls: list[tuple[float, float, tuple[float, ...]]] = []

        def fake_evaluator(modulation: float, phase: float, *, carrier_phase_offsets_degrees):
            calls.append((modulation, phase, carrier_phase_offsets_degrees))
            return _result(total_thd_percent=modulation * 10.0 + phase)

        points = (
            OperatingPoint(1, 0.5, 0.2),
            OperatingPoint(2, 0.8, 0.4),
            OperatingPoint(3, 0.7, 0.1),
        )
        result = PhaseOptimizationProblem(points, evaluator=fake_evaluator).evaluate((200.0, 30.0))
        expected = np.mean([5.2, 8.4, 7.1])
        self.assertAlmostEqual(result.mean_total_thd_percent, expected)
        self.assertEqual(result.record_count, 3)
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(call[2] == (0.0, 30.0, 200.0) for call in calls))

    def test_tracking_constraint_is_hard_and_baseline_incumbent_is_preserved(self) -> None:
        def fake_evaluator(_modulation: float, _phase: float, *, carrier_phase_offsets_degrees):
            phases = carrier_phase_offsets_degrees[1:]
            if phases == (60.0, 120.0):
                return _result(20.0, 0.0)
            return _result(0.01, 0.051)

        points = (OperatingPoint(1, 0.7, 0.0), OperatingPoint(2, 0.8, 1.2))
        problem = PhaseOptimizationProblem(points, evaluator=fake_evaluator)
        infeasible = problem.evaluate((10.0, 200.0))
        self.assertFalse(infeasible.feasible)
        self.assertGreater(infeasible.maximum_tracking_error_percent, 0.05)

        config = CircularPSOConfig(particles=5, iterations=4, seeds=(9,))
        result = CircularPhasePSO(config).optimize(problem, seed=9)
        self.assertTrue(result.best.feasible)
        self.assertEqual(result.best.canonical_phases_degrees, (60.0, 120.0))
        self.assertEqual(result.best.mean_total_thd_percent, 20.0)
        self.assertTrue(all(entry.feasible for entry in result.convergence_history))

    def test_deterministic_for_fixed_seed(self) -> None:
        def fake_evaluator(modulation: float, phase: float, *, carrier_phase_offsets_degrees):
            second, third = carrier_phase_offsets_degrees[1:]
            thd = 20.0 + abs(second - 45.0) / 100.0 + abs(third - 210.0) / 100.0
            return _result(thd + modulation * 0.01 + phase * 0.001)

        points = (OperatingPoint(1, 0.7, 0.5), OperatingPoint(2, 0.8, 1.0))
        config = CircularPSOConfig(particles=8, iterations=5, seeds=(1234,))
        optimizer = CircularPhasePSO(config)
        first = optimizer.optimize(PhaseOptimizationProblem(points, evaluator=fake_evaluator), seed=1234)
        second = optimizer.optimize(PhaseOptimizationProblem(points, evaluator=fake_evaluator), seed=1234)
        self.assertEqual(first.best.canonical_phases_degrees, second.best.canonical_phases_degrees)
        self.assertEqual(first.best.mean_total_thd_percent, second.best.mean_total_thd_percent)
        self.assertEqual(
            [entry.canonical_phases_degrees for entry in first.convergence_history],
            [entry.canonical_phases_degrees for entry in second.convergence_history],
        )

    def test_invalid_evaluator_metrics_raise_instead_of_becoming_good_scores(self) -> None:
        for invalid_result in (_result(float("nan")), _result(-1.0), _result(1.0, float("inf"))):
            with self.subTest(invalid_result=invalid_result):
                problem = PhaseOptimizationProblem(
                    (OperatingPoint(44, 0.7, 0.0),),
                    evaluator=lambda *_args, **_kwargs: invalid_result,
                )
                with self.assertRaises(CandidateEvaluationError):
                    problem.evaluate((30.0, 40.0))

    def test_evaluator_exceptions_are_reported_with_candidate_and_sample(self) -> None:
        def failing_evaluator(*_args, **_kwargs):
            raise RuntimeError("deliberate test failure")

        problem = PhaseOptimizationProblem(
            (OperatingPoint(44, 0.7, 0.0),),
            evaluator=failing_evaluator,
        )
        with self.assertRaisesRegex(CandidateEvaluationError, "SampleID 44"):
            problem.evaluate((30.0, 40.0))

    def test_stratified_subset_is_reproducible_across_modulation_and_phase(self) -> None:
        points = []
        sample_id = 1
        for modulation_group, modulation in enumerate((0.3, 0.4, 0.5, 0.6, 0.7)):
            for phase_group in range(6):
                points.append(
                    OperatingPoint(
                        sample_id,
                        modulation + phase_group * 1e-4 + modulation_group * 1e-5,
                        (phase_group + 0.5) * 2.0 * np.pi / 6.0,
                    )
                )
                sample_id += 1
        subset = select_stratified_training_subset(points)
        self.assertEqual(len(subset), 30)
        self.assertEqual(
            [point.sample_id for point in subset],
            [point.sample_id for point in select_stratified_training_subset(reversed(points))],
        )

    def test_split_loader_does_not_parse_testing_operating_condition_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split_file = root / "Sample_Splits.csv"
            conditions_file = root / "Operating_Conditions.csv"
            split_file.write_text(
                "sampleID,splitLabel\n1,training\n2,validation\n3,testing\n",
                encoding="utf-8",
            )
            conditions_file.write_text(
                "sampleID,modulationIndex,phaseAngle\n"
                "1,0.7,0.2\n"
                "2,0.8,0.3\n"
                "3,invalid-test-value,invalid-test-value\n",
                encoding="utf-8",
            )
            labels = load_split_labels(split_file)
            train = load_operating_points(conditions_file, labels, split="training")
            self.assertEqual([point.sample_id for point in train], [1])
            validation = load_operating_points(conditions_file, labels, split="validation")
            self.assertEqual([point.sample_id for point in validation], [2])
            with self.assertRaises(ValueError):
                load_operating_points(conditions_file, labels, split="testing")


if __name__ == "__main__":
    unittest.main()
