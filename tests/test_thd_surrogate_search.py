from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from evaluation.event_based_thd import evaluate_event_based_thd
from optimization.circular_phase_pso import (
    BASELINE_PHASES_DEGREES,
    OperatingPoint,
    PhaseOptimizationProblem,
)
from optimization.thd_surrogate import (
    EventTHDSurrogate,
    THD_SURROGATE_FEATURE_NAMES,
    THD_SURROGATE_TARGET_NAMES,
)
from optimization.thd_surrogate_search import (
    EventTHDSurrogateArtifact,
    load_event_thd_surrogate,
    screen_phase_candidates,
)


def _zero_model() -> EventTHDSurrogate:
    model = EventTHDSurrogate()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    return model


def _artifact() -> EventTHDSurrogateArtifact:
    return EventTHDSurrogateArtifact(
        model=_zero_model(),
        target_mean=20.0,
        target_scale=1.0,
        dataset_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        split_policy="Modulation-group holdout test fixture.",
    )


def _problem() -> PhaseOptimizationProblem:
    return PhaseOptimizationProblem(
        (
            OperatingPoint(
                sample_id=1,
                modulation_index=0.75,
                phase_angle_rad=0.0,
            ),
        )
    )


class THDSurrogateSearchTests(unittest.TestCase):
    def test_checkpoint_loader_preserves_separate_model_and_provenance(self) -> None:
        model = _zero_model()
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "separate-surrogate.pt"
            torch.save(
                {
                    "model_type": "EventTHDSurrogate",
                    "feature_names": list(THD_SURROGATE_FEATURE_NAMES),
                    "target_names": list(THD_SURROGATE_TARGET_NAMES),
                    "model_config": {
                        "input_features": 7,
                        "hidden_width": 128,
                        "output_features": 1,
                    },
                    "model_state": model.state_dict(),
                    "target_mean": [20.0],
                    "target_scale": [2.0],
                    "dataset_sha256": "c" * 64,
                    "split_policy": "Modulation-group holdout test fixture.",
                },
                checkpoint_path,
            )
            artifact = load_event_thd_surrogate(checkpoint_path)
            self.assertEqual(artifact.target_mean, 20.0)
            self.assertEqual(artifact.target_scale, 2.0)
            self.assertEqual(artifact.dataset_sha256, "c" * 64)
            self.assertEqual(len(artifact.checkpoint_sha256), 64)

    def test_screening_is_bounded_reproducible_feasible_and_exactly_verified(self) -> None:
        problem = _problem()
        baseline = problem.evaluate(BASELINE_PHASES_DEGREES[1:])
        arguments = {
            "seed": 19,
            "baseline_evaluation": baseline,
            "candidate_count": 16,
            "finalist_count": 4,
        }
        first = screen_phase_candidates(problem, _artifact(), **arguments)
        second = screen_phase_candidates(problem, _artifact(), **arguments)

        self.assertEqual(first.screened_candidate_count, 16)
        self.assertEqual(first.objective_requests, 4)
        self.assertEqual(first.unique_candidate_evaluations, 4)
        self.assertEqual(first.cache_hits, 0)
        self.assertEqual(first.dataset_sha256, "a" * 64)
        self.assertEqual(first.checkpoint_sha256, "b" * 64)
        self.assertTrue(first.best.feasible)
        self.assertEqual(first.best.canonical_phases_degrees, second.best.canonical_phases_degrees)
        self.assertEqual(
            [item.mean_total_thd_percent for item in first.convergence_history],
            [item.mean_total_thd_percent for item in second.convergence_history],
        )
        self.assertEqual(len(first.convergence_history), len(first.feasibility_history))
        self.assertTrue(
            all(
                0.0 <= phase < 360.0
                for phase in first.best.canonical_phases_degrees
            )
        )

        independently_evaluated = evaluate_event_based_thd(
            0.75,
            0.0,
            carrier_phase_offsets_degrees=(
                0.0,
                *first.best.canonical_phases_degrees,
            ),
        )
        self.assertAlmostEqual(
            first.best.mean_total_thd_percent,
            independently_evaluated.total_thd_percent,
            places=10,
        )
        self.assertAlmostEqual(
            first.best.maximum_tracking_error_percent,
            independently_evaluated.fundamental_phasor_error_percent,
            places=10,
        )

    def test_invalid_budget_seed_baseline_and_surrogate_output_raise(self) -> None:
        problem = _problem()
        baseline = problem.evaluate(BASELINE_PHASES_DEGREES[1:])
        with self.assertRaises(ValueError):
            screen_phase_candidates(
                problem,
                _artifact(),
                seed=-1,
                baseline_evaluation=baseline,
            )
        with self.assertRaises(ValueError):
            screen_phase_candidates(
                problem,
                _artifact(),
                seed=1,
                baseline_evaluation=baseline,
                candidate_count=1,
            )
        with self.assertRaises(ValueError):
            screen_phase_candidates(
                problem,
                _artifact(),
                seed=1,
                baseline_evaluation=problem.evaluate((0.0, 0.0)),
            )
        negative_artifact = EventTHDSurrogateArtifact(
            model=_zero_model(),
            target_mean=-1.0,
            target_scale=1.0,
            dataset_sha256="a" * 64,
            checkpoint_sha256="b" * 64,
            split_policy="Modulation-group holdout test fixture.",
        )
        with self.assertRaises(RuntimeError):
            screen_phase_candidates(
                problem,
                negative_artifact,
                seed=1,
                baseline_evaluation=baseline,
                candidate_count=4,
                finalist_count=2,
            )


if __name__ == "__main__":
    unittest.main()
