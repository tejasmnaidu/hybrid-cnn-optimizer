from __future__ import annotations

import unittest

import numpy as np

from evaluation.live_phase_optimization import optimize_training_phase_subset
from preprocessing.preprocess import NormalizationStats, PreparedDataset


def _prepared_training_fixture() -> PreparedDataset:
    sample_ids: list[int] = []
    splits: list[str] = []
    conditions: list[list[float]] = []
    for modulation_group in range(5):
        for phase_bin in range(6):
            for repetition in range(2):
                sample_ids.append(len(sample_ids) + 1)
                splits.append("training")
                modulation = (
                    0.56
                    + modulation_group * 0.07
                    + phase_bin * 0.0005
                    + repetition * 0.0001
                )
                phase = (phase_bin + 0.5) * 2.0 * np.pi / 6.0
                conditions.append([modulation, 30.0, 0.03, phase])
    for split in ("validation", "testing"):
        sample_ids.append(len(sample_ids) + 1)
        splits.append(split)
        conditions.append([0.75, 30.0, 0.03, 0.0])

    stats = NormalizationStats(
        input_voltage_mean=0.0,
        input_voltage_std=1.0,
        load_current_mean=0.0,
        load_current_std=1.0,
        operating_mean=[0.0, 0.0, 0.0, 0.0],
        operating_std=[1.0, 1.0, 1.0, 1.0],
        target_voltage_mean=0.0,
        target_voltage_std=1.0,
    )
    count = len(sample_ids)
    return PreparedDataset(
        sample_ids=np.asarray(sample_ids, dtype=np.int64),
        splits=np.asarray(splits),
        input_voltage=np.zeros((count, 400), dtype=np.float32),
        load_current=np.zeros((count, 400), dtype=np.float32),
        operating_conditions=np.asarray(conditions, dtype=np.float32),
        target_voltage=np.zeros((count, 400), dtype=np.float32),
        stats=stats,
        operating_feature_names=(
            "modulationIndex",
            "loadResistance",
            "loadInductance",
            "phaseAngle",
        ),
    )


class LivePhaseOptimizationTests(unittest.TestCase):
    def test_uses_only_fixed_stratified_training_subset_and_preserves_baseline(self) -> None:
        prepared = _prepared_training_fixture()
        result = optimize_training_phase_subset(
            prepared,
            seed=20261120,
            particles=2,
            iterations=1,
        )

        self.assertEqual(len(result.training_sample_ids), 30)
        selected_splits = {
            prepared.splits[np.flatnonzero(prepared.sample_ids == sample_id)[0]]
            for sample_id in result.training_sample_ids
        }
        self.assertEqual(selected_splits, {"training"})
        self.assertTrue(result.baseline_training_evaluation.feasible)
        self.assertTrue(result.run_result.best.feasible)
        self.assertLessEqual(
            result.run_result.best.mean_total_thd_percent,
            result.baseline_training_evaluation.mean_total_thd_percent + 1e-12,
        )
        self.assertEqual(result.run_result.seed, 20261120)

    def test_live_runtime_limits_are_enforced(self) -> None:
        for parameters in (
            {"seed": -1},
            {"seed": 1, "particles": 13},
            {"seed": 1, "iterations": 13},
        ):
            arguments = {"seed": 1, "particles": 8, "iterations": 8}
            arguments.update(parameters)
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                optimize_training_phase_subset(_prepared_training_fixture(), **arguments)


if __name__ == "__main__":
    unittest.main()
