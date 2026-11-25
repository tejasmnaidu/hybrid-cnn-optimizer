from __future__ import annotations

import unittest

import numpy as np

from evaluation.cnn_honest_eval import (
    aggregate_metrics,
    per_record_metrics,
    verify_disjoint_split_indices,
)
from preprocessing.preprocess import NormalizationStats, PreparedDataset


class CNNHonestEvaluationTests(unittest.TestCase):
    def test_per_record_and_pooled_metrics_are_in_physical_units(self) -> None:
        targets = np.asarray([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]], dtype=np.float64)
        predictions = np.asarray([[0.0, 2.0, 2.0], [1.0, 1.0, 3.0]], dtype=np.float64)

        mae, mse, rmse, r2 = per_record_metrics(targets, predictions)
        np.testing.assert_allclose(mae, [1.0 / 3.0, 1.0 / 3.0])
        np.testing.assert_allclose(mse, [1.0 / 3.0, 1.0 / 3.0])
        np.testing.assert_allclose(rmse, np.sqrt([1.0 / 3.0, 1.0 / 3.0]))
        self.assertTrue(np.isfinite(r2).all())

        aggregate = aggregate_metrics(targets, predictions)
        self.assertAlmostEqual(aggregate["pooled_point_metrics"]["mae"], 1.0 / 3.0)
        self.assertAlmostEqual(aggregate["pooled_point_metrics"]["mse"], 1.0 / 3.0)

    def test_split_indices_are_disjoint_and_have_expected_counts(self) -> None:
        count = 7000 + 1500 + 1500
        splits = np.asarray(
            ["training"] * 7000 + ["validation"] * 1500 + ["testing"] * 1500
        )
        prepared = PreparedDataset(
            sample_ids=np.arange(1, count + 1, dtype=np.int64),
            splits=splits,
            input_voltage=np.zeros((count, 400), dtype=np.float32),
            load_current=np.zeros((count, 400), dtype=np.float32),
            operating_conditions=np.zeros((count, 4), dtype=np.float32),
            target_voltage=np.zeros((count, 400), dtype=np.float32),
            stats=NormalizationStats(
                input_voltage_mean=0.0,
                input_voltage_std=1.0,
                load_current_mean=0.0,
                load_current_std=1.0,
                operating_mean=[0.0, 0.0, 0.0, 0.0],
                operating_std=[1.0, 1.0, 1.0, 1.0],
                target_voltage_mean=0.0,
                target_voltage_std=1.0,
            ),
            operating_feature_names=(
                "modulationIndex",
                "loadResistance",
                "loadInductance",
                "phaseAngle",
            ),
        )
        counts = verify_disjoint_split_indices(prepared)
        self.assertEqual(
            counts,
            {"training": 7000, "validation": 1500, "testing": 1500},
        )

    def test_overlap_split_is_rejected(self) -> None:
        splits = np.asarray(["training"] * 7000 + ["validation"] * 1500 + ["testing"] * 1500)
        prepared = PreparedDataset(
            sample_ids=np.arange(1, len(splits) + 1, dtype=np.int64),
            splits=splits,
            input_voltage=np.zeros((len(splits), 400), dtype=np.float32),
            load_current=np.zeros((len(splits), 400), dtype=np.float32),
            operating_conditions=np.zeros((len(splits), 4), dtype=np.float32),
            target_voltage=np.zeros((len(splits), 400), dtype=np.float32),
            stats=NormalizationStats(
                input_voltage_mean=0.0,
                input_voltage_std=1.0,
                load_current_mean=0.0,
                load_current_std=1.0,
                operating_mean=[0.0, 0.0, 0.0, 0.0],
                operating_std=[1.0, 1.0, 1.0, 1.0],
                target_voltage_mean=0.0,
                target_voltage_std=1.0,
            ),
            operating_feature_names=("modulationIndex", "loadResistance", "loadInductance", "phaseAngle"),
        )
        prepared.indices("testing")
        prepared.splits[0] = "testing"
        with self.assertRaisesRegex(ValueError, "Unexpected split counts"):
            verify_disjoint_split_indices(prepared)


if __name__ == "__main__":
    unittest.main()
