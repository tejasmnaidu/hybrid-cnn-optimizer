from __future__ import annotations

import unittest

import numpy as np

from evaluation.cnn_grouped_eval import (
    compare_group,
    grouped_test_metrics,
    modulation_band_indices,
    training_modulation_edges,
)


class CNNGroupedEvaluationTests(unittest.TestCase):
    def test_group_comparison_computed_from_cnn_and_analytical_rmse(self) -> None:
        targets = np.asarray([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]])
        cnn = np.asarray([[0.1, 1.1, 2.1], [1.1, 2.1, 3.1]])
        analytical = np.asarray([[0.2, 1.2, 2.2], [1.2, 2.2, 3.2]])

        result = compare_group(targets, cnn, analytical, minimum_records=20)
        self.assertTrue(result["below_minimum_record_count"])
        self.assertEqual(result["comparison_by_pooled_rmse"], "CNN better")

    def test_training_modulation_quantiles_and_test_group_metrics(self) -> None:
        training_modulation = np.asarray([0.55, 0.60, 0.65, 0.70, 0.75])
        training = np.asarray(
            [
                [0.55, 10.0, 0.01],
                [0.60, 20.0, 0.02],
                [0.65, 30.0, 0.03],
                [0.70, 40.0, 0.05],
                [0.75, 30.0, 0.03],
            ]
        )
        test = np.asarray(
            [
                [0.58, 10.0, 0.01],
                [0.68, 30.0, 0.03],
            ]
        )
        edges = training_modulation_edges(training_modulation, band_count=2)
        bands = modulation_band_indices(test[:, 0], edges)
        self.assertTrue(np.isin(bands, (0, 1)).all())

        result = grouped_test_metrics(
            training_conditions=training,
            test_conditions=test,
            targets_v=np.asarray([[0.0, 1.0], [1.0, 2.0]]),
            cnn_predictions_v=np.asarray([[0.1, 1.1], [1.1, 2.1]]),
            analytical_predictions_v=np.asarray([[0.2, 1.2], [1.2, 2.2]]),
            feature_names=("modulationIndex", "loadResistance", "loadInductance"),
            minimum_records=20,
            modulation_band_count=2,
        )
        self.assertIn("modulation_band", {group["category"] for group in result["groups"]})
        self.assertIn("loadResistance", {group["category"] for group in result["groups"]})
        self.assertIn("loadInductance", {group["category"] for group in result["groups"]})

    def test_mismatched_shapes_raise_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "identical"):
            compare_group(
                np.zeros((2, 4)),
                np.zeros((2, 3)),
                np.zeros((2, 4)),
            )


if __name__ == "__main__":
    unittest.main()
