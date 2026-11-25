from __future__ import annotations

import unittest

import numpy as np
import torch

from optimization.thd_surrogate import (
    EventTHDSurrogate,
    encode_thd_surrogate_features,
    spearman_rank_correlation,
)
from optimization.thd_surrogate_data import generate_thd_surrogate_dataset


class THDSurrogateTests(unittest.TestCase):
    def test_features_wrap_and_canonicalize_carrier_swap(self) -> None:
        candidates = np.asarray(
            (
                (0.75, 360.0, 60.0, 120.0),
                (0.75, 0.0, 420.0, -240.0),
                (0.75, 0.0, 120.0, 60.0),
            )
        )
        features = encode_thd_surrogate_features(candidates)
        np.testing.assert_allclose(features[0], features[1], atol=1e-7)
        np.testing.assert_allclose(features[0], features[2], atol=1e-7)

    def test_input_validation_and_prediction_shape(self) -> None:
        with self.assertRaises(ValueError):
            encode_thd_surrogate_features(np.asarray(((0.5, 0.0, 60.0, 120.0),)))
        with self.assertRaises(ValueError):
            encode_thd_surrogate_features(np.asarray(((0.75, 0.0, np.nan, 120.0),)))

        model = EventTHDSurrogate()
        prediction = model(torch.zeros((3, 7), dtype=torch.float32))
        self.assertEqual(tuple(prediction.shape), (3, 1))
        with self.assertRaises(ValueError):
            model(torch.zeros((3, 6), dtype=torch.float32))

    def test_generated_labels_are_exact_evaluator_outputs_and_reproducible(self) -> None:
        first = generate_thd_surrogate_dataset(sample_count=12, seed=123)
        second = generate_thd_surrogate_dataset(sample_count=12, seed=123)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (12, 6))
        np.testing.assert_array_equal(first[0, :4], (0.75, 0.0, 60.0, 120.0))
        self.assertTrue(np.isfinite(first).all())
        self.assertTrue(np.all(first[:, 4:] >= 0.0))

    def test_rank_correlation_accounts_for_ties(self) -> None:
        self.assertAlmostEqual(
            spearman_rank_correlation(
                np.asarray((1.0, 1.0, 2.0, 3.0)),
                np.asarray((2.0, 2.0, 4.0, 6.0)),
            ),
            1.0,
        )
        self.assertEqual(
            spearman_rank_correlation(
                np.asarray((1.0, 1.0)),
                np.asarray((2.0, 2.0)),
            ),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
