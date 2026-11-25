from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from optimization.cnn_guided_search import (
    run_cnn_guided_hybrid,
    run_standalone_hybrid,
    screen_and_rank_candidates,
    spearman_correlation,
)
from preprocessing.preprocess import NormalizationStats


class CNNGuidedSearchTests(unittest.TestCase):
    def test_spearman_perfect_positive_and_negative(self) -> None:
        values = np.asarray([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(spearman_correlation(values, values), 1.0)
        self.assertAlmostEqual(spearman_correlation(values, -values), -1.0)

    def test_spearman_ties_use_average_ranks(self) -> None:
        correlation = spearman_correlation(
            np.asarray([1.0, 1.0, 2.0]),
            np.asarray([1.0, 2.0, 2.0]),
        )
        self.assertGreater(correlation, 0.0)

    def test_candidate_ranking_is_by_cnn_score_not_phase(self) -> None:
        candidates = np.asarray([[300.0, 120.0], [60.0, 120.0], [200.0, 10.0]])
        scores = np.asarray([0.4, 0.1, 0.9])
        indices = screen_and_rank_candidates(candidates, scores, top_k=2)
        np.testing.assert_array_equal(indices, [1, 0])

    def test_candidate_ranking_rejects_invalid_top_k(self) -> None:
        candidates = np.zeros((3, 2))
        with self.assertRaisesRegex(ValueError, "top_k"):
            screen_and_rank_candidates(candidates, np.zeros(3), top_k=4)

    def test_guided_hybrid_is_reproducible_bounded_and_rechecks_finalist(self) -> None:
        normalization = NormalizationStats(
            input_voltage_mean=0.0,
            input_voltage_std=1.0,
            load_current_mean=0.0,
            load_current_std=1.0,
            operating_mean=[0.0] * 4,
            operating_std=[1.0] * 4,
            target_voltage_mean=0.0,
            target_voltage_std=1.0,
        )
        operating_point = {
            "modulation_index": 0.75,
            "dc_voltage_per_cell_v": 100.0,
            "load_resistance_ohm": 30.0,
            "load_inductance_h": 0.03,
            "phase_angle_degrees": 0.0,
        }

        def deterministic_score(_model, _stats, **kwargs):
            offsets = kwargs["carrier_offsets_degrees"]
            return (offsets[1] - 60.0) ** 2 + (offsets[2] - 120.0) ** 2

        with patch(
            "optimization.cnn_guided_search.cnn_candidate_score",
            side_effect=deterministic_score,
        ) as score:
            first = run_cnn_guided_hybrid(
                nn.Identity(),
                normalization,
                seed=19,
                operating_point=operating_point,
                candidate_budget=8,
                top_k=1,
            )
            second = run_cnn_guided_hybrid(
                nn.Identity(),
                normalization,
                seed=19,
                operating_point=operating_point,
                candidate_budget=8,
                top_k=1,
            )

        self.assertEqual(first["candidate"], second["candidate"])
        self.assertEqual(score.call_count, 16)
        self.assertTrue(all(0.0 <= phase < 360.0 for phase in first["candidate"]))
        self.assertEqual(first["waveform_generations"], 9)
        self.assertEqual(first["full_thd_evaluations"], 2)

    def test_standalone_hybrid_returns_simulator_rechecked_metrics(self) -> None:
        result = run_standalone_hybrid(
            seed=19,
            operating_point={
                "modulation_index": 0.75,
                "dc_voltage_per_cell_v": 100.0,
                "load_resistance_ohm": 30.0,
                "load_inductance_h": 0.03,
                "phase_angle_degrees": 0.0,
            },
            candidate_budget=8,
        )
        self.assertTrue(all(0.0 <= phase < 360.0 for phase in result["candidate"]))
        self.assertTrue(np.isfinite(result["event_thd"]))
        self.assertTrue(np.isfinite(result["voltage_thd"]))
        self.assertTrue(np.isfinite(result["current_thd"]))
        self.assertEqual(result["waveform_generations"], 1)
        self.assertGreaterEqual(result["full_thd_evaluations"], 2)


if __name__ == "__main__":
    unittest.main()
