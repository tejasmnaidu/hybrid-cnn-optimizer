from __future__ import annotations

import unittest

import numpy as np
import torch
from torch import nn

from evaluation.cnn_analytical_comparison import _predict_split, _split_metrics
from evaluation.scientific_validation import analytical_reference_waveforms
from preprocessing.preprocess import NormalizationStats, PreparedDataset


class CnnAnalyticalComparisonTests(unittest.TestCase):
    def test_analytical_reference_uses_modulation_phase_and_project_sampling(self) -> None:
        conditions = np.asarray(
            [
                [0.5, 10.0, 0.01, 0.0],
                [1.0, 20.0, 0.02, np.pi / 2.0],
            ],
            dtype=np.float64,
        )
        waves = analytical_reference_waveforms(
            conditions,
            ("modulationIndex", "loadResistance", "loadInductance", "phaseAngle"),
        )
        expected = np.stack(
            [
                300.0 * conditions[index, 0]
                * np.sin(2.0 * np.pi * 50.0 * np.arange(400) / 20_000.0 + conditions[index, 3])
                for index in range(2)
            ]
        )
        self.assertEqual(waves.shape, (2, 400))
        np.testing.assert_allclose(waves, expected, rtol=0.0, atol=1e-12)

    def test_split_predictor_reads_only_requested_split_in_inference_mode(self) -> None:
        class EchoModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.batch_sizes: list[int] = []
                self.inference_modes: list[bool] = []

            def forward(self, input_voltage, load_current, operating_conditions):
                self.batch_sizes.append(int(input_voltage.shape[0]))
                self.inference_modes.append(torch.is_inference_mode_enabled())
                return input_voltage

        ids = np.asarray([11, 12, 13, 14], dtype=np.int64)
        waveforms = np.arange(16, dtype=np.float32).reshape(4, 4)
        prepared = PreparedDataset(
            sample_ids=ids,
            splits=np.asarray(["training", "validation", "testing", "testing"]),
            input_voltage=waveforms,
            load_current=np.zeros_like(waveforms),
            operating_conditions=np.zeros((4, 4), dtype=np.float32),
            target_voltage=waveforms,
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
            operating_feature_names=("m", "R", "L", "phase"),
        )
        model = EchoModel()
        selected_ids, _, predictions = _predict_split(
            model,
            prepared,
            split="validation",
            device=torch.device("cpu"),
            batch_size=1,
        )
        np.testing.assert_array_equal(selected_ids, [12])
        np.testing.assert_array_equal(predictions, waveforms[1:2])
        self.assertEqual(model.batch_sizes, [1])
        self.assertEqual(model.inference_modes, [True])

    def test_pooled_and_per_record_metrics_are_separately_reported(self) -> None:
        targets = np.asarray([[0.0, 1.0], [0.0, 2.0]])
        predictions = np.asarray([[1.0, 1.0], [0.0, 4.0]])
        result = _split_metrics(targets, predictions)
        self.assertEqual(result["record_count"], 2)
        self.assertAlmostEqual(result["pooled_point_metrics"]["mae"], 0.75)
        self.assertAlmostEqual(result["pooled_point_metrics"]["rmse"], np.sqrt(1.25))
        self.assertAlmostEqual(result["mean_per_record_mae_v"], 0.75)
        self.assertAlmostEqual(
            result["mean_per_record_rmse_v"],
            (np.sqrt(0.5) + np.sqrt(2.0)) / 2.0,
        )


if __name__ == "__main__":
    unittest.main()
