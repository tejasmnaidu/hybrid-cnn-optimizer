from __future__ import annotations

import unittest

import numpy as np
import torch
from torch import nn

from evaluation.cnn_test_evaluator import (
    CNNTestEvaluationError,
    evaluate_cnn_test_set,
)
from preprocessing.preprocess import NormalizationStats, PreparedDataset


def _prepared_dataset(
    *,
    testing_count: int = 3,
    waveform_points: int = 4,
) -> PreparedDataset:
    record_count = 2 + testing_count
    sample_ids = np.arange(100, 100 + record_count, dtype=np.int64)
    splits = np.asarray(["training", "validation"] + ["testing"] * testing_count)
    waveforms = np.stack(
        [
            np.arange(waveform_points, dtype=np.float32) + row * 10
            for row in range(record_count)
        ]
    )
    return PreparedDataset(
        sample_ids=sample_ids,
        splits=splits,
        input_voltage=waveforms / 10.0,
        load_current=np.zeros_like(waveforms),
        operating_conditions=np.zeros((record_count, 4), dtype=np.float32),
        target_voltage=(waveforms - 10.0) / 2.0,
        stats=NormalizationStats(
            input_voltage_mean=0.0,
            input_voltage_std=1.0,
            load_current_mean=0.0,
            load_current_std=1.0,
            operating_mean=[0.0, 0.0, 0.0, 0.0],
            operating_std=[1.0, 1.0, 1.0, 1.0],
            target_voltage_mean=10.0,
            target_voltage_std=2.0,
        ),
        operating_feature_names=("m", "R", "L", "phase"),
    )


class IdentityInputModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []
        self.inference_modes: list[bool] = []

    def forward(self, input_voltage, load_current, operating_conditions):
        self.batch_sizes.append(int(input_voltage.shape[0]))
        self.inference_modes.append(torch.is_inference_mode_enabled())
        return input_voltage


class CNNTestEvaluatorTests(unittest.TestCase):
    def test_only_testing_split_is_evaluated(self) -> None:
        prepared = _prepared_dataset(testing_count=3)
        model = IdentityInputModel()
        result = evaluate_cnn_test_set(model, prepared, batch_size=2, device="cpu")
        np.testing.assert_array_equal(result.sample_ids, [102, 103, 104])
        self.assertEqual(result.record_count, 3)
        self.assertEqual(model.batch_sizes, [2, 1])
        self.assertTrue(all(model.inference_modes))

    def test_batch_inference_and_cpu_execution(self) -> None:
        model = IdentityInputModel()
        result = evaluate_cnn_test_set(
            model,
            _prepared_dataset(testing_count=5),
            batch_size=2,
            device=torch.device("cpu"),
        )
        self.assertEqual(model.batch_sizes, [2, 2, 1])
        self.assertEqual(result.batch_size, 2)

    def test_checkpoint_normalization_denormalizes_predictions_and_targets(self) -> None:
        prepared = _prepared_dataset(testing_count=1)
        result = evaluate_cnn_test_set(IdentityInputModel(), prepared, batch_size=1)
        np.testing.assert_allclose(
            result.predictions_v[0],
            prepared.input_voltage[-1] * 2.0 + 10.0,
        )
        np.testing.assert_allclose(
            result.targets_v[0],
            prepared.target_voltage[-1] * 2.0 + 10.0,
        )

    def test_micro_metrics_are_pooled_and_macro_metrics_average_records(self) -> None:
        prepared = _prepared_dataset(testing_count=2, waveform_points=4)
        prepared.target_voltage[2] = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        prepared.target_voltage[3] = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        prepared.input_voltage[2] = prepared.target_voltage[2]
        prepared.input_voltage[3] = prepared.target_voltage[3] + 5.0
        result = evaluate_cnn_test_set(IdentityInputModel(), prepared, batch_size=2)
        self.assertAlmostEqual(result.micro_metrics.mae, 5.0)
        self.assertAlmostEqual(result.macro_metrics.mae, 5.0)
        self.assertAlmostEqual(result.micro_metrics.mse, 50.0)
        self.assertAlmostEqual(result.macro_metrics.mse, 50.0)
        self.assertAlmostEqual(
            result.micro_metrics.rmse,
            np.sqrt(result.micro_metrics.mse),
        )
        expected_global_r2 = 1.0 - np.sum(result.errors_v**2) / np.sum(
            (result.targets_v - np.mean(result.targets_v)) ** 2
        )
        self.assertAlmostEqual(result.micro_metrics.r_squared, expected_global_r2)
        self.assertNotEqual(
            result.micro_metrics.r_squared,
            result.macro_metrics.r_squared,
        )
        self.assertEqual(len(result.per_record_metrics), 2)
        self.assertEqual(result.error_distribution.maximum, 10.0)
        self.assertEqual(result.micro_metrics.maximum_absolute_error, 10.0)
        self.assertEqual(result.macro_metrics.maximum_absolute_error, 5.0)

    def test_empty_testing_split_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "testing split is empty"):
            evaluate_cnn_test_set(
                IdentityInputModel(),
                _prepared_dataset(testing_count=0),
            )

    def test_invalid_model_output_shape_and_numerics_are_rejected(self) -> None:
        class WrongShape(nn.Module):
            def forward(self, input_voltage, load_current, operating_conditions):
                return input_voltage[:, :-1]

        class NonFinite(nn.Module):
            def forward(self, input_voltage, load_current, operating_conditions):
                return torch.full_like(input_voltage, float("nan"))

        for model in (WrongShape(), NonFinite()):
            with self.subTest(model=type(model).__name__):
                with self.assertRaises(CNNTestEvaluationError):
                    evaluate_cnn_test_set(model, _prepared_dataset(testing_count=1))

    def test_model_training_state_is_restored_after_inference(self) -> None:
        model = IdentityInputModel()
        model.train()
        evaluate_cnn_test_set(model, _prepared_dataset(testing_count=1))
        self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()
