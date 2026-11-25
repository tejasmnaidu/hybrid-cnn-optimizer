"""Held-out test evaluation for the trained CNN reference-waveform predictor."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from preprocessing.preprocess import NormalizationStats, PreparedDataset
from training.dataset import WaveformDataset


ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class RegressionSummary:
    """Regression metrics with an explicit aggregation population."""

    mae: float
    mse: float
    rmse: float
    r_squared: float | None
    correlation: float | None
    maximum_absolute_error: float


@dataclass(frozen=True)
class ErrorDistribution:
    """Pooled pointwise prediction-error distribution in volts."""

    minimum: float
    percentile_1: float
    percentile_5: float
    percentile_25: float
    median: float
    percentile_75: float
    percentile_95: float
    percentile_99: float
    maximum: float
    mean: float
    standard_deviation: float


@dataclass(frozen=True)
class PerRecordMetrics:
    sample_id: int
    mae: float
    mse: float
    rmse: float
    r_squared: float | None
    maximum_absolute_error: float


@dataclass(frozen=True)
class CNNTestEvaluation:
    """Complete test results; arrays are denormalized physical voltages."""

    sample_ids: np.ndarray
    targets_v: np.ndarray
    predictions_v: np.ndarray
    errors_v: np.ndarray
    per_record_metrics: tuple[PerRecordMetrics, ...]
    micro_metrics: RegressionSummary
    macro_metrics: RegressionSummary
    error_distribution: ErrorDistribution
    record_count: int
    waveform_points: int
    batch_size: int


class CNNTestEvaluationError(RuntimeError):
    """Raised when test-set inference cannot produce finite, valid outputs."""


def _summary(target: np.ndarray, prediction: np.ndarray, *, label: str) -> RegressionSummary:
    actual = np.asarray(target, dtype=np.float64).reshape(-1)
    predicted = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if actual.size == 0 or actual.shape != predicted.shape:
        raise CNNTestEvaluationError(f"{label} metrics require non-empty paired arrays.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise CNNTestEvaluationError(f"{label} metrics contain non-finite values.")
    error = predicted - actual
    mse_value = float(np.mean(np.square(error), dtype=np.float64))
    target_variance = float(np.sum(np.square(actual - np.mean(actual)), dtype=np.float64))
    r_squared_value = (
        float(1.0 - np.sum(np.square(error), dtype=np.float64) / target_variance)
        if target_variance > 0.0
        else None
    )
    target_std = float(np.std(actual, dtype=np.float64))
    prediction_std = float(np.std(predicted, dtype=np.float64))
    correlation_value = (
        float(np.corrcoef(actual, predicted)[0, 1])
        if target_std > 0.0 and prediction_std > 0.0
        else None
    )
    values = (
        float(np.mean(np.abs(error))),
        mse_value,
        math.sqrt(mse_value),
        float(np.max(np.abs(error))),
    )
    if not np.isfinite(values).all():
        raise CNNTestEvaluationError(f"{label} metrics produced a non-finite result.")
    if r_squared_value is not None and not np.isfinite(r_squared_value):
        raise CNNTestEvaluationError(f"{label} R-squared is non-finite.")
    if correlation_value is not None and not np.isfinite(correlation_value):
        raise CNNTestEvaluationError(f"{label} correlation is non-finite.")
    return RegressionSummary(
        mae=values[0],
        mse=values[1],
        rmse=values[2],
        r_squared=r_squared_value,
        correlation=correlation_value,
        maximum_absolute_error=values[3],
    )
def _error_distribution(errors: np.ndarray) -> ErrorDistribution:
    values = np.asarray(errors, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise CNNTestEvaluationError("Prediction errors must be non-empty and finite.")
    percentiles = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])
    return ErrorDistribution(
        minimum=float(np.min(values)),
        percentile_1=float(percentiles[0]),
        percentile_5=float(percentiles[1]),
        percentile_25=float(percentiles[2]),
        median=float(percentiles[3]),
        percentile_75=float(percentiles[4]),
        percentile_95=float(percentiles[5]),
        percentile_99=float(percentiles[6]),
        maximum=float(np.max(values)),
        mean=float(np.mean(values, dtype=np.float64)),
        standard_deviation=float(np.std(values, dtype=np.float64)),
    )


def evaluate_cnn_test_set(
    model: nn.Module,
    prepared: PreparedDataset,
    *,
    normalization: NormalizationStats | None = None,
    device: torch.device | str = "cpu",
    batch_size: int = 64,
    progress_callback: ProgressCallback | None = None,
) -> CNNTestEvaluation:
    """Evaluate every testing record with inference-only batches.

    Micro metrics flatten all waveform points across all test records.
    Macro metrics average the corresponding per-record metric; macro R-squared
    and correlation average only defined per-record values and are ``None``
    when no record has a defined value.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    test_indices = prepared.indices("testing")
    if test_indices.size == 0:
        raise ValueError("The testing split is empty; no test evaluation was run.")
    stats = normalization or prepared.stats
    device_obj = torch.device(device)
    test_dataset = WaveformDataset(prepared, "testing")
    loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    was_training = model.training
    model.to(device_obj)
    model.eval()
    prediction_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    sample_id_batches: list[np.ndarray] = []
    completed = 0
    total = int(test_indices.size)
    try:
        with torch.inference_mode():
            for batch in loader:
                output = model(
                    batch["input_voltage"].to(device_obj),
                    batch["load_current"].to(device_obj),
                    batch["operating_conditions"].to(device_obj),
                )
                expected_shape = batch["target_voltage"].shape
                if not isinstance(output, torch.Tensor) or output.shape != expected_shape:
                    shape = tuple(output.shape) if isinstance(output, torch.Tensor) else type(output).__name__
                    raise CNNTestEvaluationError(
                        f"Model output shape/type {shape} does not match target shape "
                        f"{tuple(expected_shape)}."
                    )
                if not torch.isfinite(output).all():
                    raise CNNTestEvaluationError("Model returned non-finite normalized predictions.")
                prediction_batches.append(output.detach().cpu().numpy())
                target_batches.append(batch["target_voltage"].cpu().numpy())
                sample_id_batches.append(batch["sample_id"].cpu().numpy())
                completed += int(output.shape[0])
                if progress_callback is not None:
                    progress_callback(completed, total)
    finally:
        model.train(was_training)

    normalized_predictions = np.concatenate(prediction_batches, axis=0)
    normalized_targets = np.concatenate(target_batches, axis=0)
    sample_ids = np.concatenate(sample_id_batches).astype(np.int64, copy=False)
    predictions_v = np.asarray(
        stats.denormalize_target(normalized_predictions),
        dtype=np.float64,
    )
    targets_v = np.asarray(stats.denormalize_target(normalized_targets), dtype=np.float64)
    if predictions_v.shape != targets_v.shape or not np.isfinite(predictions_v).all() or not np.isfinite(targets_v).all():
        raise CNNTestEvaluationError("Denormalized test predictions/targets are invalid.")
    errors_v = predictions_v - targets_v

    per_record: list[PerRecordMetrics] = []
    record_summaries: list[RegressionSummary] = []
    for row_index, sample_id in enumerate(sample_ids):
        record_summary = _summary(
            targets_v[row_index],
            predictions_v[row_index],
            label=f"SampleID {int(sample_id)}",
        )
        record_summaries.append(record_summary)
        per_record.append(
            PerRecordMetrics(
                sample_id=int(sample_id),
                mae=record_summary.mae,
                mse=record_summary.mse,
                rmse=record_summary.rmse,
                r_squared=record_summary.r_squared,
                maximum_absolute_error=record_summary.maximum_absolute_error,
            )
        )

    micro = _summary(targets_v, predictions_v, label="Micro")
    defined_record_r2 = [
        summary.r_squared
        for summary in record_summaries
        if summary.r_squared is not None
    ]
    macro_correlations = [
        summary.correlation
        for summary in record_summaries
        if summary.correlation is not None
    ]
    macro = RegressionSummary(
        mae=float(np.mean([record.mae for record in per_record], dtype=np.float64)),
        mse=float(np.mean([record.mse for record in per_record], dtype=np.float64)),
        rmse=float(np.mean([record.rmse for record in per_record], dtype=np.float64)),
        r_squared=float(np.mean(defined_record_r2, dtype=np.float64)) if defined_record_r2 else None,
        correlation=float(np.mean(macro_correlations, dtype=np.float64)) if macro_correlations else None,
        maximum_absolute_error=float(
            np.mean(
                [record.maximum_absolute_error for record in per_record],
                dtype=np.float64,
            )
        ),
    )
    return CNNTestEvaluation(
        sample_ids=sample_ids,
        targets_v=targets_v,
        predictions_v=predictions_v,
        errors_v=errors_v,
        per_record_metrics=tuple(per_record),
        micro_metrics=micro,
        macro_metrics=macro,
        error_distribution=_error_distribution(errors_v),
        record_count=len(sample_ids),
        waveform_points=int(targets_v.shape[1]),
        batch_size=batch_size,
    )


__all__ = [
    "CNNTestEvaluation",
    "CNNTestEvaluationError",
    "ErrorDistribution",
    "PerRecordMetrics",
    "RegressionSummary",
    "evaluate_cnn_test_set",
]
