"""Regression metrics for waveform arrays."""

from __future__ import annotations

import numpy as np


def _paired_arrays(target: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(target, dtype=np.float64)
    predicted = np.asarray(prediction, dtype=np.float64)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("target and prediction must be non-empty arrays with identical shapes.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("target and prediction must contain only finite values.")
    return actual, predicted


def mae(target: np.ndarray, prediction: np.ndarray) -> float:
    actual, predicted = _paired_arrays(target, prediction)
    return float(np.mean(np.abs(actual - predicted)))


def mse(target: np.ndarray, prediction: np.ndarray) -> float:
    actual, predicted = _paired_arrays(target, prediction)
    return float(np.mean(np.square(actual - predicted)))


def rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(mse(target, prediction)))


def max_absolute_error(target: np.ndarray, prediction: np.ndarray) -> float:
    actual, predicted = _paired_arrays(target, prediction)
    return float(np.max(np.abs(actual - predicted)))


def r_squared(target: np.ndarray, prediction: np.ndarray) -> float:
    actual, predicted = _paired_arrays(target, prediction)
    residual_sum = np.sum(np.square(actual - predicted))
    total_sum = np.sum(np.square(actual - np.mean(actual)))
    if total_sum == 0:
        raise ValueError("R-squared is undefined when the target is constant.")
    return float(1 - residual_sum / total_sum)


def correlation(target: np.ndarray, prediction: np.ndarray) -> float:
    actual, predicted = _paired_arrays(target, prediction)
    actual_flat, predicted_flat = actual.ravel(), predicted.ravel()
    if np.std(actual_flat) == 0 or np.std(predicted_flat) == 0:
        raise ValueError("Correlation is undefined for a constant array.")
    return float(np.corrcoef(actual_flat, predicted_flat)[0, 1])


def calculate_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "mae": mae(target, prediction),
        "mse": mse(target, prediction),
        "rmse": rmse(target, prediction),
        "r_squared": r_squared(target, prediction),
        "maximum_absolute_error": max_absolute_error(target, prediction),
        "correlation": correlation(target, prediction),
    }
