"""Model evaluation helpers and reusable plotting functions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from preprocessing.preprocess import NormalizationStats
from .metrics import calculate_metrics
from .thd_analysis import calculate_thd


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device | str,
    normalization: NormalizationStats | None = None,
) -> dict[str, Any]:
    """Evaluate the records provided by the caller; pass the test loader only once."""
    model.eval()
    device_obj = torch.device(device)
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    sample_ids: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            prediction = model(
                batch["input_voltage"].to(device_obj),
                batch["load_current"].to(device_obj),
                batch["operating_conditions"].to(device_obj),
            )
            predictions.append(prediction.cpu().numpy())
            targets.append(batch["target_voltage"].cpu().numpy())
            if "sample_id" in batch:
                sample_ids.append(batch["sample_id"].cpu().numpy())
    if not predictions:
        raise ValueError("Cannot evaluate an empty DataLoader.")
    prediction_array = np.concatenate(predictions)
    target_array = np.concatenate(targets)
    if normalization is not None:
        prediction_array = normalization.denormalize_target(prediction_array)
        target_array = normalization.denormalize_target(target_array)
    return {
        "metrics": calculate_metrics(target_array, prediction_array),
        "predictions": prediction_array,
        "targets": target_array,
        "sample_ids": np.concatenate(sample_ids) if sample_ids else None,
    }


def plot_training_history(history: list[dict[str, float]], path: str | Path | None = None):
    if not history:
        raise ValueError("Training history is empty.")
    figure, axis = plt.subplots()
    axis.plot([row["epoch"] for row in history], [row["train_loss"] for row in history], label="Training")
    axis.plot(
        [row["epoch"] for row in history],
        [row["validation_loss"] for row in history],
        label="Validation",
    )
    axis.set(xlabel="Epoch", ylabel="Loss", title="Training and validation loss")
    axis.legend()
    figure.tight_layout()
    if path is not None:
        figure.savefig(path, dpi=160)
    return figure


def plot_waveform_comparison(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    sample_index: int = 0,
    sampling_frequency_hz: float = 20_000.0,
    path: str | Path | None = None,
):
    actual = np.asarray(target)
    predicted = np.asarray(prediction)
    if actual.shape != predicted.shape or actual.ndim != 2:
        raise ValueError("target and prediction must have matching (samples, points) shapes.")
    if not 0 <= sample_index < len(actual):
        raise IndexError("sample_index is outside the provided arrays.")
    time_ms = np.arange(actual.shape[1]) / sampling_frequency_hz * 1000
    figure, axis = plt.subplots()
    axis.plot(time_ms, actual[sample_index], label="Reference target")
    axis.plot(time_ms, predicted[sample_index], label="CNN prediction", alpha=0.8)
    axis.set(xlabel="Time (ms)", ylabel="Voltage (V)", title="Reference and predicted waveform")
    axis.legend()
    figure.tight_layout()
    if path is not None:
        figure.savefig(path, dpi=160)
    return figure


def plot_harmonic_spectrum(
    waveform: np.ndarray,
    *,
    sampling_frequency_hz: float = 20_000.0,
    fundamental_frequency_hz: float = 50.0,
    max_harmonic: int = 100,
    path: str | Path | None = None,
):
    thd, spectrum = calculate_thd(
        waveform,
        sampling_frequency_hz=sampling_frequency_hz,
        fundamental_frequency_hz=fundamental_frequency_hz,
        max_harmonic=max_harmonic,
        return_spectrum=True,
    )
    figure, axis = plt.subplots()
    axis.stem(spectrum.harmonic_orders[1:], spectrum.amplitudes[1:])
    axis.set(xlabel="Harmonic order", ylabel="Amplitude", title=f"Harmonic spectrum (THD {thd:.4g}%)")
    figure.tight_layout()
    if path is not None:
        figure.savefig(path, dpi=160)
    return figure


def plot_thd_comparison(
    reference_waveform: np.ndarray,
    predicted_waveform: np.ndarray,
    *,
    sampling_frequency_hz: float = 20_000.0,
    fundamental_frequency_hz: float = 50.0,
    max_harmonic: int = 100,
    reference_label: str = "Reference target waveform",
    prediction_label: str = "CNN-predicted target waveform",
    path: str | Path | None = None,
):
    reference_thd = calculate_thd(
        reference_waveform,
        sampling_frequency_hz=sampling_frequency_hz,
        fundamental_frequency_hz=fundamental_frequency_hz,
        max_harmonic=max_harmonic,
    )
    prediction_thd = calculate_thd(
        predicted_waveform,
        sampling_frequency_hz=sampling_frequency_hz,
        fundamental_frequency_hz=fundamental_frequency_hz,
        max_harmonic=max_harmonic,
    )
    figure, axis = plt.subplots()
    axis.bar([reference_label, prediction_label], [reference_thd, prediction_thd])
    axis.set(ylabel="THD (%)", title="Waveform THD comparison (not measured inverter output)")
    axis.tick_params(axis="x", labelrotation=15)
    figure.tight_layout()
    if path is not None:
        figure.savefig(path, dpi=160)
    return figure


def plot_error_distribution(
    target: np.ndarray,
    prediction: np.ndarray,
    path: str | Path | None = None,
):
    if np.shape(target) != np.shape(prediction):
        raise ValueError("target and prediction must have matching shapes.")
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    figure, axis = plt.subplots()
    axis.hist(error.ravel(), bins=50)
    axis.set(xlabel="Prediction error (V)", ylabel="Count", title="Waveform error distribution")
    figure.tight_layout()
    if path is not None:
        figure.savefig(path, dpi=160)
    return figure
