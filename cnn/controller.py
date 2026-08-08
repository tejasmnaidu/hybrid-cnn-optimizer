from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from cnn.predict import load_cnn_model, predict_switching_state_batch
from multilevel.eight_level_model import EightLevelInverterParams, generate_8_level_waveform, get_8_level_values
from signal_processing.evaluation import evaluate_inverter_waveform
from signal_processing.fft import compute_fft
from signal_processing.thd import calculate_thd


@dataclass
class CNNControllerResult:
    time: np.ndarray
    reference_voltage: np.ndarray
    reference_levels: np.ndarray
    cnn_voltage: np.ndarray
    cnn_levels: np.ndarray
    fft_reference: tuple[np.ndarray, np.ndarray]
    fft_cnn: tuple[np.ndarray, np.ndarray]
    thd_reference: float
    thd_cnn: float
    thd_difference: float


def reconstruct_cnn_waveform(
    reference: np.ndarray,
    dc_voltage: float,
    model_path: Path,
) -> tuple[np.ndarray, np.ndarray, float, int, int]:
    model, window_size = load_cnn_model(model_path)
    level_values = np.sort(get_8_level_values(dc_voltage))
    cnn_levels = np.zeros(len(reference), dtype=int)
    cnn_voltage = np.zeros(len(reference), dtype=float)

    if len(reference) <= window_size:
        return cnn_voltage, cnn_levels, 0.0, 0, window_size

    windows = np.stack([reference[idx - window_size : idx] for idx in range(window_size, len(reference))])
    start_time = perf_counter()
    predictions = predict_switching_state_batch(model, windows)
    end_time = perf_counter()

    inference_time = float(end_time - start_time)
    num_windows = len(windows)

    cnn_levels[window_size:] = predictions
    cnn_levels[:window_size] = cnn_levels[window_size]

    for idx in range(window_size, len(reference)):
        cnn_voltage[idx] = float(level_values[cnn_levels[idx]])
    cnn_voltage[:window_size] = float(level_values[cnn_levels[window_size]])

    return cnn_voltage, cnn_levels, inference_time, num_windows, window_size


def _count_transitions(levels: np.ndarray) -> int:
    if levels.ndim == 2:
        encoded = levels.astype(int)
        transitions = np.any(encoded[1:] != encoded[:-1], axis=1)
        return int(np.count_nonzero(transitions))
    return int(np.count_nonzero(levels[1:] != levels[:-1]))


def _tracking_metrics(reference_voltage: np.ndarray, output_voltage: np.ndarray) -> dict[str, float]:
    error = output_voltage - reference_voltage
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mae = float(np.mean(np.abs(error)))
    correlation = float(np.corrcoef(reference_voltage, output_voltage)[0, 1]) if reference_voltage.size > 1 else float("nan")
    return {
        "rmse": rmse,
        "mae": mae,
        "correlation": correlation,
    }


def _relative_improvement(conventional: float, cnn: float, lower_is_better: bool) -> float:
    if conventional == 0.0:
        return float("nan")
    if lower_is_better:
        return float((conventional - cnn) / conventional * 100.0)
    return float((cnn - conventional) / conventional * 100.0)


def _class_labels_from_voltage(dc_voltage: float, voltage: np.ndarray) -> np.ndarray:
    levels = np.sort(get_8_level_values(dc_voltage))
    labels = np.searchsorted(np.round(levels, 8), np.round(voltage, 8), side="left")
    return np.clip(labels, 0, len(levels) - 1)


def compare_conventional_vs_cnn(
    dc_voltage: float,
    fundamental_frequency: float,
    sampling_frequency: float,
    duration: float,
    modulation_index: float,
    phase_shift: float,
    model_path: Path,
) -> dict[str, Any]:
    params = EightLevelInverterParams(
        dc_voltage=dc_voltage,
        fundamental_frequency=fundamental_frequency,
        switching_frequency=2000.0,
        sampling_frequency=sampling_frequency,
        simulation_duration=duration,
        modulation_index=modulation_index,
        phase_shift=phase_shift,
    )
    waveform = generate_8_level_waveform(params)
    reference_voltage = waveform.voltage
    reference_levels = _class_labels_from_voltage(dc_voltage, reference_voltage)

    reference_eval = evaluate_inverter_waveform(reference_voltage, sampling_frequency, fundamental_frequency)
    reference_signal = modulation_index * np.sin(2.0 * np.pi * fundamental_frequency * waveform.time + phase_shift)
    cnn_voltage, cnn_levels, inference_time, num_windows, window_size = reconstruct_cnn_waveform(reference=reference_signal, dc_voltage=dc_voltage, model_path=model_path)
    cnn_eval = evaluate_inverter_waveform(cnn_voltage, sampling_frequency, fundamental_frequency)

    valid_slice = slice(window_size, len(reference_voltage))
    accuracy = float(np.mean(cnn_levels[valid_slice] == reference_levels[valid_slice])) if len(reference_levels[valid_slice]) > 0 else 0.0

    transitions_reference = _count_transitions(reference_levels)
    transitions_cnn = _count_transitions(cnn_levels)
    num_cycles = float(duration * fundamental_frequency)
    transitions_per_cycle_ref = float(transitions_reference / num_cycles) if num_cycles > 0 else float("nan")
    transitions_per_cycle_cnn = float(transitions_cnn / num_cycles) if num_cycles > 0 else float("nan")
    transition_frequency_ref = float(transitions_reference / duration) if duration > 0 else float("nan")
    transition_frequency_cnn = float(transitions_cnn / duration) if duration > 0 else float("nan")
    tracking = _tracking_metrics(reference_voltage, cnn_voltage)
    inference_latency_ms = float((inference_time / num_windows) * 1000.0) if num_windows > 0 else float("nan")

    comparison_rows = [
        {
            "metric": "RMS output voltage",
            "conventional": reference_eval["rms"],
            "cnn": cnn_eval["rms"],
            "improvement_pct": float("nan"),
            "better": "neutral",
        },
        {
            "metric": "Peak voltage",
            "conventional": reference_eval["peak"],
            "cnn": cnn_eval["peak"],
            "improvement_pct": float("nan"),
            "better": "neutral",
        },
        {
            "metric": "Fundamental magnitude",
            "conventional": reference_eval["fundamental_magnitude"],
            "cnn": cnn_eval["fundamental_magnitude"],
            "improvement_pct": float("nan"),
            "better": "neutral",
        },
        {
            "metric": "THD (%)",
            "conventional": reference_eval["thd_percent"],
            "cnn": cnn_eval["thd_percent"],
            "improvement_pct": _relative_improvement(reference_eval["thd_percent"], cnn_eval["thd_percent"], lower_is_better=True),
            "better": "CNN" if cnn_eval["thd_percent"] < reference_eval["thd_percent"] else "Conventional" if cnn_eval["thd_percent"] > reference_eval["thd_percent"] else "neutral",
        },
        {
            "metric": "Switching accuracy (%)",
            "conventional": 100.0,
            "cnn": accuracy * 100.0,
            "improvement_pct": _relative_improvement(100.0, accuracy * 100.0, lower_is_better=False),
            "better": "neutral" if np.isclose(accuracy, 1.0) else "Conventional",
        },
        {
            "metric": "Switching transitions",
            "conventional": transitions_reference,
            "cnn": transitions_cnn,
            "improvement_pct": _relative_improvement(transitions_reference, transitions_cnn, lower_is_better=True),
            "better": "CNN" if transitions_cnn < transitions_reference else "Conventional" if transitions_cnn > transitions_reference else "neutral",
        },
        {
            "metric": "Transitions per cycle",
            "conventional": transitions_per_cycle_ref,
            "cnn": transitions_per_cycle_cnn,
            "improvement_pct": _relative_improvement(transitions_per_cycle_ref, transitions_per_cycle_cnn, lower_is_better=True),
            "better": "CNN" if transitions_per_cycle_cnn < transitions_per_cycle_ref else "Conventional" if transitions_per_cycle_cnn > transitions_per_cycle_ref else "neutral",
        },
        {
            "metric": "RMSE",
            "conventional": 0.0,
            "cnn": tracking["rmse"],
            "improvement_pct": float("nan"),
            "better": "Conventional" if tracking["rmse"] > 0 else "neutral",
        },
        {
            "metric": "MAE",
            "conventional": 0.0,
            "cnn": tracking["mae"],
            "improvement_pct": float("nan"),
            "better": "Conventional" if tracking["mae"] > 0 else "neutral",
        },
        {
            "metric": "Correlation coefficient",
            "conventional": 1.0,
            "cnn": tracking["correlation"],
            "improvement_pct": _relative_improvement(1.0, tracking["correlation"], lower_is_better=False),
            "better": "Conventional" if tracking["correlation"] < 1.0 else "neutral",
        },
        {
            "metric": "Inference latency (ms/window)",
            "conventional": 0.0,
            "cnn": inference_latency_ms,
            "improvement_pct": float("nan"),
            "better": "Conventional" if inference_latency_ms > 0 else "neutral",
        },
    ]

    return {
        "params": params,
        "reference_eval": reference_eval,
        "cnn_eval": cnn_eval,
        "reference_levels": reference_levels,
        "cnn_levels": cnn_levels,
        "reference_voltage": reference_voltage,
        "cnn_voltage": cnn_voltage,
        "accuracy": accuracy,
        "transitions_reference": transitions_reference,
        "transitions_cnn": transitions_cnn,
        "transitions_per_cycle_reference": transitions_per_cycle_ref,
        "transitions_per_cycle_cnn": transitions_per_cycle_cnn,
        "transition_frequency_reference": transition_frequency_ref,
        "transition_frequency_cnn": transition_frequency_cnn,
        "tracking": tracking,
        "inference_time": inference_time,
        "num_windows": num_windows,
        "inference_latency_ms": inference_latency_ms,
        "comparison_table": comparison_rows,
    }


def run_cnn_controller(
    dc_voltage: float,
    fundamental_frequency: float,
    sampling_frequency: float,
    duration: float,
    modulation_index: float,
    phase_shift: float,
    model_path: Path,
) -> CNNControllerResult:
    params = EightLevelInverterParams(
        dc_voltage=dc_voltage,
        fundamental_frequency=fundamental_frequency,
        switching_frequency=2000.0,
        sampling_frequency=sampling_frequency,
        simulation_duration=duration,
        modulation_index=modulation_index,
        phase_shift=phase_shift,
    )
    waveform = generate_8_level_waveform(params)
    reference = modulation_index * np.sin(2.0 * np.pi * fundamental_frequency * waveform.time + phase_shift)
    cnn_voltage, cnn_levels, _, _, _ = reconstruct_cnn_waveform(reference, dc_voltage, model_path)

    fft_reference = compute_fft(waveform.voltage, sampling_frequency)
    fft_cnn = compute_fft(cnn_voltage, sampling_frequency)
    thd_reference, _, _, _, _ = calculate_thd(waveform.voltage, sampling_frequency, fundamental_frequency)
    thd_cnn, _, _, _, _ = calculate_thd(cnn_voltage, sampling_frequency, fundamental_frequency)

    return CNNControllerResult(
        time=waveform.time,
        reference_voltage=waveform.voltage,
        reference_levels=waveform.switching_states,
        cnn_voltage=cnn_voltage,
        cnn_levels=cnn_levels,
        fft_reference=fft_reference,
        fft_cnn=fft_cnn,
        thd_reference=thd_reference,
        thd_cnn=thd_cnn,
        thd_difference=thd_cnn - thd_reference,
    )
