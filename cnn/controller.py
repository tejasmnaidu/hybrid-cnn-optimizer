from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

from cnn.dataset import WINDOW_SIZE
from cnn.model import EightLevelCNN
from cnn.predict import predict_switching_state
from multilevel.eight_level_model import EightLevelInverterParams, generate_8_level_waveform, get_8_level_values
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
) -> tuple[np.ndarray, np.ndarray]:
    level_values = np.sort(get_8_level_values(dc_voltage))
    cnn_levels = np.zeros(len(reference), dtype=int)
    cnn_voltage = np.zeros(len(reference), dtype=float)

    for idx in range(WINDOW_SIZE // 2, len(reference) - WINDOW_SIZE // 2):
        window = reference[idx - WINDOW_SIZE // 2 : idx + WINDOW_SIZE // 2]
        result = predict_switching_state(model_path, window)
        cnn_levels[idx] = result["predicted_class"]
        cnn_voltage[idx] = float(level_values[cnn_levels[idx]])

    cnn_levels[: WINDOW_SIZE // 2] = cnn_levels[WINDOW_SIZE // 2]
    cnn_levels[-WINDOW_SIZE // 2 :] = cnn_levels[-WINDOW_SIZE // 2 - 1]
    cnn_voltage[: WINDOW_SIZE // 2] = float(level_values[cnn_levels[WINDOW_SIZE // 2]])
    cnn_voltage[-WINDOW_SIZE // 2 :] = float(level_values[cnn_levels[-WINDOW_SIZE // 2 - 1]])

    return cnn_voltage, cnn_levels


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
    cnn_voltage, cnn_levels = reconstruct_cnn_waveform(reference, dc_voltage, model_path)

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
