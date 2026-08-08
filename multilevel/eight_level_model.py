from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import itertools
from typing import Any

import numpy as np
import pandas as pd
from signal_processing.fft import compute_fft


@dataclass
class EightLevelInverterParams:
    """Configuration parameters for the 8-level inverter simulation."""

    num_levels: int = 8
    dc_voltage: float = 400.0
    fundamental_frequency: float = 50.0
    switching_frequency: float = 2000.0
    sampling_frequency: float = 10000.0
    simulation_duration: float = 0.02
    modulation_index: float = 1.0
    phase_shift: float = 0.0

    def validate(self) -> None:
        if self.num_levels != 8:
            raise ValueError("This inverter engine only supports 8 levels.")
        if self.dc_voltage <= 0:
            raise ValueError("DC voltage must be positive.")
        if self.fundamental_frequency <= 0:
            raise ValueError("Fundamental frequency must be positive.")
        if self.switching_frequency <= 0:
            raise ValueError("Switching frequency must be positive.")
        if self.sampling_frequency <= 2 * self.fundamental_frequency:
            raise ValueError("Sampling frequency must be greater than twice the fundamental frequency.")
        if self.simulation_duration <= 0:
            raise ValueError("Simulation duration must be positive.")
        if not 0 < self.modulation_index <= 1.0:
            raise ValueError("Modulation index must be between 0 and 1.")


@dataclass
class EightLevelWaveform:
    time: np.ndarray
    voltage: np.ndarray
    switching_states: np.ndarray
    switching_state_labels: np.ndarray
    params: EightLevelInverterParams


def _generate_switching_state_map(dc_voltage: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the switching state matrix, output voltage levels, and labels."""
    voltage_unit = dc_voltage / 7.0
    cell_weights = np.array([1.0, 2.0, 4.0], dtype=float) * voltage_unit
    state_combinations = np.array(list(itertools.product([-1, 1], repeat=3)), dtype=int)
    output_levels = state_combinations.dot(cell_weights)
    labels = np.array(
        [
            ".".join(["+" if value > 0 else "-" for value in state])
            for state in state_combinations
        ],
        dtype=str,
    )
    return state_combinations, output_levels, labels


def generate_8_level_waveform(params: EightLevelInverterParams) -> EightLevelWaveform:
    """Generate an 8-level staircase waveform from inverter parameters."""
    params.validate()

    num_points = int(round(params.simulation_duration * params.sampling_frequency))
    if num_points < 2:
        raise ValueError("Simulation duration and sampling frequency produce too few samples.")

    time = np.linspace(0.0, params.simulation_duration, num_points, endpoint=False)
    reference = params.modulation_index * np.sin(
        2.0 * np.pi * params.fundamental_frequency * time + params.phase_shift
    )

    carrier_phases = np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0], dtype=float)
    carrier_waveforms = np.zeros((3, num_points), dtype=float)
    for cell_idx, phase in enumerate(carrier_phases):
        carrier_waveforms[cell_idx, :] = np.abs(
            2.0 * (np.remainder((params.switching_frequency * time + phase / (2.0 * np.pi)), 1.0) - 0.5)
        ) * 2.0 - 1.0

    states, levels, labels = _generate_switching_state_map(params.dc_voltage)
    waveform_voltage = np.zeros(num_points, dtype=float)
    switching_states = np.zeros((num_points, 3), dtype=int)
    switching_labels = np.empty(num_points, dtype=object)

    for index in range(num_points):
        normalized_ref = reference[index]
        cell_states = np.where(normalized_ref >= carrier_waveforms[:, index], 1, -1)
        waveform_voltage[index] = cell_states.dot(np.array([params.dc_voltage / 7.0, 2.0 * params.dc_voltage / 7.0, 4.0 * params.dc_voltage / 7.0]))
        switching_states[index, :] = cell_states

        matching_index = int(np.where((states == cell_states).all(axis=1))[0][0])
        switching_labels[index] = labels[matching_index]

    return EightLevelWaveform(
        time=time,
        voltage=waveform_voltage,
        switching_states=switching_states,
        switching_state_labels=switching_labels,
        params=params,
    )


def get_8_level_values(dc_voltage: float) -> np.ndarray:
    """Return the sorted 8-level output voltage values for the given DC voltage."""
    _, levels, _ = _generate_switching_state_map(dc_voltage)
    return np.sort(np.unique(levels))


def validate_waveform(waveform: EightLevelWaveform) -> dict[str, Any]:
    """Validate the generated waveform and return summary statistics."""
    voltage = waveform.voltage
    unique_levels = np.unique(np.round(voltage, decimals=8))
    max_voltage = float(np.max(voltage))
    min_voltage = float(np.min(voltage))
    sample_count = int(len(voltage))
    detected_levels = int(len(unique_levels))

    frequencies, magnitudes = compute_fft(voltage, waveform.params.sampling_frequency)
    peak_index = int(np.argmax(magnitudes[1:]) + 1)
    fundamental_frequency = float(frequencies[peak_index])

    return {
        "detected_levels": detected_levels,
        "unique_voltage_levels": unique_levels.tolist(),
        "max_voltage": max_voltage,
        "min_voltage": min_voltage,
        "sample_count": sample_count,
        "fundamental_frequency": fundamental_frequency,
        "switching_frequency": waveform.params.switching_frequency,
        "sampling_frequency": waveform.params.sampling_frequency,
        "simulation_duration": waveform.params.simulation_duration,
    }


def save_waveform_csv(file_path: Path | str, waveform: EightLevelWaveform) -> None:
    """Save the generated waveform to a CSV file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "time": waveform.time,
            "voltage": waveform.voltage,
            "switching_state": waveform.switching_state_labels,
        }
    )
    df.to_csv(path, index=False)


# Legacy scaffold preserved for reference
class EightLevelWaveformGenerator:
    """Legacy scaffold for prior 8-level waveform experiments."""

    def __init__(self, switch_params: dict[str, float] | None = None) -> None:
        self.switch_params = switch_params or {
            "level_1": 0.1,
            "level_2": 0.3,
            "level_3": 0.5,
            "level_4": 0.7,
        }

    def generate_waveform(self, num_points: int = 1024, frequency: float = 50.0, duration: float = 0.02) -> np.ndarray:
        t = np.linspace(0, duration, num_points, endpoint=False)
        base = np.sin(2 * np.pi * frequency * t)
        thresholds = sorted(self.switch_params.values())
        waveform = np.zeros_like(base)
        for i, threshold in enumerate(thresholds, start=1):
            waveform += np.where(base >= threshold, i, 0)
            waveform -= np.where(base <= -threshold, i, 0)
        return waveform
