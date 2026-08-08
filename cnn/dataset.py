from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from multilevel.eight_level_model import EightLevelInverterParams, generate_8_level_waveform, get_8_level_values


WINDOW_SIZE = 32


def generate_cnn_dataset(
    output_path: Path,
    num_waveforms: int = 200,
    stride: int = 8,
    dc_voltage_range: tuple[float, float] = (80.0, 120.0),
    modulation_range: tuple[float, float] = (0.35, 1.0),
    frequency_range: tuple[float, float] = (45.0, 55.0),
    phase_range: tuple[float, float] = (0.0, np.pi / 2.0),
    sampling_frequency: float = 10000.0,
    duration: float = 0.04,
) -> pd.DataFrame:
    records: list[dict[str, float | int]] = []

    for sample_id in range(num_waveforms):
        dc_voltage = float(np.random.uniform(*dc_voltage_range))
        modulation_index = float(np.random.uniform(*modulation_range))
        frequency = float(np.random.uniform(*frequency_range))
        phase = float(np.random.uniform(*phase_range))

        params = EightLevelInverterParams(
            dc_voltage=dc_voltage,
            fundamental_frequency=frequency,
            switching_frequency=2000.0,
            sampling_frequency=sampling_frequency,
            simulation_duration=duration,
            modulation_index=modulation_index,
            phase_shift=phase,
        )
        waveform = generate_8_level_waveform(params)

        reference = params.modulation_index * np.sin(
            2.0 * np.pi * params.fundamental_frequency * waveform.time + params.phase_shift
        )
        level_values = get_8_level_values(dc_voltage)
        sorted_levels = np.sort(level_values)
        target_classes = np.searchsorted(np.round(sorted_levels, 8), np.round(waveform.voltage, 8), side="left")
        target_classes = np.clip(target_classes, 0, len(sorted_levels) - 1)

        for idx in range(WINDOW_SIZE // 2, len(reference) - WINDOW_SIZE // 2, stride):
            window = reference[idx - WINDOW_SIZE // 2 : idx + WINDOW_SIZE // 2]
            record = {
                "sample_id": sample_id,
                "dc_voltage": dc_voltage,
                "modulation_index": modulation_index,
                "frequency": frequency,
                "phase": phase,
                "target_level": int(target_classes[idx]),
            }
            for j, value in enumerate(window):
                record[f"window_{j}"] = float(value)
            records.append(record)

    df = pd.DataFrame.from_records(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def build_dataset_from_dataframe(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    features = df[[f"window_{i}" for i in range(WINDOW_SIZE)]].to_numpy(dtype=float)
    targets = df["target_level"].to_numpy(dtype=int)
    return features, targets
