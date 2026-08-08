from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from multilevel.eight_level_model import EightLevelInverterParams, generate_8_level_waveform, get_8_level_values


WINDOW_SIZE = 32


def _split_waveform_ids(
    num_waveforms: int,
    seed: int,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> dict[int, str]:
    if train_fraction + validation_fraction + test_fraction != 1.0:
        raise ValueError("Train, validation, and test fractions must sum to 1.0.")

    rng = np.random.default_rng(seed)
    waveform_ids = np.arange(num_waveforms)
    rng.shuffle(waveform_ids)

    train_cut = int(round(train_fraction * num_waveforms))
    val_cut = train_cut + int(round(validation_fraction * num_waveforms))
    assignment: dict[int, str] = {}
    for idx in waveform_ids[:train_cut]:
        assignment[int(idx)] = "train"
    for idx in waveform_ids[train_cut:val_cut]:
        assignment[int(idx)] = "validation"
    for idx in waveform_ids[val_cut:]:
        assignment[int(idx)] = "test"
    return assignment


def _save_dataset_files(df: pd.DataFrame, output_path: Path, metadata: dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    parquet_path = output_path.with_suffix(".parquet")
    try:
        df.to_parquet(parquet_path, index=False)
        metadata["parquet_saved"] = True
    except Exception:
        metadata["parquet_saved"] = False
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2))


def load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    parquet_path = path.with_suffix(".parquet")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    raise FileNotFoundError(f"Dataset not found at {path} or {csv_path} or {parquet_path}.")


def generate_cnn_dataset(
    output_path: Path,
    num_waveforms: int = 200,
    window_size: int = WINDOW_SIZE,
    stride: int = 8,
    dc_voltage_range: tuple[float, float] = (80.0, 120.0),
    modulation_range: tuple[float, float] = (0.35, 1.0),
    amplitude_range: tuple[float, float] = (0.7, 1.0),
    frequency_range: tuple[float, float] = (45.0, 55.0),
    phase_range: tuple[float, float] = (0.0, np.pi / 2.0),
    sampling_frequency_range: tuple[float, float] = (9000.0, 11000.0),
    duration_range: tuple[float, float] = (0.03, 0.05),
    switching_frequency: float = 2000.0,
    noise_std: float = 0.0,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 0,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    if num_waveforms <= 0:
        raise ValueError("num_waveforms must be positive.")
    if window_size <= 1:
        raise ValueError("window_size must be greater than 1.")
    if stride <= 0 or stride > window_size:
        raise ValueError("stride must be between 1 and window_size.")
    if validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError("validation and test fractions must be positive.")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be less than 1.0.")

    rng = np.random.default_rng(seed)
    split_assignment = _split_waveform_ids(
        num_waveforms=num_waveforms,
        seed=seed,
        train_fraction=1.0 - validation_fraction - test_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )

    records: list[dict[str, float | int | str]] = []
    metadata_waveforms: list[dict[str, float]] = []

    for waveform_id in range(num_waveforms):
        dc_voltage = float(rng.uniform(*dc_voltage_range))
        modulation_index = float(rng.uniform(*modulation_range))
        amplitude = float(rng.uniform(*amplitude_range))
        effective_modulation = float(np.clip(modulation_index * amplitude, 0.0, 1.0))
        frequency = float(rng.uniform(*frequency_range))
        phase = float(rng.uniform(*phase_range))
        sampling_frequency = float(rng.uniform(*sampling_frequency_range))
        duration = float(rng.uniform(*duration_range))

        params = EightLevelInverterParams(
            dc_voltage=dc_voltage,
            fundamental_frequency=frequency,
            switching_frequency=switching_frequency,
            sampling_frequency=sampling_frequency,
            simulation_duration=duration,
            modulation_index=effective_modulation,
            phase_shift=phase,
        )
        waveform = generate_8_level_waveform(params)

        reference = params.modulation_index * np.sin(
            2.0 * np.pi * params.fundamental_frequency * waveform.time + params.phase_shift
        )
        if noise_std > 0.0:
            reference = reference + rng.normal(0.0, noise_std, size=reference.shape)

        sorted_level_values = np.sort(get_8_level_values(dc_voltage))
        target_classes = np.searchsorted(np.round(sorted_level_values, 8), np.round(waveform.voltage, 8), side="left")
        target_classes = np.clip(target_classes, 0, len(sorted_level_values) - 1)

        metadata_waveforms.append(
            {
                "waveform_id": waveform_id,
                "split": split_assignment[waveform_id],
                "dc_voltage": dc_voltage,
                "modulation_index": modulation_index,
                "amplitude": amplitude,
                "effective_modulation": params.modulation_index,
                "frequency": frequency,
                "phase": phase,
                "sampling_frequency": sampling_frequency,
                "duration": duration,
                "noise_std": noise_std,
                "samples": int(len(reference)),
            }
        )

        for idx in range(window_size, len(reference), stride):
            window = reference[idx - window_size : idx]
            record = {
                "waveform_id": waveform_id,
                "split": split_assignment[waveform_id],
                "dc_voltage": dc_voltage,
                "modulation_index": modulation_index,
                "amplitude": amplitude,
                "effective_modulation": params.modulation_index,
                "frequency": frequency,
                "phase": phase,
                "sampling_frequency": sampling_frequency,
                "duration": duration,
                "noise_std": noise_std,
                "target_level": int(target_classes[idx]),
                "target_label": str(waveform.switching_state_labels[idx]),
            }
            for j, value in enumerate(window):
                record[f"window_{j}"] = float(value)
            records.append(record)

        if progress_callback is not None:
            progress_callback(waveform_id + 1, num_waveforms)

    df = pd.DataFrame.from_records(records)
    metadata: dict[str, object] = {
        "generation_timestamp": time.time(),
        "seed": seed,
        "num_waveforms": num_waveforms,
        "window_size": window_size,
        "num_samples": int(len(df)),
        "stride": stride,
        "dc_voltage_range": dc_voltage_range,
        "modulation_range": modulation_range,
        "amplitude_range": amplitude_range,
        "frequency_range": frequency_range,
        "phase_range": phase_range,
        "sampling_frequency_range": sampling_frequency_range,
        "duration_range": duration_range,
        "switching_frequency": switching_frequency,
        "noise_std": noise_std,
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
        "split_counts": {
            "train": sum(1 for v in split_assignment.values() if v == "train"),
            "validation": sum(1 for v in split_assignment.values() if v == "validation"),
            "test": sum(1 for v in split_assignment.values() if v == "test"),
        },
        "class_mapping": {str(i): int(i) for i in range(8)},
        "waveform_metadata": metadata_waveforms,
    }
    _save_dataset_files(df, output_path, metadata)
    return df


def build_dataset_from_dataframe(
    df: pd.DataFrame,
    split: Optional[str] = None,
    window_size: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    if split is not None and "split" in df.columns:
        df = df[df["split"] == split]
    if window_size is None:
        window_columns = [col for col in df.columns if col.startswith("window_")]
        window_size = len(window_columns)
    features = df[[f"window_{i}" for i in range(window_size)]].to_numpy(dtype=float)
    targets = df["target_level"].to_numpy(dtype=int)
    return features, targets


def analyze_dataset(df: pd.DataFrame) -> dict[str, object]:
    total_samples = len(df)
    waveform_count = int(df["waveform_id"].nunique()) if "waveform_id" in df.columns else 0
    splits = df["split"].value_counts().to_dict() if "split" in df.columns else {}
    class_counts = df["target_level"].value_counts().sort_index()
    class_percentages = (class_counts / total_samples * 100.0).round(2).to_dict()

    condition_columns = [
        "dc_voltage",
        "modulation_index",
        "amplitude",
        "effective_modulation",
        "frequency",
        "phase",
        "sampling_frequency",
        "duration",
        "noise_std",
    ]
    operating_ranges: dict[str, dict[str, float]] = {}
    for column in condition_columns:
        if column in df.columns:
            series = df[column].astype(float)
            operating_ranges[column] = {
                "min": float(series.min()),
                "max": float(series.max()),
                "mean": float(series.mean()),
                "std": float(series.std()),
            }

    window_columns = [col for col in df.columns if col.startswith("window_")]
    metadata = {
        "total_samples": total_samples,
        "waveform_count": waveform_count,
        "split_counts": splits,
        "class_counts": class_counts.to_dict(),
        "class_percentages": class_percentages,
        "feature_shape": (total_samples, len(window_columns)),
        "operating_ranges": operating_ranges,
    }
    return metadata
