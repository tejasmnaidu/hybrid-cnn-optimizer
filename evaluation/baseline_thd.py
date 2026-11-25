"""Reconstruct and summarize held-out input-voltage THD."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.data_loader import DatasetTables
from preprocessing.data_validation import _id_column

from .thd_analysis import calculate_thd


WAVEFORM_POINTS = 400
CELL_COUNT = 3
CELL_VOLTAGE_V = 100.0
EXPECTED_TEST_SAMPLES = 1500


def _ordered_signal_matrix(
    frame: pd.DataFrame,
    *,
    table_name: str,
    column_prefix: str,
    column_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    id_column = _id_column(frame, table_name)
    sample_ids = pd.to_numeric(frame[id_column], errors="raise")
    if sample_ids.isna().any() or (sample_ids % 1 != 0).any() or sample_ids.duplicated().any():
        raise ValueError(f"{table_name} SampleIDs must be unique integers.")

    signal_columns = [column for column in frame.columns if column != id_column]
    expected_columns = [f"{column_prefix}{index}" for index in range(1, column_count + 1)]
    if [column.casefold() for column in signal_columns] != [
        column.casefold() for column in expected_columns
    ]:
        raise ValueError(
            f"{table_name} must contain {column_count} signal columns in the expected order."
        )
    try:
        values = frame[signal_columns].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{table_name} contains non-numeric signal values.") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"{table_name} contains missing or non-finite signal values.")

    order = np.argsort(sample_ids.to_numpy(), kind="stable")
    return sample_ids.to_numpy(dtype=np.int64)[order], values[order]


def _ordered_numeric_column(
    frame: pd.DataFrame,
    *,
    table_name: str,
    column_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    id_column = _id_column(frame, table_name)
    matching_columns = [column for column in frame if column.casefold() == column_name.casefold()]
    if len(matching_columns) != 1:
        raise ValueError(f"{table_name} must contain exactly one {column_name} column.")
    sample_ids = pd.to_numeric(frame[id_column], errors="raise")
    if sample_ids.isna().any() or (sample_ids % 1 != 0).any() or sample_ids.duplicated().any():
        raise ValueError(f"{table_name} SampleIDs must be unique integers.")
    try:
        values = pd.to_numeric(frame[matching_columns[0]], errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{table_name} {column_name} values must be numeric.") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"{table_name} {column_name} values must be finite.")
    order = np.argsort(sample_ids.to_numpy(), kind="stable")
    return sample_ids.to_numpy(dtype=np.int64)[order], values[order]


def evaluate_test_input_voltage_thd(
    tables: DatasetTables,
    cell_states_path: str | Path,
    output_dir: str | Path,
    *,
    sampling_frequency_hz: float = 20_000.0,
    fundamental_frequency_hz: float = 50.0,
    max_harmonic: int = 100,
    expected_test_count: int = EXPECTED_TEST_SAMPLES,
) -> dict[str, Any]:
    """Evaluate reconstructed input-voltage THD for the supplied testing split only."""
    if expected_test_count < 1:
        raise ValueError("expected_test_count must be positive.")
    input_ids, input_values = _ordered_signal_matrix(
        tables["input_voltage"],
        table_name="Input_Voltage.csv",
        column_prefix="inputVoltage",
        column_count=WAVEFORM_POINTS,
    )
    split_frame = tables["sample_splits"]
    split_id_column = _id_column(split_frame, "Sample_Splits.csv")
    split_column_names = [column for column in split_frame if column.casefold() == "splitlabel"]
    if len(split_column_names) != 1:
        raise ValueError("Sample_Splits.csv must contain exactly one splitLabel column.")
    split_ids = pd.to_numeric(split_frame[split_id_column], errors="raise")
    if split_ids.isna().any() or (split_ids % 1 != 0).any() or split_ids.duplicated().any():
        raise ValueError("Sample_Splits.csv SampleIDs must be unique integers.")
    split_order = np.argsort(split_ids.to_numpy(), kind="stable")
    sorted_split_ids = split_ids.to_numpy(dtype=np.int64)[split_order]
    if not np.array_equal(input_ids, sorted_split_ids):
        raise ValueError("Sample_Splits.csv SampleIDs do not align with Input_Voltage.csv.")
    split_labels = (
        split_frame[split_column_names[0]].astype(str).str.strip().str.lower().to_numpy()[split_order]
    )
    if not set(split_labels).issubset({"training", "validation", "testing"}):
        raise ValueError("Sample_Splits.csv contains an unsupported split label.")
    test_indices = np.flatnonzero(split_labels == "testing")
    if test_indices.size != expected_test_count:
        raise ValueError(
            f"Expected {expected_test_count} testing samples, found {test_indices.size}."
        )

    states_path = Path(cell_states_path)
    if not states_path.is_file():
        raise FileNotFoundError(f"Cell_States.csv does not exist: {states_path}")
    try:
        states_frame = pd.read_csv(states_path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not load {states_path}: {exc}") from exc
    state_ids, state_values = _ordered_signal_matrix(
        states_frame,
        table_name="Cell_States.csv",
        column_prefix="stateData",
        column_count=CELL_COUNT * WAVEFORM_POINTS,
    )
    if not np.array_equal(input_ids, state_ids):
        raise ValueError("Cell_States.csv SampleIDs do not align with Input_Voltage.csv.")
    test_states = state_values[test_indices]
    if not np.isin(test_states, (-1.0, 0.0, 1.0)).all():
        raise ValueError("Testing cell states must contain only -1, 0, or 1.")

    state_blocks = test_states.reshape(-1, CELL_COUNT, WAVEFORM_POINTS)
    reconstructed_voltage = CELL_VOLTAGE_V * np.sum(state_blocks, axis=1)
    input_test_voltage = input_values[test_indices]
    reconstruction_error = np.abs(reconstructed_voltage - input_test_voltage)
    reconstruction_exact = bool(np.array_equal(reconstructed_voltage, input_test_voltage))
    if not reconstruction_exact:
        raise ValueError(
            "Reconstructed testing input voltage does not exactly match Input_Voltage.csv "
            f"(maximum absolute error {float(np.max(reconstruction_error)):.12g} V)."
        )

    thd_ids, recorded_thd_values = _ordered_numeric_column(
        tables["thd_results"],
        table_name="THD_Results.csv",
        column_name="inputVoltageTHD",
    )
    if not np.array_equal(input_ids, thd_ids):
        raise ValueError("THD_Results.csv SampleIDs do not align with Input_Voltage.csv.")
    recorded_thd = recorded_thd_values[test_indices]
    if (recorded_thd < 0).any():
        raise ValueError("Recorded inputVoltageTHD values must be non-negative.")
    calculated_thd = np.asarray(
        [
            calculate_thd(
                waveform,
                sampling_frequency_hz=sampling_frequency_hz,
                fundamental_frequency_hz=fundamental_frequency_hz,
                max_harmonic=max_harmonic,
            )
            for waveform in reconstructed_voltage
        ],
        dtype=np.float64,
    )
    absolute_difference = np.abs(calculated_thd - recorded_thd)
    sample_standard_deviation = (
        float(np.std(calculated_thd, ddof=1)) if calculated_thd.size > 1 else None
    )
    distribution = {
        "mean_percent": float(np.mean(calculated_thd)),
        "standard_deviation_percent": sample_standard_deviation,
        "standard_deviation_method": (
            "sample standard deviation (ddof=1)"
            if sample_standard_deviation is not None
            else "undefined for a single sample"
        ),
        "median_percent": float(np.median(calculated_thd)),
        "minimum_percent": float(np.min(calculated_thd)),
        "maximum_percent": float(np.max(calculated_thd)),
        "25th_percentile_percent": float(np.percentile(calculated_thd, 25)),
        "75th_percentile_percent": float(np.percentile(calculated_thd, 75)),
    }

    per_sample = pd.DataFrame(
        {
            "SampleID": input_ids[test_indices],
            "calculated_input_voltage_thd_percent": calculated_thd,
            "recorded_input_voltage_thd_percent": recorded_thd,
            "absolute_difference_percent_points": absolute_difference,
        }
    )
    summary: dict[str, Any] = {
        "result_type": "testing_split_reconstructed_input_voltage_baseline_thd",
        "configuration": {
            "split_label": "testing",
            "cell_state_layout": "three consecutive cell-major blocks of 400 points",
            "voltage_reconstruction": "100 V * (state1 + state2 + state3)",
            "cell_voltage_v_per_state": CELL_VOLTAGE_V,
            "cell_count": CELL_COUNT,
            "waveform_points": WAVEFORM_POINTS,
            "sampling_frequency_hz": float(sampling_frequency_hz),
            "fundamental_frequency_hz": float(fundamental_frequency_hz),
            "max_harmonic": int(max_harmonic),
            "thd_harmonic_orders": f"2 through {int(max_harmonic)}",
            "thd_unit": "percent",
            "expected_test_sample_count": int(expected_test_count),
        },
        "test_sample_count": int(test_indices.size),
        "reconstruction_check": {
            "exact_match": reconstruction_exact,
            "checked_samples": int(test_indices.size),
            "checked_waveform_points_per_sample": WAVEFORM_POINTS,
            "checked_values": int(reconstructed_voltage.size),
            "maximum_absolute_error_v": float(np.max(reconstruction_error)),
        },
        "comparison_errors_percent_points": {
            "mean_absolute_difference": float(np.mean(absolute_difference)),
            "root_mean_square_difference": float(np.sqrt(np.mean(np.square(absolute_difference)))),
            "maximum_absolute_difference": float(np.max(absolute_difference)),
        },
        "calculated_thd_distribution": distribution,
        "interpretation": (
            "Descriptive baseline for reconstructed dataset input-voltage waveforms; "
            "not post-optimization THD reduction or measured inverter-output THD."
        ),
        "output_files": {
            "per_sample_csv": "input_voltage_baseline_thd_test.csv",
            "summary_json": "input_voltage_baseline_thd_summary.json",
        },
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    per_sample.to_csv(output_path / summary["output_files"]["per_sample_csv"], index=False)
    (output_path / summary["output_files"]["summary_json"]).write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return summary
