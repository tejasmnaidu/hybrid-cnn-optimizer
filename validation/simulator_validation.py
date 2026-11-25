"""Compare the ideal inverter simulator with supplied waveform CSV records."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import os
from pathlib import Path
import zipfile
from typing import Any

import numpy as np
import pandas as pd

from multilevel.inverter_simulator import (
    CELL_COUNT,
    DC_VOLTAGE_PER_CELL_V,
    FUNDAMENTAL_FREQUENCY_HZ,
    SAMPLING_FREQUENCY_HZ,
    SAMPLES_PER_CYCLE,
    InverterSimulation,
    simulate_inverter,
)


_REFERENCE_FILES = {
    "conditions": "Operating_Conditions.csv",
    "states": "Cell_States.csv",
    "gates": "Switching_Gates.csv",
    "voltage": "Input_Voltage.csv",
    "current": "Load_Current.csv",
}
_COMPARISONS = {
    "cell_states": ("states", 0.0, 0.0),
    "switching_gates": ("gates", 0.0, 0.0),
    "output_voltage": ("voltage", 1e-12, 0.0),
    "load_current": ("current", 1e-12, 0.0),
}
_ORIGINAL_SOURCE_SUFFIXES = {".m", ".mlx", ".slx", ".mdl", ".mat"}


def compare_arrays(
    reference: np.ndarray,
    simulated: np.ndarray,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> dict[str, Any]:
    """Compare two same-shaped finite arrays and report mismatch/error details."""
    expected = np.asarray(reference)
    actual = np.asarray(simulated)
    if expected.shape != actual.shape:
        raise ValueError(
            f"Reference shape {expected.shape} does not match simulator shape "
            f"{actual.shape}."
        )
    if expected.size == 0:
        raise ValueError("Reference and simulator arrays must not be empty.")
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError("Reference and simulator arrays must contain finite values.")
    if not np.isfinite(atol) or atol < 0.0 or not np.isfinite(rtol) or rtol < 0.0:
        raise ValueError("Comparison tolerances must be finite and non-negative.")

    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    matches = np.isclose(actual, expected, atol=atol, rtol=rtol)
    mismatch_positions = np.argwhere(~matches)
    first_index = (
        [int(index) for index in mismatch_positions[0]]
        if mismatch_positions.size
        else None
    )
    return {
        "element_count": int(expected.size),
        "mismatch_count": int(mismatch_positions.shape[0]),
        "atol": float(atol),
        "rtol": float(rtol),
        "max_absolute_error": float(np.max(difference)),
        "rms_error": float(np.sqrt(np.mean(np.square(difference)))),
        "first_mismatch_index": first_index,
    }


def compare_simulation(
    simulation: InverterSimulation,
    reference: Mapping[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    """Compare one simulator result against reference states, gates, Vout and I."""
    results: dict[str, dict[str, Any]] = {}
    for simulated_name, (reference_name, atol, rtol) in _COMPARISONS.items():
        if reference_name not in reference:
            raise ValueError(f"Reference data is missing {reference_name!r}.")
        results[simulated_name] = compare_arrays(
            reference[reference_name],
            getattr(simulation, simulated_name),
            atol=atol,
            rtol=rtol,
        )
    return results


def _column_name(columns: list[str], expected: str, table: str) -> str:
    found = [column for column in columns if column.lower() == expected.lower()]
    if len(found) != 1:
        raise ValueError(
            f"{table} must contain exactly one {expected!r} column; found {found}."
        )
    return found[0]


def _signal_columns(columns: list[str], table: str) -> tuple[str, list[str]]:
    id_column = _column_name(columns, "sampleID", table)
    return id_column, [column for column in columns if column != id_column]


def _artifact_inventory(project_root: Path) -> dict[str, Any]:
    ignored_parts = {".venv", ".git", "__pycache__"}
    source_files: list[str] = []
    waveform_csvs: list[str] = []
    archive_sources: list[str] = []
    bench_data_files: list[str] = []
    bench_terms = ("bench", "experimental", "measurement", "oscilloscope", "scope_data")

    for current_root, directories, filenames in os.walk(project_root):
        directories[:] = [name for name in directories if name not in ignored_parts]
        for filename in filenames:
            path = Path(current_root) / filename
            relative = str(path.relative_to(project_root)).replace("\\", "/")
            if path.suffix.lower() in _ORIGINAL_SOURCE_SUFFIXES:
                source_files.append(relative)
            if path.suffix.lower() == ".csv" and path.parent.name.lower().startswith(
                "seven_level_inverter_project"
            ):
                waveform_csvs.append(relative)
            if any(term in relative.lower() for term in bench_terms):
                bench_data_files.append(relative)
            if path.suffix.lower() == ".zip":
                try:
                    with zipfile.ZipFile(path) as archive:
                        archive_sources.extend(
                            f"{relative}!{member}"
                            for member in archive.namelist()
                            if Path(member).suffix.lower() in _ORIGINAL_SOURCE_SUFFIXES
                        )
                except (OSError, zipfile.BadZipFile):
                    continue

    return {
        "original_matlab_simulink_or_mat_files": sorted(source_files),
        "original_sources_in_archives": sorted(archive_sources),
        "reference_waveform_csv_files": sorted(waveform_csvs),
        "reference_data_status": (
            "reference waveform CSVs provided"
            if waveform_csvs
            else "reference data not provided"
        ),
        "bench_data_files": sorted(bench_data_files),
    }


def validate_dataset_reference(
    dataset_root: str | Path,
    *,
    project_root: str | Path,
    chunksize: int = 16,
) -> dict[str, Any]:
    """Compare all supplied dataset records against the ideal simulator."""
    root = Path(dataset_root)
    if chunksize < 1:
        raise ValueError("chunksize must be positive.")
    paths = {name: root / filename for name, filename in _REFERENCE_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required reference CSV files are missing: {missing}")

    header_frames = {
        name: pd.read_csv(path, nrows=0) for name, path in paths.items()
    }
    id_columns: dict[str, str] = {}
    signal_columns: dict[str, list[str]] = {}
    for name, frame in header_frames.items():
        id_columns[name], signal_columns[name] = _signal_columns(
            list(frame.columns), paths[name].name
        )

    expected_signal_widths = {
        "states": CELL_COUNT * SAMPLES_PER_CYCLE,
        "gates": CELL_COUNT * 4 * SAMPLES_PER_CYCLE,
        "voltage": SAMPLES_PER_CYCLE,
        "current": SAMPLES_PER_CYCLE,
    }
    for name, width in expected_signal_widths.items():
        actual_width = len(signal_columns[name])
        if actual_width != width:
            raise ValueError(
                f"{paths[name].name} has {actual_width} signal columns; expected {width}."
            )
    condition_columns = {
        field: _column_name(
            list(header_frames["conditions"].columns),
            field,
            paths["conditions"].name,
        )
        for field in (
            "modulationIndex",
            "loadResistance",
            "loadInductance",
            "phaseAngle",
        )
    }

    readers = {
        name: pd.read_csv(path, chunksize=chunksize)
        for name, path in paths.items()
    }
    totals = {
        name: {
            "element_count": 0,
            "mismatch_count": 0,
            "atol": atol,
            "rtol": rtol,
            "max_absolute_error": 0.0,
            "sum_squared_error": 0.0,
            "first_mismatch_sample_id": None,
            "first_mismatch_index": None,
        }
        for name, (_, atol, rtol) in _COMPARISONS.items()
    }
    record_count = 0
    seen_sample_ids: set[int] = set()

    while True:
        chunks = {
            name: next(reader, None) for name, reader in readers.items()
        }
        finished = [chunk is None for chunk in chunks.values()]
        if all(finished):
            break
        if any(finished):
            raise ValueError("Reference CSV files contain different numbers of records.")
        record_ids = {
            name: pd.to_numeric(chunk[id_columns[name]], errors="raise").to_numpy()
            for name, chunk in chunks.items()
        }
        first_ids = next(iter(record_ids.values()))
        if any(not np.array_equal(first_ids, ids) for ids in record_ids.values()):
            raise ValueError("Reference CSV SampleID columns are not aligned.")
        numeric_ids = np.asarray(first_ids, dtype=np.float64)
        if not np.isfinite(numeric_ids).all() or np.any(numeric_ids % 1 != 0):
            raise ValueError("Reference CSV SampleID values must be finite integers.")
        integer_ids = numeric_ids.astype(np.int64)
        if np.unique(integer_ids).size != integer_ids.size or seen_sample_ids.intersection(
            integer_ids.tolist()
        ):
            raise ValueError("Reference CSV SampleID values must be unique.")
        seen_sample_ids.update(integer_ids.tolist())

        for row_offset in range(len(chunks["conditions"])):
            sample_id = int(first_ids[row_offset])
            conditions = chunks["conditions"].iloc[row_offset]
            simulation = simulate_inverter(
                float(conditions[condition_columns["modulationIndex"]]),
                float(conditions[condition_columns["loadResistance"]]),
                float(conditions[condition_columns["loadInductance"]]),
                float(conditions[condition_columns["phaseAngle"]]),
            )
            reference = {
                "states": chunks["states"]
                .iloc[row_offset][signal_columns["states"]]
                .to_numpy(dtype=np.int8)
                .reshape(CELL_COUNT, SAMPLES_PER_CYCLE),
                "gates": chunks["gates"]
                .iloc[row_offset][signal_columns["gates"]]
                .to_numpy(dtype=np.uint8)
                .reshape(CELL_COUNT, 4, SAMPLES_PER_CYCLE),
                "voltage": chunks["voltage"]
                .iloc[row_offset][signal_columns["voltage"]]
                .to_numpy(dtype=np.float64),
                "current": chunks["current"]
                .iloc[row_offset][signal_columns["current"]]
                .to_numpy(dtype=np.float64),
            }
            comparisons = compare_simulation(simulation, reference)
            for name, comparison in comparisons.items():
                aggregate = totals[name]
                aggregate["element_count"] += comparison["element_count"]
                aggregate["mismatch_count"] += comparison["mismatch_count"]
                aggregate["max_absolute_error"] = max(
                    aggregate["max_absolute_error"],
                    comparison["max_absolute_error"],
                )
                aggregate["sum_squared_error"] += (
                    comparison["rms_error"] ** 2 * comparison["element_count"]
                )
                if (
                    comparison["first_mismatch_index"] is not None
                    and aggregate["first_mismatch_sample_id"] is None
                ):
                    aggregate["first_mismatch_sample_id"] = sample_id
                    aggregate["first_mismatch_index"] = comparison[
                        "first_mismatch_index"
                    ]
            record_count += 1

    for aggregate in totals.values():
        count = aggregate.pop("element_count")
        sum_squared_error = aggregate.pop("sum_squared_error")
        aggregate["element_count"] = count
        aggregate["rms_error"] = float(np.sqrt(sum_squared_error / count))

    inventory = _artifact_inventory(Path(project_root))
    all_passed = all(value["mismatch_count"] == 0 for value in totals.values())
    return {
        "validation_type": "ideal simulator versus supplied exported waveform CSVs",
        "status": "PASS" if all_passed else "MISMATCHES DETECTED",
        "reference_provenance": (
            "Provided project dataset CSVs; original MATLAB/Simulink source and "
            "bench measurements were not found."
        ),
        "reference_inventory": inventory,
        "conditions": {
            "records_compared": record_count,
            "phase_angle_interpretation": "radians, as accepted by simulate_inverter",
            "carrier_phase_assumption": (
                "The reference CSVs do not record carrier phase offsets; simulator "
                "defaults (0, 60, 120 degrees) were assumed."
            ),
            "carrier_phase_offsets_degrees": [0.0, 60.0, 120.0],
            "dc_voltage_per_cell_v": DC_VOLTAGE_PER_CELL_V,
            "fundamental_frequency_hz": FUNDAMENTAL_FREQUENCY_HZ,
            "sample_rate_hz": SAMPLING_FREQUENCY_HZ,
            "samples_per_cycle": SAMPLES_PER_CYCLE,
        },
        "comparison_tolerances": {
            "cell_states": {"atol": 0.0, "rtol": 0.0, "comparison": "exact"},
            "switching_gates": {"atol": 0.0, "rtol": 0.0, "comparison": "exact"},
            "output_voltage": {"atol": 1e-12, "rtol": 0.0, "units": "V"},
            "load_current": {"atol": 1e-12, "rtol": 0.0, "units": "A"},
        },
        "signals": totals,
        "validated": [
            "sampled ideal cell states against supplied Cell_States.csv",
            "sampled ideal gate signals against supplied Switching_Gates.csv",
            "sampled ideal output voltage against supplied Input_Voltage.csv",
            "sampled ideal RL load current against supplied Load_Current.csv",
        ],
        "not_validated": [
            "agreement with original MATLAB/Simulink source (not provided)",
            "hardware or bench behavior (no bench data provided)",
            "dead time, switching losses, device drops, parasitics, thermal behavior, or other non-ideal effects",
            "physical correctness of the supplied switching convention beyond agreement with the supplied CSV records",
        ],
    }


def _format_mismatch_row(name: str, metrics: Mapping[str, Any]) -> str:
    first = metrics["first_mismatch_index"]
    first_text = (
        "none"
        if first is None
        else f"SampleID {metrics['first_mismatch_sample_id']}, index {first}"
    )
    unit = "V" if name == "output_voltage" else "A" if name == "load_current" else "—"
    return (
        f"| {name} | {metrics['mismatch_count']} / {metrics['element_count']} | "
        f"{metrics['max_absolute_error']:.12g} {unit} | "
        f"{metrics['rms_error']:.12g} {unit} | {first_text} |"
    )


def write_validation_reports(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write a machine-readable result and a concise human-readable report."""
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    inventory = report["reference_inventory"]
    source_text = (
        ", ".join(inventory["original_matlab_simulink_or_mat_files"])
        or "No original MATLAB/Simulink source or .mat files found."
    )
    archive_text = (
        ", ".join(inventory["original_sources_in_archives"])
        or "No original MATLAB/Simulink source found in supplied ZIP archives."
    )
    bench_text = (
        ", ".join(inventory["bench_data_files"])
        if inventory["bench_data_files"]
        else "No bench/measurement data files found."
    )
    csv_text = "\n".join(
        f"- `{name}`" for name in inventory["reference_waveform_csv_files"]
    ) or "- No reference waveform CSV files found."
    rows = "\n".join(
        _format_mismatch_row(name, metrics)
        for name, metrics in report["signals"].items()
    )
    validated = "\n".join(f"- {item}" for item in report["validated"])
    not_validated = "\n".join(f"- {item}" for item in report["not_validated"])
    markdown = f"""# Inverter Simulator Validation

**Result:** {report['status']}  
**Records compared:** {report['conditions']['records_compared']}  
**Scope:** Ideal sampled simulator compared with the supplied exported waveform CSVs. This is not hardware validation.

## Reference inventory

- Original MATLAB/Simulink or `.mat` source files: {source_text}
- Source files inside supplied archives: {archive_text}
- Bench/measurement data: {bench_text}
- Exported reference waveform CSVs:
{csv_text}

The CSV waveforms are available as comparison references, so validation proceeded. Their presence does not establish independent provenance or substitute for the absent original source/bench measurements.

## Comparison conditions

- Carrier shifts: {report['conditions']['carrier_phase_offsets_degrees']} degrees
- DC voltage: {report['conditions']['dc_voltage_per_cell_v']} V/cell
- Fundamental frequency: {report['conditions']['fundamental_frequency_hz']} Hz
- Sampling rate: {report['conditions']['sample_rate_hz']} Hz; {report['conditions']['samples_per_cycle']} samples/cycle
- Phase angle interpreted as radians, matching the simulator API.
- Carrier offsets are not recorded in the supplied operating-condition CSV; the simulator defaults were assumed.
- State and gate comparisons are exact. Voltage and current use absolute tolerance 1e-12 V/A and zero relative tolerance.

## Sample-by-sample comparison

| Signal | Mismatches | Maximum absolute error | RMS error | First mismatch |
|---|---:|---:|---:|---|
{rows}

For arrays with cell/switch dimensions, the first-mismatch index is reported in the simulator array's native axis order, followed by sample index.

## Validated

{validated}

## Not validated

{not_validated}

The validation is limited to agreement with the supplied dataset records using the stated parameters and conventions. It does not validate physical hardware behavior. The ideal switching model contains no dead time, losses, or device effects.
"""
    markdown_output.write_text(markdown, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=project_root / "data" / "raw" / "Seven_Level_Inverter_Project_extracted",
    )
    parser.add_argument("--chunksize", type=int, default=16)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=project_root / "results" / "simulator_validation.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=project_root / "docs" / "SIMULATOR_VALIDATION.md",
    )
    arguments = parser.parse_args()
    report = validate_dataset_reference(
        arguments.dataset_root,
        project_root=project_root,
        chunksize=arguments.chunksize,
    )
    write_validation_reports(
        report,
        json_path=arguments.json_output,
        markdown_path=arguments.markdown_output,
    )
    print(json.dumps({"status": report["status"], "signals": report["signals"]}, indent=2))


if __name__ == "__main__":
    main()
