"""Schema, value, ID alignment, split, and auxiliary signal validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_loader import (
    OPTIONAL_AUXILIARY_FILES,
    TABLE_FILES,
    DatasetTables,
    load_dataset,
)

WAVEFORM_COLUMNS = {
    "input_voltage": ("inputVoltage", 400),
    "target_voltage": ("targetVoltage", 400),
    "load_current": ("inputCurrent", 400),
}
AUXILIARY_LAYOUTS = {
    "switching_gates": (4800, {0.0, 1.0}),
    "cell_states": (1200, {-1.0, 0.0, 1.0}),
}


class DataValidationError(ValueError):
    """Raised when one or more dataset validation checks fail."""


@dataclass
class ValidationReport:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def _id_column(frame: pd.DataFrame, table_name: str) -> str:
    matches = [column for column in frame if column.lower() in {"sampleid", "sample_id"}]
    if len(matches) != 1:
        raise ValueError(f"{table_name} must contain exactly one SampleID column.")
    return matches[0]


def _numeric_matrix(frame: pd.DataFrame, columns: list[str], table_name: str) -> np.ndarray:
    try:
        values = frame[columns].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{table_name} contains non-numeric signal values.") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"{table_name} contains missing or non-finite signal values.")
    return values


def _inspect_auxiliary(path: Path, key: str, expected_samples: set[int]) -> dict[str, Any]:
    expected_points, allowed = AUXILIARY_LAYOUTS[key]
    seen: set[int] = set()
    value_counts: dict[str, int] = {}
    row_count = 0
    signal_dtype_counts: dict[str, int] = {}
    id_dtype: str | None = None
    try:
        header = pd.read_csv(path, nrows=0).columns.tolist()
        id_col = _id_column(pd.DataFrame(columns=header), path.name)
        signal_columns = [column for column in header if column != id_col]
        if len(signal_columns) != expected_points:
            raise ValueError(
                f"{path.name} has {len(signal_columns)} signal columns; expected {expected_points}."
            )
        for chunk in pd.read_csv(path, chunksize=128):
            if row_count == 0:
                id_dtype = str(chunk[id_col].dtype)
            ids = pd.to_numeric(chunk[id_col], errors="coerce")
            if ids.isna().any() or (ids % 1 != 0).any():
                raise ValueError(f"{path.name} contains missing or non-integer SampleID values.")
            int_ids = ids.astype(np.int64)
            if int_ids.duplicated().any() or seen.intersection(int_ids.tolist()):
                raise ValueError(f"{path.name} contains duplicate SampleID values.")
            seen.update(int_ids.tolist())
            values = _numeric_matrix(chunk, signal_columns, path.name)
            dtypes, counts = np.unique(
                [str(chunk[c].dtype) for c in signal_columns], return_counts=True
            )
            for dtype, count in zip(dtypes, counts):
                signal_dtype_counts[str(dtype)] = max(
                    signal_dtype_counts.get(str(dtype), 0), int(count)
                )
            unexpected = set(np.unique(values)) - allowed
            if unexpected:
                raise ValueError(f"{path.name} contains invalid values: {sorted(unexpected)[:10]}.")
            vals, counts = np.unique(values, return_counts=True)
            for value, count in zip(vals, counts):
                label = str(int(value)) if value.is_integer() else str(value)
                value_counts[label] = value_counts.get(label, 0) + int(count)
            row_count += len(chunk)
    except (OSError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not validate {path}: {exc}") from exc
    if seen != expected_samples:
        raise ValueError(f"{path.name} SampleIDs do not match the main dataset.")
    return {
        "rows": row_count,
        "signal_columns": expected_points,
        "column_count": expected_points + 1,
        "id_column": id_col,
        "id_dtype": id_dtype,
        "signal_column_names_first": signal_columns[:12],
        "signal_column_names_last": signal_columns[-4:],
        "signal_dtype_counts": signal_dtype_counts,
        "values": value_counts,
        "sample_ids_aligned": True,
        "sample_id_min": min(seen) if seen else None,
        "sample_id_max": max(seen) if seen else None,
    }


def validate_dataset(
    source: DatasetTables | str | Path,
    *,
    report_path: str | Path | None = None,
    check_auxiliary: bool = True,
    raise_on_error: bool = True,
) -> ValidationReport:
    """Validate actual tables and optionally save a machine-readable report."""
    errors: list[str] = []
    warnings: list[str] = []
    summary: dict[str, Any] = {}
    tables = source if isinstance(source, DatasetTables) else load_dataset(source)

    try:
        frames = dict(tables.tables)
        required = set(TABLE_FILES) - {"configuration"}
        missing = required - set(frames)
        if missing:
            raise ValueError(f"Loaded tables are missing: {sorted(missing)}")
        config_frame = frames["configuration"]
        config_cols = {column.lower(): column for column in config_frame.columns}
        if not {"parameter", "value"}.issubset(config_cols):
            raise ValueError("Configuration.csv must contain parameter and value columns.")
        config = {
            str(row[config_cols["parameter"]]).strip().lower(): str(row[config_cols["value"]]).strip()
            for _, row in config_frame.iterrows()
        }
        summary["configuration"] = config
        required_config = {
            "total waveforms",
            "samples per cycle",
            "fundamental frequency",
            "sampling frequency",
            "number of voltage levels",
            "training samples",
            "validation samples",
            "testing samples",
            "voltage levels",
        }
        missing_config = required_config - set(config)
        if missing_config:
            raise ValueError(f"Configuration.csv is missing parameters: {sorted(missing_config)}")
        sample_count = int(float(config["total waveforms"]))
        points = int(float(config["samples per cycle"]))
        fundamental = float(config["fundamental frequency"])
        sampling = float(config["sampling frequency"])
        if sample_count < 1 or points < 1 or fundamental <= 0 or sampling <= 0:
            raise ValueError("Configured sample counts and frequencies must be positive.")
        if not np.isclose(sampling / fundamental, points):
            raise ValueError(
                f"Configuration mismatch: Fs/fundamental={sampling / fundamental:g}, "
                f"but Samples Per Cycle={points}."
            )
        if points != 400:
            raise ValueError(f"Dataset declares {points} waveform points; this project expects 400.")
        configured_level_count = int(float(config["number of voltage levels"]))
        configured_levels = np.fromstring(config["voltage levels"].strip().strip("[]"), sep=" ")
        if (
            configured_level_count != 7
            or configured_levels.size != configured_level_count
            or not np.isfinite(configured_levels).all()
            or np.unique(configured_levels).size != configured_level_count
        ):
            raise ValueError("Configuration does not declare a seven-level inverter.")

        input_frame = frames["input_voltage"]
        input_id_col = _id_column(input_frame, "input_voltage")
        input_ids = pd.to_numeric(input_frame[input_id_col], errors="coerce")
        if input_ids.isna().any() or (input_ids % 1 != 0).any() or input_ids.duplicated().any():
            raise ValueError("input_voltage contains missing, non-integer, or duplicate SampleID values.")
        reference_ids = np.sort(input_ids.to_numpy(dtype=np.int64))
        numeric_summary: dict[str, Any] = {}
        validation_arrays: dict[str, np.ndarray] = {}
        for name, frame in frames.items():
            if name == "configuration":
                summary.setdefault("table_schemas", {})[name] = {
                    "rows": len(frame),
                    "columns": len(frame.columns),
                    "column_names": [str(c) for c in frame.columns],
                    "dtypes": {str(c): str(t) for c, t in frame.dtypes.items()},
                }
                continue
            id_col = _id_column(frame, name)
            ids = pd.to_numeric(frame[id_col], errors="coerce")
            if ids.isna().any() or (ids % 1 != 0).any():
                raise ValueError(f"{name} contains missing or non-integer SampleID values.")
            ids_array = ids.to_numpy(dtype=np.int64)
            if pd.Index(ids_array).has_duplicates:
                raise ValueError(f"{name} contains duplicate SampleID values.")
            if len(frame) != sample_count:
                raise ValueError(f"{name} has {len(frame)} rows; expected {sample_count}.")
            if name != "input_voltage" and not np.array_equal(np.sort(ids_array), reference_ids):
                raise ValueError(f"{name} SampleIDs do not match input_voltage.")
            summary.setdefault("table_schemas", {})[name] = {
                "rows": len(frame),
                "columns": len(frame.columns),
                "column_names": [str(c) for c in frame.columns],
                "dtypes": {str(c): str(t) for c, t in frame.dtypes.items()},
                "sample_id_min": int(ids_array.min()) if len(ids_array) else None,
                "sample_id_max": int(ids_array.max()) if len(ids_array) else None,
                "sample_ids_unique": True,
            }

        expected_ids = np.arange(1, sample_count + 1, dtype=np.int64)
        if not np.array_equal(reference_ids, expected_ids):
            raise ValueError("SampleIDs must be exactly 1..configured sample count.")
        by_split = frames["sample_splits"]
        split_col = next((c for c in by_split if c.lower() == "splitlabel"), None)
        if split_col is None:
            raise ValueError("Sample_Splits.csv must contain splitLabel.")
        split_values = by_split[split_col].astype(str).str.strip().str.lower()
        allowed_splits = {"training", "validation", "testing"}
        if not set(split_values).issubset(allowed_splits):
            raise ValueError(f"Unexpected split labels: {sorted(set(split_values) - allowed_splits)}")
        split_counts = split_values.value_counts().to_dict()
        expected_counts = {
            "training": int(float(config["training samples"])),
            "validation": int(float(config["validation samples"])),
            "testing": int(float(config["testing samples"])),
        }
        if split_counts != expected_counts:
            raise ValueError(f"Split counts {split_counts} do not match configuration {expected_counts}.")
        summary["split_counts"] = split_counts

        allowed_levels = set(configured_levels.tolist())
        for name, (_, width) in WAVEFORM_COLUMNS.items():
            frame = frames[name]
            id_col = _id_column(frame, name)
            frame_ids = pd.to_numeric(frame[id_col], errors="raise").to_numpy(dtype=np.int64)
            columns = [c for c in frame if c != id_col]
            if len(columns) != width:
                raise ValueError(f"{TABLE_FILES[name]} has {len(columns)} signal columns; expected {width}.")
            values = _numeric_matrix(frame, columns, name)
            validation_arrays[name] = values[np.argsort(frame_ids)]
            numeric_summary[name] = {
                "shape": list(values.shape),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            }
        input_values = validation_arrays["input_voltage"]
        levels = np.unique(input_values)
        if not set(levels.tolist()).issubset(allowed_levels) or len(levels) != 7:
            raise ValueError(f"Input voltage must contain exactly the configured seven levels; got {levels.tolist()}.")
        numeric_summary["input_voltage"]["levels"] = levels.tolist()
        summary["signals"] = numeric_summary

        operating = frames["operating_conditions"]
        operating_columns = [
            c for c in operating.columns if c != _id_column(operating, "operating_conditions")
        ]
        if len(operating_columns) != 4:
            raise ValueError(f"Operating conditions has {len(operating_columns)} features; expected 4.")
        operating_values = _numeric_matrix(operating, operating_columns, "operating_conditions")
        operating_order = np.argsort(
            pd.to_numeric(operating[_id_column(operating, "operating_conditions")], errors="raise")
            .to_numpy(dtype=np.int64)
        )
        validation_arrays["operating_conditions"] = operating_values[operating_order]
        numeric_summary["operating_conditions"] = {
            "shape": [len(operating), len(operating_columns)],
            "feature_names": operating_columns,
        }
        thd = frames["thd_results"]
        thd_cols = [c for c in thd if c != _id_column(thd, "thd_results")]
        if len(thd_cols) != 3:
            raise ValueError(f"THD_Results.csv has {len(thd_cols)} measurements; expected 3.")
        thd_values = _numeric_matrix(thd, thd_cols, "thd_results")
        if (thd_values < 0).any():
            raise ValueError("THD measurements cannot be negative.")
        summary["thd_columns"] = thd_cols
        summary["thd_measurements"] = {
            column: {
                "minimum": float(thd_values[:, index].min()),
                "maximum": float(thd_values[:, index].max()),
                "mean": float(thd_values[:, index].mean()),
            }
            for index, column in enumerate(thd_cols)
        }

        split_ids = pd.to_numeric(by_split[_id_column(by_split, "sample_splits")], errors="raise")
        split_by_id = dict(
            zip(
                split_ids.astype(np.int64),
                by_split[split_col].astype(str).str.strip().str.lower(),
            )
        )
        duplicate_splits: dict[bytes, set[str]] = {}
        duplicate_ids: dict[bytes, int] = {}
        for index, sample_id in enumerate(expected_ids):
            digest = hashlib.blake2b(digest_size=16)
            for key in ("input_voltage", "load_current", "operating_conditions"):
                digest.update(validation_arrays[key][index].astype("<f4", copy=False).tobytes())
            key = digest.digest()
            duplicate_splits.setdefault(key, set()).add(split_by_id[int(sample_id)])
            duplicate_ids[key] = duplicate_ids.get(key, 0) + 1
        cross_split_duplicates = sum(
            duplicate_ids[key] for key, labels in duplicate_splits.items() if len(labels) > 1
        )
        summary["data_leakage_check"] = {
            "checked": True,
            "exact_duplicate_model_inputs_across_splits": cross_split_duplicates,
            "duplicate_key": "input voltage, load current, and operating conditions",
        }
        if cross_split_duplicates:
            warnings.append(
                f"{cross_split_duplicates} records have exact duplicate model inputs across distinct splits."
            )
        summary["data_types"] = {
            name: {str(c): str(t) for c, t in frame.dtypes.items()}
            for name, frame in frames.items()
        }

        aux_summary: dict[str, Any] = {}
        if check_auxiliary:
            for key, filename in OPTIONAL_AUXILIARY_FILES.items():
                candidates = list(tables.root.rglob(filename))
                if not candidates:
                    warnings.append(f"{filename} was not found; auxiliary validation skipped.")
                    continue
                if len(candidates) > 1:
                    raise ValueError(f"Multiple {filename} files found under {tables.root}.")
                aux_summary[key] = _inspect_auxiliary(candidates[0], key, set(expected_ids.tolist()))
            warnings.append(
                "Gate-to-cell-state physical mapping is not established by the supplied schema; "
                "the 4,800 gate and 1,200 cell-state points are validated separately."
            )
        summary["auxiliary"] = aux_summary
        summary["sample_id_range"] = [int(expected_ids[0]), int(expected_ids[-1])]
        summary["sampling_frequency_hz"] = sampling
        summary["fundamental_frequency_hz"] = fundamental
        summary["samples_per_cycle"] = points
    except (ValueError, KeyError, TypeError, IndexError, OSError, OverflowError) as exc:
        errors.append(str(exc))

    report = ValidationReport(valid=not errors, errors=errors, warnings=warnings, summary=summary)
    if report_path is not None:
        report.save(report_path)
    if errors and raise_on_error:
        raise DataValidationError("; ".join(errors))
    return report
