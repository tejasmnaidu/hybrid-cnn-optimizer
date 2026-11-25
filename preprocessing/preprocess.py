"""Training-split-only normalization and waveform tensor preparation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_loader import DatasetTables, load_dataset
from .data_validation import _id_column


@dataclass(frozen=True)
class NormalizationStats:
    input_voltage_mean: float
    input_voltage_std: float
    load_current_mean: float
    load_current_std: float
    operating_mean: list[float]
    operating_std: list[float]
    target_voltage_mean: float
    target_voltage_std: float

    def normalize_inputs(
        self, voltage: np.ndarray, current: np.ndarray, conditions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            (voltage - self.input_voltage_mean) / self.input_voltage_std,
            (current - self.load_current_mean) / self.load_current_std,
            (conditions - self.operating_mean) / self.operating_std,
        )

    def normalize_target(self, target: np.ndarray) -> np.ndarray:
        return (target - self.target_voltage_mean) / self.target_voltage_std

    def denormalize_target(self, target: np.ndarray) -> np.ndarray:
        return target * self.target_voltage_std + self.target_voltage_mean


@dataclass(frozen=True)
class PreparedDataset:
    sample_ids: np.ndarray
    splits: np.ndarray
    input_voltage: np.ndarray
    load_current: np.ndarray
    operating_conditions: np.ndarray
    target_voltage: np.ndarray
    stats: NormalizationStats
    operating_feature_names: tuple[str, ...]

    def indices(self, split: str) -> np.ndarray:
        normalized = split.strip().lower()
        if normalized not in {"training", "validation", "testing"}:
            raise ValueError(f"Unknown split name: {split}")
        return np.flatnonzero(self.splits == normalized)

    def metadata(self) -> dict[str, Any]:
        return {
            "sample_count": int(len(self.sample_ids)),
            "waveform_points": int(self.target_voltage.shape[1]),
            "splits": {
                name: int(np.count_nonzero(self.splits == name))
                for name in ("training", "validation", "testing")
            },
            "normalization": asdict(self.stats),
            "operating_feature_names": list(self.operating_feature_names),
            "sample_ids_preserved": True,
            "statistics_fit_on": "training split only",
        }


def _ordered_matrix(frame: pd.DataFrame, label: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    id_col = _id_column(frame, label)
    id_values = pd.to_numeric(frame[id_col], errors="raise")
    if id_values.isna().any() or (id_values % 1 != 0).any() or id_values.duplicated().any():
        raise ValueError(f"{label} SampleIDs must be unique integers.")
    order = np.argsort(id_values.to_numpy())
    ordered = frame.iloc[order]
    ids = id_values.iloc[order].to_numpy(dtype=np.int64)
    columns = [column for column in ordered.columns if column != id_col]
    values = ordered[columns].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains missing or non-finite values.")
    return ids, values, columns


def _safe_std(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    result = np.std(values, axis=axis, dtype=np.float64)
    if np.any(~np.isfinite(result)) or np.any(result <= 0):
        raise ValueError("Cannot normalize a constant or invalid feature.")
    return result


def preprocess_dataset(
    source: DatasetTables | str | Path,
    *,
    metadata_path: str | Path | None = None,
) -> PreparedDataset:
    """Align the waveform task by SampleID and fit all statistics on training IDs."""
    tables = source if isinstance(source, DatasetTables) else load_dataset(source)
    ids: np.ndarray | None = None
    matrices: dict[str, np.ndarray] = {}
    columns_by_name: dict[str, list[str]] = {}
    for name in ("input_voltage", "load_current", "operating_conditions", "target_voltage"):
        frame_ids, values, columns = _ordered_matrix(tables[name], name)
        if ids is None:
            ids = frame_ids
        elif not np.array_equal(ids, frame_ids):
            raise ValueError(f"{name} SampleIDs do not align with the other tables.")
        matrices[name] = values
        columns_by_name[name] = columns

    split_frame = tables["sample_splits"]
    split_id_col = _id_column(split_frame, "sample_splits")
    split_col = next((c for c in split_frame if c.lower() == "splitlabel"), None)
    if split_col is None:
        raise ValueError("Sample_Splits.csv must contain splitLabel.")
    split_id_values = pd.to_numeric(split_frame[split_id_col], errors="raise")
    if (
        split_id_values.isna().any()
        or (split_id_values % 1 != 0).any()
        or split_id_values.duplicated().any()
    ):
        raise ValueError("Sample_Splits.csv SampleIDs must be unique integers.")
    split_order = np.argsort(split_id_values.to_numpy())
    split_ordered = split_frame.iloc[split_order]
    split_ids = split_id_values.iloc[split_order].to_numpy(dtype=np.int64)
    if ids is None or not np.array_equal(ids, split_ids):
        raise ValueError("Sample_Splits.csv SampleIDs do not align with waveform records.")
    splits = split_ordered[split_col].astype(str).str.strip().str.lower().to_numpy()
    if not set(splits).issubset({"training", "validation", "testing"}):
        raise ValueError("Sample_Splits.csv contains an unsupported label.")
    train_mask = splits == "training"
    if not train_mask.any() or not (splits == "validation").any() or not (splits == "testing").any():
        raise ValueError("Training, validation, and testing splits must all be non-empty.")

    voltage = matrices["input_voltage"]
    current = matrices["load_current"]
    conditions = matrices["operating_conditions"]
    target = matrices["target_voltage"]
    if voltage.shape[1] != 400 or current.shape[1] != 400 or target.shape[1] != 400:
        raise ValueError("Input voltage, load current, and target must each have 400 points.")
    if conditions.shape[1] != 4:
        raise ValueError("Expected exactly four operating-condition features.")
    stats = NormalizationStats(
        input_voltage_mean=float(np.mean(voltage[train_mask], dtype=np.float64)),
        input_voltage_std=float(_safe_std(voltage[train_mask])),
        load_current_mean=float(np.mean(current[train_mask], dtype=np.float64)),
        load_current_std=float(_safe_std(current[train_mask])),
        operating_mean=np.mean(conditions[train_mask], axis=0, dtype=np.float64).tolist(),
        operating_std=_safe_std(conditions[train_mask], axis=0).tolist(),
        target_voltage_mean=float(np.mean(target[train_mask], dtype=np.float64)),
        target_voltage_std=float(_safe_std(target[train_mask])),
    )
    prepared = PreparedDataset(
        sample_ids=ids,
        splits=splits,
        input_voltage=((voltage - stats.input_voltage_mean) / stats.input_voltage_std).astype(np.float32),
        load_current=((current - stats.load_current_mean) / stats.load_current_std).astype(np.float32),
        operating_conditions=(
            (conditions - np.asarray(stats.operating_mean))
            / np.asarray(stats.operating_std)
        ).astype(np.float32),
        target_voltage=(
            (target - stats.target_voltage_mean) / stats.target_voltage_std
        ).astype(np.float32),
        stats=stats,
        operating_feature_names=tuple(columns_by_name["operating_conditions"]),
    )
    if metadata_path is not None:
        output = Path(metadata_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(prepared.metadata(), indent=2), encoding="utf-8")
    return prepared
