"""On-demand gate and cell-state access kept separate from waveform tensors."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from optimization.switching_constraints import reshape_cell_states, reshape_gate_data
from .data_loader import extract_dataset_zip


@dataclass(frozen=True)
class SwitchingRecord:
    sample_id: int
    flattened_gates: np.ndarray
    flattened_cell_states: np.ndarray

    def gates(self, *, layout: str) -> np.ndarray:
        return reshape_gate_data(self.flattened_gates, time_points=400, layout=layout)

    def cell_states(self, *, layout: str) -> np.ndarray:
        return reshape_cell_states(self.flattened_cell_states, time_points=400, layout=layout)


def _read_record(path: Path, sample_id: int, expected_width: int) -> np.ndarray:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        header = next(reader, None)
        if not header:
            raise ValueError(f"{path} has no CSV header.")
        id_matches = [i for i, name in enumerate(header) if name.lower() in {"sampleid", "sample_id"}]
        if len(id_matches) != 1 or len(header) - 1 != expected_width:
            raise ValueError(f"{path.name} has an unexpected SampleID or signal-column layout.")
        id_index = id_matches[0]
        for row in reader:
            if len(row) != len(header):
                raise ValueError(f"{path.name} contains a malformed CSV row.")
            if int(row[id_index]) == sample_id:
                try:
                    return np.asarray(
                        [float(value) for index, value in enumerate(row) if index != id_index],
                        dtype=np.float64,
                    )
                except ValueError as exc:
                    raise ValueError(f"{path.name} sample {sample_id} has invalid signal values.") from exc
    raise KeyError(f"SampleID {sample_id} was not found in {path.name}.")


def load_switching_record(source: str | Path, sample_id: int) -> SwitchingRecord:
    """Load one row from the high-dimensional auxiliary CSV files."""
    if sample_id < 1:
        raise ValueError("sample_id must be positive.")
    root = Path(source)
    if root.is_file() and root.suffix.lower() == ".zip":
        root = extract_dataset_zip(root, root.with_name(f"{root.stem}_extracted"))
    if not root.is_dir():
        raise FileNotFoundError(f"Expected a dataset directory or ZIP: {root}")
    gate_paths = list(root.rglob("Switching_Gates.csv"))
    state_paths = list(root.rglob("Cell_States.csv"))
    if len(gate_paths) != 1 or len(state_paths) != 1:
        raise FileNotFoundError("Switching_Gates.csv and Cell_States.csv are both required.")
    gate_path, state_path = gate_paths[0], state_paths[0]
    gates = _read_record(gate_path, sample_id, expected_width=4800)
    states = _read_record(state_path, sample_id, expected_width=1200)
    if not np.isin(gates, (0, 1)).all():
        raise ValueError(f"SampleID {sample_id} has non-binary switching-gate values.")
    if not np.isin(states, (-1, 0, 1)).all():
        raise ValueError(f"SampleID {sample_id} has invalid cell-state values.")
    return SwitchingRecord(sample_id, gates, states)
