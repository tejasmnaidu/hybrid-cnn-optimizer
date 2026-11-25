"""Explicit electrical and encoding checks without assumed gate/state mapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class SwitchingConstraintSpec:
    allowed_output_levels: tuple[float, ...] = (-300, -200, -100, 0, 100, 200, 300)
    maximum_transitions: int | None = None
    switch_count: int = 12
    cell_count: int = 3

    def __post_init__(self) -> None:
        if not self.allowed_output_levels:
            raise ValueError("allowed_output_levels cannot be empty.")
        if self.maximum_transitions is not None and self.maximum_transitions < 0:
            raise ValueError("maximum_transitions cannot be negative.")
        if self.switch_count < 1 or self.cell_count < 1:
            raise ValueError("switch_count and cell_count must be positive.")


def reshape_gate_data(
    flattened: np.ndarray,
    *,
    time_points: int,
    switch_count: int = 12,
    layout: str,
) -> np.ndarray:
    """Reshape flattened gates only after the caller declares storage ordering."""
    data = np.asarray(flattened)
    if data.size != time_points * switch_count:
        raise ValueError(
            f"Expected {time_points * switch_count} gate values, got {data.size}."
        )
    if not np.isin(data, (0, 1)).all():
        raise ValueError("Gate data must contain only binary 0/1 values.")
    if layout == "time_major":
        return data.reshape(time_points, switch_count)
    if layout == "switch_major":
        return data.reshape(switch_count, time_points).T
    raise ValueError("layout must be explicitly set to 'time_major' or 'switch_major'.")


def reshape_cell_states(
    flattened: np.ndarray,
    *,
    time_points: int,
    cell_count: int = 3,
    layout: str,
) -> np.ndarray:
    data = np.asarray(flattened)
    if data.size != time_points * cell_count:
        raise ValueError(f"Expected {time_points * cell_count} cell-state values, got {data.size}.")
    if not np.isin(data, (-1, 0, 1)).all():
        raise ValueError("Cell states must contain only -1, 0, or 1.")
    if layout == "time_major":
        return data.reshape(time_points, cell_count)
    if layout == "cell_major":
        return data.reshape(cell_count, time_points).T
    raise ValueError("layout must be explicitly set to 'time_major' or 'cell_major'.")


def validate_output_levels(
    output_voltage: np.ndarray,
    allowed_levels: tuple[float, ...] = (-300, -200, -100, 0, 100, 200, 300),
    *,
    atol: float = 1e-6,
) -> bool:
    values = np.asarray(output_voltage, dtype=np.float64)
    return bool(
        values.size
        and np.isfinite(values).all()
        and np.isclose(values[..., None], np.asarray(allowed_levels), atol=atol).any(axis=-1).all()
    )


def validate_gate_to_cell_mapping(
    gates: np.ndarray,
    cell_states: np.ndarray,
    mapping: Callable[[np.ndarray], np.ndarray],
) -> bool:
    """Compare using a caller-supplied, documented physical gate mapping."""
    gate_array = np.asarray(gates)
    state_array = np.asarray(cell_states)
    if gate_array.ndim != 2 or state_array.ndim != 2:
        raise ValueError("gates and cell_states must be 2D (time, channels) arrays.")
    if gate_array.shape[0] != state_array.shape[0]:
        raise ValueError("Gate and cell-state time dimensions are not aligned.")
    predicted = np.asarray(mapping(gate_array))
    if predicted.shape != state_array.shape:
        raise ValueError("The supplied gate mapping returned an incompatible cell-state shape.")
    return bool(np.array_equal(predicted, state_array))
