"""Pure dashboard helpers for live presets and software playback state."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence


@dataclass(frozen=True)
class OperatingPreset:
    name: str
    modulation_index: float
    dc_voltage_per_cell_v: float
    load_resistance_ohm: float
    load_inductance_h: float
    phase_angle_degrees: float


LIVE_PRESETS = (
    OperatingPreset("Normal", 0.75, 100.0, 30.0, 0.03, 0.0),
    OperatingPreset("Lower modulation", 0.60, 100.0, 30.0, 0.03, 0.0),
    OperatingPreset("Higher modulation", 0.90, 100.0, 30.0, 0.03, 0.0),
    OperatingPreset("Lower resistance", 0.75, 100.0, 10.0, 0.03, 0.0),
    OperatingPreset("Lower inductance", 0.75, 100.0, 30.0, 0.01, 0.0),
)


def normalize_live_conditions(
    modulation_index: float,
    dc_voltage_per_cell_v: float,
    load_resistance_ohm: float,
    load_inductance_h: float,
    phase_angle_degrees: float,
) -> tuple[float, float, float, float, float]:
    """Validate and return live conditions exactly as physical user inputs."""
    values = (
        float(modulation_index),
        float(dc_voltage_per_cell_v),
        float(load_resistance_ohm),
        float(load_inductance_h),
        float(phase_angle_degrees),
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Live operating conditions must be finite.")
    if not 0.55 <= values[0] <= 0.95:
        raise ValueError("modulation_index must be in [0.55, 0.95].")
    if values[1] <= 0.0:
        raise ValueError("dc_voltage_per_cell_v must be positive.")
    if values[2] not in (10.0, 20.0, 30.0, 40.0):
        raise ValueError("Unsupported load resistance.")
    if values[3] not in (0.01, 0.02, 0.03, 0.05):
        raise ValueError("Unsupported load inductance.")
    if not 0.0 <= values[4] <= 360.0:
        raise ValueError("phase_angle_degrees must be in [0, 360].")
    return values


def advance_playback(
    index: int,
    status: str,
    frame_count: int,
    *,
    step: int = 1,
) -> int:
    """Advance playback by a configured number of computed samples."""
    if frame_count < 1:
        raise ValueError("frame_count must be positive.")
    if step < 1:
        raise ValueError("step must be positive.")
    if status != "running":
        return max(0, min(index, frame_count - 1))
    return (max(0, index) + step) % frame_count


def synchronized_playback_index(
    session_state,
    status: str,
    frame_count: int,
    *,
    manual_selection: bool = False,
    step: int = 1,
) -> int:
    """Keep the playback control and the cursor widget on the same sample."""
    index = int(
        session_state.get(
            "live_workspace_cursor_index",
            session_state.get("live_playback_index", 0),
        )
    )
    if status == "running" and not manual_selection:
        index = advance_playback(index, status, frame_count, step=step)
    else:
        index = max(0, min(index, frame_count - 1))
    session_state["live_workspace_cursor_index"] = index
    session_state["live_playback_index"] = index
    return index


def normalize_carrier_phase_offsets(offsets) -> tuple[float, float, float]:
    """Validate and wrap three carrier phases to the simulator's canonical range."""
    values = tuple(float(value) for value in offsets)
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError("Exactly three finite carrier phase offsets are required.")
    return tuple(value % 360.0 for value in values)


def optimizer_baseline_supported(
    carrier_phase_offsets_degrees,
    baseline_offsets_degrees: tuple[float, float, float],
) -> bool:
    """Return whether the live phase optimizer's fixed baseline matches the run."""
    selected = normalize_carrier_phase_offsets(carrier_phase_offsets_degrees)
    baseline = normalize_carrier_phase_offsets(baseline_offsets_degrees)
    return all(
        math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12)
        for value, expected in zip(selected, baseline)
    )


def cnn_guidance_operating_point_matches(
    operating_conditions: tuple[float, float, float, float, float],
    validated_operating_point: dict[str, float],
    carrier_phase_offsets_degrees,
    baseline_offsets_degrees: tuple[float, float, float],
) -> bool:
    """Limit CNN guidance to its measured operating point and baseline phases."""
    condition_keys = (
        "modulation_index",
        "dc_voltage_per_cell_v",
        "load_resistance_ohm",
        "load_inductance_h",
        "phase_angle_degrees",
    )
    selected = tuple(float(value) for value in operating_conditions)
    if len(selected) != len(condition_keys) or not all(
        math.isfinite(value) for value in selected
    ):
        raise ValueError("Exactly five finite guidance operating conditions are required.")
    reference = tuple(float(validated_operating_point[key]) for key in condition_keys)
    if not all(math.isfinite(value) for value in reference):
        raise ValueError("Validated guidance operating conditions must be finite.")
    return optimizer_baseline_supported(
        carrier_phase_offsets_degrees,
        baseline_offsets_degrees,
    ) and all(
        math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12)
        for value, expected in zip(selected, reference)
    )


def thd_surrogate_operating_point_supported(
    modulation_index: float,
    dc_voltage_per_cell_v: float,
    phase_angle_degrees: float,
    carrier_phase_offsets_degrees: Sequence[float],
    baseline_offsets_degrees: tuple[float, float, float],
) -> bool:
    """Apply the exact modulation/phase/DC scope covered by screening tests."""
    modulation, dc_voltage, phase = (
        float(modulation_index),
        float(dc_voltage_per_cell_v),
        float(phase_angle_degrees),
    )
    if not all(math.isfinite(value) for value in (modulation, dc_voltage, phase)):
        return False
    tested_modulations = tuple(0.55 + 0.05 * index for index in range(9))
    wrapped_phase = phase % 360.0
    tested_phases = tuple(45.0 * index for index in range(8))
    return (
        any(
            math.isclose(modulation, value, rel_tol=0.0, abs_tol=1e-12)
            for value in tested_modulations
        )
        and math.isclose(dc_voltage, 100.0, rel_tol=0.0, abs_tol=1e-12)
        and any(
            math.isclose(wrapped_phase, value, rel_tol=0.0, abs_tol=1e-12)
            for value in tested_phases
        )
        and optimizer_baseline_supported(
            carrier_phase_offsets_degrees,
            baseline_offsets_degrees,
        )
    )


def live_configuration_matches(
    result,
    operating_conditions: tuple[float, float, float, float, float],
    carrier_phase_offsets_degrees,
) -> bool:
    """Compare selected live controls with the inputs stored in a simulation result."""
    selected_conditions = tuple(float(value) for value in operating_conditions)
    if len(selected_conditions) != 5 or not all(
        math.isfinite(value) for value in selected_conditions
    ):
        raise ValueError("Exactly five finite live operating conditions are required.")
    result_conditions = (
        float(result.modulation_index),
        float(result.dc_voltage_per_cell_v),
        float(result.load_resistance_ohm),
        float(result.load_inductance_h),
        math.radians(float(result.phase_angle_degrees)),
    )
    selected_offsets = normalize_carrier_phase_offsets(carrier_phase_offsets_degrees)
    stored_offsets = normalize_carrier_phase_offsets(
        result.carrier_phase_offsets_degrees
    )
    return all(
        math.isclose(stored, selected, rel_tol=0.0, abs_tol=1e-12)
        for stored, selected in zip(result_conditions, selected_conditions)
    ) and all(
        math.isclose(stored, selected, rel_tol=0.0, abs_tol=1e-12)
        for stored, selected in zip(stored_offsets, selected_offsets)
    )


@dataclass(frozen=True)
class PlaybackSnapshot:
    sample_index: int
    time_ms: float
    output_voltage_v: float
    load_current_a: float
    cell_states: tuple[int, int, int]
    switching_gates: tuple[tuple[int, int, int, int], ...]


def playback_snapshot(simulation, index: int) -> PlaybackSnapshot:
    """Read one deterministic monitor snapshot from a real simulator result."""
    import numpy as np

    frame_count = int(np.asarray(simulation.output_voltage).size)
    if frame_count < 1:
        raise ValueError("simulation must contain at least one sample.")
    if not 0 <= index < frame_count:
        raise ValueError("index is outside the simulation frame range.")
    return PlaybackSnapshot(
        sample_index=index,
        time_ms=float(simulation.time_s[index] * 1000.0),
        output_voltage_v=float(simulation.output_voltage[index]),
        load_current_a=float(simulation.load_current[index]),
        cell_states=tuple(int(value) for value in simulation.cell_states[:, index]),
        switching_gates=tuple(
            tuple(int(value) for value in simulation.switching_gates[cell_index, :, index])
            for cell_index in range(simulation.switching_gates.shape[0])
        ),
    )


def gate_vector_for_state(state: int) -> tuple[int, int, int, int]:
    """Map an existing simulator cell state to the project gate truth table."""
    if state == 1:
        return (1, 0, 0, 1)
    if state == -1:
        return (0, 1, 1, 0)
    if state == 0:
        return (0, 1, 0, 1)
    raise ValueError("cell state must be -1, 0, or 1.")


def verify_cell_voltage_sum(simulation, index: int) -> tuple[float, float, bool]:
    """Compare sum-of-cells at a sample with the simulator output array."""
    import numpy as np

    if not 0 <= index < simulation.output_voltage.size:
        raise ValueError("cursor index is outside the simulator range.")
    dc_voltage = float(simulation.dc_voltage_per_cell_v)
    cell_voltages = tuple(dc_voltage * int(state) for state in simulation.cell_states[:, index])
    sum_voltage = float(sum(cell_voltages))
    output_voltage = float(simulation.output_voltage[index])
    return sum_voltage, output_voltage, bool(np.isclose(sum_voltage, output_voltage, rtol=0.0, atol=1e-9))
