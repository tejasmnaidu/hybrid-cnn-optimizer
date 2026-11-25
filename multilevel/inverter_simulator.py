"""Ideal phase-shifted PWM and periodic RL simulation for a seven-level inverter.

The switching convention and gate truth table in this module implement the
algorithm supplied for this project. They are not verified against an original
MATLAB source file, which was not present in the project archive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DC_VOLTAGE_PER_CELL_V = 100.0
CELL_COUNT = 3
FUNDAMENTAL_FREQUENCY_HZ = 50.0
CARRIER_FREQUENCY_HZ = 1_000.0
SAMPLING_FREQUENCY_HZ = 20_000.0
SAMPLES_PER_CYCLE = 400
_SWITCH_COUNT_PER_CELL = 4
_VOLTAGE_LEVELS_V = np.arange(-CELL_COUNT, CELL_COUNT + 1, dtype=np.float64) * DC_VOLTAGE_PER_CELL_V


@dataclass(frozen=True)
class InverterSimulation:
    """One fundamental cycle of ideal inverter signals.

    Array axes:
      - ``carriers`` and ``cell_states``: (cell, time)
      - ``switching_gates``: (cell, switch S1..S4, time)
      - ``cell_voltages``: (cell, time)
      - ``time_s``, ``reference``, ``output_voltage``, ``load_current``: (time,)

    ``phase_angle_rad`` affects the sinusoidal reference. Gate signals use the
    supplied truth table and do not include dead time or non-ideal effects.
    """

    time_s: np.ndarray
    reference: np.ndarray
    carriers: np.ndarray
    cell_states: np.ndarray
    switching_gates: np.ndarray
    cell_voltages: np.ndarray
    output_voltage: np.ndarray
    load_current: np.ndarray
    periodic_initial_current_a: float
    dc_voltage_per_cell_v: float = DC_VOLTAGE_PER_CELL_V


def _finite_scalar(value: float, name: str) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite scalar.") from exc
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be a finite scalar.")
    return scalar


def _periodic_rl_current(
    voltage: np.ndarray,
    resistance_ohm: float,
    inductance_h: float,
    time_step_s: float,
) -> tuple[np.ndarray, float]:
    """Apply the specified exact-ZOH recurrence with a periodic initial state."""
    decay = resistance_ohm * time_step_s / inductance_h
    if not np.isfinite(decay) or decay <= 0:
        raise ValueError("R * dt / L must be finite and positive for the RL recurrence.")

    alpha = float(np.exp(-decay))
    beta = float(-np.expm1(-decay) / resistance_ohm)
    sample_count = voltage.size
    denominator = float(-np.expm1(-decay * sample_count))
    if denominator <= 0 or not np.isfinite(denominator):
        raise ValueError("The periodic RL initial condition is numerically undefined.")

    powers = np.arange(sample_count - 1, -1, -1, dtype=np.int64)
    weighted_voltage = np.power(alpha, powers) * voltage
    initial_current = beta * float(np.sum(weighted_voltage, dtype=np.float64)) / denominator

    current = np.empty(sample_count, dtype=np.float64)
    current[0] = initial_current
    for index in range(1, sample_count):
        current[index] = alpha * current[index - 1] + beta * voltage[index - 1]
    return current, initial_current


def simulate_inverter(
    modulation_index: float,
    load_resistance_ohm: float,
    load_inductance_h: float,
    phase_angle_rad: float,
    *,
    carrier_phase_offsets_degrees: tuple[float, ...] = (0.0, 60.0, 120.0),
    dc_voltage_per_cell_v: float = DC_VOLTAGE_PER_CELL_V,
) -> InverterSimulation:
    """Simulate one sampled fundamental cycle using the supplied conventions.

    Carrier offsets default to ``(0, 60, 120)`` degrees; callers may supply
    three explicit offsets for the phase-optimization comparison.
    The reference is ``m * sin(2*pi*50*t + phase_angle_rad)`` and each cell's
    state is ``(reference >= carrier) - (-reference >= carrier)``.

    Gate order for each cell is ``[S1, S2, S3, S4]``:
      - state +1: [1, 0, 0, 1]
      - state -1: [0, 1, 1, 0]
      - state  0: [0, 1, 0, 1]

    The returned gate array uses (cell, switch, time) axes, matching the
    cell-major/switch-major blocks in the supplied CSV representation.
    """
    modulation = _finite_scalar(modulation_index, "modulation_index")
    if not 0.0 <= modulation <= 1.0:
        raise ValueError("modulation_index must be in [0.0, 1.0].")
    resistance = _finite_scalar(load_resistance_ohm, "load_resistance_ohm")
    inductance = _finite_scalar(load_inductance_h, "load_inductance_h")
    phase = _finite_scalar(phase_angle_rad, "phase_angle_rad")
    dc_voltage = _finite_scalar(dc_voltage_per_cell_v, "dc_voltage_per_cell_v")
    if dc_voltage <= 0.0:
        raise ValueError("dc_voltage_per_cell_v must be positive.")
    if len(carrier_phase_offsets_degrees) != CELL_COUNT:
        raise ValueError(f"Exactly {CELL_COUNT} carrier phase offsets are required.")
    carrier_phase_offsets = tuple(
        _finite_scalar(offset, f"carrier_phase_offsets_degrees[{index}]") % 360.0
        for index, offset in enumerate(carrier_phase_offsets_degrees)
    )
    if resistance <= 0:
        raise ValueError("load_resistance_ohm must be positive.")
    if inductance <= 0:
        raise ValueError("load_inductance_h must be positive.")

    time_step = 1.0 / SAMPLING_FREQUENCY_HZ
    time_s = np.arange(SAMPLES_PER_CYCLE, dtype=np.float64) * time_step
    reference = modulation * np.sin(
        2.0 * np.pi * FUNDAMENTAL_FREQUENCY_HZ * time_s + phase
    )

    carriers = np.empty((CELL_COUNT, SAMPLES_PER_CYCLE), dtype=np.float64)
    cell_states = np.empty((CELL_COUNT, SAMPLES_PER_CYCLE), dtype=np.int8)
    for cell_index, phase_shift_degrees in enumerate(carrier_phase_offsets):
        carriers[cell_index] = 1.0 - 4.0 * np.abs(
            np.mod(
                CARRIER_FREQUENCY_HZ * time_s + phase_shift_degrees / 360.0,
                1.0,
            )
            - 0.5
        )
        upper_leg = reference >= carriers[cell_index]
        lower_leg = -reference >= carriers[cell_index]
        cell_states[cell_index] = upper_leg.astype(np.int8) - lower_leg.astype(np.int8)

    switching_gates = np.empty(
        (CELL_COUNT, _SWITCH_COUNT_PER_CELL, SAMPLES_PER_CYCLE), dtype=np.uint8
    )
    switching_gates[:, 0, :] = cell_states == 1
    switching_gates[:, 1, :] = cell_states != 1
    switching_gates[:, 2, :] = cell_states == -1
    switching_gates[:, 3, :] = cell_states != -1

    shoot_through = (
        (switching_gates[:, 0, :] & switching_gates[:, 1, :])
        | (switching_gates[:, 2, :] & switching_gates[:, 3, :])
    )
    if np.any(shoot_through):
        raise RuntimeError("The generated gate pattern contains shoot-through.")

    cell_voltages = dc_voltage * (
        switching_gates[:, 0, :].astype(np.int8)
        * switching_gates[:, 3, :].astype(np.int8)
        - switching_gates[:, 1, :].astype(np.int8)
        * switching_gates[:, 2, :].astype(np.int8)
    )
    expected_cell_voltages = dc_voltage * cell_states
    if not np.array_equal(cell_voltages, expected_cell_voltages):
        raise RuntimeError("Generated gates do not reconstruct their cell states.")

    output_voltage = np.sum(cell_voltages, axis=0)
    allowed_levels = (
        np.arange(-CELL_COUNT, CELL_COUNT + 1, dtype=np.float64) * dc_voltage
    )
    if not np.isin(output_voltage, allowed_levels).all():
        raise RuntimeError("Generated inverter voltage is outside the seven allowed levels.")

    load_current, initial_current = _periodic_rl_current(
        output_voltage,
        resistance,
        inductance,
        time_step,
    )
    return InverterSimulation(
        time_s=time_s,
        reference=reference,
        carriers=carriers,
        cell_states=cell_states,
        switching_gates=switching_gates,
        cell_voltages=cell_voltages,
        output_voltage=output_voltage,
        load_current=load_current,
        periodic_initial_current_a=initial_current,
        dc_voltage_per_cell_v=dc_voltage,
    )
