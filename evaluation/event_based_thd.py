"""Event-based Fourier analysis for the ideal seven-level inverter.

The switching convention is the one implemented by
``multilevel.inverter_simulator``. Carrier crossings are solved on each linear
half-carrier segment, then each constant-voltage interval is integrated
analytically. The returned Fourier coefficients use peak-amplitude scaling:

    C_k = (2 / T) * integral(v(t) * exp(-j * 2*pi*k*t/T), t=0..T)

For a sinusoid ``A*sin(2*pi*t/T + phase)``, ``abs(C_1)`` is its peak
amplitude and ``angle(C_1) + pi/2`` is its phase, wrapped to [-pi, pi).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from evaluation.thd_analysis import calculate_thd
from multilevel.inverter_simulator import (
    CARRIER_FREQUENCY_HZ,
    CELL_COUNT,
    DC_VOLTAGE_PER_CELL_V,
    FUNDAMENTAL_FREQUENCY_HZ,
    SAMPLING_FREQUENCY_HZ,
    SAMPLES_PER_CYCLE,
)


_CYCLE_S = 1.0 / FUNDAMENTAL_FREQUENCY_HZ
_EVENT_MERGE_TOLERANCE_S = 64.0 * np.finfo(np.float64).eps * _CYCLE_S
_DEFAULT_PHASE_OFFSETS_DEGREES = (0.0, 60.0, 120.0)


@dataclass(frozen=True)
class EventBasedHarmonicResult:
    """Analytic harmonic metrics and the intervals used to calculate them."""

    interval_start_s: np.ndarray
    interval_end_s: np.ndarray
    interval_voltage_v: np.ndarray
    transition_times_s: np.ndarray
    harmonic_orders: np.ndarray
    fourier_coefficients: np.ndarray
    harmonic_peak_amplitudes_v: np.ndarray
    harmonic_phase_rad: np.ndarray
    fundamental_peak_amplitude_v: float
    fundamental_phase_rad: float
    mean_voltage_v: float
    mean_square_voltage_v2: float
    ac_rms_voltage_v: float
    fundamental_rms_voltage_v: float
    total_thd_percent: float
    target_fundamental_peak_amplitude_v: float
    target_fundamental_phase_rad: float
    fundamental_phasor_error_percent: float
    thd_percent: float
    sampled_thd_percent: float


def _finite_scalar(value: float, name: str) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite scalar.") from exc
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be a finite scalar.")
    return scalar


def _nonnegative_energy(value: float, scale: float, label: str) -> float:
    if not np.isfinite(value) or not np.isfinite(scale):
        raise ValueError(f"{label} is not finite.")
    tolerance = 64.0 * np.finfo(np.float64).eps * max(scale, np.finfo(np.float64).tiny)
    if value < -tolerance:
        raise ValueError(
            f"{label} is materially negative ({value:.12g} V^2); "
            "the Fourier and interval energy calculations are inconsistent."
        )
    return max(value, 0.0)


def _carrier_value(time_s: float, phase_offset_degrees: float) -> float:
    carrier_phase = math.fmod(
        CARRIER_FREQUENCY_HZ * time_s + phase_offset_degrees / 360.0,
        1.0,
    )
    if carrier_phase < 0:
        carrier_phase += 1.0
    return 1.0 - 4.0 * abs(carrier_phase - 0.5)


def _reference_value(time_s: float, modulation_index: float, phase_angle_rad: float) -> float:
    return modulation_index * math.sin(
        2.0 * math.pi * FUNDAMENTAL_FREQUENCY_HZ * time_s + phase_angle_rad
    )


def _comparison_value(
    time_s: float,
    modulation_index: float,
    phase_angle_rad: float,
    phase_offset_degrees: float,
    polarity: float,
) -> float:
    reference = _reference_value(time_s, modulation_index, phase_angle_rad)
    return polarity * reference - _carrier_value(time_s, phase_offset_degrees)


def _bisect_crossing(
    left: float,
    right: float,
    left_value: float,
    right_value: float,
    *,
    modulation_index: float,
    phase_angle_rad: float,
    phase_offset_degrees: float,
    polarity: float,
) -> float:
    """Solve one bracketed crossing to the representable time precision."""
    if left_value == 0.0:
        return left
    if right_value == 0.0:
        return right
    if (left_value < 0.0) == (right_value < 0.0):
        raise ValueError("A crossing must be bracketed by endpoints of opposite sign.")

    lo, hi = left, right
    lo_value = left_value
    for _ in range(96):
        midpoint = lo + (hi - lo) * 0.5
        if midpoint == lo or midpoint == hi:
            break
        mid_value = _comparison_value(
            midpoint,
            modulation_index,
            phase_angle_rad,
            phase_offset_degrees,
            polarity,
        )
        if mid_value == 0.0:
            return midpoint
        if (mid_value < 0.0) == (lo_value < 0.0):
            lo, lo_value = midpoint, mid_value
        else:
            hi = midpoint
    return lo + (hi - lo) * 0.5


def _carrier_segment_knots(phase_offset_degrees: float) -> np.ndarray:
    """Return all triangle extrema in [0, T], including both cycle endpoints."""
    phase_cycles = phase_offset_degrees / 360.0
    first_half_cycle = math.floor(2.0 * phase_cycles) - 1
    last_half_cycle = math.ceil(
        2.0 * (CARRIER_FREQUENCY_HZ * _CYCLE_S + phase_cycles)
    ) + 1
    half_cycles = np.arange(first_half_cycle, last_half_cycle + 1, dtype=np.float64)
    knots = (half_cycles * 0.5 - phase_cycles) / CARRIER_FREQUENCY_HZ
    interior = knots[(knots > 0.0) & (knots < _CYCLE_S)]
    return np.concatenate(([0.0], interior, [_CYCLE_S]))


def _find_transition_times(
    modulation_index: float,
    phase_angle_rad: float,
    phase_offsets_degrees: tuple[float, ...],
) -> np.ndarray:
    roots: list[float] = []
    for phase_offset in phase_offsets_degrees:
        knots = _carrier_segment_knots(phase_offset)
        for left, right in zip(knots[:-1], knots[1:]):
            for polarity in (1.0, -1.0):
                left_value = _comparison_value(
                    float(left),
                    modulation_index,
                    phase_angle_rad,
                    phase_offset,
                    polarity,
                )
                right_value = _comparison_value(
                    float(right),
                    modulation_index,
                    phase_angle_rad,
                    phase_offset,
                    polarity,
                )
                if left_value == 0.0:
                    root = float(left)
                elif right_value == 0.0:
                    root = float(right)
                elif (left_value < 0.0) != (right_value < 0.0):
                    root = _bisect_crossing(
                        float(left),
                        float(right),
                        left_value,
                        right_value,
                        modulation_index=modulation_index,
                        phase_angle_rad=phase_angle_rad,
                        phase_offset_degrees=phase_offset,
                        polarity=polarity,
                    )
                else:
                    continue
                if 0.0 < root < _CYCLE_S:
                    roots.append(root)

    boundaries = np.sort(np.asarray([0.0, _CYCLE_S, *roots], dtype=np.float64))
    unique = [float(boundaries[0])]
    for boundary in boundaries[1:]:
        value = float(boundary)
        if value - unique[-1] <= _EVENT_MERGE_TOLERANCE_S:
            if value == _CYCLE_S:
                unique[-1] = _CYCLE_S
            continue
        unique.append(value)
    if unique[-1] != _CYCLE_S:
        unique.append(_CYCLE_S)
    return np.asarray(unique, dtype=np.float64)


def _cell_states_at(
    time_s: float,
    modulation_index: float,
    phase_angle_rad: float,
    phase_offsets_degrees: tuple[float, ...],
) -> np.ndarray:
    reference = _reference_value(time_s, modulation_index, phase_angle_rad)
    states = np.empty(CELL_COUNT, dtype=np.int8)
    for cell_index, phase_offset in enumerate(phase_offsets_degrees):
        carrier = _carrier_value(time_s, phase_offset)
        upper = reference >= carrier
        lower = -reference >= carrier
        states[cell_index] = int(upper) - int(lower)
    return states


def _sampled_output(
    modulation_index: float,
    phase_angle_rad: float,
    phase_offsets_degrees: tuple[float, ...],
    sampling_frequency_hz: float,
    dc_voltage_per_cell_v: float,
) -> np.ndarray:
    samples_per_cycle = int(round(sampling_frequency_hz / FUNDAMENTAL_FREQUENCY_HZ))
    times = np.arange(samples_per_cycle, dtype=np.float64) / sampling_frequency_hz
    reference = modulation_index * np.sin(
        2.0 * np.pi * FUNDAMENTAL_FREQUENCY_HZ * times + phase_angle_rad
    )
    output = np.zeros(samples_per_cycle, dtype=np.float64)
    for phase_offset in phase_offsets_degrees:
        carrier_phase = np.mod(
            CARRIER_FREQUENCY_HZ * times + phase_offset / 360.0,
            1.0,
        )
        carrier = 1.0 - 4.0 * np.abs(carrier_phase - 0.5)
        upper = reference >= carrier
        lower = -reference >= carrier
        output += dc_voltage_per_cell_v * (
            upper.astype(np.int8) - lower.astype(np.int8)
        )
    return output


def evaluate_event_based_thd(
    modulation_index: float,
    phase_angle_rad: float,
    *,
    carrier_phase_offsets_degrees: tuple[float, ...] = _DEFAULT_PHASE_OFFSETS_DEGREES,
    max_harmonic: int = 100,
    sampled_frequency_hz: float = SAMPLING_FREQUENCY_HZ,
    dc_voltage_per_cell_v: float = DC_VOLTAGE_PER_CELL_V,
) -> EventBasedHarmonicResult:
    """Evaluate one ideal inverter cycle with analytic interval integration.

    The event solver is guaranteed to bracket every crossing for the supported
    ``0 < modulation_index <= 1`` range: on each half-carrier segment the
    carrier slope magnitude is 4*fc = 4000 per second, while the reference
    derivative magnitude is at most 2*pi*50. Events closer than
    ``64*epsilon*T`` are coalesced as numerically indistinguishable.

    ``sampled_thd_percent`` is a secondary metric from the existing 20 kHz
    sampled THD implementation (or the explicitly supplied integer-cycle
    sample rate). It is not used in the analytic ``thd_percent`` result.
    """
    modulation = _finite_scalar(modulation_index, "modulation_index")
    phase = _finite_scalar(phase_angle_rad, "phase_angle_rad")
    dc_voltage = _finite_scalar(dc_voltage_per_cell_v, "dc_voltage_per_cell_v")
    if dc_voltage <= 0.0:
        raise ValueError("dc_voltage_per_cell_v must be positive.")
    if not 0.0 < modulation <= 1.0:
        raise ValueError("modulation_index must be in the range (0, 1].")
    if len(carrier_phase_offsets_degrees) != CELL_COUNT:
        raise ValueError(f"Exactly {CELL_COUNT} carrier phase offsets are required.")
    phase_offsets = tuple(
        _finite_scalar(offset, f"carrier_phase_offsets_degrees[{index}]") % 360.0
        for index, offset in enumerate(carrier_phase_offsets_degrees)
    )
    if not isinstance(max_harmonic, int) or max_harmonic < 2:
        raise ValueError("max_harmonic must be an integer of at least 2.")
    sampled_frequency = _finite_scalar(sampled_frequency_hz, "sampled_frequency_hz")
    sample_ratio = sampled_frequency / FUNDAMENTAL_FREQUENCY_HZ
    if sampled_frequency <= 0 or not np.isclose(sample_ratio, round(sample_ratio), rtol=0.0, atol=1e-10):
        raise ValueError("sampled_frequency_hz must contain an integer number of samples per cycle.")

    transition_times = _find_transition_times(modulation, phase, phase_offsets)
    interval_start = transition_times[:-1]
    interval_end = transition_times[1:]
    interval_midpoint = interval_start + (interval_end - interval_start) * 0.5
    interval_voltage = np.empty(interval_midpoint.size, dtype=np.float64)
    for index, midpoint in enumerate(interval_midpoint):
        states = _cell_states_at(
            float(midpoint),
            modulation,
            phase,
            phase_offsets,
        )
        interval_voltage[index] = dc_voltage * float(np.sum(states))

    if (
        interval_start[0] != 0.0
        or interval_end[-1] != _CYCLE_S
        or not np.array_equal(interval_end[:-1], interval_start[1:])
        or np.any(interval_end <= interval_start)
    ):
        raise RuntimeError("Event intervals do not form a gap-free fundamental cycle.")
    allowed_levels = dc_voltage * np.arange(-CELL_COUNT, CELL_COUNT + 1)
    if not np.isin(interval_voltage, allowed_levels).all():
        raise RuntimeError("Event-derived voltage is outside the seven allowed levels.")

    orders = np.arange(1, max_harmonic + 1, dtype=np.int64)
    coefficients = np.empty(orders.size, dtype=np.complex128)
    for index, order in enumerate(orders):
        angular_frequency = 2.0 * math.pi * int(order) / _CYCLE_S
        integral_kernel = (
            np.exp(-1j * angular_frequency * interval_start)
            - np.exp(-1j * angular_frequency * interval_end)
        ) / (1j * angular_frequency)
        coefficients[index] = (
            (2.0 / _CYCLE_S)
            * np.sum(interval_voltage * integral_kernel, dtype=np.complex128)
        )

    amplitudes = np.abs(coefficients)
    phases = np.angle(coefficients) + math.pi / 2.0
    phases = np.arctan2(np.sin(phases), np.cos(phases))
    fundamental_amplitude = float(amplitudes[0])
    interval_durations = interval_end - interval_start
    mean_voltage = float(
        math.fsum(
            float(voltage) * float(duration)
            for voltage, duration in zip(interval_voltage, interval_durations)
        )
        / _CYCLE_S
    )
    mean_square_voltage = float(
        math.fsum(
            float(voltage) ** 2 * float(duration)
            for voltage, duration in zip(interval_voltage, interval_durations)
        )
        / _CYCLE_S
    )
    ac_mean_square_voltage = _nonnegative_energy(
        mean_square_voltage - mean_voltage**2,
        mean_square_voltage + mean_voltage**2,
        "AC mean-square voltage",
    )
    ac_rms_voltage = math.sqrt(ac_mean_square_voltage)
    fundamental_rms_voltage = fundamental_amplitude / math.sqrt(2.0)
    residual_harmonic_mean_square = _nonnegative_energy(
        ac_mean_square_voltage - fundamental_rms_voltage**2,
        ac_mean_square_voltage + fundamental_rms_voltage**2,
        "Residual harmonic mean-square voltage",
    )
    target_amplitude = float(CELL_COUNT * dc_voltage * modulation)
    target_phase = float(math.atan2(math.sin(phase), math.cos(phase)))
    target_coefficient = -1j * target_amplitude * np.exp(1j * phase)
    tracking_error_percent = float(
        abs(coefficients[0] - target_coefficient) / abs(target_coefficient) * 100.0
    )
    if fundamental_amplitude <= np.finfo(np.float64).eps:
        raise ValueError("THD is undefined when the fundamental amplitude is zero.")
    total_thd_percent = float(
        math.sqrt(residual_harmonic_mean_square) / fundamental_rms_voltage * 100.0
    )
    thd_percent = float(
        np.sqrt(np.sum(np.square(amplitudes[1:]))) / fundamental_amplitude * 100.0
    )

    sampled_waveform = _sampled_output(
        modulation,
        phase,
        phase_offsets,
        sampled_frequency,
        dc_voltage,
    )
    sampled_thd = float(
        calculate_thd(
            sampled_waveform,
            sampling_frequency_hz=sampled_frequency,
            fundamental_frequency_hz=FUNDAMENTAL_FREQUENCY_HZ,
            max_harmonic=max_harmonic,
        )
    )

    return EventBasedHarmonicResult(
        interval_start_s=interval_start,
        interval_end_s=interval_end,
        interval_voltage_v=interval_voltage,
        transition_times_s=transition_times,
        harmonic_orders=orders,
        fourier_coefficients=coefficients,
        harmonic_peak_amplitudes_v=amplitudes,
        harmonic_phase_rad=phases,
        fundamental_peak_amplitude_v=fundamental_amplitude,
        fundamental_phase_rad=float(phases[0]),
        mean_voltage_v=mean_voltage,
        mean_square_voltage_v2=mean_square_voltage,
        ac_rms_voltage_v=ac_rms_voltage,
        fundamental_rms_voltage_v=fundamental_rms_voltage,
        total_thd_percent=total_thd_percent,
        target_fundamental_peak_amplitude_v=target_amplitude,
        target_fundamental_phase_rad=target_phase,
        fundamental_phasor_error_percent=tracking_error_percent,
        thd_percent=thd_percent,
        sampled_thd_percent=sampled_thd,
    )


__all__ = ["EventBasedHarmonicResult", "evaluate_event_based_thd"]
