"""Read-only comparison of a measured voltage CSV with a simulated waveform."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import io
import math

import numpy as np

from evaluation.harmonic_analysis import HarmonicSpectrum
from evaluation.thd_analysis import calculate_thd


MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_SAMPLE_COUNT = 1_000_000
_REQUIRED_COLUMNS = ("time_s", "voltage_v")


@dataclass(frozen=True)
class MeasuredWaveformComparison:
    sample_count: int
    cycle_count: int
    sampling_frequency_hz: float
    fundamental_frequency_hz: float
    max_harmonic: int
    measured_mean_voltage_v: float
    simulated_mean_voltage_v: float
    measured_fundamental_amplitude_v: float
    simulated_fundamental_amplitude_v: float
    fundamental_amplitude_change_percent: float
    measured_thd_percent: float
    simulated_thd_percent: float
    thd_difference_percentage_points: float
    measured_spectrum: HarmonicSpectrum
    simulated_spectrum: HarmonicSpectrum


def _read_measurement_csv(csv_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    if not csv_bytes:
        raise ValueError("The uploaded CSV is empty.")
    if len(csv_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("CSV exceeds the 25 MB upload-analysis limit.")
    try:
        text = csv_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("CSV must be UTF-8 encoded.") from exc
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise ValueError("CSV must have a header row.")
    normalized_headers = [header.strip() for header in reader.fieldnames]
    if len(set(normalized_headers)) != len(normalized_headers):
        raise ValueError("CSV column names must be unique.")
    if not set(_REQUIRED_COLUMNS).issubset(normalized_headers):
        raise ValueError(
            "CSV must include columns named time_s and voltage_v "
            "(time in seconds, voltage in volts)."
        )
    reader.fieldnames = normalized_headers

    times: list[float] = []
    voltages: list[float] = []
    for row_number, row in enumerate(reader, start=2):
        if len(times) >= MAX_SAMPLE_COUNT:
            raise ValueError("CSV exceeds the 1,000,000 sample analysis limit.")
        if None in row:
            raise ValueError(f"CSV row {row_number} contains extra fields.")
        try:
            time_value = float(row["time_s"])
            voltage_value = float(row["voltage_v"])
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(
                f"CSV row {row_number} must contain numeric time_s and voltage_v values."
            ) from exc
        if not math.isfinite(time_value) or not math.isfinite(voltage_value):
            raise ValueError(f"CSV row {row_number} contains a non-finite value.")
        times.append(time_value)
        voltages.append(voltage_value)

    if len(times) < 4:
        raise ValueError("CSV must contain at least four numeric samples.")
    time_values = np.asarray(times, dtype=np.float64)
    voltage_values = np.asarray(voltages, dtype=np.float64)
    time_steps = np.diff(time_values)
    if np.any(time_steps <= 0.0):
        raise ValueError("time_s values must be strictly increasing.")
    median_step = float(np.median(time_steps))
    if np.max(np.abs(time_steps - median_step)) > max(
        median_step * 0.01,
        np.finfo(np.float64).eps * max(1.0, abs(float(time_values[-1]))),
    ):
        raise ValueError("Samples must be uniformly spaced within 1% of the median step.")
    return time_values, voltage_values


def compare_measured_voltage_csv(
    csv_bytes: bytes,
    simulated_voltage: np.ndarray,
    *,
    simulated_sampling_frequency_hz: float,
    fundamental_frequency_hz: float,
    simulated_fundamental_frequency_hz: float,
    max_harmonic: int = 100,
) -> MeasuredWaveformComparison:
    """Compare measured and simulated voltage spectra over a shared harmonic band.

    CSV input must contain ``time_s`` and ``voltage_v`` columns and span an
    integer number of at least two fundamental cycles. The comparison is
    phase-independent; it does not calculate time-domain error because the
    measurement trigger is not assumed to be synchronized to the simulation.
    """
    fundamental = float(fundamental_frequency_hz)
    simulated_fundamental = float(simulated_fundamental_frequency_hz)
    simulated_rate = float(simulated_sampling_frequency_hz)
    if not math.isfinite(fundamental) or fundamental <= 0.0:
        raise ValueError("fundamental_frequency_hz must be positive and finite.")
    if not math.isfinite(simulated_fundamental) or simulated_fundamental <= 0.0:
        raise ValueError("simulated_fundamental_frequency_hz must be positive and finite.")
    if not math.isclose(fundamental, simulated_fundamental, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "Measured fundamental frequency must match the simulated fundamental."
        )
    if not math.isfinite(simulated_rate) or simulated_rate <= 0.0:
        raise ValueError("simulated_sampling_frequency_hz must be positive and finite.")
    if isinstance(max_harmonic, bool) or not isinstance(max_harmonic, int) or max_harmonic < 2:
        raise ValueError("max_harmonic must be an integer of at least 2.")

    time_values, measured_voltage = _read_measurement_csv(csv_bytes)
    measured_rate = 1.0 / float(np.median(np.diff(time_values)))
    measured = measured_voltage.copy()

    sample_count_cycles = measured.size * fundamental / measured_rate
    span_cycles = (time_values[-1] - time_values[0]) * fundamental
    rounded_sample_cycles = round(sample_count_cycles)
    rounded_span_cycles = round(span_cycles)
    sample_cycle_error = abs(sample_count_cycles - rounded_sample_cycles)
    span_cycle_error = abs(span_cycles - rounded_span_cycles)
    if span_cycle_error < sample_cycle_error:
        if span_cycle_error > 0.01:
            raise ValueError(
                "Measurement must contain an integer number of fundamental cycles."
            )
        measured = measured[:-1]
        cycle_count = int(rounded_span_cycles)
    else:
        if sample_cycle_error > 0.01:
            raise ValueError(
                "Measurement must contain an integer number of fundamental cycles."
            )
        cycle_count = int(rounded_sample_cycles)
    if cycle_count < 2:
        raise ValueError("At least two complete fundamental cycles are required.")

    simulated = np.asarray(simulated_voltage, dtype=np.float64)
    if simulated.ndim != 1 or simulated.size < 4 or not np.isfinite(simulated).all():
        raise ValueError("simulated_voltage must be a finite one-dimensional waveform.")
    harmonic_limit = min(
        max_harmonic,
        int(math.floor((measured_rate / 2.0) / fundamental + 1e-10)),
        int(math.floor((simulated_rate / 2.0) / fundamental + 1e-10)),
    )
    if harmonic_limit < 2:
        raise ValueError(
            "Both waveforms must sample at least through the second harmonic."
        )

    measured_thd, measured_spectrum = calculate_thd(
        measured,
        sampling_frequency_hz=measured_rate,
        fundamental_frequency_hz=fundamental,
        max_harmonic=harmonic_limit,
        return_spectrum=True,
    )
    simulated_thd, simulated_spectrum = calculate_thd(
        simulated,
        sampling_frequency_hz=simulated_rate,
        fundamental_frequency_hz=fundamental,
        max_harmonic=harmonic_limit,
        return_spectrum=True,
    )
    assert isinstance(measured_spectrum, HarmonicSpectrum)
    assert isinstance(simulated_spectrum, HarmonicSpectrum)
    measured_fundamental = measured_spectrum.fundamental_amplitude
    simulated_fundamental = simulated_spectrum.fundamental_amplitude
    if simulated_fundamental <= np.finfo(np.float64).eps:
        raise ValueError("Simulated waveform has no measurable fundamental.")
    return MeasuredWaveformComparison(
        sample_count=int(measured.size),
        cycle_count=cycle_count,
        sampling_frequency_hz=measured_rate,
        fundamental_frequency_hz=fundamental,
        max_harmonic=harmonic_limit,
        measured_mean_voltage_v=float(np.mean(measured)),
        simulated_mean_voltage_v=float(np.mean(simulated)),
        measured_fundamental_amplitude_v=measured_fundamental,
        simulated_fundamental_amplitude_v=simulated_fundamental,
        fundamental_amplitude_change_percent=(
            (measured_fundamental - simulated_fundamental)
            / simulated_fundamental
            * 100.0
        ),
        measured_thd_percent=float(measured_thd),
        simulated_thd_percent=float(simulated_thd),
        thd_difference_percentage_points=float(measured_thd - simulated_thd),
        measured_spectrum=measured_spectrum,
        simulated_spectrum=simulated_spectrum,
    )
