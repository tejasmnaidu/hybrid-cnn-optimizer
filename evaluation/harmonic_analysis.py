"""Single-sided harmonic spectrum calculation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HarmonicSpectrum:
    frequencies_hz: np.ndarray
    amplitudes: np.ndarray
    harmonic_orders: np.ndarray
    fundamental_amplitude: float
    fundamental_frequency_hz: float


def harmonic_spectrum(
    waveform: np.ndarray,
    *,
    sampling_frequency_hz: float = 20_000.0,
    fundamental_frequency_hz: float = 50.0,
    max_harmonic: int = 100,
) -> HarmonicSpectrum:
    values = np.asarray(waveform, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("waveform must be a one-dimensional array with at least two points.")
    if not np.isfinite(values).all():
        raise ValueError("waveform must contain only finite values.")
    if sampling_frequency_hz <= 0 or fundamental_frequency_hz <= 0 or max_harmonic < 1:
        raise ValueError("Frequencies and max_harmonic must be positive.")
    frequencies = np.fft.rfftfreq(values.size, d=1.0 / sampling_frequency_hz)
    fft = np.fft.rfft(values - np.mean(values))
    amplitude = 2.0 * np.abs(fft) / values.size
    if values.size % 2 == 0:
        amplitude[-1] *= 0.5
    bin_width = sampling_frequency_hz / values.size
    fundamental_bin = int(round(fundamental_frequency_hz / bin_width))
    if fundamental_bin < 1 or fundamental_bin >= len(amplitude):
        raise ValueError("Fundamental frequency is outside the measurable FFT range.")
    actual_fundamental = frequencies[fundamental_bin]
    if not np.isclose(actual_fundamental, fundamental_frequency_hz, atol=bin_width / 2):
        raise ValueError(
            "Fundamental frequency does not align with an FFT bin; provide an integer-cycle waveform "
            "or a suitable analysis method."
        )
    max_order = min(max_harmonic, int(frequencies[-1] // actual_fundamental))
    orders = np.arange(1, max_order + 1, dtype=np.int64)
    bins = np.rint(orders * actual_fundamental / bin_width).astype(np.int64)
    return HarmonicSpectrum(
        frequencies_hz=frequencies[bins],
        amplitudes=amplitude[bins],
        harmonic_orders=orders,
        fundamental_amplitude=float(amplitude[fundamental_bin]),
        fundamental_frequency_hz=float(actual_fundamental),
    )
