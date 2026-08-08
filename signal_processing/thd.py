from __future__ import annotations

import numpy as np
from signal_processing.fft import compute_fft


def calculate_thd(
    signal: np.ndarray,
    sampling_frequency: float,
    fundamental_frequency: float,
    max_order: int = 15,
) -> tuple[float, float, list[int], list[float], list[float]]:
    """Calculate total harmonic distortion (THD) for a waveform.

    THD is defined as:
        THD = sqrt(sum(V_h^2 for h >= 2)) / V_1
    where V_h are the one-sided amplitudes of the harmonic components and
    V_1 is the one-sided amplitude of the fundamental.
    """
    if sampling_frequency <= 0:
        raise ValueError("Sampling frequency must be positive.")
    if fundamental_frequency <= 0:
        raise ValueError("Fundamental frequency must be positive.")
    if len(signal) < 2:
        raise ValueError("Signal must contain at least two samples.")

    frequencies, magnitudes = compute_fft(signal, sampling_frequency)
    frequency_resolution = frequencies[1] - frequencies[0]

    fundamental_idx = int(np.argmin(np.abs(frequencies - fundamental_frequency)))
    fundamental_error = abs(frequencies[fundamental_idx] - fundamental_frequency)
    if fundamental_error > frequency_resolution / 2.0:
        raise ValueError(
            f"Fundamental frequency {fundamental_frequency:.6f} Hz does not align with FFT bins (nearest {frequencies[fundamental_idx]:.6f} Hz)."
        )

    if fundamental_idx <= 0 or fundamental_idx >= len(magnitudes):
        raise ValueError("Fundamental frequency is outside the FFT frequency range.")

    fundamental_magnitude = float(magnitudes[fundamental_idx])
    harmonic_orders: list[int] = []
    harmonic_frequencies: list[float] = []
    harmonic_magnitudes: list[float] = []
    harmonic_power = 0.0

    max_order = max(2, max_order)
    highest_order = int(np.floor(frequencies[-1] / fundamental_frequency))
    for order in range(2, min(max_order, highest_order) + 1):
        target_freq = fundamental_frequency * order
        idx = int(np.argmin(np.abs(frequencies - target_freq)))
        if idx >= len(frequencies):
            break
        if abs(frequencies[idx] - target_freq) > frequency_resolution / 2.0:
            continue
        harmonic_orders.append(order)
        harmonic_frequencies.append(float(frequencies[idx]))
        harmonic_magnitudes.append(float(magnitudes[idx]))
        harmonic_power += magnitudes[idx] ** 2

    thd = np.sqrt(harmonic_power) / fundamental_magnitude if fundamental_magnitude != 0 else float("nan")
    return float(thd), fundamental_magnitude, harmonic_orders, harmonic_frequencies, harmonic_magnitudes
