from __future__ import annotations

import numpy as np


def calculate_thd(signal: np.ndarray, sampling_frequency: float, fundamental_frequency: float) -> tuple[float, float, list[int], list[float], list[float]]:
    """Calculate total harmonic distortion (THD) for a waveform.

    Returns:
        thd: THD ratio
        fundamental_magnitude: magnitude of the fundamental component
        harmonic_orders: ordered list of harmonic orders included
        harmonic_frequencies: corresponding frequency values for each order
        harmonic_magnitudes: corresponding magnitude for each harmonic order
    """
    if sampling_frequency <= 0:
        raise ValueError("Sampling frequency must be positive.")
    if fundamental_frequency <= 0:
        raise ValueError("Fundamental frequency must be positive.")
    if len(signal) < 2:
        raise ValueError("Signal must contain at least two samples.")

    spectrum = np.fft.rfft(signal)
    magnitudes = np.abs(spectrum) / len(signal)
    frequencies = np.fft.rfftfreq(len(signal), d=1.0 / sampling_frequency)

    fundamental_idx = int(np.argmin(np.abs(frequencies - fundamental_frequency)))
    if fundamental_idx <= 0 or fundamental_idx >= len(magnitudes):
        raise ValueError("Fundamental frequency is outside the FFT frequency range.")

    fundamental_magnitude = float(magnitudes[fundamental_idx])
    harmonic_orders = []
    harmonic_frequencies = []
    harmonic_magnitudes = []
    harmonic_power = 0.0

    for order in range(2, min(11, int(np.floor(frequencies[-1] / fundamental_frequency)) + 1)):
        target_freq = fundamental_frequency * order
        idx = int(np.argmin(np.abs(frequencies - target_freq)))
        if idx < len(magnitudes):
            harmonic_orders.append(order)
            harmonic_frequencies.append(float(frequencies[idx]))
            harmonic_magnitudes.append(float(magnitudes[idx]))
            harmonic_power += magnitudes[idx] ** 2

    thd = np.sqrt(harmonic_power) / fundamental_magnitude if fundamental_magnitude != 0 else float("nan")
    return float(thd), fundamental_magnitude, harmonic_orders, harmonic_frequencies, harmonic_magnitudes
