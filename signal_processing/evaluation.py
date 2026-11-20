from __future__ import annotations

from typing import Any

import numpy as np

from signal_processing.fft import compute_fft
from signal_processing.thd import calculate_thd


def evaluate_inverter_waveform(
    signal: np.ndarray,
    sampling_frequency: float,
    fundamental_frequency: float,
    max_order: int = 15,
) -> dict[str, Any]:
    """Evaluate waveform quality metrics for inverter output.

    Returns a dictionary with RMS, peak, fundamental component, THD, and a harmonic spectrum.
    """
    if sampling_frequency <= 0:
        raise ValueError("sampling_frequency must be positive.")
    if fundamental_frequency <= 0:
        raise ValueError("fundamental_frequency must be positive.")
    if len(signal) < 2:
        raise ValueError("Signal must contain at least two samples.")

    rms = float(np.sqrt(np.mean(signal.astype(float) ** 2)))
    peak = float(np.max(np.abs(signal)))
    thd, fundamental_magnitude, harmonic_orders, harmonic_frequencies, harmonic_magnitudes = calculate_thd(
        signal,
        sampling_frequency,
        fundamental_frequency,
        max_order=max_order,
    )
    frequencies, magnitudes = compute_fft(signal, sampling_frequency)

    harmonic_spectrum: list[dict[str, float]] = []
    total_thd_power = sum([mag ** 2 for mag in harmonic_magnitudes])
    for order, frequency, magnitude in zip(harmonic_orders, harmonic_frequencies, harmonic_magnitudes):
        contribution = (magnitude ** 2 / total_thd_power * 100.0) if total_thd_power > 0 else 0.0
        harmonic_spectrum.append(
            {
                "order": int(order),
                "frequency": float(frequency),
                "magnitude": float(magnitude),
                "relative_magnitude_pct": float(100.0 * magnitude / fundamental_magnitude if fundamental_magnitude != 0 else 0.0),
                "thd_contribution_pct": float(contribution),
            }
        )

    dominant_harmonics = sorted(
        harmonic_spectrum,
        key=lambda row: row["magnitude"],
        reverse=True,
    )[:3]

    return {
        "rms": rms,
        "peak": peak,
        "fundamental_magnitude": fundamental_magnitude,
        "fundamental_frequency": float(fundamental_frequency),
        "thd": thd,
        "thd_percent": float(thd * 100.0),
        "harmonic_spectrum": harmonic_spectrum,
        "dominant_harmonics": dominant_harmonics,
        "fft_frequencies": frequencies.tolist(),
        "fft_magnitudes": magnitudes.tolist(),
    }
