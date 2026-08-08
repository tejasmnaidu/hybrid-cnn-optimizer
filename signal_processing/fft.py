from __future__ import annotations

import numpy as np


def compute_fft(signal: np.ndarray, sampling_frequency: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute the one-sided amplitude spectrum for a real signal.

    The returned magnitude values represent the physical amplitude of each
    positive-frequency component, using a one-sided spectrum.
    """
    if sampling_frequency <= 0:
        raise ValueError("Sampling frequency must be positive.")
    if len(signal) < 2:
        raise ValueError("Signal must contain at least two samples.")

    n = len(signal)
    spectrum = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(n, d=1.0 / sampling_frequency)
    magnitudes = np.abs(spectrum) / n

    if n % 2 == 0:
        # For even-length signals, the Nyquist bin is unique and should not be doubled.
        magnitudes[1:-1] *= 2.0
    else:
        magnitudes[1:] *= 2.0

    return frequencies, magnitudes
