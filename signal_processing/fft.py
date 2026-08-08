from __future__ import annotations

import numpy as np
from typing import Any


def compute_fft(signal: np.ndarray, sampling_frequency: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute the frequency axis and magnitude spectrum for a signal."""
    if sampling_frequency <= 0:
        raise ValueError("Sampling frequency must be positive.")
    spectrum = np.fft.rfft(signal)
    magnitudes = np.abs(spectrum) / len(signal)
    frequencies = np.fft.rfftfreq(len(signal), d=1.0 / sampling_frequency)
    return frequencies, magnitudes
