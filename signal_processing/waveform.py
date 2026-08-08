from __future__ import annotations

import numpy as np


def generate_step_waveform(levels: int, amplitude: float, length: int) -> np.ndarray:
    """Generate a simple stepped waveform with the given number of levels."""
    if levels < 2:
        raise ValueError("levels must be at least 2")
    step_values = np.linspace(-amplitude, amplitude, levels)
    return np.repeat(step_values, length // levels)
