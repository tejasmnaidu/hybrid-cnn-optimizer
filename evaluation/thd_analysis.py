"""THD computation for an explicitly identified waveform."""

from __future__ import annotations

import numpy as np

from .harmonic_analysis import HarmonicSpectrum, harmonic_spectrum


def calculate_thd(
    waveform: np.ndarray,
    *,
    sampling_frequency_hz: float = 20_000.0,
    fundamental_frequency_hz: float = 50.0,
    max_harmonic: int = 100,
    return_spectrum: bool = False,
) -> float | tuple[float, HarmonicSpectrum]:
    """Return voltage/current THD in percent over harmonics 2..max_harmonic.

    The caller must state whether a waveform is a reference or a measured
    inverter output; this function does not infer that semantic label.
    """
    spectrum = harmonic_spectrum(
        waveform,
        sampling_frequency_hz=sampling_frequency_hz,
        fundamental_frequency_hz=fundamental_frequency_hz,
        max_harmonic=max_harmonic,
    )
    if spectrum.fundamental_amplitude <= np.finfo(np.float64).eps:
        raise ValueError("THD is undefined when the fundamental amplitude is zero.")
    harmonic_amplitudes = spectrum.amplitudes[spectrum.harmonic_orders >= 2]
    value = float(np.sqrt(np.sum(np.square(harmonic_amplitudes))) / spectrum.fundamental_amplitude * 100)
    return (value, spectrum) if return_spectrum else value
