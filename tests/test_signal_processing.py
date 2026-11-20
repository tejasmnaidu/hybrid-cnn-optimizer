import numpy as np

from signal_processing.fft import compute_fft
from signal_processing.evaluation import evaluate_inverter_waveform
from signal_processing.thd import calculate_thd


def test_fft_frequency_axis_and_normalization():
    sampling_frequency = 1000.0
    duration = 1.0
    t = np.linspace(0.0, duration, int(sampling_frequency * duration), endpoint=False)
    signal = np.sin(2.0 * np.pi * 5.0 * t)

    frequencies, magnitudes = compute_fft(signal, sampling_frequency)
    assert np.isclose(frequencies[0], 0.0)
    assert np.isclose(frequencies[-1], sampling_frequency / 2.0)
    assert np.isclose(frequencies[1] - frequencies[0], sampling_frequency / len(signal))
    fundamental_idx = int(np.argmin(np.abs(frequencies - 5.0)))
    assert np.isclose(magnitudes[fundamental_idx], 1.0, atol=1e-2)


def test_thd_for_pure_sine_is_zero():
    sampling_frequency = 2000.0
    duration = 1.0
    frequency = 50.0
    t = np.linspace(0.0, duration, int(sampling_frequency * duration), endpoint=False)
    signal = np.sin(2.0 * np.pi * frequency * t)

    thd, fundamental_mag, orders, freqs, mags = calculate_thd(signal, sampling_frequency, frequency)
    assert thd >= 0.0
    assert thd < 1e-3
    assert np.isclose(fundamental_mag, 1.0, atol=1e-2)
    assert orders == []
    assert freqs == []
    assert mags == []


def test_thd_with_known_harmonic():
    sampling_frequency = 2000.0
    duration = 1.0
    frequency = 50.0
    t = np.linspace(0.0, duration, int(sampling_frequency * duration), endpoint=False)
    signal = np.sin(2.0 * np.pi * frequency * t) + 0.5 * np.sin(2.0 * np.pi * 3.0 * frequency * t)

    thd, fundamental_mag, orders, freqs, mags = calculate_thd(signal, sampling_frequency, frequency)
    expected_thd = 0.5
    assert len(orders) >= 1
    assert orders[0] == 3
    assert np.isclose(mags[0], 0.5, atol=0.02)
    assert np.isclose(thd, expected_thd, atol=0.02)


def test_rms_and_peak_from_evaluate_inverter_waveform():
    sampling_frequency = 1000.0
    amplitude = 2.0
    frequency = 10.0
    duration = 1.0
    t = np.linspace(0.0, duration, int(sampling_frequency * duration), endpoint=False)
    signal = amplitude * np.sin(2.0 * np.pi * frequency * t)

    evaluation = evaluate_inverter_waveform(signal, sampling_frequency, frequency)
    assert np.isclose(evaluation["rms"], amplitude / np.sqrt(2.0), atol=1e-3)
    assert np.isclose(evaluation["peak"], amplitude, atol=1e-6)
    assert np.isclose(evaluation["fundamental_magnitude"], amplitude, atol=1e-2)
    assert np.isclose(evaluation["thd"], 0.0, atol=1e-3)
    assert evaluation["harmonic_spectrum"] == []


def test_evaluate_inverter_waveform_reports_dominant_harmonics():
    sampling_frequency = 2000.0
    duration = 1.0
    frequency = 50.0
    t = np.linspace(0.0, duration, int(sampling_frequency * duration), endpoint=False)
    signal = np.sin(2.0 * np.pi * frequency * t) + 0.3 * np.sin(2.0 * np.pi * 5.0 * frequency * t)

    evaluation = evaluate_inverter_waveform(signal, sampling_frequency, frequency)
    assert evaluation["dominant_harmonics"]
    assert any(h["order"] == 5 for h in evaluation["dominant_harmonics"])
    assert evaluation["thd_percent"] > 0.0
