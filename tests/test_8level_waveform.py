import os
from pathlib import Path

import numpy as np

from multilevel.eight_level_model import EightLevelInverterParams, EightLevelWaveform, generate_8_level_waveform, save_waveform_csv
from signal_processing.fft import compute_fft
from signal_processing.thd import calculate_thd


def test_generate_8_level_waveform():
    params = EightLevelInverterParams(
        dc_voltage=400.0,
        fundamental_frequency=50.0,
        switching_frequency=2000.0,
        sampling_frequency=10000.0,
        simulation_duration=0.02,
        modulation_index=1.0,
    )
    waveform = generate_8_level_waveform(params)

    assert waveform.time.shape[0] == int(params.simulation_duration * params.sampling_frequency)
    assert waveform.voltage.shape == waveform.time.shape
    assert waveform.switching_states.shape == (waveform.time.shape[0], 3)
    unique_levels = np.unique(np.round(waveform.voltage, decimals=8))
    assert unique_levels.shape[0] <= 8
    assert np.all(np.isfinite(waveform.voltage))
    assert np.all(np.diff(np.sort(unique_levels)) > 0)
    assert np.isclose(unique_levels.min(), -unique_levels.max())


def test_fft_computation():
    params = EightLevelInverterParams()
    waveform = generate_8_level_waveform(params)
    frequencies, magnitudes = compute_fft(waveform.voltage, params.sampling_frequency)

    assert frequencies.shape == magnitudes.shape
    assert frequencies[0] == 0.0
    assert np.all(magnitudes >= 0)
    frequency_resolution = frequencies[1] - frequencies[0]
    assert np.isclose(frequency_resolution, params.sampling_frequency / len(waveform.voltage))
    assert np.isclose(np.max(frequencies), params.sampling_frequency / 2.0)


def test_thd_calculation():
    params = EightLevelInverterParams()
    waveform = generate_8_level_waveform(params)
    thd_value, fundamental_mag, harmonic_orders, harmonic_frequencies, harmonic_magnitudes = calculate_thd(
        waveform.voltage, params.sampling_frequency, params.fundamental_frequency
    )

    assert thd_value >= 0
    assert fundamental_mag >= 0
    assert isinstance(harmonic_magnitudes, list)
    assert isinstance(harmonic_orders, list)
    assert isinstance(harmonic_frequencies, list)


def test_save_waveform_csv(tmp_path: Path):
    params = EightLevelInverterParams()
    waveform = generate_8_level_waveform(params)
    save_path = tmp_path / "waveform.csv"
    save_waveform_csv(save_path, waveform)

    assert save_path.exists()
    content = save_path.read_text()
    assert "time" in content
    assert "voltage" in content
    assert "switching_state" in content
