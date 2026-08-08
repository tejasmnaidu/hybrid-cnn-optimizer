import os
from pathlib import Path
import unittest

import numpy as np

from multilevel.eight_level_model import EightLevelInverterParams, generate_8_level_waveform, save_waveform_csv
from signal_processing.fft import compute_fft
from signal_processing.thd import calculate_thd


class TestEightLevelWaveform(unittest.TestCase):
    def test_generate_8_level_waveform(self):
        params = EightLevelInverterParams(
            dc_voltage=400.0,
            fundamental_frequency=50.0,
            switching_frequency=2000.0,
            sampling_frequency=10000.0,
            simulation_duration=0.02,
            modulation_index=1.0,
        )
        waveform = generate_8_level_waveform(params)

        self.assertEqual(waveform.time.shape[0], int(params.simulation_duration * params.sampling_frequency))
        self.assertEqual(waveform.voltage.shape, waveform.time.shape)
        self.assertEqual(waveform.switching_states.shape, (waveform.time.shape[0], 3))
        unique_levels = np.unique(np.round(waveform.voltage, decimals=8))
        self.assertLessEqual(unique_levels.shape[0], 8)
        self.assertTrue(np.all(np.isfinite(waveform.voltage)))

    def test_fft_computation(self):
        params = EightLevelInverterParams()
        waveform = generate_8_level_waveform(params)
        frequencies, magnitudes = compute_fft(waveform.voltage, params.sampling_frequency)

        self.assertEqual(frequencies.shape, magnitudes.shape)
        self.assertEqual(frequencies[0], 0.0)
        self.assertTrue(np.all(magnitudes >= 0))

    def test_thd_calculation(self):
        params = EightLevelInverterParams()
        waveform = generate_8_level_waveform(params)
        thd_value, fundamental_mag, harmonic_magnitudes = calculate_thd(
            waveform.voltage, params.sampling_frequency, params.fundamental_frequency
        )

        self.assertGreaterEqual(thd_value, 0)
        self.assertGreaterEqual(fundamental_mag, 0)
        self.assertIsInstance(harmonic_magnitudes, list)

    def test_save_waveform_csv(self):
        params = EightLevelInverterParams()
        waveform = generate_8_level_waveform(params)
        save_path = Path("tests") / "waveform_test.csv"
        save_waveform_csv(save_path, waveform)

        self.assertTrue(save_path.exists())
        content = save_path.read_text()
        self.assertIn("time", content)
        self.assertIn("voltage", content)
        self.assertIn("switching_state", content)
        save_path.unlink()


if __name__ == "__main__":
    unittest.main()
