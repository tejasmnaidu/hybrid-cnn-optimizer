from __future__ import annotations

import unittest

import numpy as np

from evaluation.event_based_thd import evaluate_event_based_thd
from evaluation.thd_analysis import calculate_thd
from multilevel.inverter_simulator import (
    SAMPLES_PER_CYCLE,
    SAMPLING_FREQUENCY_HZ,
    simulate_inverter,
)


class THDValidationTests(unittest.TestCase):
    def test_pure_sine_thd_is_zero(self) -> None:
        time = np.arange(SAMPLES_PER_CYCLE, dtype=np.float64) / SAMPLING_FREQUENCY_HZ
        waveform = 230.0 * np.sin(2.0 * np.pi * 50.0 * time)
        thd = calculate_thd(
            waveform,
            sampling_frequency_hz=SAMPLING_FREQUENCY_HZ,
            fundamental_frequency_hz=50.0,
            max_harmonic=100,
        )
        self.assertLess(thd, 1e-10)

    def test_sine_third_and_fifth_thd_matches_analytic_value(self) -> None:
        time = np.arange(SAMPLES_PER_CYCLE, dtype=np.float64) / SAMPLING_FREQUENCY_HZ
        waveform = (
            np.sin(2.0 * np.pi * 50.0 * time)
            + 0.10 * np.sin(3.0 * 2.0 * np.pi * 50.0 * time)
            + 0.05 * np.sin(5.0 * 2.0 * np.pi * 50.0 * time)
        )
        thd = calculate_thd(
            waveform,
            sampling_frequency_hz=SAMPLING_FREQUENCY_HZ,
            fundamental_frequency_hz=50.0,
            max_harmonic=100,
        )
        expected = 100.0 * np.sqrt(0.10**2 + 0.05**2)
        self.assertAlmostEqual(thd, expected, places=10)

    def test_7_level_staircase_is_reported_by_both_methods(self) -> None:
        modulation = 0.75
        phase = 0.0
        sampled = simulate_inverter(modulation, 30.0, 0.03, phase)
        event = evaluate_event_based_thd(modulation, phase)

        self.assertEqual(np.unique(sampled.output_voltage).size, 7)
        self.assertGreater(event.total_thd_percent, 0.0)
        self.assertGreater(event.sampled_thd_percent, 0.0)
        self.assertAlmostEqual(
            event.sampled_thd_percent,
            calculate_thd(
                sampled.output_voltage,
                sampling_frequency_hz=SAMPLING_FREQUENCY_HZ,
                fundamental_frequency_hz=50.0,
                max_harmonic=100,
            ),
            places=12,
        )
        high_rate = 5_000_000.0
        high_count = int(high_rate / 50.0)
        time = np.arange(high_count, dtype=np.float64) / high_rate
        reference = modulation * np.sin(2.0 * np.pi * 50.0 * time)
        high_voltage = np.zeros(high_count, dtype=np.float64)
        for phase_shift in (0.0, 60.0, 120.0):
            carrier = 1.0 - 4.0 * np.abs(
                np.mod(1000.0 * time + phase_shift / 360.0, 1.0) - 0.5
            )
            upper = reference >= carrier
            lower = -reference >= carrier
            high_voltage += 100.0 * (
                upper.astype(np.int8) - lower.astype(np.int8)
            )
        high_thd = calculate_thd(
            high_voltage,
            sampling_frequency_hz=high_rate,
            fundamental_frequency_hz=50.0,
            max_harmonic=100,
        )
        self.assertLess(abs(high_thd - event.thd_percent), 0.1)
        # 20 kHz is a separately labelled uniform sampled measurement and is
        # allowed to differ due to stair-case sampling granularity.
        self.assertGreater(event.sampled_thd_percent, event.thd_percent)

    def test_voltage_thd_is_independent_of_r_and_l_and_current_changes(self) -> None:
        modulation = 0.75
        phase = 0.2
        voltage_results = []
        current_results = []
        for resistance, inductance in ((10.0, 0.01), (40.0, 0.01), (10.0, 0.05), (40.0, 0.05)):
            simulation = simulate_inverter(modulation, resistance, inductance, phase)
            voltage_thd = calculate_thd(
                simulation.output_voltage,
                sampling_frequency_hz=SAMPLING_FREQUENCY_HZ,
                fundamental_frequency_hz=50.0,
                max_harmonic=100,
            )
            current_thd = calculate_thd(
                simulation.load_current,
                sampling_frequency_hz=SAMPLING_FREQUENCY_HZ,
                fundamental_frequency_hz=50.0,
                max_harmonic=100,
            )
            voltage_results.append(voltage_thd)
            current_results.append(current_thd)

        np.testing.assert_allclose(voltage_results, voltage_results[0], rtol=0.0, atol=1e-12)
        self.assertGreater(np.ptp(current_results), 0.1)


if __name__ == "__main__":
    unittest.main()
