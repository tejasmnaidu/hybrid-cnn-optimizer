from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from scripts.run_thd_experiment import (
    _event_current_thd_percent,
    _format_row,
    _paired_sign_test,
    cnn_guidance_is_justified,
)


class THDExperimentTests(unittest.TestCase):
    def test_event_current_thd_uses_rl_impedance_and_harmonics(self) -> None:
        orders = np.asarray([1, 2, 3], dtype=np.int64)
        voltage = np.asarray([100.0, 10.0, 5.0])
        event = SimpleNamespace(
            harmonic_orders=orders,
            harmonic_peak_amplitudes_v=voltage,
        )
        resistance = 30.0
        inductance = 0.03
        omega = 2.0 * np.pi * 50.0 * orders
        expected = (
            np.sqrt(
                (10.0 / np.hypot(resistance, omega[1] * inductance)) ** 2
                + (5.0 / np.hypot(resistance, omega[2] * inductance)) ** 2
            )
            / (100.0 / np.hypot(resistance, omega[0] * inductance))
            * 100.0
        )
        self.assertAlmostEqual(
            _event_current_thd_percent(
                event,
                resistance_ohm=resistance,
                inductance_h=inductance,
            ),
            expected,
        )

    def test_reduction_is_negative_when_optimized_thd_is_worse(self) -> None:
        row = _format_row(
            seed=1,
            modulation_index=0.75,
            method="PSO",
            phases=(70.0, 130.0),
            baseline_thd=25.0,
            metrics={
                "event_voltage_total": 27.5,
                "event_voltage_2_100": 26.0,
                "sampled_voltage": 24.0,
                "event_current_2_100": 3.0,
                "sampled_current": 3.1,
                "feasible": True,
            },
            unique_evaluations=10,
            objective_requests=12,
            full_event_evaluations=12,
            waveform_generations=1,
            evaluation_budget=72,
            runtime=1.0,
        )
        self.assertAlmostEqual(row["percent_reduction_event_voltage_total_thd"], -10.0)
        self.assertEqual(row["optimized_event_voltage_total_thd_percent"], 27.5)
        self.assertFalse(row["improved_on_baseline"])

    def test_paired_sign_test_matches_seed_and_modulation(self) -> None:
        baseline = [
            {"seed": 1, "modulation_index": 0.55, "optimized_event_voltage_total_thd_percent": 20.0},
            {"seed": 1, "modulation_index": 0.65, "optimized_event_voltage_total_thd_percent": 30.0},
        ]
        optimized = [
            {"seed": 1, "modulation_index": 0.55, "optimized_event_voltage_total_thd_percent": 19.0},
            {"seed": 1, "modulation_index": 0.65, "optimized_event_voltage_total_thd_percent": 32.0},
        ]
        mean_difference, p_value = _paired_sign_test(baseline, optimized)
        self.assertAlmostEqual(mean_difference, -0.5)
        self.assertEqual(p_value, 1.0)

    def test_cnn_guidance_gate_uses_existing_correlation_report(self) -> None:
        justified, correlation = cnn_guidance_is_justified()
        self.assertGreaterEqual(correlation, 0.70)
        self.assertTrue(justified)


if __name__ == "__main__":
    unittest.main()
