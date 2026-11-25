from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from evaluation.event_based_thd import evaluate_event_based_thd
from evaluation.live_phase_optimization import optimize_live_operating_point
from evaluation.live_simulation import run_live_simulation
from multilevel.inverter_simulator import simulate_inverter
from optimization.circular_phase_pso import candidate_to_carrier_offsets
from preprocessing.preprocess import NormalizationStats


class ZeroWaveformModel(nn.Module):
    def forward(self, input_voltage, load_current, operating_conditions):
        return torch.zeros_like(input_voltage)


def _arguments(**overrides: float | int) -> dict[str, float | int]:
    arguments: dict[str, float | int] = {
        "modulation_index": 0.75,
        "dc_voltage_per_cell_v": 100.0,
        "load_resistance_ohm": 30.0,
        "load_inductance_h": 0.03,
        "phase_angle_degrees": 15.0,
        "seed": 20261120,
        "particles": 2,
        "iterations": 1,
    }
    arguments.update(overrides)
    return arguments


class LiveConditionedPSOTests(unittest.TestCase):
    def test_optimizer_selector_accepts_four_existing_methods(self) -> None:
        for name in ("PSO", "POA", "PFO", "HYBRID"):
            with self.subTest(name=name):
                result = optimize_live_operating_point(
                    **_arguments(particles=4, iterations=3),
                    optimizer_name=name,
                )
                self.assertTrue(result.run_result.best.feasible)
                self.assertEqual(result.optimizer_name, name)
    def test_evaluator_receives_each_live_operating_condition(self) -> None:
        scenarios = (
            _arguments(modulation_index=0.62, dc_voltage_per_cell_v=90.0,
                       load_resistance_ohm=10.0, load_inductance_h=0.01,
                       phase_angle_degrees=42.0),
            _arguments(modulation_index=0.88, dc_voltage_per_cell_v=125.0,
                       load_resistance_ohm=40.0, load_inductance_h=0.05,
                       phase_angle_degrees=211.0),
        )
        for scenario in scenarios:
            with self.subTest(scenario=scenario):
                with patch(
                    "evaluation.live_phase_optimization.simulate_inverter",
                    wraps=simulate_inverter,
                ) as simulator:
                    optimization = optimize_live_operating_point(**scenario)
                self.assertGreaterEqual(simulator.call_count, 2)
                for call in simulator.call_args_list:
                    self.assertAlmostEqual(call.args[0], float(scenario["modulation_index"]))
                    self.assertEqual(call.args[1], float(scenario["load_resistance_ohm"]))
                    self.assertEqual(call.args[2], float(scenario["load_inductance_h"]))
                    self.assertAlmostEqual(
                        call.args[3], np.deg2rad(float(scenario["phase_angle_degrees"]))
                    )
                    self.assertEqual(
                        call.kwargs["dc_voltage_per_cell_v"],
                        float(scenario["dc_voltage_per_cell_v"]),
                    )
                self.assertEqual(optimization.load_resistance_ohm, scenario["load_resistance_ohm"])
                self.assertEqual(optimization.load_inductance_h, scenario["load_inductance_h"])

    def test_voltage_thd_definition_tracking_and_reproducibility(self) -> None:
        arguments = _arguments()
        first = optimize_live_operating_point(**arguments)
        second = optimize_live_operating_point(**arguments)
        selected_offsets = candidate_to_carrier_offsets(
            first.run_result.best.canonical_phases_degrees
        )
        expected_baseline = evaluate_event_based_thd(
            float(arguments["modulation_index"]),
            np.deg2rad(float(arguments["phase_angle_degrees"])),
            dc_voltage_per_cell_v=float(arguments["dc_voltage_per_cell_v"]),
        )
        expected_candidate = evaluate_event_based_thd(
            float(arguments["modulation_index"]),
            np.deg2rad(float(arguments["phase_angle_degrees"])),
            carrier_phase_offsets_degrees=selected_offsets,
            dc_voltage_per_cell_v=float(arguments["dc_voltage_per_cell_v"]),
        )
        self.assertAlmostEqual(
            first.baseline_evaluation.mean_total_thd_percent,
            expected_baseline.total_thd_percent,
        )
        self.assertAlmostEqual(
            first.run_result.best.mean_total_thd_percent,
            expected_candidate.total_thd_percent,
        )
        self.assertTrue(first.run_result.best.feasible)
        self.assertLessEqual(first.run_result.best.maximum_tracking_error_percent, 0.05)
        self.assertEqual(
            first.run_result.best.canonical_phases_degrees,
            second.run_result.best.canonical_phases_degrees,
        )
        self.assertEqual(
            [
                item.mean_total_thd_percent
                for item in first.run_result.convergence_history
            ],
            [
                item.mean_total_thd_percent
                for item in second.run_result.convergence_history
            ],
        )

    def test_selected_phases_are_resimulated_at_the_live_condition(self) -> None:
        arguments = _arguments(
            modulation_index=0.82,
            dc_voltage_per_cell_v=110.0,
            load_resistance_ohm=20.0,
            load_inductance_h=0.02,
            phase_angle_degrees=88.0,
        )
        optimization = optimize_live_operating_point(**arguments)
        offsets = candidate_to_carrier_offsets(
            optimization.run_result.best.canonical_phases_degrees
        )
        normalization = NormalizationStats(
            input_voltage_mean=0.0,
            input_voltage_std=100.0,
            load_current_mean=0.0,
            load_current_std=10.0,
            operating_mean=[0.7, 20.0, 0.02, 0.5],
            operating_std=[0.1, 10.0, 0.01, 2.0],
            target_voltage_mean=0.0,
            target_voltage_std=200.0,
        )
        result = run_live_simulation(
            ZeroWaveformModel(),
            normalization,
            modulation_index=float(arguments["modulation_index"]),
            dc_voltage_per_cell_v=float(arguments["dc_voltage_per_cell_v"]),
            load_resistance_ohm=float(arguments["load_resistance_ohm"]),
            load_inductance_h=float(arguments["load_inductance_h"]),
            phase_angle_degrees=float(arguments["phase_angle_degrees"]),
            device="cpu",
            carrier_phase_offsets_degrees=offsets,
        )
        direct = simulate_inverter(
            float(arguments["modulation_index"]),
            float(arguments["load_resistance_ohm"]),
            float(arguments["load_inductance_h"]),
            np.deg2rad(float(arguments["phase_angle_degrees"])),
            carrier_phase_offsets_degrees=offsets,
            dc_voltage_per_cell_v=float(arguments["dc_voltage_per_cell_v"]),
        )
        np.testing.assert_array_equal(result.simulation.output_voltage, direct.output_voltage)
        np.testing.assert_array_equal(result.simulation.switching_gates, direct.switching_gates)
        self.assertTrue(np.isfinite(result.current_thd_percent))
        self.assertEqual(result.simulation.dc_voltage_per_cell_v, 110.0)


if __name__ == "__main__":
    unittest.main()
