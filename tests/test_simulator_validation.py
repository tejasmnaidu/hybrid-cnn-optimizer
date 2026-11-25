from __future__ import annotations

import unittest

import numpy as np

from multilevel.inverter_simulator import simulate_inverter
from validation.simulator_validation import compare_arrays, compare_simulation


class SimulatorValidationTests(unittest.TestCase):
    def test_comparison_reports_exact_match_metrics(self) -> None:
        reference = np.asarray([[0, 100], [-100, 0]], dtype=np.float64)
        result = compare_arrays(reference, reference.copy())

        self.assertEqual(result["element_count"], 4)
        self.assertEqual(result["mismatch_count"], 0)
        self.assertEqual(result["max_absolute_error"], 0.0)
        self.assertEqual(result["rms_error"], 0.0)
        self.assertIsNone(result["first_mismatch_index"])

    def test_comparison_detects_deliberate_mismatch(self) -> None:
        reference = np.asarray([[0, 100], [-100, 0]], dtype=np.float64)
        simulated = reference.copy()
        simulated[1, 0] = -99.5
        result = compare_arrays(reference, simulated, atol=0.1)

        self.assertEqual(result["mismatch_count"], 1)
        self.assertEqual(result["max_absolute_error"], 0.5)
        self.assertEqual(result["first_mismatch_index"], [1, 0])

    def test_simulation_comparison_checks_states_gates_voltage_and_current(self) -> None:
        simulation = simulate_inverter(0.75, 30.0, 0.03, 0.2)
        reference = {
            "states": simulation.cell_states.copy(),
            "gates": simulation.switching_gates.copy(),
            "voltage": simulation.output_voltage.copy(),
            "current": simulation.load_current.copy(),
        }

        matching = compare_simulation(simulation, reference)
        self.assertTrue(all(result["mismatch_count"] == 0 for result in matching.values()))

        reference["gates"][1, 2, 17] = 1 - reference["gates"][1, 2, 17]
        mismatched = compare_simulation(simulation, reference)
        self.assertEqual(mismatched["switching_gates"]["mismatch_count"], 1)
        self.assertEqual(mismatched["switching_gates"]["first_mismatch_index"], [1, 2, 17])

    def test_comparison_rejects_shape_mismatch_and_nonfinite_data(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            compare_arrays(np.zeros(2), np.zeros(3))
        with self.assertRaisesRegex(ValueError, "finite"):
            compare_arrays(np.asarray([np.nan]), np.asarray([0.0]))


if __name__ == "__main__":
    unittest.main()
