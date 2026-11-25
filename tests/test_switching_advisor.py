from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from assistant import switching_advisor
from assistant.switching_advisor import (
    recommend_switching,
    validate_gate_states,
)
from multilevel.inverter_simulator import simulate_inverter
from optimization.circular_phase_pso import BASELINE_PHASES_DEGREES


def _recommend(**overrides):
    arguments = {
        "modulation_index": 0.75,
        "reference_phase_degrees": 0.0,
        "dc_voltage_v": 100.0,
        "resistance_ohm": 30.0,
        "inductance_h": 0.03,
        "objective": "voltage",
        "seed": 37,
        "population_size": 6,
        "iterations": 4,
    }
    arguments.update(overrides)
    return recommend_switching(**arguments)


class SwitchingAdvisorTests(unittest.TestCase):
    def test_valid_recommendation_returns_verified_safe_gate_states(self) -> None:
        result = _recommend()
        self.assertEqual(result.gate_states.shape, (3, 4, 400))
        self.assertTrue(np.isin(result.gate_states, (0, 1)).all())
        self.assertTrue(
            np.all(
                (result.gate_states[:, 0, :] & result.gate_states[:, 1, :]) == 0
            )
        )
        self.assertTrue(
            np.all(
                (result.gate_states[:, 2, :] & result.gate_states[:, 3, :]) == 0
            )
        )
        self.assertEqual(result.predicted_thd_percent, None)
        self.assertTrue(np.isfinite(result.simulator_verified_thd_percent))
        self.assertEqual(result.carrier_phase_offsets_degrees[0], 0.0)
        self.assertTrue(all(0.0 <= phase < 360.0 for phase in result.carrier_phase_offsets_degrees))

    def test_recommendation_is_reproducible_for_fixed_seed(self) -> None:
        first = _recommend(seed=123)
        second = _recommend(seed=123)
        self.assertEqual(
            first.carrier_phase_offsets_degrees,
            second.carrier_phase_offsets_degrees,
        )
        self.assertEqual(
            first.simulator_verified_thd_percent,
            second.simulator_verified_thd_percent,
        )
        np.testing.assert_array_equal(first.gate_states, second.gate_states)

    def test_invalid_inputs_are_rejected_without_clamping(self) -> None:
        for overrides in (
            {"modulation_index": -0.01},
            {"modulation_index": 1.01},
            {"modulation_index": 0.0},
            {"modulation_index": float("nan")},
            {"reference_phase_degrees": 361.0},
            {"dc_voltage_v": 0.0},
            {"resistance_ohm": -1.0},
            {"inductance_h": 0.0},
            {"objective": "cnn"},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                _recommend(**overrides)

    def test_gate_verification_rejects_states_outside_allowed_levels(self) -> None:
        simulation = simulate_inverter(
            0.75,
            30.0,
            0.03,
            0.0,
            carrier_phase_offsets_degrees=BASELINE_PHASES_DEGREES,
        )
        invalid = simulation.cell_states.copy()
        invalid[0, 0] = 2
        corrupted = simulation.__class__(
            **{
                **simulation.__dict__,
                "cell_states": invalid,
            }
        )
        with self.assertRaisesRegex(RuntimeError, "outside -1, 0, \\+1"):
            validate_gate_states(corrupted, 0.75)
        with self.assertRaisesRegex(ValueError, "never clamped"):
            validate_gate_states(simulation, 1.01)

    def test_optimizer_failure_returns_baseline_with_explicit_reason(self) -> None:
        with patch(
            "assistant.switching_advisor.HybridPOAPFO.optimize",
            side_effect=RuntimeError("test optimizer failure"),
        ):
            result = _recommend()
        self.assertTrue(result.used_baseline_fallback)
        self.assertEqual(
            result.carrier_phase_offsets_degrees,
            BASELINE_PHASES_DEGREES,
        )
        self.assertIn("Optimization failed", result.message)
        self.assertAlmostEqual(
            result.simulator_verified_thd_percent,
            result.baseline_thd_percent,
        )

    def test_worse_verified_candidate_falls_back_to_baseline(self) -> None:
        worse_phases = (10.0, 20.0)
        fake_best = type(
            "Best",
            (),
            {
                "canonical_phases_degrees": worse_phases,
                "feasible": True,
            },
        )()
        fake_run = type(
            "Run",
            (),
            {
                "best": fake_best,
                "unique_candidate_evaluations": 1,
                "objective_requests": 1,
                "runtime_seconds": 0.01,
            },
        )()
        with patch(
            "assistant.switching_advisor.HybridPOAPFO.optimize",
            return_value=fake_run,
        ):
            result = _recommend()
        self.assertTrue(result.used_baseline_fallback)
        self.assertEqual(result.carrier_phase_offsets_degrees, BASELINE_PHASES_DEGREES)
        self.assertIn("worse than baseline", result.message)
        self.assertEqual(result.percent_change, 0.0)

    def test_candidate_simulator_failure_returns_measured_baseline(self) -> None:
        fake_best = type(
            "Best",
            (),
            {
                "canonical_phases_degrees": (10.0, 20.0),
                "feasible": True,
            },
        )()
        fake_run = type(
            "Run",
            (),
            {
                "best": fake_best,
                "unique_candidate_evaluations": 1,
                "objective_requests": 1,
                "runtime_seconds": 0.01,
            },
        )()
        original_simulate = switching_advisor._simulate_and_measure
        calls = 0

        def fail_candidate_simulation(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("test gate verification failure")
            return original_simulate(**kwargs)

        with (
            patch(
                "assistant.switching_advisor.HybridPOAPFO.optimize",
                return_value=fake_run,
            ),
            patch(
                "assistant.switching_advisor._simulate_and_measure",
                side_effect=fail_candidate_simulation,
            ),
        ):
            result = _recommend()

        self.assertTrue(result.used_baseline_fallback)
        self.assertEqual(result.carrier_phase_offsets_degrees, BASELINE_PHASES_DEGREES)
        self.assertIn("Simulator verification failed", result.message)
        self.assertEqual(
            result.simulator_verified_thd_percent,
            result.baseline_thd_percent,
        )

    def test_current_objective_is_reported_as_simulator_verified_current_thd(self) -> None:
        result = _recommend(objective="current")
        self.assertEqual(result.objective, "current")
        self.assertAlmostEqual(
            result.simulator_verified_thd_percent,
            result.verified_metrics.event_current_harmonics_2_100_thd_percent,
        )


if __name__ == "__main__":
    unittest.main()
