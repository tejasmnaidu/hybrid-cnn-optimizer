from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from multilevel.inverter_simulator import (
    CELL_COUNT,
    DC_VOLTAGE_PER_CELL_V,
    FUNDAMENTAL_FREQUENCY_HZ,
    SAMPLING_FREQUENCY_HZ,
    SAMPLES_PER_CYCLE,
    simulate_inverter,
)


def _fundamental_phasor(signal: np.ndarray) -> complex:
    sample_indices = np.arange(signal.size)
    return complex(
        2.0
        * np.sum(
            signal * np.exp(-2j * np.pi * sample_indices / signal.size)
        )
        / signal.size
    )


def _wrapped_phase_difference(actual: float, expected: float) -> float:
    return float(np.angle(np.exp(1j * (actual - expected))))


class InverterSimulatorTests(unittest.TestCase):
    def test_modulation_index_accepts_inclusive_bounds_and_rejects_out_of_range(self) -> None:
        for modulation in (0.0, 1.0):
            with self.subTest(modulation=modulation):
                result = simulate_inverter(modulation, 30.0, 0.03, 0.0)
                self.assertEqual(result.reference.shape, (SAMPLES_PER_CYCLE,))

        for modulation in (-0.0001, 1.0001):
            with self.subTest(modulation=modulation):
                with self.assertRaisesRegex(
                    ValueError,
                    r"modulation_index must be in \[0\.0, 1\.0\]",
                ):
                    simulate_inverter(modulation, 30.0, 0.03, 0.0)

    def test_output_contains_only_seven_voltage_levels(self) -> None:
        result = simulate_inverter(1.0, 30.0, 0.03, 0.0)
        allowed = np.arange(-300.0, 301.0, 100.0)
        self.assertTrue(np.isin(result.output_voltage, allowed).all())
        np.testing.assert_array_equal(np.unique(result.output_voltage), allowed)
        self.assertEqual(result.output_voltage.shape, (SAMPLES_PER_CYCLE,))
        self.assertEqual(result.cell_states.shape, (CELL_COUNT, SAMPLES_PER_CYCLE))

    def test_gate_truth_table_reconstructs_cell_states_without_shoot_through(self) -> None:
        result = simulate_inverter(0.8, 20.0, 0.02, 0.3)
        expected_gates = np.empty_like(result.switching_gates)
        expected_gates[:, 0, :] = result.cell_states == 1
        expected_gates[:, 1, :] = result.cell_states != 1
        expected_gates[:, 2, :] = result.cell_states == -1
        expected_gates[:, 3, :] = result.cell_states != -1
        np.testing.assert_array_equal(result.switching_gates, expected_gates)

        shoot_through = (
            (result.switching_gates[:, 0, :] & result.switching_gates[:, 1, :])
            | (result.switching_gates[:, 2, :] & result.switching_gates[:, 3, :])
        )
        self.assertFalse(shoot_through.any())
        reconstructed_cell_voltage = DC_VOLTAGE_PER_CELL_V * (
            result.switching_gates[:, 0, :].astype(np.int8)
            * result.switching_gates[:, 3, :].astype(np.int8)
            - result.switching_gates[:, 1, :].astype(np.int8)
            * result.switching_gates[:, 2, :].astype(np.int8)
        )
        np.testing.assert_array_equal(
            reconstructed_cell_voltage,
            DC_VOLTAGE_PER_CELL_V * result.cell_states,
        )
        np.testing.assert_array_equal(
            result.output_voltage,
            reconstructed_cell_voltage.sum(axis=0),
        )

    def test_reference_phase_shifts_output_fundamental_without_changing_amplitude(self) -> None:
        phase = 0.2
        phase_change = 0.7
        baseline = simulate_inverter(0.75, 30.0, 0.03, phase)
        shifted = simulate_inverter(0.75, 30.0, 0.03, phase + phase_change)
        baseline_phasor = _fundamental_phasor(baseline.output_voltage)
        shifted_phasor = _fundamental_phasor(shifted.output_voltage)

        self.assertAlmostEqual(
            abs(shifted_phasor),
            abs(baseline_phasor),
            delta=abs(baseline_phasor) * 1e-12,
        )
        measured_phase_change = float(np.angle(shifted_phasor / baseline_phasor))
        self.assertLessEqual(
            abs(_wrapped_phase_difference(measured_phase_change, phase_change)),
            0.01,
        )

    def test_carrier_offsets_change_switching_and_preserve_fundamental_amplitude(self) -> None:
        arguments = (0.75, 30.0, 0.03, 0.2)
        baseline = simulate_inverter(*arguments)
        explicit_baseline = simulate_inverter(
            *arguments,
            carrier_phase_offsets_degrees=(0.0, 60.0, 120.0),
        )
        shifted = simulate_inverter(
            *arguments,
            carrier_phase_offsets_degrees=(0.0, 45.0, 135.0),
        )

        for field in ("cell_states", "switching_gates", "cell_voltages", "output_voltage", "load_current"):
            np.testing.assert_array_equal(
                getattr(baseline, field),
                getattr(explicit_baseline, field),
            )
        self.assertFalse(np.array_equal(baseline.cell_states, shifted.cell_states))
        self.assertFalse(np.array_equal(baseline.switching_gates, shifted.switching_gates))
        baseline_amplitude = abs(_fundamental_phasor(baseline.output_voltage))
        shifted_amplitude = abs(_fundamental_phasor(shifted.output_voltage))
        self.assertLessEqual(
            abs(shifted_amplitude - baseline_amplitude) / baseline_amplitude,
            0.02,
        )

    def test_resistance_and_inductance_follow_discrete_rl_fundamental_response(self) -> None:
        modulation, phase = 0.75, 0.2
        parameters = ((10.0, 0.01), (40.0, 0.01), (10.0, 0.05))
        results = [
            simulate_inverter(modulation, resistance, inductance, phase)
            for resistance, inductance in parameters
        ]
        baseline = results[0]
        baseline_current_phasor = _fundamental_phasor(baseline.load_current)

        for result, (resistance, inductance) in zip(results, parameters):
            np.testing.assert_array_equal(result.output_voltage, baseline.output_voltage)
            alpha = np.exp(
                -resistance / SAMPLING_FREQUENCY_HZ / inductance
            )
            beta = -np.expm1(
                -resistance / SAMPLING_FREQUENCY_HZ / inductance
            ) / resistance
            angular_frequency = 2.0 * np.pi / SAMPLES_PER_CYCLE
            expected_transfer = beta / (np.exp(1j * angular_frequency) - alpha)
            actual_transfer = (
                _fundamental_phasor(result.load_current)
                / _fundamental_phasor(result.output_voltage)
            )
            self.assertAlmostEqual(
                abs(actual_transfer - expected_transfer),
                0.0,
                delta=1e-12,
            )

        for changed in results[1:]:
            changed_current_phasor = _fundamental_phasor(changed.load_current)
            self.assertNotAlmostEqual(
                abs(changed_current_phasor),
                abs(baseline_current_phasor),
                delta=1e-6,
            )
            self.assertGreater(
                abs(
                    _wrapped_phase_difference(
                        float(np.angle(changed_current_phasor)),
                        float(np.angle(baseline_current_phasor)),
                    )
                ),
                1e-3,
            )

    def test_dc_voltage_scales_all_seven_output_levels(self) -> None:
        nominal = simulate_inverter(
            1.0, 30.0, 0.03, 0.0, dc_voltage_per_cell_v=100.0
        )
        half_voltage = simulate_inverter(
            1.0, 30.0, 0.03, 0.0, dc_voltage_per_cell_v=50.0
        )

        np.testing.assert_array_equal(half_voltage.output_voltage, nominal.output_voltage / 2.0)
        np.testing.assert_array_equal(
            np.unique(nominal.output_voltage),
            np.arange(-300.0, 301.0, 100.0),
        )
        np.testing.assert_array_equal(
            np.unique(half_voltage.output_voltage),
            np.arange(-150.0, 151.0, 50.0),
        )
        self.assertEqual(np.unique(half_voltage.output_voltage).size, 7)

    def test_every_gate_vector_maps_to_state_and_cell_voltage_sum(self) -> None:
        result = simulate_inverter(0.8, 20.0, 0.02, 0.3)
        state_to_gates = np.asarray(
            (
                (0, 1, 1, 0),
                (0, 1, 0, 1),
                (1, 0, 0, 1),
            ),
            dtype=np.uint8,
        )
        expected_gates = state_to_gates[result.cell_states.astype(np.int64) + 1]
        np.testing.assert_array_equal(
            result.switching_gates.transpose(0, 2, 1),
            expected_gates,
        )
        valid_gate_vectors = state_to_gates
        actual_gate_vectors = result.switching_gates.transpose(0, 2, 1).reshape(-1, 4)
        valid_vector_matches = (
            actual_gate_vectors[:, None, :] == valid_gate_vectors[None, :, :]
        ).all(axis=2)
        self.assertTrue(valid_vector_matches.any(axis=1).all())
        np.testing.assert_array_equal(
            result.output_voltage,
            result.cell_voltages.sum(axis=0),
        )

    def test_rl_recurrence_and_periodic_boundary(self) -> None:
        resistance = 27.0
        inductance = 0.04
        result = simulate_inverter(0.75, resistance, inductance, 0.2)
        dt = 1.0 / SAMPLING_FREQUENCY_HZ
        alpha = np.exp(-resistance * dt / inductance)
        beta = (1.0 - alpha) / resistance
        np.testing.assert_allclose(
            result.load_current[1:],
            alpha * result.load_current[:-1] + beta * result.output_voltage[:-1],
            rtol=1e-13,
            atol=1e-13,
        )
        periodic_next = alpha * result.load_current[-1] + beta * result.output_voltage[-1]
        self.assertAlmostEqual(
            periodic_next,
            result.periodic_initial_current_a,
            delta=1e-12,
        )

    def test_simulation_is_deterministic(self) -> None:
        first = simulate_inverter(0.65, 15.0, 0.015, 1.2)
        second = simulate_inverter(0.65, 15.0, 0.015, 1.2)
        for field in (
            "time_s",
            "reference",
            "carriers",
            "cell_states",
            "switching_gates",
            "cell_voltages",
            "output_voltage",
            "load_current",
        ):
            np.testing.assert_array_equal(getattr(first, field), getattr(second, field))
        self.assertEqual(
            first.periodic_initial_current_a,
            second.periodic_initial_current_a,
        )

    def test_live_modulation_dc_voltage_and_load_change_simulated_signals(self) -> None:
        baseline = simulate_inverter(0.65, 30.0, 0.03, 0.0)
        changed_modulation = simulate_inverter(0.9, 30.0, 0.03, 0.0)
        changed_dc = simulate_inverter(
            0.65, 30.0, 0.03, 0.0, dc_voltage_per_cell_v=120.0
        )
        changed_resistance = simulate_inverter(0.65, 10.0, 0.03, 0.0)
        changed_inductance = simulate_inverter(0.65, 30.0, 0.01, 0.0)

        self.assertFalse(np.array_equal(baseline.output_voltage, changed_modulation.output_voltage))
        self.assertFalse(np.array_equal(baseline.output_voltage, changed_dc.output_voltage))
        self.assertTrue(
            np.isin(
                np.unique(changed_dc.output_voltage),
                np.arange(-360.0, 361.0, 120.0),
            ).all()
        )
        self.assertLessEqual(changed_dc.output_voltage.max(), 360.0)
        self.assertFalse(np.allclose(baseline.load_current, changed_resistance.load_current))
        self.assertFalse(np.allclose(baseline.load_current, changed_inductance.load_current))
        self.assertEqual(changed_dc.dc_voltage_per_cell_v, 120.0)

    def test_invalid_load_and_non_finite_inputs_are_rejected(self) -> None:
        for args in (
            (0.7, 0.0, 0.03, 0.0),
            (0.7, 30.0, -0.03, 0.0),
            (float("nan"), 30.0, 0.03, 0.0),
            (0.7, 30.0, 0.03, float("inf")),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                simulate_inverter(*args)

    def test_invalid_modulation_load_and_nan_inputs_raise_clear_errors(self) -> None:
        cases = (
            ((-0.01, 30.0, 0.03, 0.0), {}, r"modulation_index must be in \[0\.0, 1\.0\]"),
            ((1.01, 30.0, 0.03, 0.0), {}, r"modulation_index must be in \[0\.0, 1\.0\]"),
            ((0.7, 0.0, 0.03, 0.0), {}, "load_resistance_ohm must be positive"),
            ((0.7, -1.0, 0.03, 0.0), {}, "load_resistance_ohm must be positive"),
            ((0.7, 30.0, 0.0, 0.0), {}, "load_inductance_h must be positive"),
            ((0.7, 30.0, -0.03, 0.0), {}, "load_inductance_h must be positive"),
            ((float("nan"), 30.0, 0.03, 0.0), {}, "modulation_index must be a finite scalar"),
            ((0.7, float("nan"), 0.03, 0.0), {}, "load_resistance_ohm must be a finite scalar"),
            ((0.7, 30.0, float("nan"), 0.0), {}, "load_inductance_h must be a finite scalar"),
            ((0.7, 30.0, 0.03, float("nan")), {}, "phase_angle_rad must be a finite scalar"),
        )
        for args, kwargs, message in cases:
            with self.subTest(args=args, message=message):
                with self.assertRaisesRegex(ValueError, message):
                    simulate_inverter(*args, **kwargs)

    def test_matches_existing_dataset_record(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        data_root = project_root / "data" / "raw" / "Seven_Level_Inverter_Project_extracted"
        required = (
            "Operating_Conditions.csv",
            "Cell_States.csv",
            "Switching_Gates.csv",
            "Input_Voltage.csv",
            "Load_Current.csv",
        )
        if not all((data_root / name).is_file() for name in required):
            self.skipTest("Project dataset CSV files are not available.")

        conditions = pd.read_csv(data_root / required[0], nrows=1).iloc[0]
        self.assertEqual(int(conditions["sampleID"]), 1)
        result = simulate_inverter(
            conditions["modulationIndex"],
            conditions["loadResistance"],
            conditions["loadInductance"],
            conditions["phaseAngle"],
        )
        states = pd.read_csv(data_root / required[1], nrows=1).iloc[0, 1:].to_numpy(dtype=np.int8)
        gates = pd.read_csv(data_root / required[2], nrows=1).iloc[0, 1:].to_numpy(dtype=np.uint8)
        input_voltage = pd.read_csv(data_root / required[3], nrows=1).iloc[0, 1:].to_numpy(dtype=np.float64)
        load_current = pd.read_csv(data_root / required[4], nrows=1).iloc[0, 1:].to_numpy(dtype=np.float64)

        np.testing.assert_array_equal(
            result.cell_states,
            states.reshape(CELL_COUNT, SAMPLES_PER_CYCLE),
        )
        np.testing.assert_array_equal(
            result.switching_gates,
            gates.reshape(CELL_COUNT, 4, SAMPLES_PER_CYCLE),
        )
        np.testing.assert_array_equal(result.output_voltage, input_voltage)
        np.testing.assert_allclose(result.load_current, load_current, rtol=1e-12, atol=1e-12)
        self.assertAlmostEqual(
            FUNDAMENTAL_FREQUENCY_HZ,
            SAMPLING_FREQUENCY_HZ / SAMPLES_PER_CYCLE,
        )


if __name__ == "__main__":
    unittest.main()
