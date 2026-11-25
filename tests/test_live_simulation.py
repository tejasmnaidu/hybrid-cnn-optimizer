from __future__ import annotations

import unittest

import numpy as np
import torch
from torch import nn

from evaluation.live_simulation import (
    operating_condition_support_warnings,
    run_live_simulation,
)
from preprocessing.preprocess import NormalizationStats


def _normalization() -> NormalizationStats:
    return NormalizationStats(
        input_voltage_mean=10.0,
        input_voltage_std=100.0,
        load_current_mean=1.0,
        load_current_std=2.0,
        operating_mean=[0.7, 20.0, 0.02, 0.5],
        operating_std=[0.1, 10.0, 0.01, 2.0],
        target_voltage_mean=5.0,
        target_voltage_std=2.0,
    )


class EchoVoltageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inference_mode_seen = False
        self.input_shapes: tuple[torch.Size, ...] | None = None
        self.conditions: np.ndarray | None = None

    def forward(
        self,
        input_voltage: torch.Tensor,
        load_current: torch.Tensor,
        operating_conditions: torch.Tensor,
    ) -> torch.Tensor:
        self.inference_mode_seen = torch.is_inference_mode_enabled()
        self.input_shapes = (
            input_voltage.shape,
            load_current.shape,
            operating_conditions.shape,
        )
        self.conditions = operating_conditions.detach().cpu().numpy()
        return input_voltage


class LiveSimulationTests(unittest.TestCase):
    def _run(self, model: nn.Module | None = None):
        return run_live_simulation(
            model or EchoVoltageModel(),
            _normalization(),
            modulation_index=0.75,
            load_resistance_ohm=30,
            load_inductance_h=0.03,
            phase_angle_degrees=60,
            device="cpu",
        )

    def test_runs_real_simulator_harmonic_evaluator_and_cpu_cnn_inference(self) -> None:
        model = EchoVoltageModel()
        result = self._run(model)

        self.assertEqual(result.simulation.output_voltage.shape, (400,))
        self.assertEqual(result.simulation.cell_states.shape, (3, 400))
        self.assertEqual(result.simulation.switching_gates.shape, (3, 4, 400))
        self.assertTrue(set(np.unique(result.simulation.output_voltage)).issubset(
            {-300.0, -200.0, -100.0, 0.0, 100.0, 200.0, 300.0}
        ))
        self.assertEqual(model.input_shapes[0], torch.Size((1, 400)))
        self.assertEqual(model.input_shapes[1], torch.Size((1, 400)))
        self.assertEqual(model.input_shapes[2], torch.Size((1, 4)))
        self.assertTrue(model.inference_mode_seen)
        self.assertTrue(np.isfinite(result.cnn_prediction_voltage).all())
        self.assertEqual(result.target_reference_voltage.shape, (400,))
        np.testing.assert_allclose(
            result.target_reference_voltage,
            result.simulation.reference * 300.0,
        )
        stats = _normalization()
        expected_conditions = (
            np.asarray([0.75, 30.0, 0.03, np.deg2rad(60.0)]) - stats.operating_mean
        ) / stats.operating_std
        np.testing.assert_allclose(model.conditions[0], expected_conditions, rtol=1e-6)
        expected_prediction = (
            (result.simulation.output_voltage - stats.input_voltage_mean)
            / stats.input_voltage_std
            * stats.target_voltage_std
            + stats.target_voltage_mean
        )
        np.testing.assert_allclose(
            result.cnn_prediction_voltage,
            expected_prediction,
            rtol=1e-6,
        )
        self.assertTrue(np.isfinite(result.harmonic_result.total_thd_percent))
        self.assertTrue(np.isfinite(result.cnn_metrics["rmse"]))
        self.assertEqual(result.phase_angle_degrees, 60.0)

    def test_restores_model_training_state(self) -> None:
        model = EchoVoltageModel()
        model.train()
        self._run(model)
        self.assertTrue(model.training)

    def test_rejects_out_of_range_or_unsupported_operating_conditions(self) -> None:
        for field, value in (
            ("modulation_index", 0.5),
            ("load_resistance_ohm", 15),
            ("load_inductance_h", 0.04),
            ("phase_angle_degrees", 361),
            ("phase_angle_degrees", float("nan")),
        ):
            arguments = {
                "modulation_index": 0.75,
                "load_resistance_ohm": 30,
                "load_inductance_h": 0.03,
                "phase_angle_degrees": 60,
            }
            arguments[field] = value
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError):
                    run_live_simulation(
                        EchoVoltageModel(),
                        _normalization(),
                        **arguments,
                        device="cpu",
                    )

    def test_rejects_invalid_cnn_output(self) -> None:
        class WrongShape(nn.Module):
            def forward(self, input_voltage, load_current, operating_conditions):
                return input_voltage[:, :-1]

        class NonFinite(nn.Module):
            def forward(self, input_voltage, load_current, operating_conditions):
                return torch.full_like(input_voltage, float("nan"))

        for model in (WrongShape(), NonFinite()):
            with self.subTest(model=type(model).__name__):
                with self.assertRaises(ValueError):
                    self._run(model)

    def test_constant_finite_prediction_has_defined_error_metrics(self) -> None:
        class ConstantOutput(nn.Module):
            def forward(self, input_voltage, load_current, operating_conditions):
                return torch.zeros_like(input_voltage)

        result = self._run(ConstantOutput())
        self.assertTrue(np.isfinite(result.cnn_metrics["mae"]))
        self.assertTrue(np.isfinite(result.cnn_metrics["rmse"]))
        self.assertTrue(np.isfinite(result.cnn_metrics["r_squared"]))

    def test_custom_carrier_phases_reach_simulator_and_event_evaluator(self) -> None:
        offsets = (0.0, 45.0, 135.0)
        result = run_live_simulation(
            EchoVoltageModel(),
            _normalization(),
            modulation_index=0.75,
            load_resistance_ohm=30.0,
            load_inductance_h=0.03,
            phase_angle_degrees=60.0,
            device="cpu",
            carrier_phase_offsets_degrees=offsets,
        )
        self.assertEqual(result.carrier_phase_offsets_degrees, offsets)
        self.assertFalse(
            np.array_equal(
                result.simulation.cell_states,
                self._run().simulation.cell_states,
            )
        )
        from evaluation.event_based_thd import evaluate_event_based_thd

        expected = evaluate_event_based_thd(
            0.75,
            np.deg2rad(60.0),
            carrier_phase_offsets_degrees=offsets,
        )
        self.assertAlmostEqual(
            result.harmonic_result.total_thd_percent,
            expected.total_thd_percent,
        )

    def test_identical_live_inputs_reproduce_identical_simulation(self) -> None:
        first = self._run()
        second = self._run()

        np.testing.assert_array_equal(
            first.simulation.output_voltage,
            second.simulation.output_voltage,
        )
        np.testing.assert_array_equal(
            first.simulation.load_current,
            second.simulation.load_current,
        )
        self.assertEqual(
            first.harmonic_result.total_thd_percent,
            second.harmonic_result.total_thd_percent,
        )

    def test_modulation_and_carrier_changes_change_simulated_waveform(self) -> None:
        baseline = self._run()
        changed_modulation = run_live_simulation(
            EchoVoltageModel(),
            _normalization(),
            modulation_index=0.85,
            load_resistance_ohm=30.0,
            load_inductance_h=0.03,
            phase_angle_degrees=60.0,
            device="cpu",
        )
        changed_carriers = run_live_simulation(
            EchoVoltageModel(),
            _normalization(),
            modulation_index=0.75,
            load_resistance_ohm=30.0,
            load_inductance_h=0.03,
            phase_angle_degrees=60.0,
            device="cpu",
            carrier_phase_offsets_degrees=(0.0, 35.0, 155.0),
        )

        self.assertFalse(
            np.array_equal(
                baseline.simulation.output_voltage,
                changed_modulation.simulation.output_voltage,
            )
        )
        self.assertFalse(
            np.array_equal(
                baseline.simulation.output_voltage,
                changed_carriers.simulation.output_voltage,
            )
        )
        self.assertNotEqual(
            baseline.harmonic_result.total_thd_percent,
            changed_modulation.harmonic_result.total_thd_percent,
        )
        self.assertNotEqual(
            baseline.harmonic_result.total_thd_percent,
            changed_carriers.harmonic_result.total_thd_percent,
        )

    def test_dc_voltage_flows_through_reference_and_current_harmonic_metrics(self) -> None:
        result = run_live_simulation(
            EchoVoltageModel(),
            _normalization(),
            modulation_index=0.75,
            dc_voltage_per_cell_v=120.0,
            load_resistance_ohm=30.0,
            load_inductance_h=0.03,
            phase_angle_degrees=60.0,
            device="cpu",
        )
        self.assertEqual(result.dc_voltage_per_cell_v, 120.0)
        np.testing.assert_allclose(
            result.target_reference_voltage,
            result.simulation.reference * 360.0,
        )
        self.assertEqual(result.harmonic_result.target_fundamental_peak_amplitude_v, 270.0)
        self.assertTrue(np.isfinite(result.current_thd_percent))
        self.assertEqual(result.current_harmonic_spectrum.harmonic_orders[0], 1)

    def test_training_marginal_support_warning_respects_categories_and_phase_wrap(self) -> None:
        feature_names = (
            "modulationIndex",
            "loadResistance",
            "loadInductance",
            "phaseAngle",
        )
        training = np.asarray(
            [
                [0.6, 10.0, 0.01, 0.01],
                [0.9, 20.0, 0.03, 2.0],
                [0.7, 30.0, 0.05, 4.0],
                [0.8, 40.0, 0.02, 6.27],
            ]
        )
        inside = np.asarray([0.75, 20.0, 0.03, 2 * np.pi])
        self.assertEqual(
            operating_condition_support_warnings(inside, training, feature_names),
            (),
        )
        outside = np.asarray([0.95, 50.0, 0.04, 0.5])
        warnings = operating_condition_support_warnings(outside, training, feature_names)
        self.assertEqual(len(warnings), 3)
        self.assertTrue(any("modulationIndex" in warning for warning in warnings))
        self.assertTrue(any("loadResistance" in warning for warning in warnings))
        self.assertTrue(any("loadInductance" in warning for warning in warnings))


if __name__ == "__main__":
    unittest.main()
