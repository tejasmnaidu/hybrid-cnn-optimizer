from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from matplotlib.axes import Axes
from evaluation.dashboard_helpers import (
    LIVE_PRESETS,
    advance_playback,
    cnn_guidance_operating_point_matches,
    gate_vector_for_state,
    live_configuration_matches,
    normalize_live_conditions,
    normalize_carrier_phase_offsets,
    optimizer_baseline_supported,
    playback_snapshot,
    synchronized_playback_index,
    thd_surrogate_operating_point_supported,
    verify_cell_voltage_sum,
)
from multilevel.inverter_simulator import simulate_inverter


class DashboardHelperTests(unittest.TestCase):
    def test_presets_are_valid_physical_values(self) -> None:
        self.assertGreaterEqual(len(LIVE_PRESETS), 5)
        for preset in LIVE_PRESETS:
            conditions = normalize_live_conditions(
                preset.modulation_index,
                preset.dc_voltage_per_cell_v,
                preset.load_resistance_ohm,
                preset.load_inductance_h,
                preset.phase_angle_degrees,
            )
            self.assertEqual(len(conditions), 5)

    def test_live_condition_validation_rejects_bad_inputs(self) -> None:
        with self.assertRaises(ValueError):
            normalize_live_conditions(0.50, 100.0, 30.0, 0.03, 0.0)
        with self.assertRaises(ValueError):
            normalize_live_conditions(0.75, 0.0, 30.0, 0.03, 0.0)

    def test_playback_advances_only_when_running(self) -> None:
        self.assertEqual(advance_playback(399, "running", 400), 0)
        self.assertEqual(advance_playback(390, "running", 400, step=25), 15)
        self.assertEqual(advance_playback(123, "paused", 400), 123)
        self.assertEqual(advance_playback(123, "stopped", 400), 123)
        with self.assertRaises(ValueError):
            advance_playback(0, "running", 400, step=0)

    def test_playback_widget_and_fragment_indices_stay_synchronized(self) -> None:
        state = {
            "live_playback_index": 23,
            "live_workspace_cursor_index": 91,
        }
        self.assertEqual(synchronized_playback_index(state, "paused", 400), 91)
        self.assertEqual(state["live_playback_index"], 91)
        state["live_workspace_cursor_index"] = 123
        self.assertEqual(
            synchronized_playback_index(
                state,
                "running",
                400,
                manual_selection=True,
            ),
            123,
        )
        self.assertEqual(state["live_playback_index"], 123)
        self.assertEqual(
            synchronized_playback_index(state, "running", 400, step=10),
            133,
        )
        self.assertEqual(state["live_workspace_cursor_index"], 133)

    def test_live_configuration_comparison_includes_all_carrier_shifts(self) -> None:
        result = SimpleNamespace(
            modulation_index=0.75,
            dc_voltage_per_cell_v=100.0,
            load_resistance_ohm=30.0,
            load_inductance_h=0.03,
            phase_angle_degrees=90.0,
            carrier_phase_offsets_degrees=(0.0, 60.0, 120.0),
        )
        conditions = (0.75, 100.0, 30.0, 0.03, 1.5707963267948966)
        self.assertTrue(
            live_configuration_matches(result, conditions, (360.0, 60.0, 120.0))
        )
        for index, changed_value in enumerate((1.0, 61.0, 121.0)):
            changed_offsets = [0.0, 60.0, 120.0]
            changed_offsets[index] = changed_value
            with self.subTest(carrier_index=index):
                self.assertFalse(
                    live_configuration_matches(
                        result,
                        conditions,
                        tuple(changed_offsets),
                    )
                )
        for index, changed_value in enumerate(
            (0.76, 110.0, 40.0, 0.05, 2.0)
        ):
            changed_conditions = list(conditions)
            changed_conditions[index] = changed_value
            with self.subTest(condition_index=index):
                self.assertFalse(
                    live_configuration_matches(
                        result,
                        tuple(changed_conditions),
                        (0.0, 60.0, 120.0),
                    )
                )
        self.assertEqual(
            normalize_carrier_phase_offsets((360.0, -300.0, 480.0)),
            (0.0, 60.0, 120.0),
        )
        self.assertTrue(
            optimizer_baseline_supported(
                (360.0, 60.0, 120.0),
                (0.0, 60.0, 120.0),
            )
        )
        self.assertFalse(
            optimizer_baseline_supported(
                (0.0, 61.0, 120.0),
                (0.0, 60.0, 120.0),
            )
        )

    def test_cnn_guidance_is_limited_to_validated_conditions_and_baseline(self) -> None:
        validated_point = {
            "modulation_index": 0.75,
            "dc_voltage_per_cell_v": 100.0,
            "load_resistance_ohm": 30.0,
            "load_inductance_h": 0.03,
            "phase_angle_degrees": 0.0,
        }
        self.assertTrue(
            cnn_guidance_operating_point_matches(
                (0.75, 100.0, 30.0, 0.03, 0.0),
                validated_point,
                (0.0, 60.0, 120.0),
                (0.0, 60.0, 120.0),
            )
        )
        self.assertFalse(
            cnn_guidance_operating_point_matches(
                (0.85, 100.0, 30.0, 0.03, 0.0),
                validated_point,
                (0.0, 60.0, 120.0),
                (0.0, 60.0, 120.0),
            )
        )
        self.assertFalse(
            cnn_guidance_operating_point_matches(
                (0.75, 100.0, 30.0, 0.03, 0.0),
                validated_point,
                (0.0, 55.0, 130.0),
                (0.0, 60.0, 120.0),
            )
        )

    def test_thd_surrogate_scope_is_limited_to_benchmarked_points(self) -> None:
        baseline = (0.0, 60.0, 120.0)
        self.assertTrue(
            thd_surrogate_operating_point_supported(
                0.75,
                100.0,
                360.0,
                baseline,
                baseline,
            )
        )
        for values in (
            (0.76, 100.0, 0.0, baseline),
            (0.75, 110.0, 0.0, baseline),
            (0.75, 100.0, 1.0, baseline),
            (0.75, 100.0, 0.0, (0.0, 59.0, 120.0)),
        ):
            with self.subTest(values=values):
                self.assertFalse(
                    thd_surrogate_operating_point_supported(
                        *values,
                        baseline,
                    )
                )

    def test_snapshot_reads_actual_simulator_arrays(self) -> None:
        result = simulate_inverter(
            modulation_index=0.75,
            load_resistance_ohm=30.0,
            load_inductance_h=0.03,
            phase_angle_rad=0.0,
        )
        snapshot = playback_snapshot(result, 123)
        self.assertEqual(snapshot.output_voltage_v, float(result.output_voltage[123]))
        self.assertEqual(snapshot.load_current_a, float(result.load_current[123]))
        self.assertEqual(snapshot.cell_states, tuple(int(value) for value in result.cell_states[:, 123]))
        self.assertEqual(
            snapshot.switching_gates,
            tuple(
                tuple(int(value) for value in result.switching_gates[cell_index, :, 123])
                for cell_index in range(3)
            ),
        )

    def test_gate_mapping_and_every_sample_voltage_sum(self) -> None:
        result = simulate_inverter(
            modulation_index=0.85,
            load_resistance_ohm=30.0,
            load_inductance_h=0.03,
            phase_angle_rad=0.0,
        )
        self.assertEqual(gate_vector_for_state(1), (1, 0, 0, 1))
        self.assertEqual(gate_vector_for_state(0), (0, 1, 0, 1))
        self.assertEqual(gate_vector_for_state(-1), (0, 1, 1, 0))
        for index in range(result.output_voltage.size):
            sum_voltage, output_voltage, matches = verify_cell_voltage_sum(result, index)
            self.assertTrue(matches)
            self.assertEqual(sum_voltage, output_voltage)

    def test_cursor_index_bounds_and_session_state_shape(self) -> None:
        from streamlit.testing.v1 import AppTest

        app_path = Path(__file__).resolve().parents[1] / "app.py"
        at = AppTest.from_file(str(app_path), default_timeout=30)
        at.run()
        self.assertFalse(at.exception)
        self.assertEqual(at.tabs[0].label, "LIVE SIMULATION WORKSPACE")
        run_buttons = [button for button in at.button if button.label == "Run New Simulation"]
        self.assertEqual(len(run_buttons), 1)
        plotted_labels: list[str] = []
        original_plot = Axes.plot

        def capture_plot_labels(axis, *args, **kwargs):
            plotted_labels.append(str(kwargs.get("label", "")))
            return original_plot(axis, *args, **kwargs)

        with patch.object(Axes, "plot", new=capture_plot_labels):
            run_buttons[0].click()
            at.run()
        self.assertFalse(at.exception)
        self.assertTrue(
            any(selectbox.key == "live_playback_rate" for selectbox in at.selectbox)
        )
        self.assertTrue(
            any(
                "Accelerated replay of this precomputed" in caption.value
                for caption in at.caption
            )
        )
        self.assertIn("Inverter output voltage", plotted_labels)
        self.assertIn("Simulated PWM output − analytical reference", plotted_labels)
        self.assertIn(
            "SIMULATED OUTPUT − REFERENCE RMS",
            {metric.label for metric in at.metric},
        )
        self.assertIn(
            "PEAK ABSOLUTE TRACKING ERROR",
            {metric.label for metric in at.metric},
        )
        held_out_metrics = {
            metric.label: metric.value for metric in at.metric
        }
        self.assertEqual(
            held_out_metrics["CNN HELD-OUT POOLED RMSE"],
            "1.6223 V",
        )
        self.assertEqual(
            held_out_metrics["ANALYTICAL HELD-OUT POOLED RMSE"],
            "0.000018 V",
        )
        self.assertTrue(
            any(
                "less accurate than the analytical reference" in warning.value
                for warning in at.warning
            )
        )
        self.assertNotIn("CNN predicted target", plotted_labels)
        live_optimizer = next(
            selectbox
            for selectbox in at.selectbox
            if selectbox.key == "live_optimizer_name"
        )
        live_optimizer.set_value("Hybrid POA-PFO")
        at.run()
        optimization_button = next(
            button
            for button in at.button
            if button.label == "RUN LIVE OPTIMIZATION"
        )
        optimization_button.click()
        at.run()
        self.assertFalse(at.exception)
        optimization_metric_labels = {metric.label for metric in at.metric}
        self.assertIn("BASELINE OBJECTIVE", optimization_metric_labels)
        self.assertIn("HYBRID POA-PFO OBJECTIVE", optimization_metric_labels)
        self.assertIn(
            "OBJECTIVE CHANGE (CANDIDATE − BASELINE)",
            optimization_metric_labels,
        )
        self.assertTrue(
            any(
                "CNN predicts the reference waveform" in message.value
                for message in at.info
            )
        )
        guided_button = next(
            button
            for button in at.button
            if button.label == "RUN CNN-GUIDED HYBRID"
        )
        self.assertFalse(guided_button.disabled)
        guided_button.click()
        at.run()
        self.assertFalse(at.exception)
        guided_result = at.session_state["live_demo_cnn_guided_optimization"]
        guided_candidate = at.session_state["live_demo_cnn_guided_candidate"]
        self.assertEqual(guided_result["unique_candidate_evaluations"], 64)
        self.assertEqual(guided_result["full_thd_evaluations"], 9)
        self.assertAlmostEqual(
            guided_candidate.harmonic_result.total_thd_percent,
            guided_result["event_thd"],
            places=8,
        )
        self.assertLessEqual(
            guided_candidate.harmonic_result.fundamental_phasor_error_percent,
            0.05,
        )
        if "Event-THD surrogate shortlist" in live_optimizer.options:
            next(
                selectbox
                for selectbox in at.selectbox
                if selectbox.key == "live_optimizer_name"
            ).set_value("Event-THD surrogate shortlist")
            at.run()
            self.assertEqual(
                next(
                    selectbox
                    for selectbox in at.selectbox
                    if selectbox.key == "live_optimizer_name"
                ).value,
                "Event-THD surrogate shortlist",
            )
            next(
                button
                for button in at.button
                if button.label == "RUN LIVE OPTIMIZATION"
            ).click()
            at.run()
            self.assertFalse(at.exception)
            surrogate_optimization = at.session_state["live_demo_optimization"]
            surrogate_candidate = at.session_state["live_demo_candidate"]
            self.assertEqual(
                surrogate_optimization.run_result.screened_candidate_count,
                64,
            )
            self.assertLessEqual(
                surrogate_optimization.run_result.unique_candidate_evaluations,
                8,
            )
            self.assertTrue(surrogate_optimization.run_result.best.feasible)
            self.assertAlmostEqual(
                surrogate_candidate.harmonic_result.total_thd_percent,
                surrogate_optimization.run_result.best.mean_total_thd_percent,
                places=8,
            )
        self.assertIn(
            "CNN-GUIDED EVENT-BASED PARSEVAL THD",
            {metric.label for metric in at.metric},
        )
        self.assertIn(
            "CNN-GUIDED FUNDAMENTAL TRACKING ERROR",
            {metric.label for metric in at.metric},
        )
        advisor_button = next(
            button
            for button in at.button
            if button.label == "GENERATE VERIFIED RECOMMENDATION"
        )
        advisor_button.click()
        at.run()
        self.assertFalse(at.exception)
        metric_labels = {metric.label for metric in at.metric}
        self.assertIn("BASELINE EVENT TOTAL VOLTAGE THD", metric_labels)
        self.assertIn(
            "SIMULATOR-VERIFIED EVENT TOTAL VOLTAGE THD",
            metric_labels,
        )
        self.assertFalse(
            any("VERIFIED VERIFIED" in label for label in metric_labels)
        )
        self.assertTrue(
            any(
                "out-of-band switching energy" in caption.value
                for caption in at.caption
            )
        )
        prediction_overlay = next(
            checkbox
            for checkbox in at.checkbox
            if checkbox.label == "Show CNN prediction overlay"
        )
        self.assertFalse(prediction_overlay.value)
        plotted_labels.clear()
        prediction_overlay = next(
            checkbox
            for checkbox in at.checkbox
            if checkbox.label == "Show CNN prediction overlay"
        )
        with patch.object(Axes, "plot", new=capture_plot_labels):
            prediction_overlay.set_value(True).run()
        self.assertFalse(at.exception)
        self.assertIn("CNN predicted target", plotted_labels)
        self.assertTrue(any(slider.label == "MODULATION INDEX" for slider in at.slider))
        stale_result = at.session_state["live_demo_result"]
        modulation_slider = next(
            slider for slider in at.slider if slider.label == "MODULATION INDEX"
        )
        modulation_slider.set_value(0.80).run()
        self.assertFalse(at.exception)
        self.assertIs(at.session_state["live_demo_result"], stale_result)
        guided_button = next(
            button
            for button in at.button
            if button.label == "RUN CNN-GUIDED HYBRID"
        )
        self.assertTrue(guided_button.disabled)
        self.assertTrue(
            any(
                info.value
                == "Controls changed since this run. Click Run New Simulation to update."
                for info in at.info
            )
        )
        next(
            button for button in at.button if button.label == "Run New Simulation"
        ).click().run()
        self.assertFalse(at.exception)
        refreshed_result = at.session_state["live_demo_result"]
        self.assertIsNot(refreshed_result, stale_result)
        self.assertAlmostEqual(refreshed_result.modulation_index, 0.80)
        self.assertFalse(
            any(
                info.value
                == "Controls changed since this run. Click Run New Simulation to update."
                for info in at.info
            )
        )
        at.button(key="live_playback_start").click().run()
        sample_slider = next(
            slider for slider in at.slider if slider.label == "Sample index"
        )
        sample_slider.set_value(123).run()
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state["live_workspace_cursor_index"], 123)
        at.run()
        self.assertEqual(at.session_state["live_workspace_cursor_index"], 133)
