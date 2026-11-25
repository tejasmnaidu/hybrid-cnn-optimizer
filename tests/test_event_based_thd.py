from __future__ import annotations

import unittest

import numpy as np

from evaluation.event_based_thd import _nonnegative_energy, evaluate_event_based_thd
from evaluation.thd_analysis import calculate_thd
from multilevel.inverter_simulator import (
    FUNDAMENTAL_FREQUENCY_HZ,
    SAMPLING_FREQUENCY_HZ,
    SAMPLES_PER_CYCLE,
    simulate_inverter,
)


def _independent_sampled_thd(
    modulation_index: float,
    phase_angle_rad: float,
    sampling_frequency_hz: float,
    phase_offsets_degrees: tuple[float, float, float] = (0.0, 60.0, 120.0),
) -> tuple[complex, float]:
    """Independent uniform-time comparator simulation and direct FFT."""
    count = int(round(sampling_frequency_hz / FUNDAMENTAL_FREQUENCY_HZ))
    time_s = np.arange(count, dtype=np.float64) / sampling_frequency_hz
    reference = modulation_index * np.sin(
        2.0 * np.pi * FUNDAMENTAL_FREQUENCY_HZ * time_s + phase_angle_rad
    )
    voltage = np.zeros(count, dtype=np.float64)
    for phase_shift in phase_offsets_degrees:
        carrier = 1.0 - 4.0 * np.abs(
            np.mod(
                1_000.0 * time_s + phase_shift / 360.0,
                1.0,
            )
            - 0.5
        )
        upper = reference >= carrier
        lower = -reference >= carrier
        voltage += 100.0 * (upper.astype(np.int8) - lower.astype(np.int8))

    centered = voltage - np.mean(voltage)
    spectrum = np.fft.rfft(centered)
    peak_amplitudes = 2.0 * np.abs(spectrum) / count
    if count % 2 == 0:
        peak_amplitudes[-1] *= 0.5
    coefficient = 2.0 / count * np.sum(
        centered * np.exp(-2j * np.pi * np.arange(count) / count)
    )
    thd_percent = (
        np.sqrt(np.sum(np.square(peak_amplitudes[2:101])))
        / peak_amplitudes[1]
        * 100.0
    )
    return complex(coefficient), float(thd_percent)


def _independent_interval_harmonic_thd(
    result: object,
    max_harmonic: int,
    *,
    block_size: int = 128,
) -> float:
    """Sum Fourier energy using a centered-interval sinc formula."""
    starts = result.interval_start_s
    ends = result.interval_end_s
    values = result.interval_voltage_v
    cycle_s = 1.0 / FUNDAMENTAL_FREQUENCY_HZ
    durations = ends - starts
    centers = (starts + ends) * 0.5
    accumulated_energy = 0.0
    fundamental_rms = result.fundamental_peak_amplitude_v / np.sqrt(2.0)
    for first in range(2, max_harmonic + 1, block_size):
        orders = np.arange(
            first,
            min(first + block_size, max_harmonic + 1),
            dtype=np.float64,
        )
        angular_frequencies = 2.0 * np.pi * orders / cycle_s
        coefficients = np.zeros(orders.size, dtype=np.complex128)
        for duration, center, voltage in zip(durations, centers, values):
            coefficients += (
                2.0
                * voltage
                * duration
                / cycle_s
                * np.sinc(angular_frequencies * duration / (2.0 * np.pi))
                * np.exp(-1j * angular_frequencies * center)
            )
        accumulated_energy += float(np.sum(np.square(np.abs(coefficients))) / 2.0)
    return float(np.sqrt(accumulated_energy) / fundamental_rms * 100.0)


class EventBasedHarmonicTests(unittest.TestCase):
    def test_intervals_cover_exactly_one_cycle_without_gaps_or_overlaps(self) -> None:
        result = evaluate_event_based_thd(1.0, 0.37)
        cycle_s = 1.0 / FUNDAMENTAL_FREQUENCY_HZ
        self.assertEqual(result.interval_start_s[0], 0.0)
        self.assertEqual(result.interval_end_s[-1], cycle_s)
        np.testing.assert_array_equal(
            result.interval_end_s[:-1],
            result.interval_start_s[1:],
        )
        self.assertTrue(np.all(result.interval_end_s > result.interval_start_s))
        self.assertAlmostEqual(
            float(np.sum(result.interval_end_s - result.interval_start_s)),
            cycle_s,
            places=14,
        )

    def test_baseline_uses_all_seven_levels_and_fixed_phase_offsets(self) -> None:
        result = evaluate_event_based_thd(
            1.0,
            0.0,
            carrier_phase_offsets_degrees=(0.0, 60.0, 120.0),
        )
        np.testing.assert_array_equal(
            np.unique(result.interval_voltage_v),
            np.arange(-300.0, 301.0, 100.0),
        )

    def test_baseline_event_and_sampled_simulator_use_same_conventions(self) -> None:
        modulation, resistance, inductance, phase = 0.73, 23.0, 0.027, 1.1
        event = evaluate_event_based_thd(modulation, phase)
        sampled = simulate_inverter(modulation, resistance, inductance, phase)
        self.assertAlmostEqual(
            event.sampled_thd_percent,
            _independent_sampled_thd(modulation, phase, SAMPLING_FREQUENCY_HZ)[1],
            places=12,
        )
        self.assertAlmostEqual(
            event.sampled_thd_percent,
            calculate_thd(
                sampled.output_voltage,
                sampling_frequency_hz=SAMPLING_FREQUENCY_HZ,
                fundamental_frequency_hz=FUNDAMENTAL_FREQUENCY_HZ,
                max_harmonic=100,
            ),
            places=12,
        )
        # Compare values at every original sample against the existing simulator.
        interval_index = np.searchsorted(
            event.interval_end_s,
            sampled.time_s,
            side="right",
        )
        interval_index = np.minimum(interval_index, event.interval_voltage_v.size - 1)
        np.testing.assert_array_equal(
            event.interval_voltage_v[interval_index],
            sampled.output_voltage,
        )

    def test_simultaneous_events_are_coalesced_without_zero_width_intervals(self) -> None:
        result = evaluate_event_based_thd(
            0.8,
            0.43,
            carrier_phase_offsets_degrees=(0.0, 0.0, 0.0),
        )
        self.assertEqual(
            len(np.unique(result.transition_times_s)),
            result.transition_times_s.size,
        )
        self.assertTrue(np.all(np.diff(result.transition_times_s) > 0.0))
        self.assertTrue(np.all(result.interval_end_s > result.interval_start_s))
        # Identical carriers switch all three equal cells together.
        self.assertTrue(set(np.unique(result.interval_voltage_v)).issubset({-300.0, 0.0, 300.0}))

    def test_transition_at_cycle_boundary_is_represented_once(self) -> None:
        result = evaluate_event_based_thd(
            0.8,
            0.0,
            carrier_phase_offsets_degrees=(90.0, 90.0, 90.0),
        )
        self.assertEqual(result.transition_times_s[0], 0.0)
        self.assertEqual(result.transition_times_s[-1], 1.0 / FUNDAMENTAL_FREQUENCY_HZ)
        self.assertEqual(np.count_nonzero(result.transition_times_s == 0.0), 1)
        self.assertEqual(np.count_nonzero(result.transition_times_s == 1.0 / FUNDAMENTAL_FREQUENCY_HZ), 1)
        self.assertTrue(np.all(result.interval_end_s > result.interval_start_s))

    def test_fourier_coefficients_match_analytic_target_fundamental(self) -> None:
        result = evaluate_event_based_thd(0.7, 0.83)
        target_coefficient = -1j * 300.0 * 0.7 * np.exp(1j * 0.83)
        np.testing.assert_array_equal(result.harmonic_orders, np.arange(1, 101))
        np.testing.assert_allclose(
            result.harmonic_peak_amplitudes_v,
            np.abs(result.fourier_coefficients),
            rtol=0.0,
            atol=0.0,
        )
        self.assertAlmostEqual(
            result.thd_percent,
            np.sqrt(np.sum(np.square(result.harmonic_peak_amplitudes_v[1:])))
            / result.harmonic_peak_amplitudes_v[0]
            * 100.0,
        )
        np.testing.assert_allclose(
            result.fourier_coefficients[0],
            target_coefficient,
            rtol=0.0,
            atol=0.2,
        )
        self.assertLess(result.fundamental_phasor_error_percent, 0.1)
        self.assertAlmostEqual(
            result.fundamental_peak_amplitude_v,
            abs(result.fourier_coefficients[0]),
        )

    def test_high_rate_fft_tracks_event_result_with_improving_resolution(self) -> None:
        conditions = (
            (0.699816047538945, 5.97351416136022),
            (0.750524679950008, 1.17248144504243),
        )
        for modulation, phase in conditions:
            with self.subTest(modulation_index=modulation):
                result = evaluate_event_based_thd(modulation, phase)
                event_coefficient = result.fourier_coefficients[0]
                thd_errors = []
                phasor_errors = []
                for frequency in (1_000_000.0, 2_000_000.0, 5_000_000.0):
                    sampled_coefficient, sampled_thd = _independent_sampled_thd(
                        modulation,
                        phase,
                        frequency,
                    )
                    thd_errors.append(abs(sampled_thd - result.thd_percent))
                    phasor_errors.append(abs(sampled_coefficient - event_coefficient))
                self.assertLess(thd_errors[1], thd_errors[0])
                self.assertLess(thd_errors[2], thd_errors[1])
                self.assertLess(min(phasor_errors), 0.03)

    def test_event_fundamental_phase_is_wrapped_consistently_with_target(self) -> None:
        result = evaluate_event_based_thd(0.7, 5.97351416136022)
        self.assertAlmostEqual(
            result.fundamental_phase_rad,
            result.target_fundamental_phase_rad,
            delta=1e-10,
        )

    def test_parseval_total_thd_matches_convergent_independent_harmonic_sum(self) -> None:
        result = evaluate_event_based_thd(
            0.699816047538945,
            5.97351416136022,
            max_harmonic=100,
        )
        durations = result.interval_end_s - result.interval_start_s
        expected_mean = np.sum(result.interval_voltage_v * durations) / (
            1.0 / FUNDAMENTAL_FREQUENCY_HZ
        )
        expected_mean_square = np.sum(
            np.square(result.interval_voltage_v) * durations
        ) / (1.0 / FUNDAMENTAL_FREQUENCY_HZ)
        self.assertAlmostEqual(result.mean_voltage_v, expected_mean, places=10)
        self.assertAlmostEqual(result.mean_square_voltage_v2, expected_mean_square, places=8)
        self.assertAlmostEqual(
            result.ac_rms_voltage_v**2,
            result.mean_square_voltage_v2 - result.mean_voltage_v**2,
            places=8,
        )
        self.assertAlmostEqual(
            result.fundamental_rms_voltage_v,
            result.fundamental_peak_amplitude_v / np.sqrt(2.0),
        )

        harmonic_sums = [
            _independent_interval_harmonic_thd(result, limit)
            for limit in (100, 500, 2_000, 10_000, 20_000)
        ]
        self.assertLess(harmonic_sums[0], harmonic_sums[1])
        self.assertLess(harmonic_sums[1], harmonic_sums[2])
        self.assertLess(harmonic_sums[2], harmonic_sums[3])
        self.assertLess(harmonic_sums[3], harmonic_sums[4])
        self.assertLess(harmonic_sums[4], result.total_thd_percent)
        self.assertLess(result.total_thd_percent - harmonic_sums[4], 0.06)

    def test_parseval_roundoff_is_clipped_but_material_negative_energy_fails(self) -> None:
        self.assertEqual(_nonnegative_energy(-np.finfo(np.float64).eps, 1.0, "test"), 0.0)
        with self.assertRaisesRegex(ValueError, "materially negative"):
            _nonnegative_energy(-1e-6, 1.0, "test")

    def test_parseval_metric_for_baseline_and_alternate_phase_configurations(self) -> None:
        configurations = (
            (0.70, 0.83, (0.0, 60.0, 120.0)),
            (0.82, 2.4, (17.0, 143.0, 281.0)),
            (0.60, 0.4, (0.0, 0.0, 0.0)),
            (0.80, 0.0, (90.0, 90.0, 90.0)),
        )
        for modulation, phase, offsets in configurations:
            with self.subTest(offsets=offsets):
                result = evaluate_event_based_thd(
                    modulation,
                    phase,
                    carrier_phase_offsets_degrees=offsets,
                )
                self.assertGreater(result.total_thd_percent, 0.0)
                self.assertGreaterEqual(
                    result.total_thd_percent + 1e-10,
                    result.thd_percent,
                )
                independently_summed = _independent_interval_harmonic_thd(
                    result,
                    5_000,
                )
                self.assertLess(independently_summed, result.total_thd_percent)
                self.assertLess(
                    result.total_thd_percent - independently_summed,
                    0.5,
                )

    def test_nonbaseline_phases_match_independent_high_rate_sampling(self) -> None:
        phases = (17.0, 143.0, 281.0)
        result = evaluate_event_based_thd(
            0.82,
            2.4,
            carrier_phase_offsets_degrees=phases,
        )
        sampled_coefficient, sampled_thd = _independent_sampled_thd(
            0.82,
            2.4,
            5_000_000.0,
            phases,
        )
        self.assertLess(
            abs(sampled_coefficient - result.fourier_coefficients[0]),
            0.3,
        )
        self.assertLess(abs(sampled_thd - result.thd_percent), 0.15)

    def test_invalid_modulation_offsets_and_harmonic_count_are_rejected(self) -> None:
        for modulation in (0.0, -0.1, 1.01, float("nan")):
            with self.subTest(modulation=modulation), self.assertRaises(ValueError):
                evaluate_event_based_thd(modulation, 0.0)
        with self.assertRaises(ValueError):
            evaluate_event_based_thd(0.7, 0.0, carrier_phase_offsets_degrees=(0.0, 60.0))
        with self.assertRaises(ValueError):
            evaluate_event_based_thd(0.7, 0.0, max_harmonic=1)


if __name__ == "__main__":
    unittest.main()
