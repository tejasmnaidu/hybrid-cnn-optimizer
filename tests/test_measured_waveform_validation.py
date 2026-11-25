from __future__ import annotations

import unittest

import numpy as np

from evaluation.measured_waveform_validation import compare_measured_voltage_csv
from multilevel.inverter_simulator import SAMPLING_FREQUENCY_HZ


def _csv_bytes(
    times: np.ndarray,
    voltage: np.ndarray,
    *,
    header: str = "time_s,voltage_v",
) -> bytes:
    rows = [header]
    rows.extend(f"{time_value:.12g},{voltage_value:.12g}" for time_value, voltage_value in zip(times, voltage))
    return ("\n".join(rows) + "\n").encode("utf-8")


class MeasuredWaveformValidationTests(unittest.TestCase):
    def test_compares_shared_harmonic_band_without_touching_model_data(self) -> None:
        sample_rate = 20_000.0
        fundamental = 50.0
        simulated_time = np.arange(400) / sample_rate
        simulated = (
            100.0 * np.sin(2.0 * np.pi * fundamental * simulated_time)
            + 10.0 * np.sin(2.0 * np.pi * 3.0 * fundamental * simulated_time)
        )
        measured_time = np.arange(800) / sample_rate
        measured = (
            110.0 * np.sin(2.0 * np.pi * fundamental * measured_time)
            + 22.0 * np.sin(2.0 * np.pi * 3.0 * fundamental * measured_time)
        )

        result = compare_measured_voltage_csv(
            _csv_bytes(measured_time, measured),
            simulated,
            simulated_sampling_frequency_hz=sample_rate,
            fundamental_frequency_hz=fundamental,
            simulated_fundamental_frequency_hz=fundamental,
        )

        self.assertEqual(result.sample_count, 800)
        self.assertEqual(result.cycle_count, 2)
        self.assertEqual(result.max_harmonic, 100)
        self.assertAlmostEqual(result.sampling_frequency_hz, sample_rate)
        self.assertAlmostEqual(result.measured_thd_percent, 20.0)
        self.assertAlmostEqual(result.simulated_thd_percent, 10.0)
        self.assertAlmostEqual(result.thd_difference_percentage_points, 10.0)
        self.assertAlmostEqual(result.fundamental_amplitude_change_percent, 10.0)
        self.assertEqual(result.measured_spectrum.harmonic_orders[-1], 100)

    def test_drops_duplicate_endpoint_and_uses_shared_nyquist_band(self) -> None:
        sample_rate = 2_000.0
        fundamental = 50.0
        time_values = np.linspace(0.0, 0.04, 81)
        measured = np.sin(2.0 * np.pi * fundamental * time_values)
        simulated_time = np.arange(400) / SAMPLING_FREQUENCY_HZ
        simulated = np.sin(2.0 * np.pi * fundamental * simulated_time)

        result = compare_measured_voltage_csv(
            _csv_bytes(time_values, measured),
            simulated,
            simulated_sampling_frequency_hz=SAMPLING_FREQUENCY_HZ,
            fundamental_frequency_hz=fundamental,
            simulated_fundamental_frequency_hz=fundamental,
        )

        self.assertEqual(result.sample_count, 80)
        self.assertEqual(result.cycle_count, 2)
        self.assertEqual(result.max_harmonic, 20)
        self.assertAlmostEqual(result.measured_thd_percent, 0.0, places=10)

    def test_rejects_bad_schema_sampling_and_non_integer_cycles(self) -> None:
        sample_rate = 20_000.0
        time_values = np.arange(800) / sample_rate
        voltage = np.sin(2.0 * np.pi * 50.0 * time_values)
        simulated = np.sin(2.0 * np.pi * 50.0 * np.arange(400) / sample_rate)
        common_arguments = {
            "simulated_sampling_frequency_hz": sample_rate,
            "fundamental_frequency_hz": 50.0,
            "simulated_fundamental_frequency_hz": 50.0,
        }

        with self.assertRaisesRegex(ValueError, "columns named"):
            compare_measured_voltage_csv(
                _csv_bytes(time_values, voltage, header="time,voltage"),
                simulated,
                **common_arguments,
            )
        with self.assertRaisesRegex(ValueError, "integer number"):
            compare_measured_voltage_csv(
                _csv_bytes(time_values[:-10], voltage[:-10]),
                simulated,
                **common_arguments,
            )
        uneven_time = time_values.copy()
        uneven_time[200] += 0.1 / sample_rate
        with self.assertRaisesRegex(ValueError, "uniformly spaced"):
            compare_measured_voltage_csv(
                _csv_bytes(uneven_time, voltage),
                simulated,
                **common_arguments,
            )

    def test_rejects_invalid_fundamental_or_too_low_sample_rate(self) -> None:
        sample_rate = 20_000.0
        simulated = np.sin(2.0 * np.pi * 50.0 * np.arange(400) / sample_rate)
        time_values = np.arange(800) / sample_rate
        voltage = np.sin(2.0 * np.pi * 50.0 * time_values)
        with self.assertRaisesRegex(ValueError, "fundamental_frequency_hz"):
            compare_measured_voltage_csv(
                _csv_bytes(time_values, voltage),
                simulated,
                simulated_sampling_frequency_hz=sample_rate,
                fundamental_frequency_hz=0.0,
                simulated_fundamental_frequency_hz=50.0,
            )
        low_rate = 150.0
        low_rate_time = np.arange(6) / low_rate
        with self.assertRaisesRegex(ValueError, "second harmonic"):
            compare_measured_voltage_csv(
                _csv_bytes(
                    low_rate_time,
                    np.sin(2.0 * np.pi * 50.0 * low_rate_time),
                ),
                simulated,
                simulated_sampling_frequency_hz=sample_rate,
                fundamental_frequency_hz=50.0,
                simulated_fundamental_frequency_hz=50.0,
            )


if __name__ == "__main__":
    unittest.main()
