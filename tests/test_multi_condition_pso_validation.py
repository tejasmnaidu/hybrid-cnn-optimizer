from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from evaluation.multi_condition_pso_validation import (
    ValidationOperatingCondition,
    _evaluate_pair,
    _load_condition_records,
    _statistics,
)


def _harmonic_result(
    *,
    thd: float,
    fundamental: float = 225.0,
    tracking: float = 0.0,
) -> SimpleNamespace:
    amplitudes = np.zeros(100, dtype=np.float64)
    amplitudes[0] = fundamental
    amplitudes[1] = thd
    return SimpleNamespace(
        total_thd_percent=thd,
        fundamental_peak_amplitude_v=fundamental,
        target_fundamental_peak_amplitude_v=225.0,
        fundamental_phasor_error_percent=tracking,
        fundamental_phase_rad=0.0,
        harmonic_orders=np.arange(1, 101, dtype=np.int64),
        harmonic_peak_amplitudes_v=amplitudes,
    )


class MultiConditionPSOValidationTests(unittest.TestCase):
    def test_validation_loader_excludes_training_and_test_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "Operating_Conditions.csv"
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "sampleID",
                        "modulationIndex",
                        "loadResistance",
                        "loadInductance",
                        "phaseAngle",
                    ],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "sampleID": 1,
                            "modulationIndex": 0.6,
                            "loadResistance": 10,
                            "loadInductance": 0.01,
                            "phaseAngle": 0.0,
                        },
                        {
                            "sampleID": 2,
                            "modulationIndex": 0.7,
                            "loadResistance": 20,
                            "loadInductance": 0.02,
                            "phaseAngle": 1.0,
                        },
                        {
                            "sampleID": 3,
                            "modulationIndex": 0.9,
                            "loadResistance": 40,
                            "loadInductance": 0.05,
                            "phaseAngle": 2.0,
                        },
                    ]
                )
            records = _load_condition_records(
                path,
                {1: "training", 2: "validation", 3: "testing"},
                split="validation",
            )
        self.assertEqual([record.sample_id for record in records], [2])

    def test_paired_evaluation_uses_same_condition_and_phase_pair(self) -> None:
        condition = ValidationOperatingCondition(
            sample_id=42,
            modulation_index=0.75,
            load_resistance_ohm=30.0,
            load_inductance_h=0.03,
            phase_angle_rad=0.5,
        )
        calls: list[tuple[float, float, tuple[float, ...]]] = []

        def fake_evaluator(modulation, phase, *, carrier_phase_offsets_degrees):
            offsets = tuple(carrier_phase_offsets_degrees)
            calls.append((modulation, phase, offsets))
            return _harmonic_result(thd=5.0 if offsets == (0.0, 60.0, 120.0) else 4.0)

        rows = _evaluate_pair((condition,), (70.0, 130.0), evaluator=fake_evaluator)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][:2], calls[1][:2])
        self.assertEqual(calls[0][2], (0.0, 60.0, 120.0))
        self.assertEqual(calls[1][2], (0.0, 70.0, 130.0))
        self.assertAlmostEqual(rows[0]["thd_change_percentage_points"], -1.0)
        self.assertAlmostEqual(rows[0]["relative_thd_change_percent"], -20.0)
        self.assertIn("baseline_harmonic_2_peak_v", rows[0])
        self.assertIn("candidate_harmonic_100_peak_v", rows[0])

    def test_summary_classifies_changes_and_reports_tracking(self) -> None:
        def row(baseline, candidate, tracking=0.01):
            return {
                "baseline_full_spectrum_parseval_thd_percent": baseline,
                "candidate_full_spectrum_parseval_thd_percent": candidate,
                "thd_change_percentage_points": candidate - baseline,
                "relative_thd_change_percent": (candidate / baseline - 1.0) * 100.0,
                "baseline_fundamental_peak_v": 225.0,
                "candidate_fundamental_peak_v": 224.99,
                "target_fundamental_peak_v": 225.0,
                "baseline_fundamental_phasor_tracking_error_percent": 0.0,
                "candidate_fundamental_phasor_tracking_error_percent": tracking,
                "baseline_fundamental_amplitude_error_percent": 0.0,
                "candidate_fundamental_amplitude_error_percent": 0.01 / 225 * 100,
            }

        summary = _statistics([row(10.0, 9.0), row(10.0, 10.0), row(10.0, 11.0)])
        self.assertEqual(summary["improved_condition_count"], 1)
        self.assertEqual(summary["unchanged_condition_count"], 1)
        self.assertEqual(summary["worsened_condition_count"], 1)
        self.assertAlmostEqual(summary["median_thd_change_percentage_points"], 0.0)
        self.assertAlmostEqual(
            summary["candidate_mean_fundamental_phasor_tracking_error_percent"],
            0.01,
        )
        self.assertAlmostEqual(summary["standard_deviation_thd_change_percentage_points_sample"], 1.0)


if __name__ == "__main__":
    unittest.main()
