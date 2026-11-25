from __future__ import annotations

import csv
import io
import json
import unittest
import zipfile

import numpy as np
import torch
from torch import nn

from evaluation.live_export import build_live_export
from evaluation.live_simulation import run_live_simulation
from preprocessing.preprocess import NormalizationStats


class ZeroWaveformModel(nn.Module):
    def forward(self, input_voltage, load_current, operating_conditions):
        return torch.zeros_like(input_voltage)


def _normalization() -> NormalizationStats:
    return NormalizationStats(
        input_voltage_mean=0.0,
        input_voltage_std=100.0,
        load_current_mean=0.0,
        load_current_std=2.0,
        operating_mean=[0.7, 20.0, 0.02, 0.5],
        operating_std=[0.1, 10.0, 0.01, 2.0],
        target_voltage_mean=0.0,
        target_voltage_std=100.0,
    )


def _run(offsets: tuple[float, float, float]):
    return run_live_simulation(
        ZeroWaveformModel(),
        _normalization(),
        modulation_index=0.75,
        load_resistance_ohm=30.0,
        load_inductance_h=0.03,
        phase_angle_degrees=30.0,
        device="cpu",
        carrier_phase_offsets_degrees=offsets,
    )


class LiveExportTests(unittest.TestCase):
    def test_exports_waveforms_metrics_harmonics_and_plot_bundle(self) -> None:
        baseline = _run((0.0, 60.0, 120.0))
        candidate = _run((0.0, 45.0, 135.0))
        bundle = build_live_export(
            baseline,
            candidate=candidate,
            run_metadata={"checkpoint_filename": "best.pt"},
        )

        rows = list(csv.DictReader(io.StringIO(bundle.waveform_csv.decode("utf-8"))))
        self.assertEqual(len(rows), 400)
        self.assertIn("baseline_gate_cell1_s1", rows[0])
        self.assertIn("candidate_gate_cell3_s4", rows[0])
        self.assertIn("candidate_cnn_prediction_error_v", rows[0])

        summary = json.loads(bundle.summary_json)
        self.assertRegex(summary["run_metadata"]["run_id"], r"^[0-9a-f]{32}$")
        self.assertEqual(summary["run_metadata"]["checkpoint_filename"], "best.pt")
        self.assertEqual(summary["baseline"]["carrier_phase_offsets_degrees"], [0.0, 60.0, 120.0])
        self.assertEqual(summary["candidate"]["carrier_phase_offsets_degrees"], [0.0, 45.0, 135.0])
        self.assertEqual(
            summary["comparison"]["thd_definition"],
            "Parseval full-spectrum THD from the event-integrated switching waveform",
        )
        self.assertTrue(np.isfinite(summary["comparison"]["candidate_minus_baseline_percentage_points"]))

        with zipfile.ZipFile(io.BytesIO(bundle.experiment_zip)) as archive:
            names = set(archive.namelist())
            self.assertTrue(
                {
                    "waveforms.csv",
                    "summary.json",
                    "harmonic_spectrum.csv",
                    "pso_convergence_history.csv",
                    "plots/live_waveforms.png",
                    "plots/harmonic_spectrum.png",
                    "current_harmonic_spectrum.csv",
                    "plots/current_harmonic_spectrum.png",
                }.issubset(names)
            )
        self.assertIn("load_current_harmonic_spectrum", summary["baseline"]["harmonics"])
        self.assertIn("baseline_current_thd_percent", summary["comparison"])

    def test_rejects_mismatched_comparison_conditions(self) -> None:
        baseline = _run((0.0, 60.0, 120.0))
        candidate = run_live_simulation(
            ZeroWaveformModel(),
            _normalization(),
            modulation_index=0.75,
            load_resistance_ohm=30.0,
            load_inductance_h=0.03,
            phase_angle_degrees=31.0,
            device="cpu",
            carrier_phase_offsets_degrees=(0.0, 45.0, 135.0),
        )
        with self.assertRaisesRegex(ValueError, "identical operating conditions"):
            build_live_export(baseline, candidate=candidate)


if __name__ == "__main__":
    unittest.main()
