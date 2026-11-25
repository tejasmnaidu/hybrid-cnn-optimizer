from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluation.metrics import calculate_metrics
from evaluation.evaluate import evaluate_model
from evaluation.thd_analysis import calculate_thd
from models.model_factory import create_model
from optimization.pso_optimizer import PSOConfig, PSOOptimizer
from optimization.switching_constraints import reshape_cell_states, reshape_gate_data
from preprocessing.data_loader import load_dataset
from preprocessing.data_validation import validate_dataset
from preprocessing.preprocess import preprocess_dataset
from training.dataset import WaveformDataset
from training.checkpoint import load_checkpoint, save_checkpoint
from torch.utils.data import DataLoader


def _write_fixture(root: Path) -> None:
    count = 6
    ids = np.arange(1, count + 1)
    wave = np.arange(400, dtype=np.float32)
    discrete_levels = np.tile(np.array([-300, -200, -100, 0, 100, 200, 300]), 58)[:400]
    signals = {
        "Input_Voltage.csv": (
            "inputVoltage",
            np.stack([np.roll(discrete_levels, int(i)) for i in ids]),
        ),
        "Target_Voltage.csv": ("targetVoltage", np.stack([np.sin(wave / 20) * (i + 2) for i in ids])),
        "Load_Current.csv": ("inputCurrent", np.stack([np.cos(wave / 30) * (i + 1) for i in ids])),
    }
    for filename, (prefix, values) in signals.items():
        frame = pd.DataFrame(values, columns=[f"{prefix}{i + 1}" for i in range(400)])
        frame.insert(0, "SampleID", ids)
        frame.iloc[::-1].to_csv(root / filename, index=False)
    conditions = pd.DataFrame(
        {
            "sampleID": ids[::-1],
            "modulationIndex": np.linspace(0.5, 0.9, count),
            "loadResistance": np.arange(count) + 10,
            "loadInductance": np.arange(count) * 0.01 + 0.01,
            "phaseAngle": np.arange(count) * 5.0,
        }
    )
    conditions.to_csv(root / "Operating_Conditions.csv", index=False)
    splits = pd.DataFrame(
        {"sampleID": ids, "splitLabel": ["Training"] * 3 + ["Validation"] * 2 + ["Testing"]}
    )
    splits.to_csv(root / "Sample_Splits.csv", index=False)
    thd = pd.DataFrame(
        {
            "sampleID": ids,
            "inputVoltageTHD": np.ones(count),
            "inputCurrentTHD": np.ones(count),
            "targetVoltageTHD": np.ones(count),
        }
    )
    thd.to_csv(root / "THD_Results.csv", index=False)
    pd.DataFrame(
        {
            "parameter": [
                "Total Waveforms",
                "Samples Per Cycle",
                "Fundamental Frequency",
                "Sampling Frequency",
                "Number of Voltage Levels",
                "Voltage Levels",
                "Training Samples",
                "Validation Samples",
                "Testing Samples",
            ],
            "value": [count, 400, 50, 20000, 7, "[-300 -200 -100 0 100 200 300]", 3, 2, 1],
            "unit": ["Count", "Samples", "Hz", "Hz", "Count", "V", "Count", "Count", "Count"],
        }
    ).to_csv(root / "Configuration.csv", index=False)


class DatasetTests(unittest.TestCase):
    def test_load_directory_and_zip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_fixture(root)
            loaded = load_dataset(root)
            self.assertEqual(len(loaded["input_voltage"]), 6)
            archive = root / "fixture.zip"
            with zipfile.ZipFile(archive, "w") as zipped:
                for file in root.glob("*.csv"):
                    zipped.write(file, f"Nested/Excel_CSV/{file.name}")
            extracted = load_dataset(archive, extract_to=root / "unpacked")
            self.assertEqual(extracted["input_voltage"].shape, (6, 401))

    def test_preprocessing_aligns_ids_and_uses_train_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_fixture(root)
            prepared = preprocess_dataset(load_dataset(root))
            np.testing.assert_array_equal(prepared.sample_ids, np.arange(1, 7))
            self.assertEqual(prepared.indices("training").size, 3)
            self.assertEqual(prepared.indices("validation").size, 2)
            self.assertEqual(prepared.indices("testing").size, 1)
            self.assertTrue(np.allclose(prepared.input_voltage[:3].mean(), 0, atol=1e-6))
            record = WaveformDataset(prepared, "testing")[0]
            self.assertEqual(tuple(record["target_voltage"].shape), (400,))
            self.assertEqual(int(record["sample_id"]), 6)

    def test_dataset_validation_checks_declared_schema_and_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_fixture(root)
            report = validate_dataset(root, check_auxiliary=False)
            self.assertTrue(report.valid, report.errors)
            self.assertEqual(report.summary["split_counts"], {"training": 3, "validation": 2, "testing": 1})
            self.assertEqual(report.summary["signals"]["input_voltage"]["levels"], [-300, -200, -100, 0, 100, 200, 300])


class ModelAndEvaluationTests(unittest.TestCase):
    def test_model_forward_shape_and_parameters(self) -> None:
        model = create_model(
            {
                "model": {
                    "branch_channels": [4, 8],
                    "condition_width": 4,
                    "fusion_channels": 8,
                    "kernel_size": 3,
                }
            }
        )
        output = model(torch.randn(2, 400), torch.randn(2, 400), torch.randn(2, 4))
        self.assertEqual(tuple(output.shape), (2, 400))
        self.assertGreater(model.parameter_count(), 0)

    def test_metrics_and_thd(self) -> None:
        target = np.array([1.0, 2.0, 3.0])
        predicted = np.array([1.0, 2.0, 2.0])
        metrics = calculate_metrics(target, predicted)
        self.assertAlmostEqual(metrics["mae"], 1 / 3)
        self.assertAlmostEqual(metrics["mse"], 1 / 3)
        samples = np.arange(400)
        waveform = np.sin(2 * np.pi * samples / 400) + 0.1 * np.sin(6 * np.pi * samples / 400)
        self.assertAlmostEqual(
            calculate_thd(waveform, sampling_frequency_hz=20000, fundamental_frequency_hz=50), 10.0, places=8
        )

    def test_evaluate_model_batch_and_checkpoint_round_trip(self) -> None:
        class CopyInput(torch.nn.Module):
            def forward(self, input_voltage, load_current, operating_conditions):
                return input_voltage

        inputs = torch.linspace(-1, 1, 400).repeat(2, 1)
        loader = DataLoader(
            [
                {
                    "input_voltage": inputs[index],
                    "load_current": torch.zeros(400),
                    "operating_conditions": torch.zeros(4),
                    "target_voltage": inputs[index] + 0.1,
                    "sample_id": torch.tensor(index + 1),
                }
                for index in range(2)
            ],
            batch_size=2,
        )
        evaluated = evaluate_model(CopyInput(), loader, device="cpu")
        self.assertEqual(evaluated["predictions"].shape, (2, 400))
        self.assertEqual(len(evaluated["metrics"]), 6)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            save_checkpoint(path, {"model_state": {}, "metadata": {"verified": True}})
            restored = load_checkpoint(path)
            self.assertTrue(restored["metadata"]["verified"])

    def test_gate_and_cell_layout_requires_declaration(self) -> None:
        gates = np.tile(np.array([0, 1]), 12)
        with self.assertRaises(TypeError):
            reshape_gate_data(gates, time_points=2)
        self.assertEqual(reshape_gate_data(gates, time_points=2, layout="time_major").shape, (2, 12))
        states = np.array([-1, 0, 1, 0, 1, -1])
        self.assertEqual(reshape_cell_states(states, time_points=2, layout="time_major").shape, (2, 3))

    def test_pso_minimizes_bounded_objective(self) -> None:
        optimizer = PSOOptimizer(
            PSOConfig(bounds=((-2.0, 2.0),), particles=12, iterations=30, seed=3)
        )
        result = optimizer.optimize(lambda position: float(position[0] ** 2))
        self.assertLess(result.objective_value, 0.05)


if __name__ == "__main__":
    unittest.main()
