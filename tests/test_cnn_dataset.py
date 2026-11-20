import unittest
from pathlib import Path

import numpy as np

from cnn.dataset import analyze_dataset, generate_cnn_dataset, load_dataset
from multilevel.eight_level_model import EightLevelInverterParams, generate_8_level_waveform


class TestCNNDataset(unittest.TestCase):
    def test_generate_cnn_dataset_uses_actual_waveform_windows(self):
        output_path = Path("tests") / "cnn_dataset_waveform_test.csv"
        if output_path.exists():
            output_path.unlink()
        parquet_path = output_path.with_suffix(".parquet")
        if parquet_path.exists():
            parquet_path.unlink()
        json_path = output_path.with_suffix(".json")
        if json_path.exists():
            json_path.unlink()

        df = generate_cnn_dataset(
            output_path,
            num_waveforms=1,
            window_size=32,
            stride=8,
            dc_voltage_range=(100.0, 100.0),
            modulation_range=(0.8, 0.8),
            amplitude_range=(1.0, 1.0),
            frequency_range=(50.0, 50.0),
            phase_range=(0.0, 0.0),
            sampling_frequency_range=(10000.0, 10000.0),
            duration_range=(0.05, 0.05),
            noise_std=0.0,
            seed=123,
        )

        params = EightLevelInverterParams(
            dc_voltage=100.0,
            fundamental_frequency=50.0,
            switching_frequency=2000.0,
            sampling_frequency=10000.0,
            simulation_duration=0.05,
            modulation_index=0.8,
            phase_shift=0.0,
        )
        waveform = generate_8_level_waveform(params)
        expected_window = waveform.voltage[:32]
        actual_window = df[[f"window_{i}" for i in range(32)]].iloc[0].to_numpy(dtype=float)

        self.assertTrue(np.allclose(actual_window, expected_window, atol=1e-8))

        if output_path.exists():
            output_path.unlink()
        if parquet_path.exists():
            parquet_path.unlink()
        if json_path.exists():
            json_path.unlink()

    def test_generate_cnn_dataset_splits_and_labels(self):
        output_path = Path("tests") / "cnn_dataset_test.csv"
        if output_path.exists():
            output_path.unlink()
        parquet_path = output_path.with_suffix(".parquet")
        if parquet_path.exists():
            parquet_path.unlink()
        json_path = output_path.with_suffix(".json")
        if json_path.exists():
            json_path.unlink()

        df = generate_cnn_dataset(
            output_path,
            num_waveforms=50,
            stride=8,
            dc_voltage_range=(80.0, 120.0),
            modulation_range=(0.35, 1.0),
            amplitude_range=(0.7, 1.0),
            frequency_range=(45.0, 55.0),
            phase_range=(0.0, np.pi / 2.0),
            sampling_frequency_range=(9000.0, 11000.0),
            duration_range=(0.03, 0.05),
            noise_std=0.0,
            seed=42,
        )

        self.assertTrue(output_path.exists())
        self.assertTrue(json_path.exists())
        self.assertGreater(len(df), 0)
        self.assertEqual(df["waveform_id"].nunique(), 50)
        self.assertTrue(all(split in {"train", "validation", "test"} for split in df["split"].unique()))
        analysis = analyze_dataset(df)
        self.assertEqual(analysis["waveform_count"], 50)
        self.assertEqual(analysis["total_samples"], len(df))
        self.assertEqual(len(analysis["class_counts"]), 8)
        self.assertEqual(sorted(df["target_level"].unique().tolist()), list(range(8)))

        train_ids = set(df[df["split"] == "train"]["waveform_id"].unique())
        val_ids = set(df[df["split"] == "validation"]["waveform_id"].unique())
        test_ids = set(df[df["split"] == "test"]["waveform_id"].unique())
        self.assertTrue(train_ids.isdisjoint(val_ids))
        self.assertTrue(train_ids.isdisjoint(test_ids))
        self.assertTrue(val_ids.isdisjoint(test_ids))

        loaded_df = load_dataset(output_path)
        self.assertEqual(len(loaded_df), len(df))
        self.assertEqual(loaded_df["waveform_id"].nunique(), 50)

        if output_path.exists():
            output_path.unlink()
        if parquet_path.exists():
            parquet_path.unlink()
        if json_path.exists():
            json_path.unlink()


if __name__ == "__main__":
    unittest.main()
