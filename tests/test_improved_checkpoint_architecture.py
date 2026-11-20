import unittest
from pathlib import Path

import numpy as np
import torch

from cnn.model import EightLevelCNN, ImprovedEightLevelCNN, LegacyEightLevelCNN
from cnn.predict import load_cnn_model


class TestCheckpointArchitecture(unittest.TestCase):
    def test_legacy_checkpoint_loads_legacy_architecture(self):
        checkpoint_path = Path("models/cnn_8level_controller_test.pth")
        model, _ = load_cnn_model(checkpoint_path)
        self.assertIsInstance(model, LegacyEightLevelCNN)

    def test_improved_checkpoint_loads_improved_architecture(self):
        checkpoint_path = Path("models/cnn_improved_experiment.pth")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_keys = list(checkpoint["model_state_dict"].keys())

        self.assertTrue(
            any(key.startswith("stem.") for key in state_keys),
            "Expected the improved checkpoint to use the ImprovedEightLevelCNN stem/block architecture.",
        )

        model, _ = load_cnn_model(checkpoint_path)
        self.assertIsInstance(model, ImprovedEightLevelCNN)

    def test_incompatible_checkpoint_fails_clearly(self):
        bad_state = {
            "model_state_dict": {
                "conv1.weight": torch.randn(16, 1, 5),
                "conv1.bias": torch.randn(16),
                "conv2.weight": torch.randn(32, 16, 5),
                "conv2.bias": torch.randn(32),
                "fc1.weight": torch.randn(64, 32),
                "fc1.bias": torch.randn(64),
                "fc2.weight": torch.randn(8, 64),
                "fc2.bias": torch.randn(8),
            },
            "architecture": "ImprovedEightLevelCNN",
            "num_classes": 8,
            "dropout_rate": 0.2,
            "window_size": 32,
        }
        bad_path = Path("models/test_incompatible_checkpoint.pth")
        torch.save(bad_state, bad_path)
        try:
            with self.assertRaisesRegex(ValueError, "architecture metadata|architecture mismatch|Unsupported checkpoint architecture|Unable to infer"):
                load_cnn_model(bad_path)
        finally:
            if bad_path.exists():
                bad_path.unlink()


if __name__ == "__main__":
    unittest.main()
