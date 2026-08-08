from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from cnn.model import EightLevelCNN


def load_cnn_model(model_path: Path) -> tuple[EightLevelCNN, int]:
    checkpoint = torch.load(model_path, map_location="cpu")
    num_classes = int(checkpoint.get("num_classes", 8))
    dropout_rate = float(checkpoint.get("dropout_rate", 0.2))
    window_size = int(checkpoint.get("window_size", checkpoint.get("input_length", 32)))
    model = EightLevelCNN(input_channels=1, num_classes=num_classes, dropout_rate=dropout_rate)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, window_size


def predict_switching_state(
    model_path: Path,
    waveform_window: np.ndarray,
    model: Optional[EightLevelCNN] = None,
    window_size: Optional[int] = None,
) -> dict[str, Any]:
    if model is None or window_size is None:
        model, window_size = load_cnn_model(model_path)

    if waveform_window.shape != (window_size,):
        raise ValueError(f"Expected input shape ({window_size},), got {waveform_window.shape}")

    input_tensor = torch.tensor(waveform_window, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1).squeeze(0).numpy()
        predicted_class = int(torch.argmax(logits, dim=1).item())
        confidence = float(np.max(probabilities))

    return {
        "predicted_class": predicted_class,
        "probabilities": probabilities.tolist(),
        "confidence": confidence,
    }


def predict_switching_state_batch(
    model: EightLevelCNN,
    waveform_windows: np.ndarray,
) -> np.ndarray:
    if waveform_windows.ndim != 2:
        raise ValueError("waveform_windows must be a 2D array of shape (batch, window_size).")
    input_tensor = torch.tensor(waveform_windows, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        logits = model(input_tensor)
        return torch.argmax(logits, dim=1).cpu().numpy()
