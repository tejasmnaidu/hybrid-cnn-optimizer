from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from cnn.model import EightLevelCNN
from cnn.dataset import WINDOW_SIZE


def predict_switching_state(
    model_path: Path,
    waveform_window: np.ndarray,
) -> dict[str, Any]:
    if waveform_window.shape != (WINDOW_SIZE,):
        raise ValueError(f"Expected input shape ({WINDOW_SIZE},), got {waveform_window.shape}")

    checkpoint = torch.load(model_path, map_location="cpu")
    model = EightLevelCNN(input_length=checkpoint["input_length"], num_classes=checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

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
