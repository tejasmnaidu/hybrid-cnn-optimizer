from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from cnn.model import EightLevelCNN, ImprovedEightLevelCNN, LegacyEightLevelCNN


def _normalize_features(features: np.ndarray, feature_mean: Optional[np.ndarray], feature_std: Optional[np.ndarray]) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if feature_mean is None or feature_std is None:
        return features
    mean = np.asarray(feature_mean, dtype=np.float32)
    std = np.asarray(feature_std, dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return ((features - mean) / std).astype(np.float32)


def _detect_architecture_name(state_dict: dict[str, torch.Tensor], checkpoint: dict[str, Any] | None = None) -> str:
    keys = set(state_dict.keys())
    inferred_name = None
    if any(key.startswith("stem.") for key in keys):
        inferred_name = "ImprovedEightLevelCNN"
    elif any(key.startswith("bn1.") or key.startswith("conv3.") for key in keys):
        inferred_name = "EightLevelCNN"
    elif any(
        key.startswith("conv1.") or key.startswith("conv2.") or key.startswith("fc1.") or key.startswith("fc2.")
        for key in keys
    ):
        inferred_name = "LegacyEightLevelCNN"

    explicit = checkpoint.get("architecture") if isinstance(checkpoint, dict) else None
    if explicit is not None:
        explicit_name = str(explicit)
        if inferred_name is not None and explicit_name != inferred_name:
            raise ValueError(
                f"Checkpoint architecture metadata '{explicit_name}' does not match the state_dict signature '{inferred_name}'."
            )
        return explicit_name

    if inferred_name is None:
        raise ValueError(
            "Unable to infer checkpoint architecture from state_dict keys. "
            f"Observed keys: {sorted(keys)[:20]}"
        )
    return inferred_name


def _instantiate_model_for_checkpoint(checkpoint: dict[str, Any]) -> torch.nn.Module:
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    architecture_name = _detect_architecture_name(state_dict, checkpoint)
    num_classes = int(checkpoint.get("num_classes", 8))
    dropout_rate = float(checkpoint.get("dropout_rate", 0.2))
    input_channels = int(checkpoint.get("input_channels", 1))

    if architecture_name == "ImprovedEightLevelCNN":
        model = ImprovedEightLevelCNN(input_channels=input_channels, num_classes=num_classes, dropout_rate=dropout_rate)
    elif architecture_name == "EightLevelCNN":
        model = EightLevelCNN(input_channels=input_channels, num_classes=num_classes, dropout_rate=dropout_rate)
    elif architecture_name == "LegacyEightLevelCNN":
        model = LegacyEightLevelCNN(input_channels=input_channels, num_classes=num_classes, dropout_rate=dropout_rate)
    else:
        raise ValueError(f"Unsupported checkpoint architecture '{architecture_name}'.")

    expected = set(model.state_dict().keys())
    actual = set(state_dict.keys())
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(
            f"Checkpoint architecture mismatch for {architecture_name}: "
            f"missing keys={missing[:10]}, unexpected keys={unexpected[:10]}"
        )
    model.load_state_dict(state_dict, strict=True)
    return model


def load_cnn_model(model_path: Path) -> tuple[torch.nn.Module, int]:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = _instantiate_model_for_checkpoint(checkpoint)
    window_size = int(checkpoint.get("window_size", checkpoint.get("input_length", 32)))
    feature_mean = checkpoint.get("feature_mean")
    feature_std = checkpoint.get("feature_std")
    model.feature_mean = np.asarray(feature_mean if feature_mean is not None else np.zeros(window_size, dtype=np.float32), dtype=np.float32)
    model.feature_std = np.asarray(feature_std if feature_std is not None else np.ones(window_size, dtype=np.float32), dtype=np.float32)
    model.eval()
    return model, window_size


def predict_switching_state(
    model_path: Path,
    waveform_window: np.ndarray,
    model: Optional[torch.nn.Module] = None,
    window_size: Optional[int] = None,
) -> dict[str, Any]:
    if model is None or window_size is None:
        model, window_size = load_cnn_model(model_path)

    waveform_window = _normalize_features(
        waveform_window,
        getattr(model, "feature_mean", None),
        getattr(model, "feature_std", None),
    )

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
    model: torch.nn.Module,
    waveform_windows: np.ndarray,
) -> np.ndarray:
    if waveform_windows.ndim != 2:
        raise ValueError("waveform_windows must be a 2D array of shape (batch, window_size).")
    waveform_windows = _normalize_features(
        waveform_windows,
        getattr(model, "feature_mean", None),
        getattr(model, "feature_std", None),
    )
    input_tensor = torch.tensor(waveform_windows, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        logits = model(input_tensor)
        return torch.argmax(logits, dim=1).cpu().numpy()
