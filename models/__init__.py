"""Configurable 1D residual CNN models."""

from .cnn_model import CNNConfig, WaveformResidualCNN, detect_device
from .model_factory import create_model

__all__ = ["CNNConfig", "WaveformResidualCNN", "create_model", "detect_device"]
