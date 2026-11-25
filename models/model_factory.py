"""Model construction helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from project_paths import resolve_project_path

from .cnn_model import CNNConfig, WaveformResidualCNN


def create_model(config: CNNConfig | dict[str, Any] | str | Path | None = None) -> WaveformResidualCNN:
    """Create a model from a config object, model mapping, or YAML config path."""
    if config is None:
        model_config = CNNConfig()
    elif isinstance(config, CNNConfig):
        model_config = config
    else:
        values: Any = config
        if isinstance(config, (str, Path)):
            with resolve_project_path(config).open(encoding="utf-8") as file:
                values = yaml.safe_load(file)
            if not isinstance(values, dict):
                raise ValueError("YAML configuration must be a mapping.")
        if isinstance(values, dict):
            values = values.get("model", values)
        if not isinstance(values, dict):
            raise TypeError("Model configuration must be a mapping or YAML path.")
        if "branch_channels" in values:
            values = dict(values)
            values["branch_channels"] = tuple(values["branch_channels"])
        model_config = CNNConfig(**values)
    return WaveformResidualCNN(model_config)
