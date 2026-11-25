"""Small surrogate model for event-based inverter total-THD ranking."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


THD_SURROGATE_FEATURE_NAMES = (
    "modulation_index",
    "sin_reference_phase",
    "cos_reference_phase",
    "sin_cell_2_phase",
    "cos_cell_2_phase",
    "sin_cell_3_phase",
    "cos_cell_3_phase",
)
THD_SURROGATE_TARGET_NAMES = (
    "event_total_thd_percent",
)


def encode_thd_surrogate_features(candidates: np.ndarray) -> np.ndarray:
    """Encode rows of modulation, reference phase, and two carrier offsets."""
    values = np.asarray(candidates, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4 or values.shape[0] == 0:
        raise ValueError(
            "candidates must be a non-empty matrix of modulation, "
            "reference phase, cell-2 phase, and cell-3 phase."
        )
    if not np.isfinite(values).all():
        raise ValueError("THD surrogate inputs must be finite.")
    if np.any(values[:, 0] < 0.55) or np.any(values[:, 0] > 0.95):
        raise ValueError("modulation_index must be in [0.55, 0.95].")

    reference_phase = np.deg2rad(np.mod(values[:, 1], 360.0))
    carrier_phases = np.deg2rad(
        np.sort(np.mod(values[:, 2:], 360.0), axis=1)
    )
    features = np.column_stack(
        (
            values[:, 0],
            np.sin(reference_phase),
            np.cos(reference_phase),
            np.sin(carrier_phases[:, 0]),
            np.cos(carrier_phases[:, 0]),
            np.sin(carrier_phases[:, 1]),
            np.cos(carrier_phases[:, 1]),
        )
    )
    if not np.isfinite(features).all():
        raise ValueError("Encoded THD surrogate features are non-finite.")
    return features.astype(np.float32)


class EventTHDSurrogate(nn.Module):
    """Predict event-based total THD only."""

    def __init__(
        self,
        *,
        input_features: int = len(THD_SURROGATE_FEATURE_NAMES),
        hidden_width: int = 128,
        output_features: int = len(THD_SURROGATE_TARGET_NAMES),
    ) -> None:
        super().__init__()
        if input_features != len(THD_SURROGATE_FEATURE_NAMES):
            raise ValueError("input_features does not match the encoded feature schema.")
        if hidden_width < 1:
            raise ValueError("hidden_width must be positive.")
        if output_features != len(THD_SURROGATE_TARGET_NAMES):
            raise ValueError("output_features does not match the target schema.")
        self.network = nn.Sequential(
            nn.Linear(input_features, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, hidden_width // 2),
            nn.GELU(),
            nn.Linear(hidden_width // 2, output_features),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != len(
            THD_SURROGATE_FEATURE_NAMES
        ):
            raise ValueError(
                f"features must have shape (batch, {len(THD_SURROGATE_FEATURE_NAMES)})."
            )
        return self.network(features)


def spearman_rank_correlation(first: np.ndarray, second: np.ndarray) -> float:
    """Calculate Spearman correlation with average ranks for ties."""
    left = np.asarray(first, dtype=np.float64).reshape(-1)
    right = np.asarray(second, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or left.size < 2:
        raise ValueError("Rank-correlation inputs must be equal arrays with at least two values.")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Rank-correlation inputs must be finite.")

    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        result = np.empty(values.size, dtype=np.float64)
        start = 0
        while start < values.size:
            end = start + 1
            while end < values.size and sorted_values[end] == sorted_values[start]:
                end += 1
            result[order[start:end]] = (start + 1 + end) / 2.0
            start = end
        return result

    left_ranks = ranks(left)
    right_ranks = ranks(right)
    left_ranks -= left_ranks.mean()
    right_ranks -= right_ranks.mean()
    denominator = math.sqrt(
        float(np.sum(np.square(left_ranks)) * np.sum(np.square(right_ranks)))
    )
    if denominator == 0.0:
        return 0.0
    return float(np.sum(left_ranks * right_ranks) / denominator)
