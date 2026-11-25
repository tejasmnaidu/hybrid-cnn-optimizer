"""Waveform regression losses."""

from __future__ import annotations

import torch
from torch import nn


class WaveformLoss(nn.Module):
    """MAE or MSE with an optional first-difference penalty."""

    def __init__(
        self,
        kind: str = "mae",
        derivative_weight: float = 0.0,
    ) -> None:
        super().__init__()
        normalized = kind.strip().lower()
        if normalized not in {"mae", "mse"}:
            raise ValueError("kind must be 'mae' or 'mse'.")
        if derivative_weight < 0:
            raise ValueError("derivative_weight cannot be negative.")
        self.kind = normalized
        self.derivative_weight = derivative_weight
        self.base_loss = nn.L1Loss() if normalized == "mae" else nn.MSELoss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError("prediction and target must have identical shapes.")
        loss = self.base_loss(prediction, target)
        if self.derivative_weight:
            if prediction.shape[-1] < 2:
                raise ValueError("Derivative loss requires at least two waveform points.")
            loss = loss + self.derivative_weight * self.base_loss(
                torch.diff(prediction, dim=-1), torch.diff(target, dim=-1)
            )
        return loss
