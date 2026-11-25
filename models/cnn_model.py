"""Separate-branch residual CNN for inverter waveform regression."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .residual_blocks import ResidualBlock1D


@dataclass(frozen=True)
class CNNConfig:
    waveform_points: int = 400
    operating_features: int = 4
    branch_channels: tuple[int, ...] = (32, 64)
    condition_width: int = 32
    fusion_channels: int = 64
    kernel_size: int = 5

    def __post_init__(self) -> None:
        if self.waveform_points < 1 or self.operating_features < 1:
            raise ValueError("waveform_points and operating_features must be positive.")
        if not self.branch_channels or any(channel < 1 for channel in self.branch_channels):
            raise ValueError("branch_channels must contain positive channel widths.")
        if self.condition_width < 1 or self.fusion_channels < 1:
            raise ValueError("condition_width and fusion_channels must be positive.")
        if self.kernel_size < 1 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")


def _branch(channels: tuple[int, ...], kernel_size: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = 1
    for width in channels:
        layers.append(ResidualBlock1D(previous, width, kernel_size))
        previous = width
    return nn.Sequential(*layers)


class WaveformResidualCNN(nn.Module):
    """Predict normalized target voltage from voltage, current, and conditions."""

    def __init__(self, config: CNNConfig | None = None) -> None:
        super().__init__()
        self.config = config or CNNConfig()
        self.voltage_branch = _branch(self.config.branch_channels, self.config.kernel_size)
        self.current_branch = _branch(self.config.branch_channels, self.config.kernel_size)
        branch_width = self.config.branch_channels[-1]
        self.condition_encoder = nn.Sequential(
            nn.Linear(self.config.operating_features, self.config.condition_width),
            nn.GELU(),
            nn.Linear(self.config.condition_width, self.config.condition_width),
            nn.GELU(),
        )
        fused_width = 2 * branch_width + self.config.condition_width
        self.fusion = nn.Sequential(
            ResidualBlock1D(fused_width, self.config.fusion_channels, self.config.kernel_size),
            ResidualBlock1D(
                self.config.fusion_channels, self.config.fusion_channels, self.config.kernel_size
            ),
            nn.Conv1d(self.config.fusion_channels, 1, kernel_size=1),
        )

    @staticmethod
    def _waveform_shape(waveform: torch.Tensor, label: str) -> torch.Tensor:
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError(f"{label} must have shape (batch, points) or (batch, 1, points).")
        return waveform

    def forward(
        self,
        input_voltage: torch.Tensor,
        load_current: torch.Tensor,
        operating_conditions: torch.Tensor,
    ) -> torch.Tensor:
        voltage = self._waveform_shape(input_voltage, "input_voltage")
        current = self._waveform_shape(load_current, "load_current")
        if voltage.shape != current.shape:
            raise ValueError("input_voltage and load_current must have matching shapes.")
        if voltage.shape[-1] != self.config.waveform_points:
            raise ValueError(
                f"Expected {self.config.waveform_points} waveform points; got {voltage.shape[-1]}."
            )
        if operating_conditions.ndim != 2 or operating_conditions.shape != (
            voltage.shape[0],
            self.config.operating_features,
        ):
            raise ValueError(
                f"operating_conditions must have shape (batch, {self.config.operating_features})."
            )
        voltage_features = self.voltage_branch(voltage)
        current_features = self.current_branch(current)
        condition_features = self.condition_encoder(operating_conditions).unsqueeze(-1)
        condition_features = condition_features.expand(-1, -1, voltage.shape[-1])
        fused = torch.cat((voltage_features, current_features, condition_features), dim=1)
        return self.fusion(fused).squeeze(1)

    def parameter_count(self, *, trainable_only: bool = True) -> int:
        parameters = self.parameters()
        return sum(parameter.numel() for parameter in parameters if not trainable_only or parameter.requires_grad)


def detect_device() -> torch.device:
    """Select CUDA when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
