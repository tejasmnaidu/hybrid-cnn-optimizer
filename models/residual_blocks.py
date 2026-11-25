"""Residual building blocks for sequence-to-sequence 1D CNNs."""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock1D(nn.Module):
    """Two-convolution residual block that preserves temporal length."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.projection = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.projection(inputs)
        output = self.activation(self.norm1(self.conv1(inputs)))
        output = self.norm2(self.conv2(output))
        return self.activation(output + residual)
