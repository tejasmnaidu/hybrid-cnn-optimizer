"""PyTorch datasets and split-aware DataLoader construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from preprocessing.preprocess import PreparedDataset


class WaveformDataset(Dataset[dict[str, torch.Tensor]]):
    """A split-specific view over normalized waveform records."""

    def __init__(self, data: PreparedDataset, split: str) -> None:
        normalized = split.strip().lower()
        if normalized not in {"training", "validation", "testing"}:
            raise ValueError(f"Unknown split name: {split}")
        self.data = data
        self.indices = data.indices(normalized)
        if self.indices.size == 0:
            raise ValueError(f"The {normalized} split is empty.")
        self.split = normalized

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = int(self.indices[index])
        return {
            "input_voltage": torch.from_numpy(self.data.input_voltage[row]),
            "load_current": torch.from_numpy(self.data.load_current[row]),
            "operating_conditions": torch.from_numpy(self.data.operating_conditions[row]),
            "target_voltage": torch.from_numpy(self.data.target_voltage[row]),
            "sample_id": torch.tensor(self.data.sample_ids[row], dtype=torch.int64),
        }


@dataclass(frozen=True)
class DataLoaders:
    train: DataLoader[dict[str, torch.Tensor]]
    validation: DataLoader[dict[str, torch.Tensor]]
    test: DataLoader[dict[str, torch.Tensor]]


def create_dataloaders(
    data: PreparedDataset,
    *,
    batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 42,
    pin_memory: bool | None = None,
) -> DataLoaders:
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative.")
    generator = torch.Generator()
    generator.manual_seed(seed)
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    return DataLoaders(
        train=DataLoader(
            WaveformDataset(data, "training"),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            generator=generator,
        ),
        validation=DataLoader(
            WaveformDataset(data, "validation"),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        test=DataLoader(
            WaveformDataset(data, "testing"),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    )
