from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset

from cnn.dataset import WINDOW_SIZE, build_dataset_from_dataframe
from cnn.model import EightLevelCNN


class WaveformDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx].unsqueeze(0)
        y = self.targets[idx]
        return x, y


def train_cnn(
    dataset_csv: Path,
    model_path: Path,
    history_path: Path,
    confusion_path: Path,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    validation_split: float = 0.2,
    progress_callback: Optional[Callable[[int, dict[str, float]], None]] = None,
) -> dict[str, Any]:
    df = pd.read_csv(dataset_csv)
    features, targets = build_dataset_from_dataframe(df)

    dataset = WaveformDataset(features, targets)
    val_count = int(len(dataset) * validation_split)
    train_count = len(dataset) - val_count
    train_dataset, val_dataset = random_split(dataset, [train_count, val_count])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = EightLevelCNN(input_length=WINDOW_SIZE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        train_preds = []
        train_targets = []

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_preds.extend(torch.argmax(outputs, dim=1).tolist())
            train_targets.extend(y_batch.tolist())

        train_loss = float(np.mean(train_losses))
        train_accuracy = float(accuracy_score(train_targets, train_preds))

        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                val_losses.append(loss.item())
                val_preds.extend(torch.argmax(outputs, dim=1).tolist())
                val_targets.extend(y_batch.tolist())

        val_loss = float(np.mean(val_losses))
        val_accuracy = float(accuracy_score(val_targets, val_preds))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)

        # Call progress callback if provided
        if progress_callback is not None:
            progress_callback(epoch, {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
            })

    torch.save({"model_state_dict": model.state_dict(), "input_length": WINDOW_SIZE, "num_classes": 8}, model_path)

    conf_matrix = confusion_matrix(val_targets, val_preds)
    np.savetxt(confusion_path, conf_matrix, delimiter=",", fmt="%d")

    history_df = pd.DataFrame(
        {
            "epoch": list(range(1, epochs + 1)),
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
            "train_accuracy": history["train_accuracy"],
            "val_accuracy": history["val_accuracy"],
        }
    )
    history_df.to_csv(history_path, index=False)

    return {
        "status": "trained",
        "model_path": str(model_path),
        "history_path": str(history_path),
        "confusion_path": str(confusion_path),
        "train_accuracy": history["train_accuracy"][-1],
        "val_accuracy": history["val_accuracy"][-1],
        "train_loss": history["train_loss"][-1],
        "val_loss": history["val_loss"][-1],
        "epochs": epochs,
        "train_samples": train_count,
        "validation_samples": val_count,
    }
