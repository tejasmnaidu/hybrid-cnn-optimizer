from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset, random_split

from cnn.dataset import WINDOW_SIZE, build_dataset_from_dataframe, load_dataset
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


def _infer_window_size(df: pd.DataFrame) -> int:
    return len([col for col in df.columns if col.startswith("window_")])


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _compute_classification_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    labels = list(range(8))
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    recall = float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def train_cnn(
    dataset_csv: Path,
    model_path: Path,
    history_path: Path,
    confusion_path: Path,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0,
    dropout_rate: float = 0.2,
    validation_split: float = 0.2,
    patience: int = 5,
    min_delta: float = 1e-4,
    scheduler_step_size: int = 5,
    scheduler_gamma: float = 0.5,
    seed: int = 0,
    progress_callback: Optional[Callable[[int, dict[str, float]], None]] = None,
) -> dict[str, Any]:
    df = load_dataset(dataset_csv)
    window_size = _infer_window_size(df)

    if "split" in df.columns:
        train_df = df[df["split"] == "train"].reset_index(drop=True)
        val_df = df[df["split"] == "validation"].reset_index(drop=True)
        test_df = df[df["split"] == "test"].reset_index(drop=True)
        train_features, train_targets = build_dataset_from_dataframe(train_df, window_size=window_size)
        val_features, val_targets = build_dataset_from_dataframe(val_df, window_size=window_size)
        train_dataset = WaveformDataset(train_features, train_targets)
        val_dataset = WaveformDataset(val_features, val_targets)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = None
        if len(test_df) > 0:
            test_features, test_targets = build_dataset_from_dataframe(test_df, window_size=window_size)
            test_loader = DataLoader(WaveformDataset(test_features, test_targets), batch_size=batch_size, shuffle=False)
        train_count = len(train_dataset)
        val_count = len(val_dataset)
        test_count = len(test_loader.dataset) if test_loader is not None else 0
    else:
        features_all, targets_all = build_dataset_from_dataframe(df, window_size=window_size)
        dataset = WaveformDataset(features_all, targets_all)
        val_count = int(len(dataset) * validation_split)
        train_count = len(dataset) - val_count
        train_dataset, val_dataset = random_split(dataset, [train_count, val_count])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = None
        test_count = 0

    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EightLevelCNN(input_channels=1, num_classes=8, dropout_rate=dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_no_improve = 0

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "train_precision": [],
        "val_precision": [],
        "train_recall": [],
        "val_recall": [],
        "train_f1": [],
        "val_f1": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        train_preds: list[int] = []
        train_targets: list[int] = []

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_preds.extend(torch.argmax(outputs, dim=1).tolist())
            train_targets.extend(y_batch.tolist())

        train_loss = float(np.mean(train_losses))
        train_metrics = _compute_classification_metrics(train_targets, train_preds)

        model.eval()
        val_losses: list[float] = []
        val_preds: list[int] = []
        val_targets: list[int] = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                val_losses.append(loss.item())
                val_preds.extend(torch.argmax(outputs, dim=1).tolist())
                val_targets.extend(y_batch.tolist())

        val_loss = float(np.mean(val_losses))
        val_metrics = _compute_classification_metrics(val_targets, val_preds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["train_precision"].append(train_metrics["precision"])
        history["val_precision"].append(val_metrics["precision"])
        history["train_recall"].append(train_metrics["recall"])
        history["val_recall"].append(val_metrics["recall"])
        history["train_f1"].append(train_metrics["f1"])
        history["val_f1"].append(val_metrics["f1"])

        if val_loss + min_delta < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if progress_callback is not None:
            progress_callback(epoch, {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_metrics["accuracy"],
                "val_accuracy": val_metrics["accuracy"],
                "train_precision": train_metrics["precision"],
                "val_precision": val_metrics["precision"],
                "train_recall": train_metrics["recall"],
                "val_recall": val_metrics["recall"],
                "train_f1": train_metrics["f1"],
                "val_f1": val_metrics["f1"],
            })

        scheduler.step()

        if epochs_no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if not model_path.parent.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_channels": 1,
            "num_classes": 8,
            "dropout_rate": dropout_rate,
            "window_size": window_size,
            "seed": seed,
        },
        model_path,
    )

    final_test_metrics: dict[str, Any] = {}
    final_conf_matrix = None
    if test_loader is not None:
        test_preds: list[int] = []
        test_targets: list[int] = []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                outputs = model(x_batch)
                test_preds.extend(torch.argmax(outputs, dim=1).tolist())
                test_targets.extend(y_batch.tolist())
        final_test_metrics = _compute_classification_metrics(test_targets, test_preds)
        final_conf_matrix = confusion_matrix(test_targets, test_preds, labels=list(range(8)))
    else:
        final_conf_matrix = confusion_matrix(val_targets, val_preds, labels=list(range(8)))

    np.savetxt(confusion_path, final_conf_matrix, delimiter=",", fmt="%d")

    history_df = pd.DataFrame(
        {
            "epoch": list(range(1, len(history["train_loss"]) + 1)),
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
            "train_accuracy": history["train_accuracy"],
            "val_accuracy": history["val_accuracy"],
            "train_precision": history["train_precision"],
            "val_precision": history["val_precision"],
            "train_recall": history["train_recall"],
            "val_recall": history["val_recall"],
            "train_f1": history["train_f1"],
            "val_f1": history["val_f1"],
        }
    )
    history_df.to_csv(history_path, index=False)

    parameter_count = sum(p.numel() for p in model.parameters())
    return {
        "status": "trained",
        "model_path": str(model_path),
        "history_path": str(history_path),
        "confusion_path": str(confusion_path),
        "train_accuracy": history["train_accuracy"][-1],
        "val_accuracy": history["val_accuracy"][-1],
        "train_precision": history["train_precision"][-1],
        "val_precision": history["val_precision"][-1],
        "train_recall": history["train_recall"][-1],
        "val_recall": history["val_recall"][-1],
        "train_f1": history["train_f1"][-1],
        "val_f1": history["val_f1"][-1],
        "train_loss": history["train_loss"][-1],
        "val_loss": history["val_loss"][-1],
        "epochs": len(history["train_loss"]),
        "best_val_loss": best_val_loss,
        "train_samples": train_count,
        "validation_samples": val_count,
        "test_samples": test_count,
        "test_accuracy": final_test_metrics.get("accuracy"),
        "test_precision": final_test_metrics.get("precision"),
        "test_recall": final_test_metrics.get("recall"),
        "test_f1": final_test_metrics.get("f1"),
        "model_parameters": int(parameter_count),
        "window_size": window_size,
        "dropout_rate": dropout_rate,
        "weight_decay": weight_decay,
        "scheduler_step_size": scheduler_step_size,
        "scheduler_gamma": scheduler_gamma,
        "seed": seed,
    }
