from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from cnn.dataset import build_dataset_from_dataframe, load_dataset
from cnn.model import ImprovedEightLevelCNN
from cnn.predict import load_cnn_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "generated" / "cnn_test_dataset.csv"
BASELINE_MODEL_PATH = PROJECT_ROOT / "models" / "cnn_8level_controller_test.pth"
NEW_MODEL_PATH = PROJECT_ROOT / "models" / "cnn_improved_experiment.pth"
SUMMARY_PATH = PROJECT_ROOT / "results" / "cnn_improved_experiment_summary.json"
HISTORY_PATH = PROJECT_ROOT / "results" / "cnn_improved_experiment_history.csv"


class WaveformDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx].unsqueeze(0), self.targets[idx]


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | list[float]]:
    labels = list(range(8))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class_recall": recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def _normalize_window_features(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std = np.where(std < 1e-6, 1.0, std)
    return ((features - mean) / std).astype(np.float32)


def _evaluate_split(model: nn.Module, features: np.ndarray, targets: np.ndarray, device: torch.device) -> dict[str, float | list[float]]:
    model.eval()
    inputs = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(1)
    with torch.no_grad():
        logits = model(inputs)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    return _compute_metrics(np.asarray(targets), preds)


def _run_logistic_regression(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, dict[str, object]]:
    feature_cols = [f"window_{i}" for i in range(32)]

    def make_features(df: pd.DataFrame) -> np.ndarray:
        rows = []
        for row in df[feature_cols].to_numpy():
            win = np.asarray(row, dtype=np.float32)
            rows.append([
                win.mean(),
                win.std(),
                win.min(),
                win.max(),
                win[0],
                win[-1],
                np.sqrt(np.mean(win ** 2)),
                (win[-1] - win[0]) / max(1, len(win) - 1),
                np.mean(np.abs(np.diff(win))),
                np.ptp(win),
            ])
        return np.asarray(rows, dtype=np.float32)

    X_train = make_features(train_df)
    X_val = make_features(val_df)
    X_test = make_features(test_df)
    y_train = train_df["target_level"].to_numpy(dtype=int)
    y_val = val_df["target_level"].to_numpy(dtype=int)
    y_test = test_df["target_level"].to_numpy(dtype=int)

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    model.fit(X_train, y_train)
    summary: dict[str, dict[str, object]] = {}
    for name, X, y in [("validation", X_val, y_val), ("test", X_test, y_test)]:
        pred = model.predict(X)
        summary[name] = _compute_metrics(y, pred)
    return summary


def _evaluate_existing_cnn(val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, dict[str, object]]:
    baseline_model, window_size = load_cnn_model(BASELINE_MODEL_PATH)
    baseline_model.eval()
    summaries: dict[str, dict[str, object]] = {}
    for name, split_df in [("validation", val_df), ("test", test_df)]:
        features, targets = build_dataset_from_dataframe(split_df, window_size=window_size)
        feature_mean = getattr(baseline_model, "feature_mean", None)
        feature_std = getattr(baseline_model, "feature_std", None)
        if feature_mean is not None and feature_std is not None:
            features = _normalize_window_features(features, np.asarray(feature_mean, dtype=np.float32), np.asarray(feature_std, dtype=np.float32))
        inputs = torch.tensor(features, dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            logits = baseline_model(inputs)
            preds = torch.argmax(logits, dim=1).numpy()
        summaries[name] = _compute_metrics(np.asarray(targets), preds)
    return summaries


def main() -> None:
    df = load_dataset(DATASET_PATH)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "validation"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    train_features, train_targets = build_dataset_from_dataframe(train_df, window_size=32)
    val_features, val_targets = build_dataset_from_dataframe(val_df, window_size=32)
    test_features, test_targets = build_dataset_from_dataframe(test_df, window_size=32)

    train_mean = train_features.mean(axis=0, keepdims=True)
    train_std = train_features.std(axis=0, keepdims=True)
    train_features = _normalize_window_features(train_features, train_mean, train_std)
    val_features = _normalize_window_features(val_features, train_mean, train_std)
    test_features = _normalize_window_features(test_features, train_mean, train_std)

    class_counts = np.bincount(train_targets, minlength=8).astype(np.float32)
    class_weight = np.sum(class_counts) / (8.0 * np.where(class_counts == 0, 1.0, class_counts))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(WaveformDataset(train_features, train_targets), batch_size=64, shuffle=True)
    val_loader = DataLoader(WaveformDataset(val_features, val_targets), batch_size=64, shuffle=False)
    test_loader = DataLoader(WaveformDataset(test_features, test_targets), batch_size=64, shuffle=False)

    model = ImprovedEightLevelCNN(input_channels=1, num_classes=8, dropout_rate=0.2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weight, dtype=torch.float32, device=device))

    best_state: dict[str, torch.Tensor] | None = None
    best_val_f1 = -1.0
    best_epoch = -1
    history: list[dict[str, float]] = []

    for epoch in range(1, 31):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds: list[int] = []
        val_true: list[int] = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                logits = model(x_batch.to(device))
                val_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
                val_true.extend(y_batch.cpu().tolist())

        val_metrics = _compute_metrics(np.asarray(val_true), np.asarray(val_preds))
        epoch_record = {
            "epoch": epoch,
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "val_recall": float(np.mean(val_metrics["per_class_recall"])),
        }
        history.append(epoch_record)
        if float(val_metrics["macro_f1"]) > best_val_f1:
            best_val_f1 = float(val_metrics["macro_f1"])
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("No valid model state was selected.")

    selected_model = ImprovedEightLevelCNN(input_channels=1, num_classes=8, dropout_rate=0.2)
    selected_model.load_state_dict(best_state)
    selected_model.to(device)
    selected_model.eval()

    val_metrics = _evaluate_split(selected_model, val_features, val_targets, device)
    test_metrics = _evaluate_split(selected_model, test_features, test_targets, device)

    logistic_summary = _run_logistic_regression(train_df, val_df, test_df)
    existing_cnn_summary = _evaluate_existing_cnn(val_df, test_df)

    summary = {
        "selection_rule": "validation macro-F1",
        "selected_epoch": best_epoch,
        "selected_validation_macro_f1": best_val_f1,
        "improved_cnn": {
            "validation": val_metrics,
            "test": test_metrics,
        },
        "existing_cnn": {
            "validation": existing_cnn_summary["validation"],
            "test": existing_cnn_summary["test"],
        },
        "logistic_regression": logistic_summary,
        "history": history,
    }

    NEW_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "architecture": selected_model.__class__.__name__,
        "model_config": {
            "input_channels": 1,
            "num_classes": 8,
            "dropout_rate": 0.2,
            "window_size": 32,
        },
        "model_state_dict": selected_model.state_dict(),
        "feature_mean": train_mean.reshape(-1).astype(np.float32),
        "feature_std": train_std.reshape(-1).astype(np.float32),
        "num_classes": 8,
        "dropout_rate": 0.2,
        "window_size": 32,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_val_f1,
    }, NEW_MODEL_PATH)
    pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    print("=== Improvement CNN experiment ===")
    print(f"selected_epoch={best_epoch}")
    print(f"selected_validation_macro_f1={best_val_f1:.6f}")
    print("validation_improved_cnn", val_metrics)
    print("test_improved_cnn", test_metrics)
    print("validation_existing_cnn", existing_cnn_summary["validation"])
    print("test_existing_cnn", existing_cnn_summary["test"])
    print("validation_logistic", logistic_summary["validation"])
    print("test_logistic", logistic_summary["test"])
    print(f"saved_new_checkpoint={NEW_MODEL_PATH}")
    print(f"saved_summary={SUMMARY_PATH}")


if __name__ == "__main__":
    main()
