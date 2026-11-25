"""Training loop with validation-only model selection and resumable checkpoints."""

from __future__ import annotations

import csv
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from models.cnn_model import detect_device
from .checkpoint import load_checkpoint, save_checkpoint
from .losses import WaveformLoss


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 15
    min_delta: float = 1e-5
    loss: str = "mae"
    derivative_weight: float = 0.0
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    minimum_learning_rate: float = 1e-6
    gradient_clip_norm: float | None = 1.0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.patience < 1 or self.scheduler_patience < 0:
            raise ValueError("epochs and early-stopping patience must be positive; scheduler patience non-negative.")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.min_delta < 0:
            raise ValueError("Learning rate must be positive; weight decay and min_delta non-negative.")
        if not 0 < self.scheduler_factor < 1 or self.minimum_learning_rate < 0:
            raise ValueError("scheduler_factor must lie in (0, 1) and minimum_learning_rate be non-negative.")
        if self.minimum_learning_rate > self.learning_rate:
            raise ValueError("minimum_learning_rate cannot exceed learning_rate.")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive when set.")


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: TrainConfig | None = None,
        *,
        device: torch.device | str | None = None,
        checkpoint_path: str | Path = "checkpoints/best.pt",
        history_path: str | Path = "results/training_history.csv",
        optimizer: Optimizer | None = None,
        checkpoint_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.config = config or TrainConfig()
        set_reproducibility(self.config.seed)
        self.device = torch.device(device) if device is not None else detect_device()
        self.model = model.to(self.device)
        self.loss_fn = WaveformLoss(self.config.loss, self.config.derivative_weight)
        self.optimizer = optimizer or AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.config.scheduler_factor,
            patience=self.config.scheduler_patience,
            min_lr=self.config.minimum_learning_rate,
        )
        self.checkpoint_path = Path(checkpoint_path)
        self.last_checkpoint_path = self.checkpoint_path.with_name("last.pt")
        self.history_path = Path(history_path)
        self.checkpoint_metadata = checkpoint_metadata or {}
        self.history: list[dict[str, float]] = []
        self.best_validation_loss = float("inf")
        self.start_epoch = 0
        self.epochs_without_improvement = 0

    def resume(self, path: str | Path | None = None) -> int:
        resume_path = Path(path) if path is not None else (
            self.last_checkpoint_path if self.last_checkpoint_path.is_file() else self.checkpoint_path
        )
        state = load_checkpoint(resume_path, map_location=self.device)
        self.model.load_state_dict(state["model_state"])
        if "optimizer_state" in state:
            self.optimizer.load_state_dict(state["optimizer_state"])
        if "scheduler_state" in state:
            self.scheduler.load_state_dict(state["scheduler_state"])
        self.history = list(state.get("history", []))
        self.best_validation_loss = float(state.get("best_validation_loss", float("inf")))
        self.epochs_without_improvement = int(state.get("epochs_without_improvement", 0))
        self.start_epoch = int(state.get("epoch", -1)) + 1
        return self.start_epoch

    def _run_epoch(self, loader: DataLoader, *, training: bool) -> float:
        if training:
            self.model.train()
        else:
            self.model.eval()
        total_loss = 0.0
        total_examples = 0
        for batch in loader:
            voltage = batch["input_voltage"].to(self.device, non_blocking=True)
            current = batch["load_current"].to(self.device, non_blocking=True)
            conditions = batch["operating_conditions"].to(self.device, non_blocking=True)
            target = batch["target_voltage"].to(self.device, non_blocking=True)
            if training:
                self.optimizer.zero_grad(set_to_none=True)
                prediction = self.model(voltage, current, conditions)
                loss = self.loss_fn(prediction, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Training loss became non-finite.")
                loss.backward()
                if self.config.gradient_clip_norm is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                self.optimizer.step()
            else:
                with torch.no_grad():
                    prediction = self.model(voltage, current, conditions)
                    loss = self.loss_fn(prediction, target)
            if not torch.isfinite(prediction).all() or not torch.isfinite(loss):
                raise FloatingPointError("Model prediction or loss became non-finite.")
            count = voltage.shape[0]
            total_loss += float(loss.detach().item()) * count
            total_examples += count
        if total_examples == 0:
            raise ValueError("Cannot run an epoch with an empty DataLoader.")
        return total_loss / total_examples

    def _save_history(self) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "validation_loss", "learning_rate"])
            writer.writeheader()
            writer.writerows(self.history)

    def fit(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        *,
        resume: bool = False,
    ) -> list[dict[str, float]]:
        """Fit using only training data and validation data for selection.

        A test loader is deliberately not accepted here.
        """
        if resume:
            self.resume()
        if getattr(getattr(train_loader, "dataset", None), "split", None) not in (None, "training"):
            raise ValueError("train_loader must contain only training records.")
        if getattr(getattr(validation_loader, "dataset", None), "split", None) not in (None, "validation"):
            raise ValueError("validation_loader must contain only validation records.")

        for epoch in range(self.start_epoch, self.config.epochs):
            train_loss = self._run_epoch(train_loader, training=True)
            validation_loss = self._run_epoch(validation_loader, training=False)
            self.scheduler.step(validation_loss)
            learning_rate = float(self.optimizer.param_groups[0]["lr"])
            record = {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": learning_rate,
            }
            self.history.append(record)
            improved = validation_loss < self.best_validation_loss - self.config.min_delta
            if improved:
                self.best_validation_loss = validation_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            checkpoint_state = {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "best_validation_loss": self.best_validation_loss,
                "epochs_without_improvement": self.epochs_without_improvement,
                "history": self.history,
                "train_config": asdict(self.config),
                "metadata": self.checkpoint_metadata,
            }
            save_checkpoint(self.last_checkpoint_path, checkpoint_state)
            if improved:
                save_checkpoint(self.checkpoint_path, checkpoint_state)
            self._save_history()
            if self.epochs_without_improvement >= self.config.patience:
                break
        if self.checkpoint_path.is_file():
            best_state = load_checkpoint(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(best_state["model_state"])
        return self.history
