"""Atomic checkpoint persistence."""

from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, state: dict[str, Any]) -> Path:
    """Atomically save a checkpoint so interrupted writes do not corrupt it."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(state, temporary)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        state = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except (OSError, RuntimeError, EOFError, ValueError, pickle.UnpicklingError) as exc:
        raise ValueError(f"Could not load checkpoint {checkpoint_path}: {exc}") from exc
    if not isinstance(state, dict) or "model_state" not in state:
        raise ValueError(f"{checkpoint_path} is not a project training checkpoint.")
    return state
