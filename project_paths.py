"""Resolve project resources independently of the process working directory."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_project_path(
    path: str | Path,
    project_root: str | Path = PROJECT_ROOT,
) -> Path:
    """Resolve absolute paths as given and relative paths from the project root."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path(project_root) / candidate
    return candidate.resolve()
