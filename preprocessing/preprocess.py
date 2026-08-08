from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd


def load_datasets(data_dir: str) -> dict[str, Any]:
    """Load raw CSV datasets from the specified directory."""
    base_path = Path(data_dir)
    records: dict[str, Any] = {}

    for filename in [
        "Solar_Energy_Generation.csv",
        "Weather_Data_reordered_all.csv",
        "Solar_Site_Details.csv",
        "Monthly_Summary_Solar.csv",
    ]:
        path = base_path / filename
        if path.exists():
            records[filename] = pd.read_csv(path)
        else:
            records[filename] = f"File not found: {path}"

    return records


def preprocess_data(input_dir: str, output_dir: str) -> dict[str, Any]:
    """Perform high-level preprocessing and save processed output."""
    os.makedirs(output_dir, exist_ok=True)
    loaded = load_datasets(input_dir)
    return {"loaded_files": list(loaded.keys()), "status": "placeholder"}
