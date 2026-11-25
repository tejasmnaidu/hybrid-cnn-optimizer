"""Loading and ID-based alignment helpers for the inverter CSV dataset."""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd


TABLE_FILES = {
    "input_voltage": "Input_Voltage.csv",
    "target_voltage": "Target_Voltage.csv",
    "load_current": "Load_Current.csv",
    "operating_conditions": "Operating_Conditions.csv",
    "thd_results": "THD_Results.csv",
    "sample_splits": "Sample_Splits.csv",
    "configuration": "Configuration.csv",
}
OPTIONAL_AUXILIARY_FILES = {
    "switching_gates": "Switching_Gates.csv",
    "cell_states": "Cell_States.csv",
}


class DatasetLoadError(ValueError):
    """Raised when dataset files cannot be located or loaded."""


@dataclass(frozen=True)
class DatasetTables:
    """Loaded CSV tables. Auxiliary gate/state matrices are opt-in due to size."""

    tables: Mapping[str, pd.DataFrame]
    root: Path

    def __getitem__(self, name: str) -> pd.DataFrame:
        return self.tables[name]


def extract_dataset_zip(zip_path: str | Path, destination: str | Path) -> Path:
    """Extract the known project CSVs from a dataset ZIP into a flat directory."""
    archive = Path(zip_path)
    target = Path(destination)
    if not archive.is_file():
        raise FileNotFoundError(f"Dataset ZIP does not exist: {archive}")
    expected = {name.lower() for name in (*TABLE_FILES.values(), *OPTIONAL_AUXILIARY_FILES.values())}
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        members: dict[str, zipfile.ZipInfo] = {}
        for info in source.infolist():
            basename = Path(info.filename.replace("\\", "/")).name
            if basename.lower() in expected and not info.is_dir():
                if basename.lower() in members:
                    raise DatasetLoadError(f"ZIP contains multiple files named {basename}.")
                members[basename.lower()] = info
        missing = {name for name in TABLE_FILES.values() if name.lower() not in members}
        if missing:
            raise DatasetLoadError(f"ZIP is missing required CSV files: {sorted(missing)}")
        for lowered, info in members.items():
            output = target / Path(info.filename.replace("\\", "/")).name
            with source.open(info) as file_in, output.open("wb") as file_out:
                while chunk := file_in.read(1024 * 1024):
                    file_out.write(chunk)
    return target


def _find_files(root: Path, include_auxiliary: bool) -> dict[str, Path]:
    expected = dict(TABLE_FILES)
    if include_auxiliary:
        expected.update(OPTIONAL_AUXILIARY_FILES)
    files = list(root.rglob("*.csv"))
    by_name: dict[str, list[Path]] = {}
    for path in files:
        by_name.setdefault(path.name.lower(), []).append(path)
    found: dict[str, Path] = {}
    for key, filename in expected.items():
        matches = by_name.get(filename.lower(), [])
        if len(matches) > 1:
            raise DatasetLoadError(f"Multiple {filename} files found under {root}.")
        if matches:
            found[key] = matches[0]
    missing = set(expected) - set(found)
    if missing:
        names = [expected[name] for name in sorted(missing)]
        raise DatasetLoadError(f"Missing required dataset files: {names}")
    return found


def load_dataset(
    source: str | Path,
    *,
    include_auxiliary: bool = False,
    extract_to: str | Path | None = None,
) -> DatasetTables:
    """Load project CSVs, extracting a ZIP automatically when needed.

    Gate/state CSVs are excluded by default because they contain tens of millions
    of values and are validated in chunks by the validation module.
    """
    path = Path(source)
    if path.is_file() and path.suffix.lower() == ".zip":
        destination = Path(extract_to) if extract_to else path.with_name(f"{path.stem}_extracted")
        extract_dataset_zip(path, destination)
        root = destination
    elif path.is_dir():
        root = path
    else:
        raise FileNotFoundError(f"Dataset source must be a directory or ZIP file: {path}")

    files = _find_files(root, include_auxiliary)
    tables: dict[str, pd.DataFrame] = {}
    for name, file_path in files.items():
        try:
            tables[name] = pd.read_csv(file_path)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            raise DatasetLoadError(f"Could not load {file_path}: {exc}") from exc
    return DatasetTables(tables=tables, root=root)
