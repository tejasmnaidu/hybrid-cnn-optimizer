"""Generate a separate, reproducible dataset from the event-based evaluator."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from evaluation.event_based_thd import evaluate_event_based_thd


DATASET_COLUMNS = (
    "modulation_index",
    "phase_angle_degrees",
    "cell_2_phase_degrees",
    "cell_3_phase_degrees",
    "event_total_thd_percent",
    "fundamental_phasor_error_percent",
)


def generate_thd_surrogate_dataset(
    *,
    sample_count: int,
    seed: int,
) -> np.ndarray:
    """Return candidate inputs and exact event-evaluator labels."""
    if sample_count < 2:
        raise ValueError("sample_count must be at least 2.")
    if not 0 <= seed <= 2**31 - 1:
        raise ValueError("seed must be in [0, 2**31 - 1].")
    rng = np.random.default_rng(seed)
    unit_samples = np.empty((sample_count, 4), dtype=np.float64)
    for column in range(unit_samples.shape[1]):
        strata = (np.arange(sample_count) + rng.random(sample_count)) / sample_count
        unit_samples[:, column] = strata[rng.permutation(sample_count)]
    candidates = np.empty_like(unit_samples)
    candidates[:, 0] = 0.55 + 0.40 * unit_samples[:, 0]
    candidates[:, 1:] = 360.0 * unit_samples[:, 1:]
    candidates[:, 2:] = np.sort(candidates[:, 2:], axis=1)
    candidates[0] = (0.75, 0.0, 60.0, 120.0)

    rows = np.empty((sample_count, len(DATASET_COLUMNS)), dtype=np.float64)
    for index, candidate in enumerate(candidates):
        modulation, phase_degrees, cell_2, cell_3 = candidate
        evaluation = evaluate_event_based_thd(
            float(modulation),
            np.deg2rad(float(phase_degrees)),
            carrier_phase_offsets_degrees=(0.0, float(cell_2), float(cell_3)),
        )
        rows[index] = (
            *candidate,
            evaluation.total_thd_percent,
            evaluation.fundamental_phasor_error_percent,
        )
    if not np.isfinite(rows).all():
        raise RuntimeError("Event evaluator generated non-finite surrogate data.")
    return rows


def write_thd_surrogate_dataset(
    output_path: str | Path,
    *,
    sample_count: int,
    seed: int,
) -> Path:
    """Write a new dataset without replacing any existing file."""
    path = Path(output_path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing surrogate dataset: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = generate_thd_surrogate_dataset(sample_count=sample_count, seed=seed)
    with path.open("x", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(DATASET_COLUMNS)
        writer.writerows(rows.tolist())
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/thd_surrogate_v1/dataset.csv"),
    )
    parser.add_argument("--samples", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=20261125)
    arguments = parser.parse_args()
    path = write_thd_surrogate_dataset(
        arguments.output,
        sample_count=arguments.samples,
        seed=arguments.seed,
    )
    print(f"Wrote {arguments.samples} rows to {path}")


if __name__ == "__main__":
    main()
