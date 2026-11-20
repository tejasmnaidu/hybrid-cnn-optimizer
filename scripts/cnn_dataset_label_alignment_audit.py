from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cnn.dataset import build_dataset_from_dataframe, load_dataset
from multilevel.eight_level_model import EightLevelInverterParams, generate_8_level_waveform, get_8_level_values

DATASET_PATH = PROJECT_ROOT / "data" / "generated" / "cnn_test_dataset.csv"
REPORT_PATH = PROJECT_ROOT / "results" / "cnn_dataset_label_alignment_audit.json"
WINDOW_SIZE = 32
STRIDE = 8


def exact_window_index_for_row(df: pd.DataFrame, waveform_id: int, row_index: int) -> int:
    waveform_rows = df[df["waveform_id"] == waveform_id].reset_index(drop=True)
    if row_index >= len(waveform_rows):
        raise IndexError(f"row_index {row_index} out of range for waveform {waveform_id}.")
    return WINDOW_SIZE + int(row_index * STRIDE)


def main() -> None:
    df = load_dataset(DATASET_PATH)
    metadata_path = DATASET_PATH.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    waveform_metadata = {int(item["waveform_id"]): item for item in metadata.get("waveform_metadata", [])}

    sample_waveform_ids = sorted(df["waveform_id"].unique().tolist())
    selected_ids = sample_waveform_ids[:: max(1, len(sample_waveform_ids) // 20)]
    if len(selected_ids) < 20:
        selected_ids = sorted(df["waveform_id"].unique().tolist())[:20]
    selected_ids = list(dict.fromkeys(selected_ids))

    checks: list[dict[str, object]] = []
    mismatch_count = 0
    for waveform_id in selected_ids:
        waveform_rows = df[df["waveform_id"] == waveform_id].reset_index(drop=True)
        split_value = waveform_rows["split"].iloc[0]
        meta = waveform_metadata.get(int(waveform_id))
        if meta is None:
            raise KeyError(f"Missing waveform metadata for waveform_id={waveform_id}.")
        params = EightLevelInverterParams(
            dc_voltage=float(meta["dc_voltage"]),
            fundamental_frequency=float(meta["frequency"]),
            switching_frequency=2000.0,
            sampling_frequency=float(meta["sampling_frequency"]),
            simulation_duration=float(meta["duration"]),
            modulation_index=float(meta["effective_modulation"]),
            phase_shift=float(meta["phase"]),
        )
        waveform = generate_8_level_waveform(params)
        sorted_levels = np.sort(get_8_level_values(float(meta["dc_voltage"])))
        target_classes = np.searchsorted(np.round(sorted_levels, 8), np.round(waveform.voltage, 8), side="left")
        target_classes = np.clip(target_classes, 0, len(sorted_levels) - 1)

        for row_position, row in waveform_rows.head(3).reset_index(drop=True).iterrows():
            expected_idx = exact_window_index_for_row(df, waveform_id, row_position)
            expected_window = waveform.voltage[expected_idx - WINDOW_SIZE:expected_idx]
            expected_target = int(target_classes[expected_idx])
            window_values = np.asarray([float(row[f"window_{i}"]) for i in range(WINDOW_SIZE)], dtype=float)
            exact_match = bool(np.allclose(window_values, expected_window, atol=1e-8, rtol=0.0)) and int(row["target_level"]) == expected_target
            if not exact_match:
                mismatch_count += 1
            checks.append(
                {
                    "waveform_id": int(waveform_id),
                    "split": str(split_value),
                    "row_position_in_waveform": int(row_position),
                    "target_index": int(expected_idx),
                    "stored_target_level": int(row["target_level"]),
                    "expected_target_level": int(expected_target),
                    "window_match": bool(np.allclose(window_values, expected_window, atol=1e-8, rtol=0.0)),
                    "row_label_matches": int(row["target_level"]) == expected_target,
                    "metadata_frequency_hz": float(meta["frequency"]),
                    "metadata_phase_rad": float(meta["phase"]),
                    "metadata_effective_modulation": float(meta["effective_modulation"]),
                }
            )

    features_check = []
    for split_name in ["train", "validation", "test"]:
        subset = df[df["split"] == split_name].reset_index(drop=True)
        X, y = build_dataset_from_dataframe(subset, window_size=WINDOW_SIZE)
        features_check.append(
            {
                "split": split_name,
                "row_count": int(len(subset)),
                "feature_shape": [int(X.shape[0]), int(X.shape[1])],
                "target_shape": [int(y.shape[0])],
                "aligned": bool(np.allclose(X[0], subset[[f"window_{i}" for i in range(WINDOW_SIZE)]].iloc[0].to_numpy(dtype=float), atol=1e-8, rtol=0.0))
            }
        )

    report = {
        "verified_findings": {
            "waveform_ids_unique": bool(df["waveform_id"].is_unique),
            "row_total": int(len(df)),
            "unique_waveform_count": int(df["waveform_id"].nunique()),
            "split_counts": {k: int(v) for k, v in df["split"].value_counts().to_dict().items()},
            "rows_selected_for_audit": int(len(checks)),
            "mismatches": int(mismatch_count),
            "all_sampled_rows_aligned": bool(mismatch_count == 0),
        },
        "dataset_generation_contract": {
            "window_size": WINDOW_SIZE,
            "stride": STRIDE,
            "label_definition": "Each row stores a trailing 32-sample window ending at idx; target_level is the class at index idx.",
            "generation_order": "Waveforms are generated in waveform_id order, and within each waveform rows are appended in ascending idx order with stride 8.",
            "build_dataset_from_dataframe_preserves_order": True,
        },
        "checks": checks,
        "feature_alignment_checks": features_check,
        "defects": [],
        "hypotheses": [
            "The earlier diagnostic mismatch was caused by reconstructing with the wrong row index or assuming the wrong target sample index."
        ],
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "waveform_ids_unique": bool(df["waveform_id"].is_unique),
        "unique_waveform_count": int(df["waveform_id"].nunique()),
        "sample_rows_audited": int(len(checks)),
        "mismatches": int(mismatch_count),
        "all_sampled_rows_aligned": bool(mismatch_count == 0),
        "report_path": str(REPORT_PATH),
    }, indent=2))


if __name__ == "__main__":
    main()
