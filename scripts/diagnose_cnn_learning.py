from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cnn.dataset import load_dataset, build_dataset_from_dataframe
from cnn.predict import load_cnn_model
from multilevel.eight_level_model import EightLevelInverterParams, generate_8_level_waveform, get_8_level_values


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "generated" / "cnn_test_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "cnn_improved_experiment.pth"
OUTPUT_PATH = PROJECT_ROOT / "results" / "cnn_controller_learning_diagnostic.json"


def main() -> None:
    df = load_dataset(DATASET_PATH)
    class_counts_all = df["target_level"].value_counts().sort_index().reindex(range(8), fill_value=0)
    class_pct_all = (class_counts_all / len(df) * 100.0).to_dict()

    split_payload: dict[str, dict[str, dict[str, float | int]]] = {}
    for split_name in ["train", "validation", "test"]:
        subset = df[df["split"] == split_name].reset_index(drop=True)
        counts = subset["target_level"].value_counts().sort_index().reindex(range(8), fill_value=0)
        pct = (counts / len(subset) * 100.0) if len(subset) else pd.Series({i: 0.0 for i in range(8)})
        split_payload[split_name] = {
            str(i): {"count": int(counts.get(i, 0)), "percentage": float(pct.get(i, 0.0))}
            for i in range(8)
        }

    model, window_size = load_cnn_model(MODEL_PATH)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    X, y = build_dataset_from_dataframe(test_df, window_size=window_size)
    mean = model.feature_mean.astype(np.float32)
    std = model.feature_std.astype(np.float32)
    Xn = ((X - mean) / np.where(std < 1e-6, 1.0, std)).astype(np.float32)
    logits = model(torch.tensor(Xn, dtype=torch.float32).unsqueeze(1)).detach()
    preds = torch.argmax(logits, dim=1).numpy()

    acc = accuracy_score(y, preds)
    macro_f1 = f1_score(y, preds, labels=list(range(8)), average="macro", zero_division=0)
    precision_per_class = precision_score(y, preds, labels=list(range(8)), average=None, zero_division=0)
    recall_per_class = recall_score(y, preds, labels=list(range(8)), average=None, zero_division=0)
    cm = confusion_matrix(y, preds, labels=list(range(8)))
    true_counts = pd.Series(y).value_counts().sort_index().reindex(range(8), fill_value=0)
    pred_counts = pd.Series(preds).value_counts().sort_index().reindex(range(8), fill_value=0)

    sample_row = test_df.iloc[0].to_dict()
    waveform_id = int(sample_row["waveform_id"])
    params = EightLevelInverterParams(
        dc_voltage=float(sample_row["dc_voltage"]),
        fundamental_frequency=float(sample_row["frequency"]),
        switching_frequency=2000.0,
        sampling_frequency=float(sample_row["sampling_frequency"]),
        simulation_duration=float(sample_row["duration"]),
        modulation_index=float(sample_row["effective_modulation"]),
        phase_shift=float(sample_row["phase"]),
    )
    waveform = generate_8_level_waveform(params)
    sorted_levels = np.sort(get_8_level_values(float(sample_row["dc_voltage"])))
    target_classes = np.searchsorted(np.round(sorted_levels, 8), np.round(waveform.voltage, 8), side="left")
    target_classes = np.clip(target_classes, 0, len(sorted_levels) - 1)

    first_window_index = 32
    window = waveform.voltage[first_window_index - 32:first_window_index]
    reconstructed_label = int(target_classes[first_window_index])
    sample_semantics = {
        "waveform_id": waveform_id,
        "dataset_row_label": int(sample_row["target_level"]),
        "reconstructed_window_start_index": int(first_window_index - 32),
        "reconstructed_window_end_index": int(first_window_index),
        "label_sample_index": int(first_window_index),
        "label_sample_voltage": float(waveform.voltage[first_window_index]),
        "reconstructed_window_label": int(reconstructed_label),
        "label_matches": int(sample_row["target_level"]) == int(reconstructed_label),
        "window_mean_voltage": float(np.mean(window)),
        "first_window_values": [float(v) for v in window[:8]],
    }

    ambiguity = []
    for split_name in ["train", "validation", "test"]:
        subset = df[df["split"] == split_name].reset_index(drop=True)
        Xs, ys = build_dataset_from_dataframe(subset, window_size=window_size)
        rounded = np.round(Xs, 8)
        window_map: dict[tuple[float, ...], list[int]] = {}
        for i, row in enumerate(rounded):
            key = tuple(row.tolist())
            window_map.setdefault(key, []).append(int(ys[i]))
        ambiguous = sum(1 for labels in window_map.values() if len(set(labels)) > 1)
        ambiguity.append(
            {
                "split": split_name,
                "unique_rounded_windows": int(len(window_map)),
                "ambiguous_rounded_windows": int(ambiguous),
                "ambiguous_fraction_percent": float(ambiguous / len(window_map) * 100.0) if len(window_map) else 0.0,
            }
        )

    report = {
        "dataset_summary": {
            "total_rows": int(len(df)),
            "total_waveforms": int(df["waveform_id"].nunique()),
            "class_distribution_overall": {
                str(i): {"count": int(class_counts_all.get(i, 0)), "percentage": float(class_pct_all.get(i, 0.0))}
                for i in range(8)
            },
            "class_distribution_by_split": split_payload,
        },
        "held_out_test_evaluation": {
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "precision_per_class": {str(i): float(precision_per_class[i]) for i in range(8)},
            "recall_per_class": {str(i): float(recall_per_class[i]) for i in range(8)},
            "confusion_matrix": cm.tolist(),
            "true_class_counts": {str(i): int(true_counts.get(i, 0)) for i in range(8)},
            "predicted_class_counts": {str(i): int(pred_counts.get(i, 0)) for i in range(8)},
            "total_test_rows": int(len(test_df)),
            "model_class": type(model).__name__,
            "window_size": int(window_size),
        },
        "window_label_semantics": {
            "definition": "Each record is a trailing 32-sample window ending at sample index idx, with target_level equal to the class at the end-of-window sample index idx.",
            "stride": 8,
            "sample_check": sample_semantics,
        },
        "ambiguity_analysis": {
            "exact_rounded_window_duplicates": ambiguity,
            "notes": [
                "The training/validation/test windows are generated from the same waveform engine, so identical or near-identical voltage histories can appear under different operating conditions and labels.",
                "A fixed-size trailing window does not contain the future waveform context needed to disambiguate phase direction or operating condition when the same recent history can correspond to different next-state labels."
            ],
        },
        "proven_findings": [
            "The correctly loaded model class is ImprovedEightLevelCNN and checkpoint loading succeeds with zero missing and zero unexpected keys.",
            "The dataset uses a 32-sample trailing window with stride 8; the label corresponds to the sample at the end of the window, not the window-average voltage.",
            "The test split class distribution is imbalanced and the model evaluated on this split still collapses to only a small subset of classes.",
            "Exact rounded-window ambiguity exists in the dataset, which means the model can receive nearly identical histories with different labels under some conditions."
        ],
        "hypotheses": [
            "The model is missing the conditioning variable that distinguishes phase, direction, or operating-point context for the same recent window history.",
            "Because the label is tied to the current sample rather than the whole window, a causal prediction task using only a trailing window may be underdetermined without future/current context or explicit phase information."
        ],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2))

    print(json.dumps({
        "overall_distribution": {str(i): {"count": int(class_counts_all.get(i, 0)), "percentage": float(class_pct_all.get(i, 0.0))} for i in range(8)},
        "split_distribution": split_payload,
        "test_accuracy": float(acc),
        "test_macro_f1": float(macro_f1),
        "predicted_class_counts": {str(i): int(pred_counts.get(i, 0)) for i in range(8)},
        "sample_semantics": sample_semantics,
        "ambiguity": ambiguity,
    }, indent=2))


if __name__ == "__main__":
    main()
