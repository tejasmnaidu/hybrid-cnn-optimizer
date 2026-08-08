from __future__ import annotations

import traceback
from pathlib import Path
import sys
import os
import numpy as np
import pandas as pd

# Ensure project root is on sys.path so local packages (cnn, multilevel, signal_processing) import correctly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cnn.dataset import generate_cnn_dataset
from cnn.train import train_cnn
from cnn.controller import run_cnn_controller
from cnn.predict import predict_switching_state


def main() -> int:
    try:
        base = Path.cwd()
        dataset_path = base / "data" / "generated" / "cnn_smoke_test.csv"
        model_path = base / "models" / "cnn_smoke_test.pth"
        history_path = base / "results" / "cnn_smoke_history.csv"
        confusion_path = base / "results" / "cnn_smoke_confusion.csv"

        print("Generating dataset...")
        df = generate_cnn_dataset(dataset_path, num_waveforms=50)
        print(f"created {dataset_path}")

        def progress(epoch: int, metrics: dict[str, float]) -> None:
            print(f"epoch={epoch} metrics={metrics}")

        print("Training CNN...")
        train_res = train_cnn(
            dataset_csv=dataset_path,
            model_path=model_path,
            history_path=history_path,
            confusion_path=confusion_path,
            epochs=3,
            batch_size=16,
            learning_rate=0.001,
            progress_callback=progress,
        )

        # Report
        print('\n=== CNN SMOKE TEST ===\n')
        print('Dataset:')
        print('  rows:', len(df))
        classes = sorted(df['target_level'].unique().tolist())
        print('  classes:', classes)

        print('\nTraining:')
        print('  epochs:', train_res.get('epochs'))
        print('  final training accuracy:', train_res.get('train_accuracy'))
        print('  final validation accuracy:', train_res.get('val_accuracy'))
        print('  final training loss:', train_res.get('train_loss'))
        print('  final validation loss:', train_res.get('val_loss'))

        print('\nModel:')
        print('  path:', str(model_path))
        print('  exists:', model_path.exists())
        if model_path.exists():
            print('  size:', model_path.stat().st_size)

        # Pick a representative waveform to run controller/inference.
        # Use params from the first sample_id in the generated dataframe.
        first_sample = df.iloc[0]
        # The dataset records store per-window rows; retrieve the waveform-level params
        dc_voltage = float(first_sample['dc_voltage'])
        modulation_index = float(first_sample['modulation_index'])
        frequency = float(first_sample['frequency'])
        phase = float(first_sample['phase'])
        sampling_frequency = 10000.0
        duration = 0.04

        print('\nRunning controller/inference...')
        controller_res = run_cnn_controller(
            dc_voltage=dc_voltage,
            fundamental_frequency=frequency,
            sampling_frequency=sampling_frequency,
            duration=duration,
            modulation_index=modulation_index,
            phase_shift=phase,
            model_path=model_path,
        )

        predicted_states = np.array(controller_res.cnn_levels)
        predicted_voltage = np.array(controller_res.cnn_voltage)

        unique_states = np.unique(predicted_states)

        print('\nInference:')
        print('  prediction successful:', predicted_states.size > 0)
        print('  number of predicted states:', int(predicted_states.size))
        print('  unique predicted states:', unique_states.tolist())
        rms = float(np.sqrt(np.mean(predicted_voltage ** 2)))
        print('  RMS voltage:', rms)
        try:
            print('  THD (cnn):', float(controller_res.thd_cnn))
            print('  THD (reference):', float(controller_res.thd_reference))
            print('  THD difference:', float(controller_res.thd_difference))
        except Exception:
            print('  THD/reference metrics: not available')

        # Verifications
        print('\nVerifications:')
        print('  dataset exists:', dataset_path.exists())
        print('  model exists:', model_path.exists())
        print('  history exists:', history_path.exists())
        print('  confusion exists:', confusion_path.exists())

        # Validate predicted classes are in 0..7
        valid_classes = set(range(8))
        predicted_valid = set(unique_states.tolist()).issubset(valid_classes)
        print('  prediction classes valid (0..7):', predicted_valid)

        # Brief summary of learning
        print('\nSummary:')
        # Check if training improved beyond random (very basic check)
        val_acc = train_res.get('val_accuracy', 0.0)
        print('  final validation accuracy:', val_acc)
        if val_acc > 0.2:
            print('  model appears to have learned patterns (val_acc > 0.2)')
        else:
            print('  model did not learn much (val_acc <= 0.2)')

        return 0

    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        print('\n=== ERROR DURING SMOKE TEST ===')
        print(tb)
        # Identify offending frame
        tb_list = traceback.extract_tb(e.__traceback__)
        if tb_list:
            last = tb_list[-1]
            print('Error location:', last.filename, 'function', last.name)
        return 2


if __name__ == '__main__':
    sys.exit(main())
