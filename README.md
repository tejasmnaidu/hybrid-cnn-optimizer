# Hybrid CNN Inverter Optimization

Research project for a seven-level cascaded H-bridge inverter, analytical
simulation, waveform prediction, and simulator-verified carrier-phase
optimization. The CNN predicts a reference waveform; it does not predict
switching states or control the inverter. Training is not started implicitly:
the Colab notebook and CLI contain explicit training cells/commands.

## Verified dataset

The supplied `Seven_Level_Inverter_Project.zip` contains CSVs under
`Seven_Level_Inverter_Project/Excel_CSV/`. Its `Configuration.csv` specifies:

- 10,000 records: 7,000 training, 1,500 validation, and 1,500 testing
- 400 points per waveform, 20 kHz sampling, 50 Hz fundamental (400 samples/cycle)
- 3 x 100 V DC sources, three H-bridge cells, 12 switches, seven levels
- Input-voltage levels: -300, -200, -100, 0, 100, 200, and 300 V
- Gate table: 4,800 signal values per record (12 x 400 by count)
- Cell-state table: 1,200 values per record (3 x 400 by count)

The dimensional counts alone do not establish the physical gate-driver
polarity, device assignment, or switching dead time. The project includes an
ideal inverter simulator in `multilevel/`; its digital cell-state and gate
mapping is checked against the supplied software dataset. This establishes
software agreement, not hardware validation or physical wiring verification.
`preprocessing.switching_data` loads auxiliary records separately; reshape
methods require an explicit layout.

`THD_Results.csv` contains `inputVoltageTHD`, `inputCurrentTHD`, and
`targetVoltageTHD`. They are dataset-reported input/reference quantities, not
measured post-CNN or post-inverter output THD. The FFT helpers calculate THD
from the waveform passed to them; results must be labeled by the caller.

## Setup and validation

Use Python 3.10 or newer.

Keep the dataset ZIP at `data/raw/Seven_Level_Inverter_Project.zip` (or supply a
different ZIP/directory path). ZIP loading extracts the known CSV files into a
flat sibling directory. Install dependencies and validate/preprocess without
training:

```powershell
python -m pip install -r requirements.txt
python -c "from preprocessing.data_validation import validate_dataset; validate_dataset('data/raw/Seven_Level_Inverter_Project.zip', report_path='results/dataset_validation.json')"
python -c "from preprocessing.preprocess import preprocess_dataset; preprocess_dataset('data/raw/Seven_Level_Inverter_Project.zip', metadata_path='data/processed/preprocessing_metadata.json')"
python -m unittest discover -s tests -v
```

Normalization statistics are fitted only on the supplied training split.
Sample IDs are retained and tables are aligned by ID. Test records are exposed
through their own DataLoader and are not accepted by `Trainer.fit`.

## Training in Google Colab

Upload this whole project directory to Google Drive, open
`notebooks/colab_train.ipynb` in Colab, set `PROJECT_ROOT` and `DATASET_ZIP` in
the configuration cell, then run cells in order. Mount Drive to persist the
best checkpoint and experiment results. GPU is selected when available; the
same model runs on CPU. The notebook only evaluates the test split after
training and validation-based checkpoint selection.

For local training only when deliberately requested:

```powershell
python -m training.train --config configs/config.yaml --dataset data/raw/Seven_Level_Inverter_Project.zip
```

Resume an existing experiment by passing its directory:

```powershell
python -m training.train --config configs/config.yaml --dataset data/raw/Seven_Level_Inverter_Project.zip --experiment-dir experiments/<run-directory> --resume
```

## Streamlit

From the project directory, run `streamlit run app.py`. The app automatically
loads `colab_runs/baseline_20261120_080836/checkpoints/best.pt` when present;
an uploaded project checkpoint can be used to override it. The checkpoint must
contain model configuration and training-split normalization metadata.
Prediction comparisons use the selected dataset row's reference target; they
are not inverter-output measurements.

For the local Windows virtual environment, launch with:

```powershell
.\.venv\Scripts\python.exe -B -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

The **LIVE DEMONSTRATION** tab runs the ideal inverter at the exact operating
conditions entered by the user. Optimization candidates use the same live
conditions and must not be interpreted as hardware validation.

### Local AI project chat

The **AI PROJECT CHAT ASSISTANT** uses Ollama on this computer; it does not call
a cloud API. On Windows, install Ollama and download the default chat model:

```powershell
winget install --exact --id Ollama.Ollama
ollama pull qwen2.5:3b
```

Keep Ollama running while using the dashboard. In the live demonstration tab,
open the chat panel, select the installed model tag, and use **CHECK LOCAL
MODEL** to confirm availability. The assistant can explain project context and
request a switching recommendation through the existing Hybrid POA-PFO
optimizer. Recommended phases, gate states, and THD are returned by the
optimizer and inverter simulator, not generated by the language model. If the
verified result is not better, the existing advisor reports its baseline
fallback. The chatbot is not hardware control, real-time actuation, or a
protection system. Recommendation calls may take time because optimization and
simulator verification are synchronous.

## Reproducing recorded evaluations

Run commands from the project root after installing `requirements.txt`.
The existing dataset and checkpoint are treated as read-only by the evaluation
workflows.

- **CNN held-out evaluation:** use the dashboard's *CNN TEST EVALUATION* tab
  with the existing checkpoint. The saved inference-only, condition-grouped
  result is [`results/cnn_grouped_test_eval_20261124.json`](./results/cnn_grouped_test_eval_20261124.json).
- **Optimizer grid and multi-seed validation:** see
  [`docs/optimizer_validation.md`](./docs/optimizer_validation.md) and
  [`results/optimizer_grid_search.csv`](./results/optimizer_grid_search.csv).
  The current project has no standalone grid-regeneration CLI; the saved
  result is not regenerated by the unit tests.
- **CNN-guided ranking correlation and ten-seed comparison:** run
  the correlation helper and benchmark below. Run from a fresh working copy
  with the read-only checkpoint available and the correlation/benchmark output
  files absent; the report writers refuse to overwrite existing files. Execute
  the correlation code in a Python interpreter started from the project root.

  First regenerate the 500-candidate correlation report (the benchmark script
  consumes this report; it does not recalculate it):

  ```python
  import numpy as np
  from scripts.run_thd_experiment import _read_cnn
  from optimization.cnn_guided_search import (
      CORRELATION_OUTPUT,
      measure_correlation,
      write_correlation_report,
  )

  model, normalization = _read_cnn()
  candidates = np.random.default_rng(20261124).uniform(0.0, 360.0, (500, 2))
  report = measure_correlation(
      model,
      normalization,
      candidates,
      operating_point={
          "modulation_index": 0.75,
          "dc_voltage_per_cell_v": 100.0,
          "load_resistance_ohm": 30.0,
          "load_inductance_h": 0.03,
          "phase_angle_degrees": 0.0,
      },
      seed=20261124,
  )
  write_correlation_report(report, CORRELATION_OUTPUT)
  ```

  Then run `python -m scripts.run_thd_experiment` to generate the
  baseline/PSO/POA/PFO/Hybrid/CNN-guided comparison, CSV, summary, and figures.
  The CNN-guided method is included only when the saved correlation passes its
  predeclared threshold.
- **All tests:** `python -m unittest discover -s tests -v`. This validates
  code behavior and does not train the CNN or regenerate the dataset.

The checked-in THD CSV and figures are described in
[`results/summary.md`](./results/summary.md). Before comparing THD values,
check their labels: event-based total voltage THD, event-based H2–100
harmonics, and sampled H2–100 voltage/current THD have different definitions.

## Project and exhibition documents

- [Project overview](./PROJECT_OVERVIEW.md)
- [Scientific validation report](./SCIENTIFIC_VALIDATION_REPORT.md)
- [Exhibition presentation guide](./EXHIBITION_PRESENTATION_GUIDE.md)
- [Project limitations](./PROJECT_LIMITATIONS.md)
- [Final measured-results report](./docs/final_report.md)

## Modules

- `preprocessing/`: archive/CSV loading, ID alignment, validation, split-safe normalization, separate gate/state access.
- `models/`: configurable residual Conv1D architecture and device detection.
- `training/`: PyTorch datasets, train/validation loop, losses, early stopping, checkpoints, and experiment CLI.
- `optimization/`: bounded PSO, POA/PFO/hybrid search, CNN-guided ranking, constraints, and evaluator interfaces.
- `evaluation/`: live simulation, event and sampled THD, model evaluation, and plotting helpers.
- `assistant/`: offline, simulator-verified carrier-phase recommendation (not a controller).
- `multilevel/`: seven-level inverter and RL-load simulator.
- `scripts/`, `results/`, `docs/`: experiment entry points, recorded outputs, and validation notes.
- `experiments/`: per-run configuration and result directories.
- `tests/`: focused unit tests; running them does not train the model.
