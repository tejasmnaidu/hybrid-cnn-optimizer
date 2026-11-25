# Project Overview

## Project

**Hybrid CNN and PSO-Based Harmonic Reduction in a Seven-Level Multilevel Inverter**

This project is a software research demonstrator for an ideal three-cell,
seven-level inverter. It combines a phase-shifted PWM and RL-load simulator,
reference-waveform CNN regression, event-based harmonic analysis, and a
circular-PSO experiment. The title describes the research objective; the saved
results do **not** establish meaningful harmonic reduction.

## Problem statement and objectives

Multilevel inverter switching can create voltage harmonics. The project
provides a reproducible environment to inspect simulated output waveforms,
estimate harmonic content, compare carrier-phase candidates, and evaluate a
trained CNN's reference-target regression.

Current objectives are to:

1. Reproduce the supplied project's ideal seven-level switching convention in
   a Python simulator.
2. Inspect output voltage, RL load current, cell states, and switching gates.
3. Calculate finite-band THD and event-based Parseval full-spectrum THD.
4. Predict the reference target voltage waveform from simulator signals and
   operating conditions using the saved CNN checkpoint.
5. Test a circular carrier-phase PSO procedure without using testing records.
6. Make the model's actual behavior and scientific limitations visible in the
   Streamlit dashboard.

## Current system workflow

Operating conditions → ideal inverter simulation → output voltage and load
current → CNN reference-target prediction → prediction-error analysis.

In a separate experiment, circular PSO searches cell 2 and cell 3 carrier
phase offsets on a fixed training subset. The selected candidate is then
compared with the baseline on the validation split. The live demo does not
apply that PSO candidate.

The CNN does **not** predict switching gates, inverter output voltage, or
optimized THD. It predicts the reference target waveform; CNN regression
accuracy is not evidence of harmonic reduction.

## System architecture

| Area | Existing implementation |
|---|---|
| Dataset handling | `preprocessing/` loads and validates the supplied CSV tables, aligns records by SampleID, and preserves explicit training, validation, and testing labels. |
| Normalization | `preprocessing/preprocess.py` fits input and target statistics on training records and stores the statistics in checkpoint metadata. |
| Inverter | `multilevel/inverter_simulator.py` implements the project's ideal, sampled three-cell PWM and periodic RL-load recurrence. |
| CNN | `models/` defines a separate-branch residual 1D CNN using inverter voltage, load current, and four operating-condition features. |
| Training | `training/` accepts training and validation loaders; validation loss selects checkpoints. No implicit training occurs when the application starts. |
| Evaluation | `evaluation/cnn_test_evaluator.py` performs batched inference on records explicitly labelled testing. |
| Harmonics | `evaluation/event_based_thd.py` analytically integrates piecewise-constant switching intervals; FFT-based utilities provide sampled finite-band THD. |
| Optimization | `optimization/` includes the dedicated circular phase-PSO backend and saved reproducible runs. |
| Interface | `app.py` provides the existing dark engineering dashboard, CNN and test-evaluation tabs, data diagnostics, a live demonstration, and a one-click exhibition view. |

## Dataset and trained model

The supplied dataset contains 10,000 records: 7,000 training, 1,500
validation, and 1,500 testing records. Each waveform contains 400 points at
20 kHz for a 50 Hz cycle.

The default saved checkpoint is:

`colab_runs/baseline_20261120_080836/checkpoints/best.pt`

The audit strictly loaded this checkpoint into the current 201,601-parameter
CNN. Its training-only normalization values match those recomputed from the
current training split. The checkpoint identifies zero-based epoch 97
(98th epoch) as the selected epoch, with validation loss 0.0076394975.
However, it does not embed an immutable source dataset hash or source-code
revision, so exact historical training-data identity is only partially
verifiable.

## Verified evaluation and optimization evidence

The complete held-out test split was evaluated once for the scientific audit:
1,500 records (600,000 waveform points). Pooled MAE was 1.224758 V, pooled
RMSE was 1.622291 V, global R² was 0.99989901, and global correlation was
0.99994951. The analytical reference formula derived from operating
conditions had held-out RMSE about 0.0000179 V, much lower than the CNN RMSE.
Thus, the CNN improves over a constant training-mean predictor, but it does
not improve on the known analytical target formula.

The saved 20-run PSO result improved the 30-record training objective by only
0.00001221 percentage points. On validation, selected-candidate THD was
0.00004996 percentage points higher than baseline; the paired bootstrap 95%
interval included zero. This does not support a meaningful harmonic-reduction
claim.

Detailed evidence and output files are in
[`SCIENTIFIC_VALIDATION_REPORT.md`](./SCIENTIFIC_VALIDATION_REPORT.md) and
`results/scientific_validation_20261120_205301/`.

## Technologies

- Python and NumPy for data, simulation, and numerical analysis
- PyTorch for CNN definition, checkpoint loading, and inference
- Matplotlib and Streamlit for the local interactive dashboard
- `unittest` for focused backend and evaluator tests

## Run locally

From the project directory, with its local virtual environment:

```powershell
.\.venv\Scripts\python.exe -B -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Open `http://127.0.0.1:8501`. The application loads the default checkpoint
when it is available and compatible. It does not train the CNN at startup.

## Further reading

- [Scientific validation report](./SCIENTIFIC_VALIDATION_REPORT.md)
- [Exhibition presentation guide](./EXHIBITION_PRESENTATION_GUIDE.md)
- [Project limitations](./PROJECT_LIMITATIONS.md)
- [Baseline feasibility audit](./BASELINE_FEASIBILITY_AUDIT.md)
- [Optimization design](./OPTIMIZATION_DESIGN.md)
