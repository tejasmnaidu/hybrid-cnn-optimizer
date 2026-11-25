# Scientific Validation Audit

**Project:** Hybrid CNN and PSO-Based Harmonic Reduction in a Seven-Level Multilevel Inverter  
**Audit scope:** Saved CNN checkpoint, the complete held-out test split, operating-condition slices, live demo, and saved PSO experiment.  
**Overall assessment:** **PARTIALLY READY for a software-simulation exhibition. The available evidence does not support a claim that the CNN reduces inverter harmonics, and the saved PSO result does not show a meaningful or validation-confirmed improvement.**

## Status key

- **VERIFIED** — directly checked against code or measured in this audit.
- **PARTIALLY VERIFIED** — supporting evidence exists, but an important provenance or coverage limitation remains.
- **NOT VERIFIED** — required evidence or validation is absent.
- **FAILED** — evidence shows that the stated claim is not met.

## 1. Training provenance

| Question | Status | Evidence and conclusion |
|---|---|---|
| Which dataset was used? | **PARTIALLY VERIFIED** | [`configs/config.yaml`](./configs/config.yaml) names `data/raw/Seven_Level_Inverter_Project.zip` and its extracted directory. [`training/train.py`](./training/train.py) uses the ZIP as its default dataset. The checkpoint is stored under `colab_runs/baseline_20261120_080836/`. The current extracted dataset has 10,000 rows and its training-only preprocessing statistics exactly match the checkpoint. However, the checkpoint metadata contains no dataset path, archive hash, or CSV hashes, so it cannot prove which exact archive produced those weights. Current dataset hashes are recorded in the audit JSON. |
| Training/validation/test counts | **VERIFIED** | The current split manifest and preprocessing report contain 7,000 training, 1,500 validation, and 1,500 testing records. The split-aware loader creates explicit subsets. |
| Was the test split used for training or checkpoint selection? | **VERIFIED for the inspected training code; PARTIALLY VERIFIED for checkpoint provenance** | [`training/trainer.py`](./training/trainer.py) accepts only train and validation loaders, checks their labels, and selects `best.pt` using validation loss. The CLI passes those two loaders to `fit`; no test loader is accepted. [`preprocessing/preprocess.py`](./preprocessing/preprocess.py) fits normalization statistics on training rows only. The existing split manifest has distinct SampleIDs and the audit found **zero exact training/testing duplicates** across the model input-voltage, load-current, and operating-condition arrays. The saved checkpoint does not identify a source-code revision or dataset hash, so this cannot independently prove the historical run used exactly this code and dataset. |
| Selected checkpoint epoch | **VERIFIED** | `best.pt` stores zero-based epoch **97** (the 98th epoch) and validation loss **0.0076394975**. Its embedded history has 98 rows and identifies epoch 97 as its minimum. The external `training_history.csv` has 100 rows; epoch 97 is its minimum-validation-loss row and epoch 99 is the final row. Thus the selected file is the best-validation checkpoint, not simply the last epoch. |
| Architecture and parameter count | **VERIFIED** | The saved model config strictly loaded into the current `WaveformResidualCNN`; PyTorch reported all state-dictionary keys matched. Architecture: 400 waveform points, 4 operating features, branch channels `[32, 64]`, condition width 32, fusion width 64, kernel size 5. Trainable parameter count: **201,601**. |
| Normalization compatibility | **VERIFIED** | All saved checkpoint normalization fields match statistics recomputed from the current training split within the app’s tolerance. Saved training statistics include input-voltage standard deviation 166.063219 V, current standard deviation 7.959998 A, target-voltage standard deviation 160.709264 V, and operating means `[0.748908, 24.907143, 0.0275943, 3.124457]` in the feature order `modulationIndex`, `loadResistance`, `loadInductance`, `phaseAngle`. |
| Training history and loss curve | **VERIFIED as stored artifacts** | `colab_runs/baseline_20261120_080836/results/training_history.csv` contains 100 epochs and the run includes `training_validation_loss.png`. Best validation loss is 0.00763950 at epoch 97; epoch 99 validation loss is 0.00773258. These are normalized-space losses, not volts. |
| Data leakage fully ruled out? | **PARTIALLY VERIFIED** | Training-only normalization, explicit split labels, no duplicate model-input rows across train and test, and validation-only checkpoint selection are supported by the code and current files. The original split-generation procedure and immutable source-data hash were not embedded in the checkpoint; shared categorical load settings across splits are present by design. No claim of independent acquisition or group-wise separation is made. |

The current checkpoint SHA-256 and hashes for every CSV in the extracted dataset are recorded in [`scientific_validation_summary.json`](./results/scientific_validation_20261120_205301/scientific_validation_summary.json). The checkpoint’s metadata itself has no dataset fingerprint.

## 2. Complete held-out test evaluation

**Status: VERIFIED.** The unchanged saved `best.pt` was loaded with `weights_only=True`, loaded strictly into the current architecture, and evaluated with `torch.inference_mode()` on rows explicitly labelled `testing`. The run used **1,500 records × 400 points = 600,000 waveform points**, in batches of 64, on CPU. No training, fine-tuning, model selection, or threshold tuning was performed on the test results.

### Global pooled-point metrics

These are calculated after concatenating all 600,000 waveform points; they are not means of record-level scores.

| Metric | Result |
|---|---:|
| Pooled MAE | **1.224758 V** |
| Pooled RMSE | **1.622291 V** |
| Global R² | **0.99989901** |
| Global Pearson correlation | **0.99994951** |
| Maximum absolute point error | **13.209122 V** |

### Per-record metrics

Each record’s metric is computed over its 400 points, then summarized over all 1,500 records.

| Metric | Mean | Median |
|---|---:|---:|
| Per-record MAE | **1.224758 V** | **1.184865 V** |
| Per-record RMSE | **1.576753 V** | **1.529406 V** |

The pooled RMSE differs from the mean per-record RMSE, as expected from the different aggregation. Global R² and correlation are computed over the concatenated points, not by averaging per-record R²/correlation.

### Pooled signed prediction-error distribution

Error is `prediction − target`; percentiles are over all 600,000 points.

| Statistic | Volts |
|---|---:|
| Mean | 0.017392 |
| Standard deviation | 1.622197 |
| Minimum / maximum | −13.078674 / 13.209122 |
| 1st / 5th percentile | −4.086793 / −2.649706 |
| 25th / median / 75th percentile | −0.924168 / 0.000107 / 0.957951 |
| 95th / 99th percentile | 2.719279 / 4.135897 |

### Worst records by per-record RMSE

| SampleID | MAE (V) | RMSE (V) | Max absolute error (V) |
|---:|---:|---:|---:|
| 4651 | 2.5539 | 3.0562 | 7.9754 |
| 4488 | 2.4546 | 2.9832 | 8.6608 |
| 6454 | 2.4651 | 2.9817 | 5.8387 |
| 4347 | 2.4007 | 2.9731 | 6.1058 |
| 4183 | 2.2364 | 2.8291 | 6.3515 |

All **1,500** per-record values are in [`cnn_test_per_record.csv`](./results/scientific_validation_20261120_205301/cnn_test_per_record.csv). The pooled summary and definitions are in the audit JSON.

## 3. Baseline comparison

**Status: VERIFIED.** Both baselines were measured on the same held-out records, without changing the model.

| Test predictor | Pooled MAE (V) | Pooled RMSE (V) | Global R² | Global correlation |
|---|---:|---:|---:|---:|
| Trained CNN | 1.224758 | 1.622291 | 0.99989901 | 0.99994951 |
| Mean training target waveform | 143.661357 | 161.415636 | 0.00017939 | 0.01467561 |
| Analytical reference from known conditions | **0.0000134** | **0.0000179** | **0.99999999999999** | **0.99999999999999** |

The analytical baseline is `300*m*sin(2*pi*50*t + phaseAngle)`, sampled at the project’s 20 kHz, 400-point cycle. Its near-zero residual is consistent with the dataset target being this known sinusoidal reference, subject to CSV/float precision. The CNN is substantially better than the constant mean-waveform baseline, but is **not a meaningful improvement over the simple analytical baseline**; it approximates the available formula less accurately. High CNN R² alone must not be presented as evidence of harmonic reduction.

## 4. Generalization and operating-condition analysis

**Status: VERIFIED for descriptive slices of the held-out test split; NOT VERIFIED beyond observed training support.** Condition bins were defined from training quantiles or known inverter settings, not selected to tune the model.

### Pooled RMSE by operating condition

| Feature | Group | Records | Pooled RMSE (V) |
|---|---|---:|---:|
| Modulation index (training deciles) | 0.5500–0.5909 | 151 | 1.6538 |
|  | 0.5909–0.6303 | 143 | 1.6013 |
|  | 0.6303–0.6701 | 148 | 1.6185 |
|  | 0.6701–0.7101 | 126 | 1.6889 |
|  | 0.7101–0.7499 | 161 | 1.6798 |
|  | 0.7499–0.7880 | 158 | 1.4751 |
|  | 0.7880–0.8260 | 144 | 1.4765 |
|  | 0.8260–0.8669 | 160 | 1.6495 |
|  | 0.8669–0.9089 | 151 | **1.7063** |
|  | 0.9089–0.9499 | 158 | 1.6575 |
| Load resistance | 10 Ω / 20 Ω / 30 Ω / 40 Ω | 379 / 356 / 393 / 372 | **1.6964 / 1.5724 / 1.5746 / 1.6416** |
| Load inductance | 0.01 H / 0.02 H / 0.03 H / 0.05 H | 378 / 354 / 384 / 384 | **1.7113 / 1.6088 / 1.5774 / 1.5886** |
| Phase angle | 0–45° / 45–90° / 90–135° / 135–180° / 180–225° / 225–270° / 270–315° / 315–360° | 207 / 187 / 187 / 179 / 192 / 179 / 195 / 174 | **1.6018 / 1.6114 / 1.6572 / 1.5948 / 1.6436 / 1.6065 / 1.6123 / 1.6517** |

The highest observed slice RMSE is in the first load-inductance category (0.01 H, 1.7113 V), the ninth training modulation decile (0.8669–0.9089, 1.7063 V), and the 10 Ω resistance category (1.6964 V). These differences are descriptive; no condition-slice hypothesis tests were run. All slices retain high R² partly because the target waveform has a large sinusoidal amplitude.

Training support spans modulation **0.550044–0.949949**, resistance **10/20/30/40 Ω**, inductance **0.01/0.02/0.03/0.05 H**, and nearly the full circular phase range (about 0.0003–6.2815 rad). Marginal ranges do not establish that every joint combination is densely represented.

### Live-demo OOD check

**Status: VERIFIED for the marginal-support warning; PARTIALLY VERIFIED as a complete OOD detector.** The LIVE DEMONSTRATION now compares the selected modulation index and discrete load values against the training records, handles phase as circular, and warns when marginal support is exceeded. It also hides prior results when submitted operating inputs differ, requiring a new run for new metrics. This check does not measure joint-density support, so “in marginal range” is not a guarantee of in-distribution behavior.

The slider endpoints 0.55 and 0.95 lie just outside the extrema actually observed in training (by less than 0.00005) and trigger a warning. The four load settings are all represented in training; phase is treated modulo 360°.

## 5. Live demonstration validation

**Status: VERIFIED.** The demo uses the loaded checkpoint and its normalization, feeds simulator-generated output voltage/current and normalized `[m, R, L, phase_rad]` inputs to the CNN, denormalizes the output, and compares it to `300 * simulation.reference` (the simulator’s actual reference scaled to physical voltage). CNN inference is explicitly inference-only. The event THD metrics describe the ideal simulated inverter output, not the CNN prediction. Each line below was independently run from simulation inputs, not from held-out records.

| m | R (Ω) | L (H) | Phase | CNN target MAE (V) | CNN target RMSE (V) | R² | Parseval output THD |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.75 | 30 | 0.03 | 0° | 0.8485 | 1.0986 | 0.999952 | 24.72348% |
| 0.55 | 10 | 0.01 | 90° | 1.2376 | 1.6755 | 0.999794 | 37.31379% |
| 0.95 | 40 | 0.05 | 270° | 1.0843 | 1.4674 | 0.999947 | 20.68910% |
| 0.80 | 20 | 0.02 | 180° | 1.0189 | 1.2995 | 0.999941 | 24.36412% |

For the two endpoint modulation values, the UI’s training-range check warns that the requested value is slightly outside the observed sample extrema. The live demo requires an explicit **RUN LIVE SIMULATION** submit, then recalculates; a prior result is not shown as current after parameters have changed. The 20 kHz sampled finite-band THD and event-based finite-band/Parseval metrics remain distinct.

Browser smoke check at `http://127.0.0.1:8501`: the default `m=0.75` run produced 1.099 V CNN target RMSE; changing modulation to `m=0.76` and submitting produced a newly labelled `m=0.76` result with 1.031 V RMSE. Submitting `m=0.95` displayed the out-of-training-range warning and a result for the new operating point. Thus the UI recalculated after submission and did not display the previous metrics as the new point’s result.

## 6. PSO and harmonic-reduction claim audit

**Status: VERIFIED for the saved experiment metrics; FAILED as evidence of meaningful, validation-confirmed reduction.**

The saved experiment ran 20 seeds (20261120–20261139), with 20 particles and 50 iterations per seed; all completed and were feasible. It optimized Parseval full-spectrum THD on a fixed 30-record training subset. The selected candidate was `[0°, 60.0780984597°, 120.0326906390°]` from seed 20261139.

| Dataset | Baseline mean THD | Selected mean THD | Selected − baseline | Interpretation |
|---|---:|---:|---:|---|
| 30-record training subset | 26.35635545% | 26.35634324% | **−0.00001221 percentage points** | Relative decrease about 0.00004634%; practically negligible |
| 1,500-record validation split | 26.28062798% | 26.28067794% | **+0.00004996 percentage points** | Slightly worse than baseline |

An independent paired recomputation over validation records gave a selected-minus-baseline mean of **+0.00004996 percentage points**. A deterministic 10,000-resample paired bootstrap 95% interval was **[−0.00007878, +0.00017655] percentage points**, including zero. The change is not statistically distinguished from no change by this interval, and its magnitude is practically negligible. The test split was not used for PSO or this comparison.

The experiment is reproducibly documented by its fixed config, seeds, training subset, run summary, convergence history, and saved results. This supports repeatability of the recorded software procedure, not global optimality or physical performance. **Do not claim that PSO has demonstrated meaningful harmonic reduction.**

## 7. Scientific limitations

- The inverter, switching, and load model are ideal software conventions. No hardware, MATLAB, or Simulink validation is established.
- The CNN predicts a reference sine waveform. It does not predict gates, calculate optimized inverter-output THD, or demonstrate harmonic reduction.
- The analytical target formula is available from the operating conditions and performs much better than the CNN on the test split.
- The held-out test results are one evaluation of this checkpoint on this supplied split; they do not prove generalization to other hardware, topologies, sampling rates, unseen load values, or other datasets.
- Operating-condition slice metrics and marginal OOD checks do not prove joint-distribution coverage or physical robustness.
- The checkpoint does not carry a dataset hash or source-code revision. Exact historical training-data identity is therefore not provable from the saved checkpoint alone.
- The test split has now been used for the requested final audit. It must remain excluded from future fitting, model selection, threshold selection, and PSO selection.

## 8. Project readiness assessment

**PARTIALLY READY** for an exhibition framed as an ideal inverter software simulation and a CNN waveform-regression demonstration. The checkpoint loads, uses matching normalization, and achieves the test-set regression metrics above.

**NOT READY to claim that the CNN reduces harmonics.** The CNN predicts a known reference waveform and is outperformed by its analytical formula. The saved PSO result is effectively unchanged on training and marginally worse on validation, with paired uncertainty spanning zero. Present the event-based inverter THD and CNN target-prediction error as separate measurements, and state these limitations explicitly.

## Reproducibility and files

Run the inference-only audit from the project root (it creates a new timestamped directory and refuses to overwrite an existing one):

```powershell
.\.venv\Scripts\python.exe -B -m evaluation.scientific_validation
```

Audit outputs for this run:

- [`scientific_validation_summary.json`](./results/scientific_validation_20261120_205301/scientific_validation_summary.json) — metrics, SHA-256 fingerprints, definitions, condition slices, live points, and paired PSO validation analysis.
- [`cnn_test_per_record.csv`](./results/scientific_validation_20261120_205301/cnn_test_per_record.csv) — all 1,500 test SampleIDs and per-record scores.
- [`cnn_test_condition_breakdown.csv`](./results/scientific_validation_20261120_205301/cnn_test_condition_breakdown.csv) — descriptive held-out condition slices.

The audit script runs inference only. It does not train, fine-tune, or change model weights, source datasets, checkpoints, or split labels.

Validation commands:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Result: **52 tests passed**. Changed Python files also passed an AST syntax check. The audit output integrity check confirmed 1,500 unique test SampleIDs, 26 condition groups, and the test-not-used-for-tuning flag.
