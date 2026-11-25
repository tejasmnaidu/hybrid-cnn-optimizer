> **SUPERSEDED — 2026-11-24**
>
> This pre-training audit predates the implemented ideal PWM/RL simulator and live optimizer integrations. Its historical warning that no simulator or electrical evaluator existed is no longer the current project status. See [docs/SIMULATOR_VALIDATION.md](./docs/SIMULATOR_VALIDATION.md), [docs/CNN_GROUPED_TEST_EVALUATION.md](./docs/CNN_GROUPED_TEST_EVALUATION.md), and [docs/CURRENT_STATUS.md](./docs/CURRENT_STATUS.md). The original report text is retained unchanged below.

# Pre-Training Technical Audit

**Project:** `Hybrid_CNN_Inverter_Optimization`  
**Audit date:** 2026-11-20  
**Scope:** Static review of the current source, configuration, requirements, and Colab notebook; focused unit tests and syntax checks. No model training was run and no project source/configuration was changed for this audit.

## Executive result

| Area | Status | Summary |
|---|---|---|
| Dataset loading, alignment, splits, leakage checks | **PASS** | The project loader and validator align the CSV records by `SampleID`; preprocessing preserves the supplied split labels. The existing validation report records the configured split counts and no exact duplicate model inputs across splits. |
| Preprocessing and normalization | **PASS** | Statistics are fitted on training records and reused for all splits; inverse target normalization is the matching affine inverse. |
| CNN architecture and tensor shapes | **PASS** | The model accepts the configured two 400-point waveforms and four conditions and returns a 400-point normalized target waveform. |
| Training and model selection | **PASS** | Training uses training batches; validation loss drives scheduler, early stopping, and best-checkpoint selection. The test loader is not passed to `Trainer.fit`. |
| Metrics and FFT/THD | **PASS** | Metric formulas and FFT bin calculations are consistent with the dataset's 20 kHz / 50 Hz / 400-point waveform. The notebook clearly compares reference and predicted waveforms, not measured inverter output. |
| Colab readiness | **WARNING** | Notebook paths/imports are internally consistent for the documented Drive layout and backend; notebook cells parse. A real fresh Colab runtime was not launched, and resume requires pointing the run path to an existing run. |
| Physical controller / harmonic-reduction claims | **WARNING — critical limitation** | The CNN is a waveform-regression model, not a gate-command controller. No verified gate/state mapping, concrete electrical evaluator, simulator, or measured post-inverter THD loop is implemented. Do not claim physical harmonic reduction from this project as it stands. |

### Critical issues before training

**None found that block training the stated supervised waveform-regression baseline.** The data-validation report already present at `results/dataset_validation.json` is valid for the supplied archive; it records 10,000 rows, 7,000/1,500/1,500 splits, aligned auxiliary IDs, and zero exact duplicate input records across splits. The eight focused unit tests pass, Python modules compile, and notebook code cells parse.

**Critical before interpreting results as inverter control or harmonic reduction:** first supply a verified physical mapping and simulator/electrical evaluation path, then evaluate actual post-inverter output waveforms. Until then, the valid interpretation is only CNN prediction of the dataset's target/reference voltage waveform.

## 1. Dataset loading, alignment, split use, and leakage

**Status: PASS**, with leakage-check scope noted below.

- [`preprocessing/data_loader.py`](./preprocessing/data_loader.py:43) extracts known CSV basenames from a ZIP, rejects duplicate matching basenames, and loads the required non-auxiliary tables. Auxiliary gate/state CSVs are excluded from eager loading by default.
- [`preprocessing/data_validation.py`](./preprocessing/data_validation.py:127) validates required tables, unique integer IDs, table row counts, configured sample IDs, and CSV values. The IDs of each main table are compared against the input-voltage IDs at lines 188–225.
- Split labels are taken from `Sample_Splits.csv`, not regenerated. Configured counts are enforced at [`data_validation.py`](./preprocessing/data_validation.py:241). The data report records training 7,000, validation 1,500, and testing 1,500.
- [`preprocessing/preprocess.py`](./preprocessing/preprocess.py:76) sorts each waveform/condition table by ID and checks alignment; lines 122–135 align `Sample_Splits.csv` by ID before producing split indices.
- The validator hashes the model inputs (input voltage, load current, and conditions) to detect exact repeated inputs assigned across different splits at [`data_validation.py`](./preprocessing/data_validation.py:306). The existing report records **0** such records.
- Gate and cell-state rows are checked in chunks against the main ID set at [`data_validation.py`](./preprocessing/data_validation.py:66), avoiding loading the full auxiliary matrices into memory during validation.

**WARNING — leakage check is not exhaustive:** the implemented audit detects exact duplicate model inputs only. It does not test near-duplicates, related/grouped records, or every possible domain-specific leakage mechanism. This does not invalidate the supplied split, but exact-duplicate absence alone is not a general proof of leakage-free sampling.

**WARNING — call validation before reusable preprocessing:** `preprocess_dataset` verifies IDs, shapes, supported labels, and nonempty splits, but does not independently enforce the configured 7,000/1,500/1,500 counts or configured seven-level values. The CLI and notebook call `validate_dataset` first; other callers should do likewise.

## 2. Preprocessing, normalization, inverse transform, and tensor types

**Status: PASS.**

- `train_mask` is formed from the supplied split at [`preprocessing/preprocess.py`](./preprocessing/preprocess.py:137). Means and standard deviations for input voltage, load current, the four condition features, and target voltage are calculated from masked training rows only at lines 149–158.
- The target inverse is `normalized * target_voltage_std + target_voltage_mean` at [`preprocess.py`](./preprocessing/preprocess.py:40), which correctly reverses the target transform at lines 163–169. `evaluation/evaluate.py` applies it to both predictions and targets when normalization stats are supplied (lines 48–49).
- Waveform records are prepared as `(samples, 400)` arrays; conditions as `(samples, 4)`. `_ordered_matrix` casts numeric values to `float32` and rejects non-finite values at [`preprocess.py`](./preprocess.py:76).
- [`training/dataset.py`](./training/dataset.py:30) returns waveform/condition tensors with their prepared `float32` dtype and preserves `sample_id` as `int64`. `WaveformDataset` and `create_dataloaders` select data using the existing split labels.
- Input and target statistics use one scalar mean/std per waveform channel over training samples and points, while operating conditions use per-feature means/stds. This is a valid global standardization choice; it is not per-sample normalization.

## 3. CNN architecture and output task

**Status: PASS for the configured regression task.**

- [`models/cnn_model.py`](./models/cnn_model.py:74) accepts `(batch, 400)` or `(batch, 1, 400)` voltage/current inputs and `(batch, 4)` conditions; it rejects mismatched shapes and non-configured waveform lengths.
- Separate voltage and current feature branches are instantiated at lines 51–52. Encoded condition features are broadcast over the temporal axis and fused with both waveform representations at lines 95–100.
- [`models/residual_blocks.py`](./models/residual_blocks.py:10) uses two padded `Conv1d` layers, batch normalization, GELU activations, and an identity or 1×1 projection for channel-matching residual addition. The padding preserves temporal length.
- The final 1×1 convolution and squeeze at [`cnn_model.py`](./models/cnn_model.py:100) yield `(batch, 400)`. The label loaded and normalized by preprocessing is `Target_Voltage.csv`; the output is **target/reference voltage prediction**, not a switch state, gate pattern, or measured physical inverter output.

**Research limitation:** the CNN output is an unconstrained continuous regression waveform. No output quantizer, modulation strategy, switching schedule, gate interlock, dead-time handling, or hardware safety layer turns it into a deployable controller. This is not a defect for the stated regression baseline, but is a blocker for direct control interpretation.

## 4. Training, validation, checkpointing, and test isolation

**Status: PASS for the standard project workflow.**

- [`training/losses.py`](./training/losses.py:10) implements MAE or MSE with an optional first-difference penalty and checks matching prediction/target shapes.
- [`training/trainer.py`](./training/trainer.py:78) uses AdamW; lines 83–89 configure `ReduceLROnPlateau`. `_run_epoch` switches train/eval modes, uses gradients only for training batches, and rejects non-finite predictions/losses at lines 115–151.
- `Trainer.fit` accepts only training and validation loaders, calculates validation loss at line 177, and uses that loss for best-checkpoint selection and early stopping at lines 187–212. It does not accept a test loader. [`training/train.py`](./training/train.py:38) and Colab both create distinct train/validation/test loaders and pass only train/validation to `fit`.
- Both `last.pt` and improved `best.pt` checkpoints are written; after fitting, the best model weights are restored at [`trainer.py`](./training/trainer.py:202). Checkpoints contain model/optimizer/scheduler state, history, training configuration, and metadata.
- [`notebooks/colab_train.ipynb`](./notebooks/colab_train.ipynb) first evaluates `LOADERS.test` at notebook JSON line 345, after the training cell at line 305. The comments explicitly reserve test data for final evaluation.

**WARNING — resume reproducibility:** resume restores model, optimizer, scheduler, history, and early-stopping counter in [`trainer.py`](./training/trainer.py:99), but it does not restore Python/NumPy/PyTorch RNG states or the DataLoader generator state. Resuming can continue from saved weights, but is not guaranteed to reproduce an uninterrupted stochastic run exactly.

**WARNING — loader provenance guard:** `Trainer.fit` checks a dataset's `.split` attribute when present (lines 168–171). A custom loader without that attribute can bypass the guard. The packaged `create_dataloaders` path tags each `WaveformDataset` and keeps the splits separate; use that path or add stronger provenance checks if arbitrary loaders will be passed.

## 5. Evaluation, FFT, THD, and interpretation

**Status: PASS for the defined waveform-level calculations.**

- [`evaluation/metrics.py`](./evaluation/metrics.py:18) computes MAE and MSE as means over all array elements; RMSE is square-root MSE at line 28. R-squared uses residual and centered-total sums at lines 37–44. Correlation is Pearson correlation over flattened waveform arrays at lines 46–52; it rejects constant arrays. All metrics check shape and finite values.
- [`evaluation/harmonic_analysis.py`](./evaluation/harmonic_analysis.py:33) derives frequencies using `rfftfreq(N, 1/Fs)`, subtracts the mean, and forms a single-sided amplitude spectrum. The configured dataset has 400 points at 20 kHz, so bin width is 50 Hz and the 50 Hz fundamental is exactly bin 1. The requested harmonics through order 100 are below the 10 kHz Nyquist frequency.
- [`evaluation/thd_analysis.py`](./evaluation/thd_analysis.py:10) computes root-sum-square harmonic amplitude for orders 2 through the configured maximum, divided by fundamental amplitude and multiplied by 100. It errors if the fundamental amplitude is effectively zero.
- [`plot_thd_comparison`](./evaluation/evaluate.py:126) and notebook JSON lines 391–400 compare the **dataset target/reference waveform** with the **CNN-predicted target waveform**. They are not reference-versus-measured-inverter-output measurements. The notebook labels that limitation explicitly.

**WARNING — FFT assumptions:** THD is a finite-record FFT estimate. The supplied 400-point, 50 Hz waveform is an integer-cycle record, which supports the configured bin calculation. Applying the same helper to arbitrary non-integer-cycle or transient records can introduce spectral leakage and biased THD; such records need an explicitly chosen windowing/measurement convention.

**Critical interpretation boundary:** neither the predicted-waveform THD nor `THD_Results.csv`'s reference/input THD demonstrates actual post-CNN or post-inverter output THD. Do not use these as evidence of physical harmonic reduction.

## 6. Google Colab, paths, imports, and dependencies

**Status: WARNING — static review only; not run in a fresh Colab runtime.**

- The notebook mounts Drive at JSON line 21 and uses a Drive-rooted `PROJECT_ROOT`, ZIP path, extracted-data path, config path, and experiment path at lines 43–53. It expects the complete project folder **and ZIP** to have been uploaded under that documented Drive folder.
- It inserts `PROJECT_ROOT` into `sys.path`, loads the project YAML, and imports the actual backend modules at lines 53 and 87–101. These match the current project modules. The notebook JSON and code cells parse; project modules compile and the focused unit tests pass in the available local environment.
- Dataset extraction, validation, and preprocessing use the shared backend at lines 127–149. GPU/CPU choice comes from `detect_device`; DataLoaders pin memory only when CUDA is active.
- Checkpoint and result paths are under `EXPERIMENT_DIR` on Drive (lines 248–286); the best checkpoint is loaded for final test evaluation at line 344.
- `requirements.txt` lists NumPy, pandas, PyYAML, PyTorch, Matplotlib, and Streamlit. The notebook additionally relies on `google.colab`, Python standard-library modules, and Colab's existing runtime. Streamlit is not needed to train in the notebook.

**WARNING — resume path:** `RUN_NAME` is newly timestamped and `EXPERIMENT_DIR` is constructed from it at notebook JSON lines 46–48. Setting `RESUME = True` alone therefore points to a new run with no checkpoint. To resume, edit `EXPERIMENT_DIR` to the existing Drive run directory as its comment instructs; the subsequent checkpoint assertion then verifies `best.pt` exists.

**WARNING — environment bootstrap ordering:** the experiment-configuration cell imports `yaml` and `torch` before the `pip install` cell (JSON lines 39–55 precede line 75). Standard Colab runtimes normally include these; a stripped runtime missing either will fail before dependency installation can repair it. The notebook itself was not executed in Colab, so fresh-runtime dependency resolution, Drive permissions, available GPU memory, and full dataset extraction were not runtime-verified here.

**Dependency reproducibility:** requirements are lower-bounded, not pinned. Colab may resolve different package versions over time; record actual versions with experiment metadata before comparing runs.

## 7. Research limitations and PSO integration

**Status: WARNING — critical for physical-control or harmonic-reduction conclusions; not a blocker to the baseline regression training.**

- The CNN is trained to predict `Target_Voltage.csv`. Its output is a reference waveform, not inverter switch commands or an evaluated physical output. Low target-prediction error alone cannot establish that an inverter can track the waveform.
- The supplied gate and cell-state data have different lengths (4,800 and 1,200 values/sample). [`preprocessing/switching_data.py`](./preprocessing/switching_data.py:21) requires an explicitly supplied layout; [`optimization/switching_constraints.py`](./optimization/switching_constraints.py:32) does not assume one. Current source does not establish device ordering, gate polarity, cell mapping, dead time, or temporal relation for physical evaluation.
- [`optimization/electrical_evaluator.py`](./optimization/electrical_evaluator.py:24) is an abstract interface only. There is no concrete verified circuit/simulator evaluator implementing `evaluate_candidate`.
- [`optimization/pso_optimizer.py`](./optimization/pso_optimizer.py:50) implements bounded generic PSO, but it does not itself know inverter parameters, switch schedules, or electrical behavior. `configs/config.yaml` has objective weights and voltage levels but no candidate parameter definitions/bounds or concrete simulator binding. PSO is not connected to the CNN training notebook/CLI.
- [`optimization/objective_function.py`](./optimization/objective_function.py:44) can score harmonic distortion, tracking error, transition count, output levels, and a constraint flag only after a concrete evaluator supplies those measured quantities. Default penalty weights are configuration choices, not physical constants.
- No simulation/HIL/bench measurement pipeline compares the actual switched inverter output before and after optimization. Therefore, actual output THD reduction, switching feasibility, thermal/electrical limits, and safety remain unvalidated.

### Required before physical claims or controller deployment

1. Establish and document the channel ordering and physical mapping of the gate and cell-state arrays, including their time base and switch/cell conventions.
2. Define PSO candidate variables, feasible bounds, and physically meaningful hard constraints from the actual converter/control design.
3. Implement and independently validate a concrete `ElectricalEvaluator` against a trusted switching model or hardware measurements.
4. Evaluate actual measured/simulated post-inverter output waveforms using a documented THD convention and compare against a baseline under the same operating conditions.
5. Add controller-level checks (including valid gate combinations, interlocks/dead time, and electrical ratings) before any deployment consideration.

These are not needed to train the CNN as an offline reference-waveform regressor, but they are necessary to connect that regressor to the project's physical harmonic-reduction objective.

## Verification performed for this audit

- `python -m unittest discover -s tests -v`: **8 tests passed**.
- `python -m compileall` over project Python modules: **passed**.
- Notebook JSON parsing and Python parsing of all code cells: **passed**.
- No notebook cells or training loops were executed. No training metrics or physical performance results are reported by this audit.
