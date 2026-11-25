> **SUPERSEDED — 2026-11-24**
>
> This earlier readiness audit predated the live phase optimizers and later dashboard reliability work. Its statement that PSO was not executed by the dashboard is historical, not a description of the present implementation. Current component status and validation limits are summarized in [docs/CURRENT_STATUS.md](./docs/CURRENT_STATUS.md). The original report text is preserved below.

# Exhibition Readiness Audit

**Project:** Hybrid CNN and PSO-Based Harmonic Reduction in a Seven-Level
Multilevel Inverter  
**Audit type:** Read-only software readiness audit. No source, dataset,
checkpoint, training, or optimization files were changed for this audit.  
**Overall assessment:** **PARTIALLY READY** as an offline, ideal-model
simulation and CNN waveform-regression exhibition. The normal live simulation
and saved PSO comparison are available, but PSO is not executed by the
dashboard and is not applied to the live operating point. The current PSO
evidence does not support a meaningful harmonic-reduction claim.

## Status definitions

- **VERIFIED** — directly checked in source, existing artifacts, tests, or
  application execution.
- **IMPLEMENTED BUT UNVERIFIED** — code exists, but this audit did not exercise
  that specific runtime path or environment condition.
- **INCOMPLETE** — a needed end-to-end feature, connection, or recovery path is
  absent.
- **FAILED** — evidence shows that the stated requirement is not met.

## 1. Project contents and offline readiness

| Feature | Status | Evidence |
|---|---|---|
| Inverter simulator | **VERIFIED** | [`multilevel/inverter_simulator.py`](./multilevel/inverter_simulator.py), `simulate_inverter`, returns 400 sampled points for voltage/current, three cell states, and 12 gate signals. |
| CNN source | **VERIFIED** | There is no `cnn/` directory; the model is in [`models/cnn_model.py`](./models/cnn_model.py) and constructed by [`models/model_factory.py`](./models/model_factory.py). |
| Dataset | **VERIFIED** | Both `data/raw/Seven_Level_Inverter_Project.zip` and the extracted required CSVs exist, including voltage/current/target, operating conditions, split labels, switching gates, and cell states. |
| Trained checkpoint and history | **VERIFIED** | `best.pt`, `last.pt`, and `colab_runs/baseline_20261120_080836/results/training_history.csv` exist. The app displayed `CNN LOADED`; the earlier scientific validation strictly loaded this checkpoint into the current architecture. |
| Local Python runtime | **VERIFIED for this workstation** | The project `.venv` runs Python 3.13.12 with NumPy 2.5.5, Pandas 3.1.1, PyYAML 6.0.3, PyTorch 2.15.0+cpu, Matplotlib 3.11.2, and Streamlit 1.67.0. |
| App startup | **VERIFIED for this checkout** | Streamlit health endpoint returned `ok` at `http://127.0.0.1:8501/_stcore/health`; browser showed the existing dashboard and loaded model. |
| Runtime internet requirement | **IMPLEMENTED BUT UNVERIFIED** | The inspected application, simulator, model, and evaluators use local project files and installed Python libraries; no remote API or hosted model/data fetch is part of the launch path. The app was not run with the network physically disabled, and a fresh offline install of `requirements.txt` has not been tested. |
| Fresh-machine offline setup | **IMPLEMENTED BUT UNVERIFIED** | Required packages are declared in `requirements.txt` and installed in this workstation's `.venv`. Availability of package wheels/cache on another offline laptop is not established. |
| `cnn/` path requested in the audit prompt | **VERIFIED absence; not a defect** | CNN functionality is implemented under `models/`; the implementation plan should target that directory rather than create a duplicate `cnn/` package. |

### Offline dependency set

`requirements.txt` declares `numpy`, `pandas`, `PyYAML`, `torch`,
`matplotlib`, and `streamlit`. These are installed in the existing project
environment. No additional dependency is indicated for the existing
simulation or exhibition flow. To guarantee a fresh laptop can be prepared
without internet, the wheel/cache or a tested environment bundle would need
to be staged locally; that packaging work has not been performed.

## 2. Current end-to-end workflow

| Workflow stage | Existing module/function | Status and finding |
|---|---|---|
| User inputs | [`app.py`](./app.py), live-demo tab | **VERIFIED.** Modulation index, load resistance, load inductance, and phase inputs and valid presets are present. Inputs are committed by an explicit simulation button. |
| Input validation | [`evaluation/live_simulation.py`](./evaluation/live_simulation.py), `_validate_operating_inputs` | **VERIFIED.** Checks finite numeric values, modulation range 0.55–0.95, supported R/L categories, and phase 0–360 degrees. |
| Seven-level PWM and load simulation | [`multilevel/inverter_simulator.py`](./multilevel/inverter_simulator.py), `simulate_inverter` | **VERIFIED for implemented conventions and checked dataset record.** Produces carriers, states, gates, cell/output voltage, and periodic RL current. |
| Switching state and gates | `simulate_inverter` | **VERIFIED in software tests.** Gate truth table reconstructs states/cell voltage; prohibited within-leg shoot-through combinations are rejected. This is not a hardware gate-driver validation. |
| CNN prediction | [`evaluation/live_simulation.py`](./evaluation/live_simulation.py), `run_live_simulation` | **VERIFIED.** Uses checkpoint normalization, `model.eval()`, `torch.inference_mode()`, and target denormalization; compares against `simulation.reference * 300 V`. |
| Event-based harmonics | [`evaluation/event_based_thd.py`](./evaluation/event_based_thd.py), `evaluate_event_based_thd` | **VERIFIED for the implemented piecewise-constant switching convention.** Calculates exact intervals, Fourier coefficients, finite-band THD, and Parseval full-spectrum THD. |
| Sampled FFT harmonics | [`evaluation/thd_analysis.py`](./evaluation/thd_analysis.py), `calculate_thd`; [`evaluation/harmonic_analysis.py`](./evaluation/harmonic_analysis.py), `harmonic_spectrum` | **VERIFIED.** Calculates finite-band sampled FFT THD from the waveform supplied. The caller must label which waveform is being analyzed. |
| Phase PSO | [`optimization/circular_phase_pso.py`](./optimization/circular_phase_pso.py), `CircularPhasePSO.optimize`; [`optimization/phase_pso_experiment.py`](./optimization/phase_pso_experiment.py), `run_experiment` | **VERIFIED as an offline backend and saved experiment.** It is not called by either live simulation tab. |
| Candidate comparison | `run_experiment` | **VERIFIED on validation records.** Baseline and selected phases are evaluated on the same validation `OperatingPoint`s. The exhibition tab displays this saved validation summary, not a fresh baseline-versus-PSO simulation at the live operating point. |
| Result export | `app.py` CNN evaluation section; `run_experiment` | **INCOMPLETE end-to-end.** Test evaluation offers CSV/JSON downloads; PSO writes CSV/JSON files under `results/`. Live and exhibition simulation results do not offer a dedicated complete result export. |

## 3. Scientific and numerical checks

### Inverter and load

| Check | Status | Evidence |
|---|---|---|
| Seven output levels | **VERIFIED for tested conditions** | `test_output_contains_only_seven_voltage_levels` checks all seven levels at `m=1`; simulator also validates every generated output sample is a member of `{-300,-200,-100,0,100,200,300} V`. |
| Gate/state consistency | **VERIFIED in code and tests** | `test_gate_truth_table_reconstructs_cell_states_without_shoot_through` tests the four gate channels, cell voltage reconstruction, output sum, and no shoot-through condition. |
| Match to supplied dataset | **VERIFIED for SampleID 1** | `test_matches_existing_dataset_record` compares simulator cell states, gate signals, output voltage, and load current to the existing dataset row. This does not establish a complete all-record or original MATLAB reproduction. |
| RL recurrence and periodic boundary | **VERIFIED for implemented recurrence** | `test_rl_recurrence_and_periodic_boundary` independently checks the exact-ZOH recurrence and cycle-boundary initial current. |
| Physical electrical validation | **NOT VERIFIED** | The simulator documentation says its switching convention is not checked against the original MATLAB source; no hardware or Simulink model is established. |
| Dataset gate mapping | **PARTIALLY VERIFIED** | `preprocessing/switching_data.py` and `optimization/switching_constraints.py` require the caller to declare flattened layout and do not infer physical gate/device mapping. The simulator's generated gate truth table is separately tested. |

### CNN, checkpoint, and data separation

| Check | Status | Evidence |
|---|---|---|
| Current architecture and checkpoint | **VERIFIED** | The prior scientific audit strictly loaded `best.pt` with all keys matching the 201,601-parameter model. |
| Normalization | **VERIFIED** | `run_live_simulation` applies `NormalizationStats.normalize_inputs` and `denormalize_target`; audit found saved statistics match current training-only statistics. |
| Prediction target | **VERIFIED** | The live evaluator compares prediction with the scaled sinusoidal reference target. It is not comparing CNN output to switching gates or inverter-output THD. |
| Training/test isolation | **VERIFIED in inspected code and tests; historical provenance is partially verified** | `Trainer.fit` accepts only training and validation loaders; `PreparedDataset.indices` separates explicit split labels; the test evaluator uses `indices("testing")`; its tests include `test_only_testing_split_is_evaluated`. The checkpoint lacks immutable source-data hash and code revision, so exact historical provenance is not fully proven. |
| PSO/test separation | **VERIFIED in implementation and saved run** | PSO split loading explicitly allows only training/validation, and `load_operating_points` rejects `testing`. The saved run records `test_records_evaluated: 0`; test data are not used in candidate fitting. |
| Model actually performs harmonic reduction | **FAILED as a supported claim** | The CNN predicts a reference waveform; it does not control switching. The audited analytical reference baseline had held-out RMSE about `0.0000179 V`, far lower than the CNN's `1.622291 V`. |

The complete 1,500-record testing split was evaluated in the existing
scientific audit, not during this readiness audit. Its pooled metrics were
MAE `1.224758 V`, RMSE `1.622291 V`, global R² `0.99989901`, and correlation
`0.99994951`. Refer to
[`SCIENTIFIC_VALIDATION_REPORT.md`](./SCIENTIFIC_VALIDATION_REPORT.md) and
`results/scientific_validation_20261120_205301/` for the full metrics and
per-record results. Do not use the test split for any new fitting, tuning,
model selection, or PSO decisions.

### Harmonic metrics

**VERIFIED, WITH AN IMPORTANT LIMITATION.** `evaluate_event_based_thd`
creates contiguous one-cycle piecewise-constant intervals and exact Fourier
coefficients. The finite-band event THD uses harmonic orders 2–100; its
fundamental is order 1. Parseval full-spectrum THD is computed from interval
mean/mean-square energy after removing DC and fundamental energy. The sampled
metric uses a 20 kHz waveform and finite orders 2–100.

At the default live point (`m=0.75`, phase `0°`), the app showed event
finite-band THD `1.019e-07%`, sampled finite-band THD `10.871%`, and Parseval
full-spectrum THD `24.723%`. The near-zero event 2–100 figure and much larger
sampled figure are **not interchangeable**: exact-event integration avoids
sampling aliasing, while the 20 kHz sampled PWM waveform has only 20 samples
per carrier period. Existing independent Fourier/FFT audit work documented
the spectral distinction and tests compare the event reconstruction and
sampled simulator conventions. The exhibition must call out sampling/alias
limitations rather than present the two finite-band values as equivalent
estimates.

## 4. Exhibition reliability

| Requirement | Status | Evidence / gap |
|---|---|---|
| Current dashboard preserved | **VERIFIED** | Existing dark styling and tabs remain; the current app has Overview, CNN Prediction, CNN Test Evaluation, Data Diagnostics, Live Demonstration, and Exhibition Mode. |
| Reproducible live preset | **VERIFIED** | Browser run of default point (`m=.75`, `R=30 Ω`, `L=.03 H`, `phase=0°`) returned 225 V fundamental peak, 1.099 V CNN target RMSE, sampled THD 10.871%, and Parseval THD 24.723%. |
| Changed inputs do not show prior metrics as current | **VERIFIED in UI flow** | The live tab compares the stored result's conditions to selected inputs, shows a rerun notice, and suppresses metrics for mismatched conditions. A lower-modulation preset (`m=.60`) was submitted and produced a distinct fresh result (180 V fundamental peak, CNN RMSE 1.026 V). |
| Reset | **VERIFIED in browser for current UI** | Reset clears the live result and restores the default controls; no old operating-point metrics remain. |
| Missing checkpoint | **IMPLEMENTED BUT UNVERIFIED** | With no checkpoint, model remains `None`, live/exhibition run controls are disabled, and an unavailable warning is shown. This branch was not simulated by removing/renaming the real checkpoint. |
| Corrupt checkpoint | **INCOMPLETE** | Most load/state errors are surfaced in the app-level error panel, but checkpoint loading is in the startup path and a bad checkpoint can prevent the dashboard from rendering. A zero-byte checkpoint can raise `EOFError`, which is not listed in the app's outer exception tuple. No corrupt-checkpoint recovery test exists. Do not test by altering the real checkpoint; use temporary fixtures. |
| Local path portability | **INCOMPLETE** | `default_checkpoint` is project-root absolute, but `default_dataset` is the relative string `data/raw/Seven_Level_Inverter_Project_extracted`; `load_dataset` resolves it against the process working directory. Running outside the project root can therefore fail to find the default dataset. |
| Dataset cache invalidation | **INCOMPLETE** | `load_prepared_dataset` is cached by its `source` argument. Live and test-result keys also use file fingerprints, but the cached prepared dataset call itself does not include that fingerprint; in-place CSV edits under the same source string may leave prepared data cached. |
| Multiple simulations | **VERIFIED for successful reruns** | A new run clears the old live result before calling `run_live_simulation`; results are also invalidated when dataset/checkpoint fingerprints change. A successful changed-preset run was observed. |
| Live/exhibition local export | **INCOMPLETE** | Test-evaluation downloads and PSO output artifacts exist. The live/exhibition tabs do not yet export conditions, raw waveforms, harmonics, and CNN metrics together. |
| Optimization failure recovery | **INCOMPLETE** | `run_experiment` creates a new timestamped output directory and writes progress artifacts, but it does not resume a failed run or mark/clean incomplete output directories. It is not currently invoked from the dashboard. |
| Offline execution | **IMPLEMENTED BUT UNVERIFIED** | Local files/dependencies are present and the app responds on loopback. No explicit disconnected-network test or clean-machine offline provisioning test was run. |

## 5. Existing experiment and claims

**VERIFIED: saved experiment exists; FAILED as evidence of useful improvement.**

The saved circular PSO experiment completed all 20 seeds with 20 particles and
50 iterations, using a fixed 30-record training subset; it used validation
once for baseline/winner comparison and evaluated zero test records. On the
training subset, mean full-spectrum THD fell by only `0.00001221` percentage
points. On the 1,500-record validation split, the selected candidate was
`0.00004996` percentage points worse; the paired bootstrap 95% interval
included zero. The current exhibition comparison is this saved validation
comparison, not a recomputation at the live condition. No meaningful PSO
harmonic-reduction claim is supported.

The standard 20-seed experiment has been recorded at approximately
230.92 seconds per run in the saved report (about 77 minutes total for the
full default set). Do not trigger that full experiment implicitly on app
startup or as a live demo action.

## 6. Existing tests and actual result

The audit ran the complete existing `unittest` suite:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

**Result: 52 tests passed in 2.169 seconds; `OK`.**

Coverage includes:

- Dataset archive/directory loading, validation, split-safe preprocessing, and
  SampleID alignment.
- CNN architecture, metrics, and checkpoint round-trip.
- Simulator seven-level output, gate truth table, shoot-through check,
  periodic RL recurrence, determinism, invalid input handling, and comparison
  against one supplied dataset row.
- Event interval construction, event/sampled convention agreement, harmonic
  integration, Parseval convergence, and invalid input handling.
- Circular PSO wrap/distance, cell symmetry, hard tracking constraint,
  baseline incumbent, deterministic seeded result, invalid evaluator, and
  split-loader isolation.
- CNN test-split-only inference, denormalization, batching, CPU execution,
  aggregation, empty split, and invalid model output.
- Live simulator/CNN inference, input validation, invalid output handling,
  support warning, and result metrics.

There are no automated Streamlit integration tests for corrupted checkpoint
recovery, changed-input stale-result behavior, reset, result-download contents,
or a fully disconnected-network launch. Browser smoke checks above establish
basic app rendering and two successful live runs, but do not replace those
regression tests.

## 7. Summary of findings

### Genuinely working now

- Local app launch and local checkpoint/dataset loading in the current
  environment.
- Ideal inverter output, gate/state generation, and RL recurrence with
  software-level tests.
- Checkpoint-normalized live CNN reference-target inference.
- Event finite-band, sampled finite-band, and full-spectrum THD as distinct
  measurements.
- Separate circular PSO experiment with training/validation split controls
  and reproducible saved artifacts.
- Full prior held-out CNN test evaluation and saved audit results.

### Must be addressed before describing the system as fully end-to-end

1. Resolve the default dataset from the project root, not the current working
   directory, and include source fingerprints in the dataset cache key.
2. Isolate dataset/checkpoint load errors so a corrupt model does not erase
   unrelated dashboard capabilities; test missing, malformed, and empty
   checkpoint fixtures without changing the saved checkpoint.
3. Add a live/exhibition export for the exact submitted operating point and
   its simulation, prediction, and harmonic results.
4. If a live baseline-versus-PSO comparison is desired, evaluate both phase
   configurations with identical operating inputs and label saved-validation
   comparison separately. Do not imply the CNN uses the PSO phase candidate.
5. Add application-level regression tests and an offline launch check.
6. Preserve the existing caveats: event vs sampled THD may differ due
   sampling/aliasing; no hardware/MATLAB/Simulink equivalence; no meaningful
   PSO improvement; no CNN harmonic-reduction evidence.

## Post-audit implementation update (2026-11-20)

The original audit above records the state before the live-workflow changes.
The following items have since been implemented without changing the existing
dashboard theme or training artifacts:

| Follow-up | Current status | Evidence |
|---|---|---|
| Live PSO action | **VERIFIED** | The Live Demonstration tab runs one configurable seeded circular-PSO run (defaults: 8 particles × 8 iterations; bounded to 2–12 × 1–12) over the deterministic 30-record training subset. |
| Candidate application and matched comparison | **VERIFIED** | The existing simulator and CNN inference path accept the selected carrier offsets. Baseline and candidate are rerun with the same live operating conditions and compared using Parseval full-spectrum THD. |
| Training/live feasibility labeling | **VERIFIED** | The optimizer retains the 0.05% hard tracking constraint on every optimization record; the live-point tracking result is checked and labeled separately. |
| Live result export | **VERIFIED** | CSV/JSON downloads and a ZIP with waveforms, gate/state channels, CNN errors, harmonic data, PSO convergence, metadata, and generated plots are available. |
| Project-root path handling | **VERIFIED** | App dataset paths and training/config paths resolve from `project_paths.py`; dataset fingerprints participate in the prepared-data cache key. The app successfully loaded its checkpoint and dataset when launched from `C:\Windows`. |
| Browser PSO smoke run | **VERIFIED, one operating point** | Seed `20261120`, 8 particles, 8 iterations, 30 training records; 64 unique candidate evaluations; runtime 16.22 s. It retained `[0°, 60°, 120°]`, with training-subset THD 26.3564% and live Parseval THD 24.7235% unchanged at the displayed precision. No meaningful improvement was observed. This is not validation evidence. |
| Follow-up regression suite | **VERIFIED** | `.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v` passed all 60 tests. |

`optimization/electrical_evaluator.py` is still an abstract interface; no
concrete implementation was invented. The workflow instead uses the existing
concrete circular `PhaseOptimizationProblem` and event evaluator, and applies
the candidate to `simulate_inverter`, which checks the generated switching
pattern, cell voltages, and seven allowed output levels. The original held-out
test split was not read or evaluated for this follow-up.

See [`EXHIBITION_IMPLEMENTATION_PLAN.md`](./EXHIBITION_IMPLEMENTATION_PLAN.md)
for an ordered, minimal implementation path.
