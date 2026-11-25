# Exhibition Implementation Plan

This plan describes a minimal next phase for the existing project. It is not
an authorization to retrain the CNN, overwrite artifacts, run the full PSO
experiment during exhibition, or change the scientific claims. The original
read-only audit made no implementation changes; the current follow-up status
is recorded below.

## Phase 1 implementation status (2026-11-20)

The requested live workflow additions are now implemented:

- The **LIVE DEMONSTRATION** tab has an explicit seeded circular-PSO action.
  It uses the existing `PhaseOptimizationProblem` and event-based evaluator on
  the deterministic 30-record training subset; UI limits are 2–12 particles
  and 1–12 iterations (defaults: 8 × 8). It does not evaluate validation or
  testing records.
- The selected phase offsets are applied through the existing simulator and
  live CNN-inference path. Baseline and candidate use identical live
  conditions and Parseval full-spectrum THD. Training-subset and live-point
  feasibility are shown separately; a live candidate that violates tracking
  is explicitly marked infeasible.
- CSV and JSON downloads, plus a ZIP with waveform/harmonic CSVs, PSO history,
  JSON metadata, and generated plots, are available for the current result.
- Dataset, YAML configuration, and training/checkpoint resource paths now
  resolve relative to the project root. The prepared-dataset cache key
  includes a source fingerprint.

The existing `optimization/electrical_evaluator.py` remains an abstract
adapter, not an executable evaluator. The live action therefore reuses the
concrete circular `PhaseOptimizationProblem`/event evaluator and then invokes
`simulate_inverter`, whose generated gate pattern checks shoot-through,
cell-voltage consistency, and allowed output levels. This implementation does
not claim hardware or external-simulator validation.

Verification completed: `.\.venv\Scripts\python.exe -B -m unittest discover
-s tests -v` passed all 60 tests. The app also loaded its dataset and trained
checkpoint from a launch directory outside the project root; its local
Streamlit health endpoint returned `ok`.

## Design principles

1. Keep the existing Streamlit theme, layout, and tabs.
2. Reuse existing simulator, CNN, event evaluator, and circular PSO backend.
3. Do not duplicate model, normalization, FFT, THD, or PSO logic in `app.py`.
4. Never load test records for training or optimization.
5. Do not run full multi-seed optimization on app startup or during a live
   presentation.
6. Every comparison must use identical operating records/conditions and
   explicitly identify whether its source is live simulation, training
   subset, or validation split.
7. Leave the existing checkpoint, source datasets, and prior result folders
   unchanged. Write new results to unique, non-overwriting paths.

## Target end-to-end workflow

| Step | User-visible behavior | Existing implementation to reuse | Planned connection / gap |
|---|---|---|---|
| 1. Operating conditions | Enter modulation, R, L, and reference phase or select a preset | `app.py` Live Demonstration form/preset controls | Already connected. Resolve values with the project root and preserve explicit submit semantics. |
| 2. Validate inputs | Reject invalid or unsupported ranges with useful feedback | `evaluation.live_simulation._validate_operating_inputs` | Already used by `run_live_simulation`; add UI-level error-path tests. |
| 3. Simulate inverter | Generate reference, three carriers, cell states, gate signals, cell voltages, output, and periodic RL current | `multilevel.inverter_simulator.simulate_inverter` | Already called by `evaluation.live_simulation.run_live_simulation`; preserve conventions and test invariants. |
| 4. Predict CNN target | Normalize signals and conditions using checkpoint statistics; run inference; denormalize target prediction | `evaluation.live_simulation.run_live_simulation`; `preprocessing.preprocess.NormalizationStats` | Already connected. The target is `simulation.reference * CELL_COUNT * DC_VOLTAGE_PER_CELL_V`; keep CNN output separate from inverter output and gates. |
| 5. Search carrier phases | Explicitly optimize cell 2/3 offsets on fixed training data; cell 1 remains zero | `evaluation.live_phase_optimization.optimize_training_phase_subset`; `optimization.circular_phase_pso.CircularPhasePSO` and `PhaseOptimizationProblem` | Implemented as a user-triggered, single-seed action over the fixed 30-record training subset; bounded to 12 particles and 12 iterations. The full multi-seed research run is not started by the app. |
| 6. Analyze harmonics | Report event finite-band, sampled finite-band, and Parseval full-spectrum THD with precise bands | `evaluation.event_based_thd.evaluate_event_based_thd`; `evaluation.thd_analysis.calculate_thd` | Already connected for live simulation. Preserve definitions and explicitly explain that sampled PWM may alias carrier energy. |
| 7. Compare baseline and selected phases | Compute both under identical m/phase and state exactly which population is used | `evaluation.live_simulation.run_live_simulation`; `PhaseOptimizationProblem.evaluate` | Implemented. The candidate is rerun through the simulator, CNN, and same event-based THD path; live-point tracking is checked separately from training-subset feasibility. |
| 8. Export results | Download conditions, waveform arrays, CNN metrics, event coefficients/THD, and exact comparison metadata | `evaluation.live_export.build_live_export` | Implemented: waveform CSV, JSON metrics/metadata, and ZIP with waveforms, harmonic spectrum, PSO history, and plots. |
| 9. Present limitations | Keep simulator scope, CNN target, and PSO validation result visible | Existing `app.py`, `SCIENTIFIC_VALIDATION_REPORT.md`, and this audit | Keep metrics labeled and do not state that CNN controls gates or that PSO has shown improvement. |

## Required implementation order

### Phase A — Startup and local resource reliability (paths implemented)

**Files:** `app.py`, potentially a focused app-resource helper under
`evaluation/`, plus a new test module.

1. **Implemented:** resolve default and user-entered relative paths against the
   project root; preserve absolute paths.
2. **Implemented:** include the dataset fingerprint in the cached prepared
   dataset call key so replacing a CSV invalidates the cache.
3. Separate dataset-load and checkpoint-load failures. If the checkpoint is
   missing/corrupt/incompatible, preserve any safe dashboard/data sections
   that do not require CNN inference; disable dependent actions and report the
   exact actionable failure.
4. Keep `torch.load(..., weights_only=True)`, normalization checks, model
   strict state-dictionary loading, and original checkpoint bytes untouched.
5. Add tests for default path resolution from a non-project current working
   directory; same-path dataset fingerprint changes; missing checkpoint;
   empty/truncated checkpoint; wrong metadata; strict state mismatch; and
   clear disabled inference controls.

### Phase B — Regression-test the live workflow

**Files:** `evaluation/live_simulation.py` only if a verified defect is found;
otherwise tests and `app.py` tests.

1. Keep `run_live_simulation` as the single entry point from operating
   conditions to simulation, harmonic metrics, and CNN prediction.
2. Test default and alternate preset conditions using the actual saved model
   where feasible, and a deterministic stub model in unit tests.
3. Test that changing each operating input either suppresses the old result
   until re-run or produces a new result after the explicit run action.
4. Test reset returns controls to defaults and clears result/error/cache keys.
5. Verify an inference failure never leaves an old result labeled as current.
6. Preserve marginal OOD language: it does not establish joint-condition
   support.

### Phase C — Exhibition PSO execution model (implemented)

**Files:** `optimization/phase_pso_experiment.py`, `app.py`, and tests only
if a real missing use case is selected.

The app's action is explicitly user-triggered and uses one configurable seed,
2–12 particles, and 1–12 iterations. It fits on the fixed 30-record training
subset only. It displays the seed, configuration, objective evaluations,
runtime, and feasibility. It does not evaluate validation or testing records.
The existing 20-seed research experiment remains separate and is not launched
by the dashboard. Errors are shown in the live tab; a subsequent run is
explicitly started by the user. Circular symmetry, hard tracking feasibility,
and the baseline incumbent remain those of the existing optimizer.

### Phase D — Matched baseline and selected-candidate comparison (implemented)

**Files:** likely a focused helper in `evaluation/` and `app.py`; test module
covering both candidate paths.

1. The live action uses the selected candidate returned by the current
   reproducible PSO run; the saved full validation comparison remains a
   separate result.
2. For a live comparison, evaluate baseline `[0°,60°,120°]` and the selected
   phases with the same modulation, reference phase, simulator constants,
   event evaluator, and harmonic indices.
3. Display baseline, selected value, selected-minus-baseline percentage-point
   delta, and relative percent delta with sign convention.
4. Keep the stored full validation-split comparison separate and identify its
   split and record count. Do not claim that saved validation means are live
   outputs.
5. Add tests showing both candidates receive identical operating conditions,
   fixed cell-1 offset, and the expected phase list; test a known non-improving
   candidate so the UI does not label it a success.

### Phase E — Local result export (implemented)

**Files:** `app.py`, preferably a small pure serializer helper under
`evaluation/`, plus tests.

1. Export a run identifier, simulator constants, modulation/R/L/phase,
   carrier offsets, split/source label, checkpoint identity, and metric
   definitions.
2. Include sampled time, reference target voltage, output voltage, load
   current, CNN prediction/error, three cell states, gate array, event
   Fourier orders/amplitudes, and the THD metrics actually calculated.
3. Provide CSV for row-oriented waveform signals and JSON for metadata and
   aggregate metrics; avoid unbounded JSON lists of large intermediate
   objects if a CSV is more appropriate.
4. Add a round-trip schema test and verify finite values/units, exact submitted
   condition, and baseline-vs-selected labels.
5. Do not include testing records in PSO export or use them to choose a
   candidate.

### Phase F — Offline and exhibition acceptance

**Files:** tests and documentation only, unless a reproduced defect is found.

1. Start the app from the project root using the local `.venv`.
2. Repeat a no-network launch test with all data/checkpoint files on disk.
3. Verify six tabs remain available and no remote resources are requested by
   the application.
4. Run default and at least two alternate live simulations; confirm reset and
   stale-result behavior.
5. Inspect checkpoint failure cases using temporary fixture files, not the
   saved checkpoint.
6. Confirm exports, saved validation comparison, and all simulator/
   evaluator/PSO tests.
7. Run the entire existing `unittest` suite.
8. Re-state the scientific limitations in the final exhibit documentation.

## Test matrix

| Component | Existing tests | Additional tests before calling exhibition-ready |
|---|---|---|
| Simulator and switching | `tests/test_inverter_simulator.py` | Additional operating points for state/gate invariants and explicit invalid combinations if a mapping changes. |
| RL current | `test_rl_recurrence_and_periodic_boundary` | Independent recurrence calculation at boundary load values; no physical plant claim. |
| Event Fourier / THD | `tests/test_event_based_thd.py` | Retain checks of event/sampled difference; add display/serialization band-index tests if formatting changes. |
| Circular PSO | `tests/test_circular_phase_pso.py` | Run-failure/partial-output behavior and no-overwrite assertion for experiment directory. |
| CNN live inference | `tests/test_live_simulation.py` | Checkpoint-backed inference smoke test and failure preserving no stale result. |
| Test-set isolation | `tests/test_cnn_test_evaluator.py` | Keep test evaluator isolated from any new optimizer entry point; no new model selection from test. |
| App startup | No dedicated Streamlit integration suite | Path, missing/corrupt checkpoint, no-network startup and six-tab availability. |
| UI live results | Current manual browser smoke only | Preset update, changed-input suppression, run success/failure, reset defaults, and no misleading stale metrics. |
| Export | Existing test CSV/JSON paths, not full live export | Schema, waveform lengths, metadata, units, integrity, unique filenames/output paths. |
| Baseline/PSO comparison | Existing saved validation CSV/JSON | Same-condition baseline/selected evaluation, delta sign/units, candidate phase and source split correctness. |

## Expected final demonstration workflow

1. Start locally with:

   ```powershell
   .\.venv\Scripts\python.exe -B -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
   ```

2. Open `http://127.0.0.1:8501`; verify dataset and checkpoint statuses.
3. Use **EXHIBITION MODE** for one-click default simulation, or **LIVE
   DEMONSTRATION** to change a preset/input and explicitly run again.
4. Explain the inverter's generated states/gates and output/current traces as
   ideal simulation signals.
5. Explain CNN prediction/error as target-waveform regression only.
6. Distinguish event finite-band THD, sampled finite-band THD, and Parseval
   full-spectrum THD; discuss sampling/aliasing where sampled and event
   finite-band values differ.
7. Show the saved baseline-versus-PSO validation comparison separately from
   live metrics. State that the current candidate is effectively unchanged
   and slightly worse on validation.
8. Export only the current completed run, with its operating conditions and
   scientific definitions included.
9. Keep test-set outputs as final audit evidence; do not tune the project from
   them.

## Definition of done for the next implementation phase

- Default startup works from any current directory and offline using local
  files and the existing environment.
- Missing/corrupt model state produces an actionable warning without hiding
  unrelated safe application sections.
- Live conditions always map to fresh, labeled simulation/inference results;
  reset and errors cannot display stale outputs.
- Matched baseline/selected comparison is scientifically labeled and never
  overstates the saved PSO evidence.
- A complete live run can be exported without overwriting prior results.
- Current unit tests and new app-path/failure/export tests pass.
- No checkpoint, dataset, or test data is modified or used for optimization.
