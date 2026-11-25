> **SUPERSEDED — 2026-11-24**
>
> This audit recorded the project state before the ideal seven-level PWM/RL simulator was implemented. The present simulator, dataset agreement, and remaining validation limits are documented in [docs/SIMULATOR_VALIDATION.md](./docs/SIMULATOR_VALIDATION.md) and [docs/CURRENT_STATUS.md](./docs/CURRENT_STATUS.md). The original text below is preserved as historical evidence and is not the current status.

# Electrical Model Audit

**Project:** `Hybrid_CNN_Inverter_Optimization`  
**Audit date:** 2026-11-20  
**Scope:** Read-only inspection of the supplied Python project, configuration, dataset, archived preprocessing script, evaluation utilities, optimization interfaces, and safe tests. No model training, PSO optimization, source edits, dataset writes, checkpoint writes, or hardware/Simulink validation were performed.

## Implementation update

After this initial audit, the project owner supplied the PWM, gate, and RL
recurrence conventions in chat. The new
[`multilevel/inverter_simulator.py`](./multilevel/inverter_simulator.py)
implements that specification and is tested against the existing CSVs. The
provided MATLAB source itself remains unavailable, so this is not a
line-by-line MATLAB verification or hardware/Simulink validation. The original
audit below describes the project state **before** those conventions and the
new simulator were supplied.

- Generated cell states match all 10,000 dataset records.
- Generated gates match all 10,000 records in `(cell, switch, time)` order.
- Gate-derived output voltage matches `Input_Voltage.csv` exactly.
- The periodic RL recurrence matches `Load_Current.csv` within floating-point
  precision (simulator comparison RMSE approximately `2.35e-14 A`, maximum
  absolute error approximately `1.83e-13 A`).
- The evaluator interface remains abstract and PSO has not been integrated
  with this simulator or run.

## Executive findings

| Question | Finding |
|---|---|
| Does a complete seven-level switching simulator exist? | **No.** There is no `multilevel/` module, carrier generator, PWM comparator, gate scheduler, or concrete switched-circuit evaluator in the supplied Python source. |
| What is `optimization/electrical_evaluator.py`? | An **abstract adapter** and result dataclass. Its `evaluate_candidate` method has no implementation. |
| Is the original MATLAB generation script included? | **No MATLAB/Simulink source was found** in the project or supplied dataset ZIP. The ZIP does contain `preprocess_seven_level_inverter.py`; it validates, aligns, normalizes, and packages already-generated CSVs. It does not generate inverter PWM or simulate the load. |
| Can the Python project read and validate the supplied electrical records? | **Yes.** It loads the waveform CSVs and can read gate/state rows when a storage layout is explicitly supplied. Existing validation reports aligned records and valid discrete values. |
| Can supplied electrical signals be reconstructed from data? | **Several relationships are reproducible, but the project has no complete signal generator.** Cell states reconstruct every recorded input-voltage point; a sinusoidal formula reproduces every target point; a periodic exact-ZOH RL recurrence reproduces the recorded current to floating-point precision. Only the test-set input-voltage reconstruction exists as a Python evaluation helper; the target and current formula checks were independent audit calculations, not existing simulator functions. These checks do not prove the missing MATLAB implementation or independently validate a physical circuit. |
| Are switching gates reproducible from the provided configuration? | **Not from the evidence supplied.** Gate values are present, but channel ordering, gate polarity/device assignment, carrier phase offsets, comparator rules, and gate-to-state mapping are not specified in an executable source. |
| Is PSO ready to optimize a physical inverter? | **No.** The optimizer is generic and bounded, but no physically defined candidate vector, concrete evaluator, verified switching constraints, or simulator binding is configured. No optimization was run. |

## Project and source inventory

- The project contains `optimization/`, `preprocessing/`, `evaluation/`, `configs/`, `tests/`, the CNN/training modules, and saved experiment artifacts.
- **No `multilevel/` directory or package is present.**
- No `.m`, `.mlx`, `.slx`, or `.mdl` file was found under the extracted project or in the supplied `Seven_Level_Inverter_Project.zip`.
- The archive contains `Seven_Level_Inverter_Project/preprocess_seven_level_inverter.py`. Its source reads the CSVs, validates and aligns records, makes training/validation/testing arrays, and writes preprocessing artifacts. It explicitly treats the target as an ideal reference and warns that actual post-CNN output/THD requires a separate simulation. It is not the partner's MATLAB switching/load-generation algorithm.
- Existing project reports and run artifacts are historical outputs. Their presence does not establish an implemented Python simulator or physical/Simulink validation.

## Configuration and dataset facts

`data/raw/Seven_Level_Inverter_Project_extracted/Configuration.csv` declares:

| Parameter | Declared value |
|---|---:|
| DC voltage per source | 100 V |
| Number of DC sources | 3 |
| Number of H-bridge cells | 3 |
| Number of switches | 12 |
| Number of voltage levels | 7 |
| Output levels | -300, -200, -100, 0, 100, 200, 300 V |
| Fundamental frequency | 50 Hz |
| Carrier frequency | 1 kHz |
| Sampling frequency | 20 kHz |
| Samples per cycle | 400 |
| Total records | 10,000 (7,000 training, 1,500 validation, 1,500 testing) |

The CSV schemas are consistent with the declared waveform dimensions:

- `Input_Voltage.csv`, `Target_Voltage.csv`, and `Load_Current.csv`: 10,000 rows with 400 signal values per row.
- `Operating_Conditions.csv`: `modulationIndex`, `loadResistance`, `loadInductance`, and `phaseAngle`.
- `Cell_States.csv`: 1,200 state values per record (3 × 400), using -1, 0, and 1.
- `Switching_Gates.csv`: 4,800 binary values per record (12 × 400).
- `Sample_Splits.csv` and `THD_Results.csv` carry aligned record IDs.

`results/dataset_validation.json` reports a valid dataset and aligned auxiliary IDs. Its warning is important: it validates the gate and state files separately because the schema does not establish the physical gate-to-cell-state mapping.

## What the Python modules implement

### Data loading and preprocessing

- `preprocessing/data_loader.py` loads the required CSV tables by filename and ID; ZIP input is extracted to a directory.
- `preprocessing/data_validation.py` checks schema, sample IDs, value domains, configured sample/split counts, and optionally the large auxiliary files in chunks.
- `preprocessing/preprocess.py` aligns waveforms and conditions by `SampleID`, uses the provided split labels, fits normalization on training records, and returns normalized arrays. It does **not** generate switched waveforms or simulate an RL load.
- `preprocessing/switching_data.py` loads one stored gate/state record. Its reshape methods require the caller to declare an ordering; that is file access, not an inferred physical mapping.

### Evaluation

- `evaluation/metrics.py`, `harmonic_analysis.py`, and `thd_analysis.py` calculate regression metrics and FFT-based waveform THD for arrays supplied by a caller.
- `evaluation/baseline_thd.py` reconstructs input voltage from cell-state rows for test records and calculates its THD. It is a dataset analysis path, not a PWM or post-optimization simulator.
- No evaluation module generates the original gates, simulates a switched inverter, or evaluates an optimized inverter output.

### Optimization

- `optimization/electrical_evaluator.py` defines `EvaluationResult` and an abstract `ElectricalEvaluator.evaluate_candidate(candidate)` method. There is no concrete subclass in the project.
- `optimization/objective_function.py` scores THD, tracking error, transition count, output-level validity, and a supplied constraints flag **after** an evaluator returns them. The default objective weights and penalties in `configs/config.yaml` are study settings, not derived circuit constants.
- `optimization/pso_optimizer.py` implements a generic bounded global-best PSO algorithm. It requires a caller-supplied `PSOConfig.bounds` and objective; it has no inverter-specific variables or model binding.
- `optimization/switching_constraints.py` provides array-shape/value checks, output-level checks, an optional transition-count limit, and a callback-based gate/state consistency check. It does not define device ordering, complementary gate pairs, dead-time/interlocks, a physical mapping, or electrical ratings.
- `configs/config.yaml` supplies candidate output levels and objective weights, but no candidate vector, variable bounds, carrier-phase settings, concrete evaluator, or transition limit (`maximum_transitions` is `null`).

## Comparison with the stated MATLAB algorithm

The MATLAB file itself is absent, so a line-by-line algorithm comparison is not possible. The following distinguishes declared settings and data-supported identities from missing generation logic.

### Three sources, three cells, seven levels

The configuration declares three 100 V sources, three H-bridge cells, and seven output levels. The saved cell states use -1/0/1 and have three groups of 400 values. A read-only check tested both possible flat-array layouts against the saved input voltage:

- Interpreting each consecutive 400-value block as one cell and summing the three states × 100 V reproduced **all 4,000,000 input-voltage samples exactly**.
- The time-major interpretation did not reproduce all points.

Thus, the stored `Cell_States.csv` is consistent with the relation

`inputVoltage[n] = 100 V × (state1[n] + state2[n] + state3[n])`

when the data are interpreted as three cell-major blocks. This verifies a relationship among these dataset columns. It does not identify which gate channels operate each cell or prove a device-level H-bridge model.

### Sampling and carrier settings

The configuration declares 50 Hz, 20 kHz sampling, and 400 points per fundamental cycle; these values satisfy `20,000 / 50 = 400`. It also declares a 1 kHz carrier, or 20 samples per carrier period at the configured sample rate.

No executable carrier generation was found. The configuration does not state the individual carrier phases, carrier waveform amplitude/offset, comparator polarity, reference scaling, switching update/sample convention, or how comparator outputs map to the 12 gate columns. A 1 kHz frequency by itself is insufficient to regenerate the gate dataset.

### Modulation and target reference

The stored target is not generated by Python source in this project; the archived preprocessing script consumes `Target_Voltage.csv`. A read-only comparison across all records found the saved target consistent to floating-point precision with:

`targetVoltage[n] = 300 × modulationIndex × sin(2π × 50 × n / 20,000 + phaseAngle)`

Here 300 V is three times the declared 100 V source, and the recorded `phaseAngle` behaves as radians for this identity. The maximum absolute residual was approximately `2.4e-12 V` (RMSE approximately `5.1e-13 V`).

This verifies the relationship of the supplied reference waveform to its stored conditions. It does **not** reproduce MATLAB's carrier-comparison/switching logic, nor does an ideal reference establish achieved inverter output or harmonic reduction.

### RL load current and periodic initial condition

The supplied project has no current-generation function. However, the saved current was independently compared against the following exact solution for a series RL equation with each input-voltage sample held constant over one sampling interval:

```
dt = 1 / Fs
a  = exp(-R * dt / L)
b  = (1 - a) / R
i[n + 1] = a * i[n] + b * inputVoltage[n]
```

For `N = 400` samples and a periodic steady state (`i[N] = i[0]`), the initial current is:

```
i[0] = b * sum(a**(N - 1 - k) * inputVoltage[k] for k in 0..N-1)
       / (1 - a**N)
```

Using each record's stored `loadResistance` and `loadInductance`, this recurrence reproduced all recorded current samples with RMSE approximately `1.25e-14 A` and maximum absolute residual approximately `6.1e-14 A`. This is strong evidence that the supplied current is consistent with this periodic, exact-zero-order-hold RL recurrence driven by the recorded input voltage. It is a data-based reproduction check, not a comparison to the missing MATLAB source or a separate hardware model validation.

### Switching gates

The supplied gate data exist and the validation report confirms 10,000 aligned rows, 4,800 binary values per row. Python can load a stored gate row. It cannot regenerate those gates from the present configuration because the gate channel order, device pairing/polarity, carrier phase offsets, comparator logic, and gate-to-cell relationship are not defined in executable Python or in the supplied MATLAB source (which is absent).

## Can the existing Python project reproduce the records?

| Record | Current evidence | Reproducibility conclusion |
|---|---|---|
| Input voltage | Exact reconstruction from the stored cell states with 100 V per cell; checked over all rows and time points. `evaluation/baseline_thd.py` implements a test-split version of this reconstruction. | **Reconstructable from saved states.** The project does not regenerate the switching decisions. |
| Cell states | Stored as 1,200 values per row; the cell-major layout is supported by the exact input-voltage reconstruction. | **Can read the stored state array, but cannot regenerate states from PWM logic.** |
| Switching gates | Stored as 4,800 binary values per row and accessible via an explicit-layout loader. | **Can read stored gates, but cannot synthesize/reproduce them from carrier/control inputs.** |
| Target reference voltage | The stated sinusoidal equation matches the saved data to floating-point precision. | **The equation reproduces the reference, but no such generator exists in the current Python project.** This is not measured output. |
| Load current | The exact-ZOH RL recurrence plus periodic initial condition matches the saved current to floating-point precision. | **The equation reproduces the saved current, but the current project does not implement the recurrence.** The missing MATLAB source prevents source-code equivalence claims. |
| Post-optimization inverter output | No concrete electrical evaluator or switched simulator exists. | **No.** No physical PSO objective can currently be measured. |

## Missing information and assumptions to resolve

Before calling a Python implementation equivalent to the partner's MATLAB generator, obtain the exact MATLAB source and document:

1. Carrier construction per cell: triangle definition, amplitude, offset, starting phase, phase-shift angles, and whether phase is normalized to carrier period.
2. Reference/modulation comparison: exact scaling and comparator inequalities, including behavior at equality and sample-boundary ordering.
3. Gate output conventions: ordering of the 12 `gateData` columns, bridge/device assignment, active polarity, complementary relationships, dead time, and safe states.
4. Cell-state convention: which gate combinations map to -1, 0, and +1 for each cell, and how the series output voltage is formed.
5. Timing convention: whether values are recorded before or after each switching update and whether the 400 samples include one period endpoint or use a half-open cycle.
6. Load model: exact discretization method, input waveform used, R/L units and ranges, and how the periodic initial condition is computed.
7. Baseline semantics: whether the recorded current is the load current caused by the switched input voltage, and whether any parasitics, snubbers, or non-ideal switch effects are included.

The dataset checks support one exact candidate target equation and one exact-ZOH periodic RL recurrence, but they do not resolve missing carrier/gate semantics. Those must be compared directly to the original source or clarified by the partner rather than guessed.

## Recommended implementation plan

1. **Acquire the authoritative generator.** Obtain the partner's MATLAB script and any referenced helper functions or model files. Record source version, units, timing, carrier phases, gate order, and comparator boundary rules before coding.
2. **Freeze a small golden fixture.** Select records spanning modulation and R/L settings across the supplied splits. Preserve their input conditions and the MATLAB-produced 400-point target, cell-state, gate, voltage, and current arrays as read-only reference data.
3. **Implement waveform/reference generation separately.** Add a clear pure-Python function for the verified ideal target equation and test every sample against the CSV. Keep reference generation distinct from switched output.
4. **Implement carrier and switching logic only from the MATLAB specification.** Make carrier phases, comparator rules, gate mapping, and time indexing explicit. Test gate dimensions/binary values, valid device combinations, dead-time/interlock rules if present in the source, cell-state decoding, output levels, and switching transitions.
5. **Implement the RL load solver and periodic boundary condition.** Use the exact recurrence verified against the dataset if MATLAB confirms it. Derive and test the periodic initial current, then compare the regenerated current to CSV records over the full dataset with stated numerical tolerances.
6. **Validate stage by stage.** Compare generated gates first, then states, output voltage, current, and finally metrics. Report per-signal max/RMSE mismatches and first differing indices; do not hide an unexplained mismatch behind a tolerance.
7. **Add a concrete electrical evaluator.** Define the PSO candidate vector and physically justified bounds only after the control parameters and switching behavior are known. The evaluator should return actual simulated waveform metrics, transition counts, levels, and explicit constraint results.
8. **Validate the objective and PSO on known cases before a study run.** Establish a reproducible baseline and hard constraints, verify objective units/weights and infeasible-candidate behavior, and only then run PSO. Any claims beyond the dataset-matching Python model require separate independent simulator or experimental evidence.

## Safe verification performed

- Read-only inspection of the project sources, config, existing validation report, CSV schemas, and archive members. No MATLAB file was present.
- Read-only dataset validation of the extracted main CSVs: **valid**, 10,000 records, 400 points, and 7,000/1,500/1,500 splits.
- Compared all 4,000,000 cell-state-derived input-voltage points against `Input_Voltage.csv`: **exact match** using three cell-major blocks and 100 V per state.
- Compared all target points against the stored-condition sinusoidal equation: max absolute residual approximately `2.4e-12 V`.
- Compared all current points against the periodic exact-ZOH RL recurrence: RMSE approximately `1.25e-14 A`, max absolute residual approximately `6.1e-14 A`.
- Ran seven selected dataset/model/evaluation/layout unit tests: **all passed**. The test that runs a toy PSO optimizer was deliberately excluded; no PSO optimization was run.
- No Python source, configuration, dataset, checkpoint, or existing result was modified by this audit.
