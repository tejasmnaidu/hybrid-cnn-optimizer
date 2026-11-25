# Baseline Feasibility Audit

**Verdict: PASS WITH LIMITATIONS**

The fixed baseline carrier offsets `[0°, 60°, 120°]` satisfy all evaluated
fundamental-phasor tracking thresholds on the complete development dataset:
7,000 training records and 1,500 validation records. The test split was not
evaluated and was not used to select or recommend a threshold.

## Scope and data handling

The dataset schema was verified from the CSV headers:

- `Sample_Splits.csv`: `sampleID`, `splitLabel`
- `Operating_Conditions.csv`: `sampleID`, `modulationIndex`, `loadResistance`,
  `loadInductance`, `phaseAngle`

The split-aware conventions in `preprocessing/preprocess.py` and
`training/dataset.py` identify `training`, `validation`, and `testing` as the
split labels. Existing preprocessing metadata reports 7,000 training, 1,500
validation, and 1,500 testing records. The split manifest was used only to
select training and validation IDs. For the operating-conditions CSV, values
of `modulationIndex` and `phaseAngle` were parsed only for those development
IDs; testing rows were skipped. No test waveform or test operating-condition
values were evaluated or used for threshold selection.

All 8,500 selected records were evaluated using
`evaluate_event_based_thd(m, phase, carrier_phase_offsets_degrees=(0, 60, 120))`
from `evaluation/event_based_thd.py`. `m` came from `modulationIndex`, and
`phase` came from `phaseAngle` in radians. Parseval full-spectrum THD was
recorded as `total_thd_percent`. Each record's result, inputs, split label,
status, and error (if any) are in
[baseline_feasibility_metrics_20261120_181949.csv](./results/baseline_feasibility_metrics_20261120_181949.csv).

## Phasor and phase calculations

The evaluator uses peak-scaled complex Fourier coefficients and the sine-phase
convention. For the target `300 m sin(2 pi 50 t + phase)`, the target peak and
phasor are:

```text
A_target = 3 * 100 V * m
C_target = -j * A_target * exp(j * phase)
```

The measured phasor tracking error is:

```text
100 * abs(C1 - C_target) / abs(C_target)
```

The phase error is independently formed from the sine-convention phases and
wrapped to `[-180°, 180°]`:

```text
delta = atan2(sin(phi_measured - phi_target),
              cos(phi_measured - phi_target))
phase_error_deg = degrees(delta)
```

For every record, the independently recomputed complex phasor error agreed
with `fundamental_phasor_error_percent` from the evaluator. The measured
fundamental peak matched `300*m` within floating-point precision. Wrapped
phase errors likewise remain at floating-point roundoff, confirming the
amplitude and phase comparison conventions. The largest measured absolute
phase error was approximately `2.04e-13°`.

## Development-set distributions

Percentiles use NumPy's default linear percentile definition. All statistics
below cover the combined training and validation records (`n=8,500`).

| Metric | Minimum | Median | Mean | P95 | P99 | Maximum |
|---|---:|---:|---:|---:|---:|---:|
| Parseval full-spectrum THD (%) | 20.672948 | 24.699010 | 26.418385 | 35.770612 | 37.008306 | 37.308004 |
| Fundamental peak amplitude (V) | 165.013228 | 224.976986 | 224.824062 | 278.750867 | 283.583202 | 284.984573 |
| Target fundamental peak amplitude (V) | 165.013228 | 224.976986 | 224.824062 | 278.750867 | 283.583202 | 284.984573 |
| Wrapped fundamental phase error (degrees) | -1.7811e-13 | 0 | -3.1321e-15 | 6.3611e-14 | 1.0178e-13 | 2.0356e-13 |
| Complex fundamental phasor tracking error (%) | 0 | 8.0559e-14 | 8.8714e-14 | 1.8228e-13 | 2.4335e-13 | 3.6851e-13 |

Split checks gave the same feasibility conclusion: the maximum phasor error was
`3.6851e-13%` in training and `3.1009e-13%` in validation; neither split had
any record above any tested threshold.

| Tracking-error threshold | Exceeding records | Percentage of valid development records |
|---:|---:|---:|
| 0.05% | 0 / 8,500 | 0% |
| 0.1% | 0 / 8,500 | 0% |
| 0.25% | 0 / 8,500 | 0% |
| 0.5% | 0 / 8,500 | 0% |
| 1% | 0 / 8,500 | 0% |

There were no evaluator exceptions, non-finite results, or invalid results:
8,500 evaluated, 8,500 valid.

## Feasibility and recommendation

A `0.1%` phasor-error constraint is feasible for every training and validation
record under the **baseline**. This does not establish feasibility for
alternative carrier phase offsets or optimized candidates, which have not
been evaluated.

For an initial PSO experiment, use a predeclared hard limit of **0.05%** rather
than relaxing to 0.1% without need. This is the strictest requested threshold
and still passes all 8,500 baseline cases by an enormous margin over the
observed floating-point error. It limits the allowable fundamental phasor
deviation to 0.05% of the target amplitude (about `0.083–0.142 V` over the
observed target-amplitude range). The baseline data establish feasibility,
not that 0.05% is a hardware or standards-based tolerance. In a later
optimization study, freeze the selected limit before running PSO and report
candidate feasibility; do not tune the limit after seeing optimization
results.

The near-zero baseline tracking error is expected for the evaluator's analytic
fundamental phasor under the current ideal switching conventions. It should
not be confused with low distortion: baseline Parseval THD is approximately
`20.67–37.31%` across these development operating points.

## Validation and limitations

Focused checks passed:

- `.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_event_based_thd.py -v`
  — **13 tests passed**.
- From the project root:
  `.\.venv\Scripts\python.exe -B -c "import sys,unittest; sys.path.insert(0,'tests'); from test_inverter_simulator import InverterSimulatorTests as T; suite=unittest.TestSuite(T(n) for n in dir(T) if n.startswith('test_') and n!='test_matches_existing_dataset_record'); result=unittest.TextTestRunner(verbosity=2).run(suite); sys.exit(0 if result.wasSuccessful() else 1)"`
  — **5 tests passed**. The dataset-backed simulator test was intentionally
  omitted to avoid reading any record whose split was not needed for this
  audit.
- Per-record output validation — **8,500 rows**, training/validation labels
  only, all statuses `ok`.

This audit covers the ideal event-based model and supplied dataset conventions
only. It does not validate physical hardware, MATLAB, or Simulink, and it does
not test PSO candidate behavior. No source code, dataset, checkpoint,
configuration, model, UI, or test-split record was modified.
