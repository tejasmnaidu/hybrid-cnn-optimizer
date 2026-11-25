# Circular phase PSO experiment report

## Result

All **20 of 20** requested seeds completed, each using 20 particles and 50
iterations, against the same 30-record training subset. The test split was not
evaluated. Following optimization, the selected candidate and the baseline
were each evaluated once on all 1,500 validation records.

The selected candidate was `[0°, 60.0780984597°, 120.0326906390°]`
(seed `20261139`). Its training-subset mean Parseval full-spectrum THD was
`26.3563432377%`, compared with the baseline's `26.3563554509%`. This is a
decrease of only `0.0000122131` percentage points (relative change
`-0.00004634%`), not evidence of a meaningful improvement.

On validation, the selected mean THD was `26.2806779435%` versus `26.2806279837%`
for baseline, a slight increase of `0.0000499596` percentage points (relative
change `+0.00019010%`). Both candidates satisfy the 0.05% phasor constraint on
every evaluated record. The optimization result therefore does **not** show
an improvement that generalizes to this validation split; it is effectively
at the baseline and marginally worse on validation.

## Fixed training subset

The deterministic selection was one nearest-centroid record from each cell in
the 5 modulation-index quantile by 6 equal-width circular phase-bin
stratification:

```text
103, 597, 1314, 1431, 1637, 1689, 1908, 2210, 2677, 2695,
2746, 2802, 2896, 2929, 3163, 3187, 4728, 5322, 5602, 5604,
6150, 6257, 6852, 7343, 7870, 8319, 8767, 8878, 9124, 9568
```

The exact inputs and selection strata are in `training_subset.csv`.

## Run and evaluation accounting

- Seeds: `20261120` through `20261139`; all completed.
- Per run: 1,020 objective requests (20 initial particles plus 50 updates of
  20 particles); cached duplicate candidates avoided repeated record runs.
- Runtime: mean `230.92 s` per seed; minimum `227.10 s`, maximum `235.29 s`.
- All run winners were feasible and no run returned a worse result than its
  baseline incumbent.
- Validation evaluated two candidates once each after optimization:
  baseline and selected winner, each on 1,500 validation records.
- Test records evaluated: **0**.

`run_summary.csv` contains the per-seed result, feasibility, evaluation counts,
runtime, and baseline objective. `convergence_history.csv` contains each
iteration's incumbent and current feasibility count. `experiment_config.json`
records the algorithm settings, selection rule and IDs; `final_results.json`
contains the final aggregate metrics.

## Caveats

This is a numerical result for the supplied ideal event-based model and
project conventions. The tiny training reduction is below a practically
meaningful distinction here, and validation is slightly worse. The experiment
does not establish global optimality, physical hardware performance, MATLAB
equivalence, or Simulink validation. It does not model dead time, semiconductor
losses, switching transients, or parasitics.
