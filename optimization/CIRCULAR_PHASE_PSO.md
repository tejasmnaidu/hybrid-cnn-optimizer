# Circular carrier-phase PSO

`circular_phase_pso.py` implements the inverter-specific PSO independently of
the generic bounded optimizer. The generic optimizer retains Euclidean
differences and clipping and is not used for phase search.

## Variables and circular symmetry

The candidate is `(phi2, phi3)` in degrees. Cell 1 remains fixed at `0°`;
positions are wrapped with modulo `360°`, never clipped at a phase boundary.
Cell 2 and cell 3 are identical, so each objective evaluation canonicalizes
the pair by wrapping both values and sorting them. Thus `(a,b)`, `(b,a)`, and
angle-equivalent values share one electrical representation.

For target phase `a` and current phase `b`, the shortest signed circular
difference is:

```text
delta(a,b) = ((a - b + 180) mod 360) - 180
```

For the two-cell pair, the PSO compares direct and swapped matchings and uses
the matching with the shorter total squared circular distance. For each
coordinate, velocity and position are updated as:

```text
v = w*v + c1*r1*delta(pbest, x) + c2*r2*delta(gbest, x)
x = (x + clip(v, -vmax, vmax)) mod 360
```

The default settings are 20 particles, 50 iterations, inertia `0.7298`,
cognitive/social coefficients `1.49618`, and maximum velocity `72°`.
They are configurable through `CircularPSOConfig` and the experiment command
line. Each run inserts the baseline `[60°,120°]` into its initial population.

## Objective and hard constraint

Each candidate is evaluated with `evaluate_event_based_thd` using the fixed
baseline operating-point subset and offsets `(0°, phi2, phi3)`. The objective
is the arithmetic mean of `total_thd_percent` (Parseval full-spectrum THD)
across the subset. A candidate is feasible only if every record's
`fundamental_phasor_error_percent` is at most `0.05%`, using the evaluator's
complex sine-phasor comparison against `-j * (300*m) * exp(j*phase)`.

Feasibility is lexicographically ranked before THD; infeasible candidates can
never replace a feasible personal/global best. Among infeasible candidates,
maximum tracking-constraint violation guides the swarm toward feasibility;
this is not a weighted penalty. Invalid/non-finite/negative evaluator metrics
and evaluator exceptions raise `CandidateEvaluationError` with candidate and
SampleID context. They are never mapped to a low score.

## Dataset selection and evaluation protocol

The experiment reads `Sample_Splits.csv` for split IDs/labels and only loads
numeric `Operating_Conditions.csv` values for training records before PSO.
It deterministically selects 30 records: one representative nearest the
within-stratum modulation median and circular phase-bin center from each of
five modulation-index quantile bins crossed with six equal-width phase bins.
Empty strata are rejected rather than silently replaced. The resulting
SampleIDs and conditions are written to `training_subset.csv` and the full
configuration is saved to `experiment_config.json`.

All 20 seeds, `20261120` through `20261139`, use the same fixed subset. Run
history records the best feasible candidate, feasibility counts, objective
requests, unique candidate evaluations, record evaluations, cache hits, and
runtime. Baseline is evaluated as a feasible incumbent. After all runs, the
best feasible training candidate and baseline are each evaluated once over
the 1,500 validation records, and their summary is written to
`validation_comparison.csv` and `final_results.json`. No testing operating
conditions or waveforms are evaluated.

Run from the project root:

```powershell
.\.venv\Scripts\python.exe -B -m optimization.phase_pso_experiment
```

Optional arguments `--particles`, `--iterations`, `--seed-start`, `--runs`,
and `--output-directory` support reproducible test runs or a new explicit
output path. The default experiment creates a timestamped, non-overwriting
directory under `results/`.

## Limitations

The experiment uses the ideal event-based simulator and the project-provided
PWM and gate conventions. The original MATLAB source was not included. It
does not model dead time, semiconductor losses, switching transients,
parasitics, hardware, or Simulink. Subset feasibility and validation
performance do not prove global optimality or generalization to unobserved
operating conditions. CNN outputs are not used in this backend.
