# Event-THD surrogate shortlist

## Purpose and separation

The live phase-optimization panel has an optional **Event-THD surrogate
shortlist** method. It is separate from the existing CNN, its checkpoint, and
the project dataset. The surrogate predicts event-based total voltage THD only.
It ranks bounded phase candidates; the existing `PhaseOptimizationProblem`
then evaluates the baseline and selected finalists with the exact event-based
evaluator and enforces the existing fundamental-phasor tracking constraint.
The best exact finalist is re-simulated before the dashboard displays it.

The dashboard offers this method only when its separate checkpoint is present
at `experiments/thd_surrogate_v1/model_run_modulation_holdout/thd_surrogate.pt`.
To avoid extrapolating beyond measured scope, it is enabled only for
modulation indices 0.55–0.95 in 0.05 steps, reference phase in 45-degree steps,
100 V per cell, and the conventional `[0, 60, 120]` degree carrier baseline.
Candidate phases remain bounded to `[0, 360)` with cell 1 fixed at zero.
Fundamental tracking feasibility and final selection are always evaluated
exactly, never inferred from the model. The live dashboard still needs its
normal CNN to render its comparison waveform; that CNN does not rank surrogate
candidates.

## Training and held-out evaluation

The separate training data contains 12,000 rows labelled by the existing
event-based THD evaluator. The model input is modulation index, reference
phase, and the two searched carrier offsets; training targets contain only
event-based total THD. The split holds out modulation bands rather than
randomly splitting adjacent modulation values. The recorded test results are:

| Metric | Held-out value |
|---|---:|
| MAE | 0.4422 percentage points |
| RMSE | 0.6284 percentage points |
| Maximum absolute error | 4.5710 percentage points |
| R² | 0.99858 |
| Spearman rank correlation | 0.99935 |

These are synthetic event-evaluator data, not measurements from hardware.
They establish neither hardware generalization nor a THD improvement.

## Wider ranking benchmark

The recorded operating-grid benchmark used 216 cases: nine modulation indices
from 0.55 through 0.95, eight reference phases from 0 through 315 degrees in
45-degree steps, and three seeds. Each case used the same seeded pool of 64
unique phase pairs, including the conventional baseline. The comparison
evaluated the full pool exactly versus surrogate-ranking the pool and exactly
evaluating eight finalists, with the baseline reserved among the finalists.
Exact finalist ranking used the same event-based total voltage THD and exact
feasibility checks.

| Measure | Result over 216 cases |
|---|---:|
| Mean Spearman correlation | 0.99823 |
| Minimum / maximum Spearman correlation | 0.99364 / 0.99940 |
| Cases with shortlist regret versus the best in the same 64-candidate pool | 0 |
| Mean / maximum shortlist regret | 0 / 0 percentage points |
| Cases better than the conventional baseline | 0 |
| Exact THD evaluations, full pool / shortlist | 64 / 8 |
| Mean event-evaluator time, full pool | 0.5579 s |
| Mean event-evaluator time, shortlist plus inference | 0.0653 s |
| Ratio of mean event-evaluator times | 8.55× |

The speed ratio is from the standalone event-evaluator benchmark and is
environment-specific; it is not a guarantee of end-to-end dashboard latency.
The shortlist matched the best candidate in each *sampled pool*, not a global
search optimum. In all 216 cases the selected candidate was the conventional
baseline. Thus the current evidence supports a promising way to screen random
candidates with fewer exact THD evaluations, but it does not show that the
surrogate or any optimizer improves on `[0, 60, 120]`.

The full recorded benchmark is
`experiments/thd_surrogate_v1/model_run_modulation_holdout/ranking_benchmark_operating_grid.json`;
the model test metrics are in the adjacent `summary.json`. These generated
artifacts are kept separate from the original CNN checkpoint and dataset.
