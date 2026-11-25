# Final measured-results report

## Objective and system

This project combines an ideal seven-level cascaded H-bridge inverter simulator,
an existing residual 1D CNN, and phase-offset search. The CNN predicts a target
voltage waveform; it does not predict switching states or directly control the
inverter. POA, PFO, their hybrid, and the existing PSO search carrier offsets
for cells 2 and 3 while cell 1 remains the reference. Every achieved THD value
reported below is simulator-based and independently recomputed.

The supplied dataset contains 10,000 records (7,000/1,500/1,500 train,
validation, and test), with 400 points per waveform. The existing CNN
checkpoint was loaded strictly in inference-only evaluations and not modified.

## THD conventions

- **Primary optimizer objective:** event-based Parseval total voltage THD,
  independently recomputed after candidate selection.
- **Event voltage/current H2–100:** event-based voltage harmonics and
  corresponding RL-derived current harmonics over orders 2–100.
- **Sampled voltage/current THD:** calculated from the simulator's 20 kHz
  sampled output over harmonics 2–100.

These are different metrics and must not be compared as interchangeable THD
values. Baseline and optimized measurements use the same evaluator within each
reported comparison.

## CNN evaluation

The saved inference-only grouped test evaluation covers the complete 1,500
record test split. Its grouping rules are derived from training conditions;
the checkpoint was loaded strictly, and neither training nor fine-tuning ran.
The recorded report found the CNN worse than the analytical reference in every
reported group. It found no held-out joint modulation/load combinations absent
from training, so this does not establish generalization to unseen joint
conditions. The CNN output remains a prediction, not an inverter measurement.

The grouped metrics and provenance are in
[`../results/cnn_grouped_test_eval_20261124.json`](../results/cnn_grouped_test_eval_20261124.json).
The protected checkpoint still has SHA-256
`9ffea39995c4bbba15f1f3572c9aaf54d49ad6ff1b1a186c4cafffe38263636c`; all
nine dataset CSV hashes recorded in that evaluation still match.

## Optimizer and baseline comparison

The 36 × 36 actual-simulator phase grid evaluated 1,296 combinations at the
validated operating point (modulation 0.75, 100 V/cell, 30 ohm, 0.03 H,
reference phase 0 degrees). Its minimum was **24.7234791510%** event-based
total voltage THD at `(60°, 120°)`, the same as the conventional baseline.
Across the three recorded seeds, PSO, POA, PFO, and Hybrid POA-PFO reached that
grid point with zero THD and phase-space gap. This validates search at that
operating point, not improvement over baseline.

The broader experiment contains 300 rows: five modulation indices
(0.55, 0.65, 0.75, 0.85, 0.95), ten paired seeds per index, and six methods
(baseline, PSO, POA, PFO, Hybrid POA-PFO, and CNN-guided hybrid). The table
below records the actual comparison protocol. The same conditions, seed
schedule, final THD definition, decision-variable bounds, and feasibility
check were used, but the evaluation budgets were **not identical**.

| Comparison factor | Shared protocol / measured values | Budget parity and qualifications |
|---|---|---|
| Operating conditions | 100 V/cell, 30 ohm, 0.03 H, reference phase 0 degrees; modulation indices 0.55, 0.65, 0.75, 0.85, and 0.95 | Same for every method at each paired condition |
| Random seeds | 20261124–20261133; 10 seeds at each of 5 modulation indices; 50 rows per method | Same seed schedule and paired conditions for all six methods |
| Decision variables | Cell 1 phase fixed at 0 degrees; cell 2 and 3 phases wrapped to [0, 360) degrees and canonicalized as a sorted interchangeable pair | Same phase representation and bounds |
| Final objective | Independently recomputed event-based Parseval total voltage THD, orders 2 through infinity | Same final metric; CNN-guided search uses a CNN score to rank candidates before event-THD finalist evaluation |
| Feasibility | Fundamental phasor tracking error no greater than 0.05%; final candidates independently rechecked | Same threshold; every recorded row is feasible |
| Baseline budget | 1 objective request, 1 unique evaluation, 1 full event-THD evaluation | Not matched to optimizers |
| PSO budget | Nominal setting 8 particles × (8 iterations + initial population) = 72 requests; observed 72 requests, 63–71 unique candidate evaluations, 65–73 full event-THD evaluations | Near 72 requests; distinct baseline/guided budgets |
| POA budget | Same nominal 72-request setting; observed 73–74 requests, 71 unique candidate evaluations, 73 full event-THD evaluations | Actual requests exceed nominal setting |
| PFO budget | Same nominal 72-request setting; observed 76 requests, 67 unique candidate evaluations, 69 full event-THD evaluations | Actual requests exceed nominal setting |
| Hybrid POA-PFO budget | Same nominal 72-request setting; observed 75 requests, 70 unique candidate evaluations, 72 full event-THD evaluations | Actual requests exceed nominal setting |
| CNN-guided Hybrid budget | 64 unique candidates ranked; 10 full event-THD evaluations; 66 simulator waveform generations, including final verification | Different budget and work profile; not a budget-matched comparison |

The CSV preserves per-run unique evaluations, objective requests, full
event-THD evaluations, waveform generations, feasibility, and runtime.

| Method | Mean event total voltage THD (%) | Mean reduction (%) | Improved runs |
|---|---:|---:|---:|
| Baseline | 27.039059 | 0.000000 | 0/50 |
| PSO | 27.025443 | 0.053772 | 12/50 |
| POA | 27.035528 | 0.015048 | 7/50 |
| PFO | 27.035979 | 0.013477 | 3/50 |
| Hybrid POA-PFO | 27.037384 | 0.007106 | 4/50 |
| CNN-guided Hybrid | 27.033504 | 0.019794 | 9/50 |

Across all 250 optimized runs, **35 improved, 215 were flat, and none were
worse** than their paired baseline. These are small, operating-condition-
specific results; they do not establish broad or physical inverter
improvement. Per-modulation results and paired statistics are in
[`../results/summary.md`](../results/summary.md) and all per-seed measurements
are in [`../results/thd_experiment.csv`](../results/thd_experiment.csv).

## CNN-guided search

On 500 seeded random phase pairs at modulation 0.75 and the same 100 V/cell,
30 ohm, 0.03 H, 0-degree conditions, CNN waveform RMSE ranking had a tie-aware
Spearman correlation of **0.9540794881** with event-based total voltage THD.
It exceeded the predeclared 0.70 gate, so a ten-seed comparison was run.

At this single operating point, both standalone and CNN-guided Hybrid POA-PFO
selected the grid-optimal baseline on all ten seeds: mean event total THD was
24.723479% for both, with no THD or phase-space gap. CNN guidance therefore
**did not improve THD**. It reduced mean full event-THD evaluations from 63 to
9 but increased mean simulator waveform generations from 1 to 65. Counts
measure different work; the data do not support a general efficiency claim.
Details are in [`cnn_guided_optimization.md`](./cnn_guided_optimization.md).

## Switching advisor

The dashboard's existing live workspace includes the offline switching
advisor. It uses Hybrid POA-PFO to recommend carrier offsets, then asks the
inverter simulator to generate and verify gate states and recompute THD. On
optimization failure or a verified result worse than baseline it returns the
baseline configuration and reports that no improvement was found. It is not
AI switch-state prediction, real-time control, a hardware controller, or a
protection system.

## Chatbot grounding and stale-result UI checks

The local `qwen2.5:3b` chatbot was probed with the same kind of computed context
the dashboard supplies (dataset split/count, model status, dashboard settings,
and live metrics when present). It answered the available dataset-count
question with 10,000 records and the 7,000/1,500/1,500 split. Asked for the
best optimizer improvement, which was not in that context, it said the
benchmark result was unavailable and did not give a number. Asked for hardware
conversion efficiency, it said that could not be determined from the provided
information. No unsupported numeric result was produced in these three probes.
This is a limited behavioral check, not a formal guarantee that a language
model can never hallucinate.

The stale-result notice remains in the live workspace. The existing
`test_dashboard_helpers.py` AppTest changes the modulation control while
retaining the previous simulation, asserts the message “Controls changed since
this run. Click Run New Simulation to update.”, then reruns the simulation and
asserts that the notice clears. Removing the separate clarification box did
not remove this stale-result state.

## Limitations and future work

The inverter and RL-load model is idealized: it does not model switching dead
time, semiconductor/device losses, parasitics, thermal behavior, or hardware
sensor effects. There is no hardware validation, and no fault protection or
closed-loop controller is implemented. CNN ranking correlation and optimizer
results are conditioned on the tested operating points and budgets; they do
not imply performance elsewhere.

Future validation should include MATLAB/Simulink cross-validation against an
authoritative model and controlled hardware experiments with appropriate
instrumentation and safety review.

## Reproduction and artifacts

Install dependencies with `python -m pip install -r requirements.txt`, run the
dashboard using `streamlit run app.py`, and run tests using
`python -m unittest discover -s tests -v`. From a fresh working copy with the
read-only checkpoint available, regenerate the 500-candidate correlation
report using the `measure_correlation`/`write_correlation_report` code in the
README, then run `python -m scripts.run_thd_experiment` to reproduce the
baseline/optimizer and CNN-guided comparison. Both report writers refuse to
overwrite existing results. The saved before/after, convergence, and
representative waveform/spectrum figures are linked from
[`../results/summary.md`](../results/summary.md).
