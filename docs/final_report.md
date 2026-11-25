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
reference phase 0 degrees). Its 10°-grid minimum was **24.7234791510%**
event-based total voltage THD at `(60°, 120°)`, the same as the conventional
baseline. Across the three recorded seeds, PSO, POA, PFO, and Hybrid POA-PFO
reached that coarse-grid point with zero THD and phase-space gap. The later
0.5° local sweep found lower Parseval and sampled H2–100 values, so the coarse
grid result does not establish that `(60°,120°)` is exactly optimal.

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

### Targeted phase checks and equal-budget comparison

At m=0.75, a 0.5° local sweep around the default (φ₂=50°–70°,
φ₃=110°–130°) found a minimum event Parseval total voltage THD of
24.6779522784% (orders 2–∞), 0.0455268727 percentage points below the
24.7234791510% default. The sampled 20 kHz FFT H2–100 minimum was 9.7099870851%,
1.1614942330 points below its 10.8714813181% default. The exact event voltage
H2–100 minimum was 0.000000101887616%, tied at the default. These are local
grid results only. The full values and heatmap are
[`../results/carrier_shift_fine_sweep.csv`](../results/carrier_shift_fine_sweep.csv)
and
[`../results/carrier_shift_fine_sweep_heatmap.png`](../results/carrier_shift_fine_sweep_heatmap.png).

At modulation 0.55, 0.65, 0.85, and 0.95, each additional 10° coarse grid
had the same eight symmetry-related minima as the default configuration under
all three THD definitions. No point meaningfully beat the default; stored
event differences below 1e-13 percentage points are floating-point variation.
The summarized values and configurations are in
[`../results/carrier_shift_vs_modulation.csv`](../results/carrier_shift_vs_modulation.csv).

The new 0.5° local sweeps at those same modulation indices show that local
minima depend on which THD definition is used. Below, Parseval means event
total voltage THD (orders 2–∞), event means exact event voltage H2–100, and
sampled means 20 kHz FFT voltage H2–100. Each improvement is baseline minus
minimum in percentage points; tied sampled/Parseval minimizers are fully
listed in the linked result CSV.

| Modulation | Parseval minimum (φ₂,φ₃); gain (pp) | Event H2–100 minimum (φ₂,φ₃); gain (pp) | Sampled H2–100 minimum (representative φ₂,φ₃); gain (pp) |
|---:|---|---|---|
| 0.55 | 37.2204859303% at (60°,118°), (62°,120°); 0.0933023354 | 0.000000000269501% at (60°,120°); 0 | 12.2156848972% at 8 ties; 1.6619062489 |
| 0.65 | 28.7790228798% at (63°,117°); 0.0022597875 | 0.000000006727596% at (60°,120°); 0 | 10.1413136237% at 4 ties; 1.9388277219 |
| 0.85 | 23.5842680915% at (58°,121.5°), (58.5°,122°); 0.1033750353 | 0.000001054736196% at (60°,120°); 0 | 8.3590753165% at 30 ties; 0.4202599047 |
| 0.95 | 20.6066895992% at (61.5°,118.5°); 0.0824123783 | 0.000008080868143% at (60°,120°); 0 | 7.0332985753% at (59°,121°); 0.4845350091 |

At m=0.75, the sampled-metric minima's apparent 1.1614942330 pp gain at
20 kHz is not a robust harmonic reduction: at the same phases, exact event
H2–100 is 3.0285–3.8179% (the default is approximately zero), and at 1 MHz
sampled H2–100 those candidates are 2.8358–3.5914 pp worse than the default.
The low-rate result is strongly affected by aliasing. The aggregate energy
analysis and sampling-rate comparison are documented in
[`THD_METHOD_COMPARISON.md`](./THD_METHOD_COMPARISON.md).

To compare optimizers under an identical budget, PSO, POA, PFO, and Hybrid
POA-PFO were rerun at the default m=0.75 operating point with seeds
20261124–20261133, the same phase bounds and baseline, and the same
0.05% phasor-error feasibility limit. Each method received exactly 32
distinct simulator-evaluated candidates per seed, including the baseline;
search stopped at the cap, followed by an independent final re-simulation.
The objective and every value below are **event-based Parseval total voltage
THD, orders 2 through infinity**. Standard deviation is sample std (ddof=1).

| Method | Mean (%) | Std (%) | Min (%) | Max (%) | Mean reduction vs default (pp) | Best reduction (pp) | Improved seeds |
|---|---:|---:|---:|---:|---:|---:|---:|
| PSO | 24.723155574266 | 0.001023239597 | 24.720243383318 | 24.723479151038 | 0.000323576772 | 0.003235767720 | 1/10 |
| POA | 24.723352272252 | 0.000401225950 | 24.722210363179 | 24.723479151038 | 0.000126878786 | 0.001268787859 | 1/10 |
| PFO | 24.723479151038 | 0.000000000000 | 24.723479151038 | 24.723479151038 | 0.000000000000 | 0.000000000000 | 0/10 |
| Hybrid POA-PFO | 24.723352272252 | 0.000401225950 | 24.722210363179 | 24.723479151038 | 0.000126878786 | 0.001268787859 | 1/10 |

The best rerun result was PSO at 24.720243383318%, 0.003235767720
percentage points below the (60°,120°) default. POA and Hybrid POA-PFO each
reached 24.722210363179% at their best; PFO did not improve on the default.
All methods remain in this comparison because the common simulator-evaluation
cap was enforceable. The complete 40-run record, including candidate phases,
feasibility, actual candidate counts, and independent final checks, is
[`../results/optimizer_equal_budget_comparison.csv`](../results/optimizer_equal_budget_comparison.csv).
These results show small reductions at this one operating point and budget;
they do not establish broad optimizer superiority.

### Phase-offset conclusion

The conventional `(φ₂,φ₃)=(60°,120°)` setting is **near-optimal, but not
exactly optimal** on the examined 0.5° local grids under event Parseval total
THD and sampled 20 kHz H2–100 THD. At m=0.75, the local Parseval gain is
0.0455268727 percentage points; the sampled H2–100 gain is 1.1614942330
points, but it disappears and reverses at higher sampling rates because of
aliasing. Exact event H2–100 remains at the default across the tested
modulation indices. At m=0.55, 0.65, 0.85, and 0.95, Parseval gains range
from 0.0022597875 to 0.1033750353 pp, while the sampled H2–100 grid gains
range from 0.4202599047 to 1.9388277219 pp. These are local-grid findings
only, not proof of continuous global optimality.

The table below compares the best **event Parseval total voltage THD**
improvement from each modulation's 0.5° local sweep against the best result
from each optimizer's equal-budget search. Improvements are percentage points
relative to the (60°,120°) baseline at that modulation. The equal-budget
optimizer experiment was run **only at m=0.75**; “not run” is shown elsewhere
rather than extrapolating optimizer performance.

| Modulation | Fine-sweep best Parseval improvement (pp) | PSO best equal-budget improvement (pp) | POA | PFO | Hybrid POA-PFO |
|---:|---:|---:|---:|---:|---:|
| 0.55 | 0.0933023354 | Not run | Not run | Not run | Not run |
| 0.65 | 0.0022597875 | Not run | Not run | Not run | Not run |
| 0.75 | 0.0455268727 | 0.0032357677 | 0.0012687879 | 0.0000000000 | 0.0012687879 |
| 0.85 | 0.1033750353 | Not run | Not run | Not run | Not run |
| 0.95 | 0.0824123783 | Not run | Not run | Not run | Not run |

At m=0.75, the best equal-budget optimizer gain (PSO) is about 7.1 times
smaller than the fine-grid gain; the fine sweep checks 1,681 local points,
whereas each optimizer searches under a 32-distinct-candidate budget. This is
an observed search gap, not evidence that the optimizer could attain the
fine-grid result with a different budget or at another modulation. Detailed
per-seed results are in
[`../results/optimizer_equal_budget_comparison.csv`](../results/optimizer_equal_budget_comparison.csv)
and the fine-grid minima are in
[`../results/carrier_shift_fine_vs_modulation.csv`](../results/carrier_shift_fine_vs_modulation.csv).

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
