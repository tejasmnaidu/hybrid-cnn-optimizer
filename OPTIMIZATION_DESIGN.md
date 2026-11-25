> **SUPERSEDED — 2026-11-24**
>
> This document was a design stage and predates implementation of live PSO, POA, PFO, and hybrid carrier-phase optimization plus independent live candidate re-simulation. It is retained as historical design rationale, not as the current implementation status. See [docs/CURRENT_STATUS.md](./docs/CURRENT_STATUS.md) and [docs/SIMULATOR_VALIDATION.md](./docs/SIMULATOR_VALIDATION.md).

# Stage 3: Inverter-Specific PSO Harmonic Optimization Design

**Status:** Design only. No PSO run, optimization result, or harmonic-improvement claim is made in this document.

## 1. Scope and verified implementation

The proposed first search changes only the relative phase offsets of the second
and third triangular carriers. Keep the existing ideal inverter topology,
100 V cell voltages, comparator equations, gate truth table, switching sample
rate, fundamental reference, and periodic RL recurrence unchanged.

The current simulator is
[`multilevel/inverter_simulator.py`](./multilevel/inverter_simulator.py).
It emits one 400-sample cycle at 20 kHz and currently hard-codes carrier
offsets to 0°, 60°, and 120°. The first carrier is the fixed phase reference.
The supplied gate mapping and voltage reconstruction have already matched all
10,000 dataset records; that validates reproduction of this ideal dataset
under the provided conventions, not physical hardware behavior.

The original MATLAB generator is not present. The simulator follows the
conventions supplied by the project owner; it has not been compared
line-by-line against MATLAB, Simulink, or hardware.

## 2. Baseline definition and fair operating conditions

### Fixed PWM baseline

The baseline carrier offsets are:

```
phi1 = 0 degrees
phi2 = 60 degrees
phi3 = 120 degrees
```

For each evaluation case, preserve exactly the same modulation index `m`,
load resistance `R`, load inductance `L`, and fundamental phase `theta` for
the baseline and every candidate. Also hold fixed:

- DC voltage: 100 V per cell; three cells.
- Fundamental reference: `m * sin(2*pi*50*t + theta)` using the simulator's
  stated normalized comparator reference.
- Desired voltage reference in physical units:
  `v_ref[n] = 3 * 100 V * m * sin(2*pi*50*t[n] + theta)`.
- Carrier frequency 1 kHz, sampling rate 20 kHz, and 400 half-open-cycle
  samples `t[n] = n / 20,000`, `n = 0..399`.
- Same gate truth table, RL recurrence, FFT implementation, and constraints.

Do not silently regenerate or change any other configuration while searching
carrier phase.

### Pilot condition

For a reproducible software pilot, use one preselected operating point and
record its source before optimization. A convenient existing point is
`SampleID 1` from `Operating_Conditions.csv`:

| Quantity | Value |
|---|---:|
| Modulation index | 0.699816047538945 |
| Load resistance | 30 ohm |
| Load inductance | 0.03 H |
| Reference phase | 5.97351416136022 rad |

These are fixed for both baseline and candidate evaluations; they are not
optimized. A one-condition pilot is only an algorithm and objective check. It
does not establish performance over the operating envelope. Before drawing
broader conclusions, preregister multiple operating points and aggregate over
the same points for every candidate. A defensible follow-on is to select
training/selection conditions in advance and reserve a disjoint, untouched
condition set for final evaluation. Do not select operating points after
looking at their candidate results.

## 3. Candidate variables and phase symmetries

### Proposed variables

Use the two relative carrier offsets:

```
x = [phi2, phi3]
phi1 = 0 degrees (fixed)
```

Carrier phase is periodic modulo 360 degrees. Full-circle offsets are
technically valid under the stated generator: changing an offset changes the
relative timing of that cell's carrier comparisons without changing the
topology, cell DC voltages, comparator rule, or gate truth table.

### Recommended representation and bounds

The physical search domain is:

```
phi2, phi3 in [0, 360) degrees
```

Since cells 2 and 3 have equal DC sources and identical switching rules,
swapping their carrier offsets only relabels the cells and produces the same
aggregate voltage/current. Use a canonical representation:

1. Reduce both angles modulo 360 degrees.
2. Sort them so `0 <= phi2 <= phi3 < 360`.
3. Cache/evaluate only this canonical pair.

The baseline `[60, 120]` is already canonical. A phase difference of 0 and
360 degrees is identical; do not treat these as distinct settings. Equal
carrier offsets are allowed by the stated equations, although they eliminate
interleaving between those cells. Do not exclude equal or near-equal offsets
unless an explicit engineering minimum-separation requirement is supplied.
If such a requirement is adopted, define it in degrees before optimization and
apply it to circular distance, not ordinary subtraction.

### PSO implementation implications

The existing [`PSOOptimizer`](./optimization/pso_optimizer.py) clips each
coordinate to its bounds and computes ordinary Euclidean coordinate
differences. That is not a suitable direct encoding for periodic angles:
359° and 1° are physically 2° apart, not 358° apart, and a clipped phase can
stick at a boundary. Sorting only the objective input also leaves the swarm
visiting equivalent `[phi2, phi3]` and `[phi3, phi2]` points.

Before a later PSO run, either:

- implement a periodic-angle PSO update using wraparound positions and shortest
  circular differences, while canonicalizing cell permutations; or
- implement a candidate encoding/projection that explicitly handles circular
  phases and canonical cell labels, with tests proving that it covers the
  intended physical domain.

Retain both the raw particle vector and canonical physical phase pair in
results. Add tests for 0°/360° equivalence, swapped-cell equivalence, wraparound
updates, and deterministic seeded runs. Do not impose an arbitrary restricted
arc such as `[0, 180]`; that would require a proof that it preserves all
distinct configurations for the fixed reference and evaluation.

## 4. Objective: reduce THD without sacrificing fundamental tracking

### Metrics

For each simulated output-voltage cycle `v[n]`, calculate its complex
fundamental phasor and harmonic amplitudes from the same 400 samples. The
desired physical reference is the 300*m-volt sinusoid above; its fundamental
peak amplitude is `V1_ref = 300*m` V.

Use a complex fundamental coefficient such as:

```
C_k(v) = (2/N) * sum((v[n] - mean(v)) * exp(-j*2*pi*k*n/N))
```

with `N = 400`; use the identical convention for the desired reference.
Define normalized fundamental tracking error as:

```
e1 = abs(C_1(v) - C_1(v_ref)) / abs(C_1(v_ref))
```

This penalizes both fundamental amplitude and phase error. Also report
amplitude-only error and phase difference separately for interpretation. Reject
or mark infeasible a case with `abs(C_1(v_ref))` below a predeclared minimum;
do not divide by zero or assign it a favorable score.

Calculate band-limited voltage THD using the existing
[`harmonic_spectrum`](./evaluation/harmonic_analysis.py) and
[`calculate_thd`](./evaluation/thd_analysis.py) conventions:

```
THD_percent = 100 * sqrt(sum(A_h**2 for h=2..100)) / A_1
H = THD_percent / 100
```

The current configuration's maximum harmonic is 100 (5 kHz at 50 Hz).
Remove DC as the helper does, use its single-sided FFT amplitude convention,
and report the resulting value as **2nd-through-100th-harmonic THD**, not
unqualified full-spectrum or hardware THD. Use the same software/version and
parameters for baseline and candidate. The existing helper returns an error
when the fundamental is effectively zero; preserve that behavior as an
infeasible evaluation, not a low-THD win.

### Recommended constrained minimization

Use normalized, dimensionless quantities:

```
minimize       J = H + lambda_track * e1**2
subject to     e1 <= epsilon_track
               all gates are binary and satisfy the stated gate truth table
               no same-leg shoot-through in the sampled gate sequence
               every output sample is one of {-300,-200,-100,0,100,200,300} V
               switching count is within a predeclared budget, if one is adopted
```

The hard tracking bound prevents a candidate from obtaining a favorable THD
by suppressing the fundamental. The squared soft term ranks feasible
candidates with better tracking ahead of otherwise similar THD candidates.
Report `H` and `e1` separately even when using the scalar score.

Do not copy the existing `HybridObjective` default weights directly. It
currently adds THD in percent to an unspecified `tracking_error` value using
unit weights. Those units/scales are not guaranteed compatible, and
`EvaluationResult` has no explicit fundamental phasor/amplitude field. Its
output-level and boolean constraint penalties are generic and do not implement
the tracking bound above.

Choose `epsilon_track` from the project's accepted fundamental-voltage
tolerance before search. If there is no approved tolerance, report a
THD-versus-tracking Pareto curve and show a sensitivity analysis over
predeclared tolerances; do not present one arbitrary tolerance as a standard.
Choose `lambda_track` on the normalized terms using a predeclared trade-off:
for example, if the team decides that the maximum tolerated error
`epsilon_track` should cost the same as a meaningful THD change `delta_H`,
set `lambda_track = delta_H / epsilon_track**2`. State `delta_H` in THD
fraction units (one percentage point is `0.01`). This calibrates the soft
penalty transparently; it does not replace the hard constraint.

For infeasible points, prefer a proper feasibility mechanism over a score that
can accidentally beat feasible candidates. The current optimizer can return
infinity from a feasibility callback, but that callback must evaluate/cache
the candidate metrics consistently; the current objective's fixed `1e6`
penalty magnitudes are not justified for these normalized metrics. If a
finite penalty is used, prove it exceeds every feasible objective over the
declared domain and test that property.

## 5. Evaluation, steady state, and constraints

### One-cycle sufficiency for this ideal model

One cycle is sufficient for the current deterministic discrete-time model when
all of the following remain true:

- The reference is 50 Hz and the carrier is 1 kHz, so exactly 20 carrier
  periods occur in one 20 ms fundamental cycle.
- The sample rate remains 20 kHz with 400 samples per cycle and the phase
  offsets are constant during the cycle.
- The simulator applies its analytic periodic RL initial condition for the
  exact-ZOH recurrence rather than starting from zero current.

Under those conditions, the sampled ideal switching voltage is periodic over
the 400 samples, and the calculated current is in periodic steady state from
sample zero. A separate transient warm-up is unnecessary for the present
model. Verify cycle closure numerically for each candidate:

```
i_next = alpha*i[-1] + beta*v[-1]
abs(i_next - i[0]) <= a documented floating-point tolerance
```

If future changes introduce incommensurate carrier/fundamental frequencies,
time-varying controls, a different sampling ratio, startup transients, or
non-ideal dynamics, use sufficient warm-up cycles and evaluate an integer
number of steady-state fundamental cycles. Do not silently reuse the
one-cycle assumption in those cases.

### Consistent metrics and switching checks

For every candidate and baseline, record:

- Fundamental voltage amplitude, complex phasor tracking error, amplitude
  error, and phase error.
- 2nd-through-100th harmonic THD and all individual harmonic amplitudes.
- Output voltage levels, RMS voltage, waveform tracking RMSE, and periodic
  current closure residual.
- Total gate transitions per cycle, per-cell/per-switch transitions, and
  transition counting convention. Count wraparound transitions from sample
  399 to sample 0 so every cyclic edge is counted consistently.
- Gate validity and same-leg shoot-through checks at all simulated sample
  points.

The current simulator checks the allowed seven voltage levels, reconstructs
cell voltage from the gate equation, and checks same-leg simultaneous-on
patterns. It does not report a transition count or enforce a maximum
transition budget. Add a count and a budget only in a later implementation;
choose any hard limit from a project requirement or report the count without
claiming a thermal/switching-loss limit. This sampled check does not model
between-sample transients or dead time.

## 6. Baseline-vs-PSO experimental protocol

1. **Freeze the case set and operating conditions.** Store the exact `m`, `R`,
   `L`, and reference phase for every case. The baseline and all candidate
   evaluations must use these identical values.
2. **Evaluate baseline first.** Run `[0,60,120]` on each case and preserve raw
   waveforms, gates, states, metrics, and code/environment versions.
3. **Use a deterministic evaluator.** Given the same canonical phase pair and
   operating point, the simulator and metric pipeline must return identical
   values. Cache by canonical phases plus all operating conditions and
   evaluation settings.
4. **Predeclare PSO settings.** Record angular representation, swarm size,
   iteration/evaluation budget, coefficients, stopping criterion, feasibility
   rule, objective weights/tolerance, and all random seeds before runs.
5. **Repeat stochastic optimization.** Use at least 20 independent seeds for
   the initial study, while ensuring the baseline `[60,120]` is explicitly
   evaluated and available as a feasible incumbent. Report each run, not only
   the best seed.
6. **Report runtime and compute budget.** For each seed include wall-clock
   duration, number of unique simulator evaluations, cache hits, iterations,
   hardware/CPU, Python/library versions, and whether parallel execution was
   used. Compare the same evaluation budget across runs.
7. **Compare paired metrics.** For each operating point, report baseline and
   candidate `H`, `e1`, fundamental amplitude/phase, switching transitions,
   and constraints. Summarize paired change in THD, feasibility rate, median
   and interquartile range across seeds, plus the selected test-condition
   results.
8. **Separate selection from final evaluation.** Select phases using only the
   predeclared optimization/selection set. Evaluate the selected candidate on
   held-out operating conditions after selection. Do not retune on that
   held-out set.

A single pilot point can validate code and protocol but is not evidence of
general performance. No harmonic improvement is established by this design.

## 7. CNN role and limits

The trained model is `WaveformResidualCNN` in
[`models/cnn_model.py`](./models/cnn_model.py), constructed by
[`models/model_factory.py`](./models/model_factory.py). Its inputs are input
voltage, load current, and four operating-condition features; its output is a
400-point normalized **target/reference voltage waveform**. The inference
interface in [`app.py`](./app.py) preprocesses a selected dataset record,
loads the checkpoint, runs inference, denormalizes the target, and compares
that prediction to the dataset's reference waveform.

The CNN does **not** accept carrier offsets or gate sequences and does not
predict switching gates, the simulated candidate output voltage, or candidate
THD. Its current prediction task cannot serve as a candidate-dependent
surrogate objective for this phase-offset PSO. Its output is also not a
measured or simulated post-inverter waveform. Since the desired sinusoidal
reference is already known analytically from `m` and `theta`, the CNN is not
needed to define the first-stage tracking target.

A future CNN-assisted PSO would require a separate, explicitly defined
candidate-aware surrogate trained on simulator-generated phase/configuration
inputs and candidate electrical outcomes (waveforms or metrics). Before using
such a surrogate, validate it on held-out phase combinations and operating
conditions, report uncertainty and out-of-domain behavior, compare its
rankings against the exact simulator, and rerun every selected candidate
through the exact simulator. Keep the current checkpoint/task unchanged and
do not label its waveform regression as CNN control or optimized harmonic
reduction.

## 8. Required later implementation work

Before running an inverter-specific PSO:

- Add a candidate-aware evaluator that passes canonical carrier phase offsets
  to the simulator, computes phasor/THD metrics, and returns all raw outcomes.
  The current `simulate_inverter` signature does not accept carrier offsets;
  extend it in that later implementation while preserving `[0,60,120]` as the
  explicit default and verifying default-output regression tests.
- Extend the result/constraint contract with explicit fundamental target and
  measured phasors, normalized tracking error, transition counts, feasibility
  reasons, and evaluation details.
- Implement periodic angle handling and candidate canonicalization in the
  optimizer or a dedicated phase-aware wrapper; test it against phase and cell
  permutation equivalences.
- Add objective tests for low-fundamental rejection, tracking tolerance,
  THD units, penalty dominance, and feasibility behavior.
- Validate the candidate evaluator against the fixed `[0,60,120]` baseline
  and known dataset operating points before any search.
- Add no PSO calls to the current CNN app; keep analysis and experiment
  reporting separate until their scope is intentionally designed.

The existing [`pso_optimizer.py`](./optimization/pso_optimizer.py) is a
generic bounded PSO and can remain useful after these additions. It currently
does not handle circular variables, inverter-specific candidate encoding,
the required phasor tracking constraint, a physically justified switching
budget, or detailed per-candidate results. The existing
[`HybridObjective`](./optimization/objective_function.py) is likewise a
generic weighted scorer and should not be treated as the final electrical
objective without the metric/interface changes above.

## 9. Limitations and claims

The simulator is an ideal sampled switching model with an ideal 100 V source
per cell and an ideal series RL recurrence. It does not model semiconductor
voltage drops, conduction/switching losses, dead time, parasitic elements,
device thermal limits, measurement noise, gate-driver behavior, or hardware
safety. The stated gate truth table's no-shoot-through check is a logic-level
sample check, not a dead-time guarantee.

No Simulink comparison, hardware validation, or physical harmonic-reduction
demonstration is available. Report all future simulation results as
**ideal-model results under the stated assumptions**. Do not describe them as
hardware-validated, Simulink-validated, or proven physical harmonic reduction.

## Recommendation

The safest first experiment is a **single-condition, fixed-operating-point
software pilot** against the exact `[0°,60°,120°]` baseline, using the
periodic-angle-aware candidate representation, an explicitly predeclared
fundamental-phasor tolerance, and a constrained THD objective. Run repeated
seeds, preserve the baseline as an incumbent, and report paired metrics and
runtime. Treat the result only as an ideal simulator-method check; require
held-out operating-condition evaluation before making any broader claim.
