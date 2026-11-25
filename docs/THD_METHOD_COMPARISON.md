# THD method comparison and carrier-phase sweep

## Scope and reproducibility

All measurements here use the existing ideal inverter and THD functions; no
simulator, CNN, dataset, checkpoint, optimizer, THD implementation, or app
behavior was changed for this analysis.

The fixed operating point is:

| Input | Value |
|---|---:|
| Modulation index | 0.75 |
| DC voltage per cell | 100 V |
| Load resistance | 30 Ω |
| Load inductance | 0.03 H |
| Reference phase | 0° |
| Carrier offsets (baseline) | φ₁=0°, φ₂=60°, φ₃=120° |
| Fundamental / sample rate | 50 Hz / 20 kHz |
| Samples per fundamental cycle | 400 |

The sweep file [`../results/carrier_shift_sweep.csv`](../results/carrier_shift_sweep.csv)
contains all 1,296 combinations on a 10° grid: φ₂ and φ₃ independently take
values 0°, 10°, …, 350°, with φ₁ fixed at 0°. Angles are periodic modulo 360°.
The heatmap is [`../results/carrier_shift_sweep_heatmap.png`](../results/carrier_shift_sweep_heatmap.png).

## What each THD value measures

At 20 kHz, Nyquist is 10 kHz, or harmonic order 200 for a 50 Hz fundamental.
The 400-point, one-cycle FFT has 50 Hz bin spacing.

| Reported metric | Existing function and basis | Harmonic range |
|---|---|---|
| **Sampled FFT voltage THD: 10.871481318%** | `calculate_thd(simulation.output_voltage, sampling_frequency_hz=20000, fundamental_frequency_hz=50, max_harmonic=100)`; 400 uniformly sampled simulator points, mean removed, rectangular real FFT | Orders 2–100 (100 Hz–5 kHz). The FFT can contain aliases of continuous-waveform harmonics above Nyquist. |
| **Event-integrated voltage THD: 0.000000102%** | `evaluate_event_based_thd(..., max_harmonic=100).thd_percent`; exact Fourier coefficients integrated over switching-event intervals | Exact continuous-time orders 2–100 (100 Hz–5 kHz); no sampling or window. |
| **Event Parseval total voltage THD: 24.723479151%** | `evaluate_event_based_thd(...).total_thd_percent`; exact interval mean-square energy with the mean and fundamental removed | All non-fundamental AC harmonic energy, orders 2 through infinity; no finite harmonic cutoff. |

The Parseval definition is

`100 * sqrt(AC mean-square voltage - fundamental RMS voltage²) / fundamental RMS voltage`.

For exact Fourier peak amplitudes `A_n`, the energy in a band is
`sum(A_n² / 2)`. The event evaluator returns the continuous interval energy,
so the energy above order 200 is the Parseval residual minus the exactly
integrated order-2-through-200 energy. The THD contributions below are RMS
components normalized to the exact event fundamental RMS; they combine in
quadrature, not by adding percentages.

## Why 10.871% sampled and 24.723% Parseval differ

The event evaluator was called with `max_harmonic=200` to expose exact
coefficients through Nyquist and its Parseval interval energy for the remaining
tail. Results at the default carrier offsets:

| Continuous-time band / quantity | Harmonic range | Voltage energy | THD RMS component |
|---|---:|---:|---:|
| Exact event energy included in the sampled FFT's nominal range | 2–100 (100 Hz–5 kHz) | `2.62771247 × 10⁻¹⁴ V²` | `1.01887616 × 10⁻⁷%` |
| Exact event energy below Nyquist but above the sampled cutoff | 101–200 (5.05–10 kHz) | `859.88931193 V²` | `18.43120609%` |
| Exact event energy above Nyquist | 201 and higher (>10 kHz) | `687.33831706 V²` | `16.47850307%` |
| Parseval non-fundamental energy, orders 2 through infinity | 2 and higher | `1547.22762900 V²` | `24.72347915%` |

Orders 2–200 account for **55.5761%** of the Parseval non-fundamental energy;
orders above 200 account for **44.4239%**. The components `18.43120609%` and
`16.47850307%` combine to the Parseval total of `24.72347915%` in RMS
quadrature.

The sampled FFT's nominal orders 2–100 contain `304.64623414 V²` of apparent
harmonic energy and report `10.871481318%`, even though the exact continuous
event energy in orders 2–100 is effectively zero. This demonstrates **strong
aliasing**: the ideal switched waveform has components above the 10 kHz
Nyquist limit, and uniform sampling folds some of them into lower FFT bins.
There is no anti-alias filter in these existing functions. The FFT's measured
fundamental peak is also `227.05139835 V`, versus the exact event fundamental
of `225 V`, consistent with sampling/alias effects.

**Both aliasing and band-limiting matter, but in different ways.** The sampled
metric's configured cutoff excludes true orders 101 and above, while its
orders-2–100 FFT bins are contaminated by aliases (the measured `10.871481318%`
versus exact event orders-2–100 `0.000000102%`). The Parseval metric retains
the true energy of orders 101 and above, yielding `24.723479151%`. Therefore
the sampled value cannot be interpreted as a simple band-limited version of
the Parseval value, nor can the percentage-point difference be assigned to
one band by subtraction. The reported sampled-to-Parseval difference is
`13.851997833` percentage points.

The sweep's sampled column uses the event evaluator's `sampled_thd_percent`
with `max_harmonic=100`, which is the same 20 kHz, 400-point H2–100 sampled
convention. When the evaluator is instead called with `max_harmonic=200`, its
sampled metric covers H2–200; that is a different metric and is not the CSV's
sampled H2–100 column.

## Carrier-phase sweep results

The existing `evaluate_event_based_thd` was run for every grid point at the
fixed operating point. The CSV labels all three THD definitions explicitly:

1. `parseval_total_voltage_thd_percent`: event interval / Parseval, orders
   2 through infinity.
2. `event_analytic_voltage_thd_h2_100_percent`: exact event Fourier amplitudes,
   orders 2–100.
3. `sampled_fft_voltage_thd_h2_100_percent`: sampled 20 kHz FFT, orders 2–100.

| Metric | Minimum on 10° grid | Default (φ₂=60°, φ₃=120°) | Is default a grid minimum? |
|---|---:|---:|---|
| Parseval total voltage THD, orders 2–∞ | 24.723479151% | 24.723479151% | Yes, tied |
| Event-integrated voltage THD, orders 2–100 | 0.000000101888% | 0.000000101888% | Yes, tied |
| Sampled FFT voltage THD, orders 2–100 | 10.871481318% | 10.871481318% | Yes, tied |

| Metric | Maximum on grid | Spread (max − min) | Grid points tied at minimum |
|---|---:|---:|---:|
| Parseval total voltage THD, orders 2–∞ | 83.630462979% | 58.906983828 percentage points | 8 |
| Event-integrated voltage THD, orders 2–100 | 74.9941841984% | 74.994184096512 percentage points | 8 |
| Sampled FFT voltage THD, orders 2–100 | 78.5080067528% | 67.636525434700 percentage points | 8 |

For all three metrics, the eight tied coordinates are
`(60°,120°), (60°,300°), (120°,60°), (120°,240°), (240°,120°),
(240°,300°), (300°,60°), (300°,240°)`. They are symmetry-related phase
permutations/reflections, not a flat region: only 8 of 1,296 points attain the
minimum, and the full-grid spreads are shown above. For event-integrated
H2–100, the default is only `1.5888 × 10⁻¹⁴` percentage points above the
smallest stored value; it is a tie at the reported precision. Thus, the
default is a grid minimum under all three methods. This establishes a minimum
only on the specified 10° grid; it does not prove a global minimum for
continuous phase angles or another operating point.

## Fine carrier-phase sweep near the default

The fine sweep evaluates all 1,681 pairs on φ₂=50°…70° and φ₃=110°…130°,
both in 0.5° steps, with φ₁=0° and the other operating conditions fixed at
the default point above. The full values are in
[`../results/carrier_shift_fine_sweep.csv`](../results/carrier_shift_fine_sweep.csv);
the three panels in
[`../results/carrier_shift_fine_sweep_heatmap.png`](../results/carrier_shift_fine_sweep_heatmap.png)
show event Parseval total voltage THD (orders 2–∞), exact event H2–100
voltage THD, and sampled 20 kHz FFT H2–100 voltage THD, respectively.

| Method | Minimum THD | Minimizing φ₂, φ₃ (degrees) | Default value at (60°, 120°) | Minimum − default |
|---|---:|---|---:|---:|
| Event Parseval total voltage THD, orders 2–∞ | 24.6779522784% | (58.5°, 120°), (60°, 121.5°) | 24.7234791510% | −0.0455268727 pp |
| Exact event voltage THD, orders 2–100 | 0.000000101887616% | (60°, 120°) | 0.000000101887616% | 0 pp |
| Sampled FFT voltage THD, orders 2–100 | 9.70998708507% | 12 tied pairs: φ₂=56.5°–57.5° at φ₃=116.5° or 117°; φ₂=63° or 63.5° at φ₃=122.5°–123.5° (listed at 0.5° grid points) | 10.8714813181% | −1.1614942330 pp |

Thus, on this fine local grid, the Parseval and sampled H2–100 objectives each
have points below the default (by 0.0455268727 and 1.1614942330 percentage
points, respectively); the exact event H2–100 objective ties the default.
These are grid minima only and do not establish improvement over the entire
continuous phase space.

## Coarse phase sweeps at additional modulation indices

The existing 10° grid (φ₂, φ₃ independently from 0° through 350°, φ₁=0°)
was reevaluated at m=0.55, 0.65, 0.85, and 0.95. Each grid contains 1,296
points. The summary table is
[`../results/carrier_shift_vs_modulation.csv`](../results/carrier_shift_vs_modulation.csv);
its three metric columns are event Parseval total voltage THD (orders 2–∞),
exact event voltage THD (orders 2–100), and sampled 20 kHz FFT voltage THD
(orders 2–100).

| Modulation | Parseval min, orders 2–∞ (%) | Event min, orders 2–100 (%) | Sampled min, orders 2–100 (%) | All three minima's configurations |
|---:|---:|---:|---:|---|
| 0.55 | 37.313788266 | 0.000000000269496 | 13.877591146 | Same eight default-symmetric configurations |
| 0.65 | 28.781282667 | 0.000000006727595 | 12.080141346 | Same eight default-symmetric configurations |
| 0.85 | 23.687643127 | 0.000001054736188 | 8.779335221 | Same eight default-symmetric configurations |
| 0.95 | 20.689101978 | 0.000008080868143 | 7.517833584 | Same eight default-symmetric configurations |

For every method at every listed modulation, the eight minimizing pairs are
`(60°,120°), (60°,300°), (120°,60°), (120°,240°), (240°,120°),
(240°,300°), (300°,60°), (300°,240°)`. Thus all grid minima remain at the
default `(60°,120°)` or its symmetric equivalents. No coarse-grid minimum
meaningfully beats the default: the largest apparent reduction is below
`1 × 10⁻¹³` percentage points and is floating-point variation in the event
calculations, not a measurable THD improvement.

## Equal-budget multi-optimizer comparison at the default point

PSO, POA, PFO, and Hybrid POA-PFO were rerun at m=0.75, 100 V/cell,
30 Ω, 0.03 H, and 0° reference phase. The objective and reported final metric
are **event-based Parseval total voltage THD, harmonic orders 2 through
infinity**. All methods used seeds 20261124–20261133, the same two carrier
phase variables in [0°, 360°), the baseline (60°, 120°), and the same hard
fundamental-phasor tracking feasibility tolerance of 0.05%.

The common budget was exactly 32 distinct candidate simulations per
method/seed, including the baseline. Each optimizer was stopped before
simulating a 33rd distinct candidate; no substitute objective values were
returned for over-budget candidates. All 40 searches reached the cap. The
selected candidate was then independently simulated and re-evaluated with the
event THD evaluator outside the search budget; all selected candidates were
feasible and all independent re-evaluations matched. Duplicate objective
requests are not additional simulator evaluations and are recorded separately.
The per-seed data are in
[`../results/optimizer_equal_budget_comparison.csv`](../results/optimizer_equal_budget_comparison.csv).

| Method | Mean final event Parseval THD (%) | Sample std, ddof=1 (%) | Minimum (%) | Maximum (%) | Mean reduction vs (60°,120°) (pp) | Best reduction (pp) | Seeds improved / 10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| PSO | 24.723155574266 | 0.001023239597 | 24.720243383318 | 24.723479151038 | 0.000323576772 | 0.003235767720 | 1 |
| POA | 24.723352272252 | 0.000401225950 | 24.722210363179 | 24.723479151038 | 0.000126878786 | 0.001268787859 | 1 |
| PFO | 24.723479151038 | 0.000000000000 | 24.723479151038 | 24.723479151038 | 0.000000000000 | 0.000000000000 | 0 |
| Hybrid POA-PFO | 24.723352272252 | 0.000401225950 | 24.722210363179 | 24.723479151038 | 0.000126878786 | 0.001268787859 | 1 |

On this shared 32-candidate budget, PSO, POA, and the hybrid each found a
candidate below the default in one of ten seeds; PFO returned the default for
all ten. The best observed result was PSO seed 20261129 at
(φ₂=61.551557°, φ₃=122.297245°), lowering event Parseval THD by
0.003235767720 percentage points. POA and the hybrid's best was
24.722210363179% (0.001268787859 percentage points lower than baseline).
These are small, point-specific outcomes, not proof of general optimizer
superiority or of a continuous global optimum. All four methods were retained
in the comparison because the equal cap was enforceable without modifying
their implementations.

## Live-workspace screenshot

[`verification/live_workspace_thd_explanation.png`](verification/live_workspace_thd_explanation.png)
was captured before the live-workspace clarification box was removed. It is a
historical screenshot and does not represent the current dashboard.

## Existing live-simulation tests reviewed

The two added tests in `tests/test_live_simulation.py` call
`run_live_simulation`, which calls the real `simulate_inverter` and event
harmonic evaluator. They do not mock or patch the simulator. The test CNN is a
small deterministic `EchoVoltageModel` fixture; it is not a simulator mock.

- `test_identical_live_inputs_reproduce_identical_simulation`
  - Runs identical conditions twice.
  - Asserts exact equality of output-voltage arrays.
  - Asserts exact equality of load-current arrays.
  - Asserts equality of event Parseval total voltage THD.
- `test_modulation_and_carrier_changes_change_simulated_waveform`
  - Compares the baseline against modulation 0.85 and carrier shifts
    (0°, 35°, 155°).
  - Asserts output voltage differs for each changed input.
  - Asserts event Parseval total voltage THD differs for each changed input.
