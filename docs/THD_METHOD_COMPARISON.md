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
| Exact event coefficient sum in the sampled FFT's nominal range (numerical residual; not resolved as physical energy) | 2–100 (100 Hz–5 kHz) | `2.62771247 × 10⁻¹⁴ V²` | nominal ratio `1.69833606 × 10⁻¹⁵%`, below double-precision resolution of the total |
| Exact event energy below Nyquist but above the sampled cutoff | 101–200 (5.05–10 kHz) | `859.88931193 V²` | `18.43120609%` |
| Exact event energy above Nyquist | 201 and higher (>10 kHz) | `687.33831706 V²` | `16.47850307%` |
| Parseval non-fundamental energy, orders 2 through infinity | 2 and higher | `1547.22762900 V²` | `24.72347915%` |

Orders 2–200 account for **55.5761%** of the Parseval non-fundamental energy;
orders above 200 account for **44.4239%**. The components `18.43120609%` and
`16.47850307%` combine to the Parseval total of `24.72347915%` in RMS
quadrature.

The sampled FFT's nominal orders 2–100 contain `304.64623414 V²` of apparent
harmonic energy and report `10.871481318%`. The direct event coefficient
energy in those orders is only `2.62771247 × 10⁻¹⁴ V²`, but that residual is
below float64 resolution at the total-energy scale and is not a resolved
physical measurement. The event-reported H2–100 THD of `0.000000102%` should
therefore also be treated as a numerical residual, not as proof that the
physical low-order energy is zero. The measured sampling-rate dependence
supports **strong aliasing**: the ideal switched waveform has components
above the 10 kHz Nyquist limit, and uniform sampling folds some of them into
lower FFT bins. There is no anti-alias filter in these existing functions.
The FFT's measured fundamental peak is also `227.05139835 V`, versus the
exact event fundamental of `225 V`, consistent with sampling/alias effects.

**Both aliasing and band-limiting matter, but in different ways.** The sampled
metric's configured cutoff excludes true orders 101 and above, while its
orders-2–100 FFT bins are contaminated by aliases (the measured `10.871481318%`
versus the event evaluator's unresolved numerical-residual output
`0.000000102%`). The Parseval metric retains
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
default is tied for the minimum on this 10° grid under all three methods.
This statement applies only to the 10° grid: the 0.5° local sweep below finds
lower Parseval and sampled H2–100 values near the default. Neither grid proves
a global minimum over continuous phase angles or another operating point.

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

### Fundamental tracking at the fine-sweep minima

The event evaluator's fundamental phasor was compared with its commanded
reference fundamental (225 V peak and 0° at m=0.75, reference phase 0°).
“Amplitude error” below is actual minus target fundamental peak amplitude;
phase error is the wrapped actual-minus-target phase. Feasibility uses the
optimizer's 0.05% phasor tracking-error tolerance.

| Fine-sweep minimum set | Number of points | Target amplitude | Maximum absolute amplitude error | Maximum absolute phase error | Maximum phasor tracking error | Feasible points |
|---|---:|---:|---:|---:|---:|---:|
| Parseval minimum: (58.5°,120°), (60°,121.5°) | 2 | 225 V peak | 1.71 × 10⁻¹³ V | 5.09 × 10⁻¹⁴° | 9.19 × 10⁻¹⁴% | 2/2 |
| Exact event H2–100 minimum: (60°,120°) | 1 | 225 V peak | 8.53 × 10⁻¹⁴ V | 0° | 4.28 × 10⁻¹⁴% | 1/1 |
| Sampled H2–100 minimum: 12 tied pairs listed above | 12 | 225 V peak | 2.28 × 10⁻¹³ V | 5.09 × 10⁻¹⁴° | 1.24 × 10⁻¹³% | 12/12 |

Thus every fine-sweep minimizer satisfies the optimizer feasibility rule by a
wide margin. Carrier offsets in these candidates do not materially alter the
fundamental phasor; the THD changes are in harmonic content, not a tracking
tradeoff. Per-candidate amplitude, phase, phasor, feasibility, and THD values
are in
[`../results/carrier_shift_fine_minimizer_tracking.csv`](../results/carrier_shift_fine_minimizer_tracking.csv).

## Fine local sweeps by modulation index

At m=0.55, 0.65, 0.85, and 0.95, a fresh 41 × 41 sweep used φ₂=50°–70°,
φ₃=110°–130°, and 0.5° increments, with φ₁=0°. The comparison is against
(60°,120°) at the same modulation. Parseval is event total voltage THD
(orders 2–∞); event H2–100 is exact event voltage THD; sampled H2–100 is the
20 kHz FFT metric. Improvements are baseline minus minimum, in percentage
points. Ties are reported to 1e-12 percentage-point tolerance; the CSV lists
all tied coordinates.

| Modulation | Parseval minimum and minimizer(s); improvement | Exact event H2–100 minimum; improvement | Sampled 20 kHz H2–100 minimum and minimizer(s); improvement |
|---:|---|---|---|
| 0.55 | 37.2204859303% at (60°,118°), (62°,120°); 0.0933023354 pp | 0.000000000269501% at (60°,120°); 0 pp | 12.2156848972% at 8 tied points; 1.6619062489 pp |
| 0.65 | 28.7790228798% at (63°,117°); 0.0022597875 pp | 0.000000006727596% at (60°,120°); 0 pp | 10.1413136237% at 4 tied points; 1.9388277219 pp |
| 0.85 | 23.5842680915% at (58°,121.5°), (58.5°,122°); 0.1033750353 pp | 0.000001054736196% at (60°,120°); 0 pp | 8.3590753165% at 30 tied points; 0.4202599047 pp |
| 0.95 | 20.6066895992% at (61.5°,118.5°); 0.0824123783 pp | 0.000008080868143% at (60°,120°); 0 pp | 7.0332985753% at (59°,121°); 0.4845350091 pp |

The complete 12-row result, including the default value, exact improvement,
all ties, and method labels is
[`../results/carrier_shift_fine_vs_modulation.csv`](../results/carrier_shift_fine_vs_modulation.csv).
Across all five modulation indices checked on local fine grids, the exact
event H2–100 minimum remains at the default; Parseval total and sampled
H2–100 have local grid points below it. The amounts depend strongly on metric.

## Aliasing check at the sampled-metric minima

The sampled H2–100 fine-sweep minimum at m=0.75 is 9.7099870851% at 12 tied
phase pairs. At those same phases, exact event H2–100 THD ranges from
3.0284698660% to 3.8178596786%, while event Parseval total THD (orders 2–∞)
ranges from 24.8445923340% to 24.8895684317%. The exact event H2–100 value at
the default is only 0.0000001018876%; the Parseval total at the default is
24.7234791510%. Thus the sample-metric minimizers are worse than the default
under both exact event H2–100 (by 3.0284697641–3.8178595768 pp) and Parseval
total (by 0.1211131830–0.1660892807 pp).

The existing evaluator supports a caller-selected integer-cycle sampling rate
through `sampled_frequency_hz`; it does not expose a longer multi-cycle window.
At 100 kHz, 200 kHz, 1 MHz and 2 MHz, the evaluator computes one 50 Hz cycle
with 2,000, 4,000, 20,000 and 40,000 samples respectively, still reporting
sampled H2–100. The following values compare the default with the 12
20 kHz-minimizing phases:

| Sampling rate | Default (60°,120°), sampled H2–100 | 12 sampled-minimum phases: min–max H2–100 |
|---:|---:|---:|
| 20 kHz | 10.8714813181% | 9.7099870851–9.7099870851% |
| 100 kHz | 1.9660621125% | 3.6477390906–4.2111496016% |
| 200 kHz | 0.8901469660% | 3.2830659164–3.9065742448% |
| 1 MHz | 0.2163033915% | 3.0521290422–3.8077364447% |
| 2 MHz | 0.1075647439% | 3.0359585772–3.8151446656% |

This aliasing conclusion is based on the explicit **20, 100, 200, 1,000,
and 2,000 kHz sampling-rate sweep** above, not on attributing the 20 kHz THD
difference to aliasing by assumption. As the sampling rate rises, the default
H2–100 sampled value changes from 10.8714813181% to 1.9660621125%,
0.8901469660%, 0.2163033915%, and 0.1075647439%; for the 12 sampled-minimum
phases, the corresponding ranges are 9.7099870851%, 3.6477390906–4.2111496016%,
3.2830659164–3.9065742448%, 3.0521290422–3.8077364447%, and
3.0359585772–3.8151446656%. This rate dependence supports alias
contamination as the explanation for the low-rate reversal. However, the
evaluator only supports a **one-cycle window**; leakage and aliasing were not
separately isolated in this experiment. Higher rates also retain finite
one-cycle FFT and sample-grid effects.

At 20 kHz, the minimum appears 1.1614942330 pp better than the default.
At 1 MHz, those same 12 candidates are instead 2.8358256507–3.5914330532 pp
worse than the default sampled result; their mean is 3.2290641529 pp worse.
Their 20 kHz values exceed their 1 MHz values by 5.9022506403–6.6578580429
pp (mean 6.2646195407 pp). At 2 MHz they remain 2.9283938333–3.7075799217
pp worse than the default. The apparent 20 kHz gain is therefore not a
sampling-rate-stable improvement: it is dominated by aliasing in the sampled
H2–100 FFT, and higher-rate samples converge toward the exact event H2–100
values. This is a metric/sampling artifact, not a verified reduction in the
event-based objective. A higher sampling rate reduces, but cannot eliminate,
aliasing from the ideal switched waveform's unbounded harmonic spectrum.

All per-phase results are in
[`../results/carrier_shift_sampled_alias_comparison.csv`](../results/carrier_shift_sampled_alias_comparison.csv).

## Independent Parseval energy check from a dense waveform FFT

As an independent numerical check at the default point, the event evaluator's
exact interval start/end times and voltage levels were sampled at uniform
midpoint times for one complete 50 Hz cycle. A one-sided real FFT was
normalized by the number of samples; harmonic-bin energy was summed by
Parseval's theorem. The 40 MHz run used 800,000 samples in that cycle, with
Nyquist at harmonic order 400,000. The event evaluator's exact comparison
values are shown alongside this independent finite-rate calculation:

| Calculation | Orders 2–100 energy | Orders 101 and above energy / coverage | Low-band fraction |
|---|---:|---:|---:|
| Direct sum of exact event coefficient energies | 2.62771247044 × 10⁻¹⁴ V² | — | nominal ratio 1.69833606 × 10⁻¹⁵% |
| Parseval total minus reconstructed orders 101–∞ energy | 0 V² at float64 precision | 1547.2276289961221 V² (orders 101–∞) | 0% after rounded subtraction |
| Independent waveform FFT at 40 MHz | 7.06272308 × 10⁻⁵ V² | 1547.3747982 V² (orders 101–400,000) | 4.56432583 × 10⁻⁶% |

The low-order energy calculation was independently checked two ways using
the event coefficients through order 3,000. First, direct summation of
`A_n²/2` for n=2…100 gives `2.6277124704402311 × 10⁻¹⁴ V²`. Second, summing
orders 101…3,000 and closing the remaining tail with the exact Parseval
residual gives an orders-101–∞ energy of `1547.2276289961221 V²`; subtracting
that from the total `1547.2276289961221 V²` gives `0 V²` at float64 precision.
The difference between subtraction-based and direct low-band results is
`−2.6277124704402311 × 10⁻¹⁴ V²`.

This subtraction is below floating-point resolution at the total-energy
scale: float64 machine epsilon times the total is
`3.4355354760953685 × 10⁻¹³ V²`, and one ULP at the total is
`2.2737367544323206 × 10⁻¹³ V²`. The direct coefficient sum is only 0.116
ULP of the total. Thus `2.6 × 10⁻¹⁴ V²` is not distinguishable from rounding
when represented as a residual of two approximately `1547 V²` energies.
The direct sum is a numerical residual from the coefficient calculations,
not a resolved physical low-order energy measurement. **Complete low-order
cancellation is unresolved**, rather than established by the `1.7 × 10⁻¹⁵%`
ratio.

The independent waveform FFT agrees qualitatively that almost all resolved
distortion energy is above order 100: at 40 MHz it assigns 99.99999544% of
resolved residual energy to orders 101–400,000. Its total residual energy
(1547.37486883 V²) is 0.14723983 V², about 0.0095%, above the exact event
residual. It is **not quantitative agreement** for the minute low-band
energy: the sampled FFT's `7.0627 × 10⁻⁵ V²` is about 2.69 billion times the
direct event coefficient sum. The finite sample grid cannot resolve that
residual and its discretization/spectral contamination dominates it. Across
the 1, 2, 10, 20 and 40 MHz checks, independent orders-2–100 energy was
0.08884556, 0.02170586, 0.00158381, 0.00020587 and 0.00007063 V²
respectively, trending downward but remaining above the event coefficient
residual.

For these five rates, a least-squares fit of `log(E₂₋₁₀₀)` against
`log(sampling rate)` gives exponent **−1.9396** (`R²=0.99288` in log space).
The measured values therefore follow an approximate power-law trend over
this limited range, close to inverse-square scaling, but the five-point fit
does not establish an asymptotic law or prove that the residual converges to
zero.

The exact event coefficients through order 3,000 also show that the lowest
harmonic with peak amplitude above **0.01% of the fundamental** is order
**107**, at **5,350 Hz**: its peak amplitude is 0.05464653 V against a
0.0225 V threshold (fundamental peak 225 V). The exact event energy
calculation, dense-FFT calculation, and convergence rows are preserved in
[`../results/parseval_energy_independent_check.csv`](../results/parseval_energy_independent_check.csv).

## Parseval distortion energy below and above order 100

At the default point, total Parseval non-fundamental energy is approximately
1547.227628996 V². The direct exact coefficient sum in orders 2–100 is
2.62771247044 × 10⁻¹⁴ V², but this is below float64 rounding resolution at the
total-energy scale (epsilon × total is 3.43553547610 × 10⁻¹³ V²). Therefore
the earlier nominal fraction of 1.7 × 10⁻¹⁵% must not be interpreted as a
physical measurement. The float64 resolution scale is about
3.44 × 10⁻¹³ V² (about 2.22 × 10⁻¹⁴% of total): contributions below this
scale cannot be resolved by subtracting the low-band tail from the total.
This is a numerical-resolution threshold, not a physical upper bound.
Nearly all resolved distortion energy is above order 100. Whether orders
2–100 cancel completely or contain energy below the resolution threshold is
unresolved. This energy assessment is for the event-integrated Parseval
measure, not aliased sampled FFT bins.

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

## Is sampled 20 kHz THD an optimization objective?

No existing dashboard optimizer uses sampled 20 kHz voltage THD as its search
objective. The live phase optimizer routes candidates through
`PhaseOptimizationProblem.evaluate`, which reads `result.total_thd_percent`
from the event-based evaluator (Parseval total voltage THD, orders 2–∞).
The dashboard switching advisor uses event Parseval voltage THD when
“voltage” is selected, or event-Fourier RL-derived current H2–100 when
“current” is selected. Its sampled voltage/current THD values are displayed
as secondary measurements after candidate evaluation, not optimized.
CNN-guided search ranks candidates by CNN waveform score and verifies finalists
using event-based THD; it does not optimize the sampled FFT metric.

The sampled 20 kHz THD is therefore not the current dashboard optimizer
objective. If used as a future optimization target, it would be unreliable
without changing the measurement convention: the 0.5° sweep's apparent
1.1614942330 pp gain at m=0.75 reverses under higher-rate sampling and against
the exact event metrics, as quantified above. This note documents the existing
behavior and does not change it.

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
