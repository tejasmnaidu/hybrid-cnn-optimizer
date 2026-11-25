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
