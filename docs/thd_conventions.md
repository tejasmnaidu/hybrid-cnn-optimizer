# THD Conventions

## Scope

This project has three related but not interchangeable THD paths:

1. `harmonic_analysis.py`: single-sided FFT harmonic amplitudes.
2. `thd_analysis.py`: sampled waveform THD using those FFT amplitudes.
3. `event_based_thd.py`: analytic Fourier coefficients and Parseval/interval
   energy for the ideal seven-level inverter voltage waveform.

These methods must keep different labels in reports. A difference between
sampled and event-based values is a methodological comparison result, not
automatic evidence of a defect.

## 1. FFT harmonic spectrum

**Implementation:** `evaluation/harmonic_analysis.py`,
`harmonic_spectrum`.

- Fundamental frequency: configurable, 50 Hz by default.
- Number of cycles: determined by the caller; a fundamental FFT bin is required.
- Sampling/windowing: real FFT over `N` points with a rectangular-window FFT
  (`np.fft.rfft`); the waveform mean is removed before FFT. No Hann/window taper
  is applied and no explicit multi-cycle averaging is performed.
- Harmonic cutoff: configurable `max_harmonic`, default 100; also limited by Nyquist
  and available FFT bins.
- DC component treatment: removed as `waveform - mean(waveform)` before FFT.
- Fundamental extraction: single-sided amplitude at the FFT bin closest to the
  configured 50 Hz bin. At 20 kHz and 400 samples/cycle, the bin spacing is 50 Hz.
- Voltage/current: the function is waveform-generic. `calculate_thd` does not
  infer whether the array is voltage or current; callers must label it explicitly.

For 400 real samples/cycle, bin 1 is 50 Hz. Higher bins correspond to 100 Hz,
150 Hz, ... through the configured harmonic limit.

## 2. Sampled THD

**Implementation:** `evaluation/thd_analysis.py`, `calculate_thd`.

- Fundamental frequency: 50 Hz default.
- Number of cycles: caller controls array length and must provide an integer
  number of fundamental cycles aligned to the FFT bins.
- Sampling/windowing: defaults to 20 kHz and uses the same rectangular FFT
  approach described above. The input waveform is mean-centered.
- Harmonic cutoff: harmonics 2 through `max_harmonic`, default 100.
- DC component treatment: excluded by subtracting the waveform mean.
- Fundamental extraction: single-sided peak amplitude at the fundamental FFT bin.
- Voltage THD:

  `THD_V = 100 * sqrt(sum(V_k^2, k=2..100)) / V_1`

- Current THD uses the same formula with current amplitudes:

  `THD_I = 100 * sqrt(sum(I_k^2, k=2..100)) / I_1`

The method is one uniform-time sample estimate. It is sensitive to sample
resolution, integer-cycle alignment, and whether every switching interval is
adequately represented.

## 3. Event-based ideal inverter THD

**Implementation:** `evaluation/event_based_thd.py`,
`evaluate_event_based_thd`.

- Fundamental frequency: exactly 50 Hz.
- Number of cycles: exactly one continuous fundamental cycle, `T = 1/50 s`.
- Sampling/windowing: no window and no uniform-time waveform estimate for the
  primary metric. Carrier/reference crossings are bracketed into piecewise-
  constant voltage intervals and integrated analytically.
- Harmonic cutoff:
  - `thd_percent` sums Fourier orders 2 through the configured `max_harmonic`,
    default 100.
  - `total_thd_percent` uses Parseval interval energy with mean removed and
    excludes the fundamental component, while retaining all AC energy beyond
    the fundamental. It is therefore not a deliberately forced equivalent to
    the 2–100 sampled metric.
- DC component treatment: mean voltage is computed over the one-cycle interval
  and removed from the AC/Parseval residual.
- Fundamental extraction: peak Fourier coefficient
  `C1 = (2/T) integral(v(t)e^(-j2πt/T)dt)`. The fundamental RMS value is
  `|C1|/sqrt(2)`.
- Voltage vs current: the event evaluator evaluates ideal inverter output
  voltage. It does not evaluate load current. Current requires the separate
  RL recurrence and sampled FFT from `calculate_thd(simulation.load_current)`.

The event evaluator also stores `sampled_thd_percent`, a compatibility metric
calculated by uniform 20 kHz sampling over the one-cycle simulator waveform.
It is not the primary `total_thd_percent` objective.

## Method comparison

For harmonics 2–100, the sampled voltage THD and event `thd_percent` are
mathematically comparable as peak-amplitude sums over the same harmonic range.
They can differ because the sampled implementation uses uniform 20 kHz samples
and a rectangular DFT, while the event method integrates exact switching
intervals. The differences should be reported, not forced to zero.

`total_thd_percent` is a full-spectrum Parseval residual THD. It can include
energy above harmonic 100 and is therefore not guaranteed to equal either
2–100 harmonic sum. At high/fully resolved harmonic content, the definitions
can converge, but they remain distinct quantities.

## Voltage and current behavior

- Voltage THD changes with modulation and carrier phase shifts; ideal
  switching output does not use R or L.
- Load-current THD depends on R and L through the periodic RL recurrence.
- R and L must not be claimed to change ideal voltage THD.

## Phase 4 optimization objective

The validated carrier-phase objective should be event-based
`total_thd_percent` or, only with an explicitly changed requirement,
an independently recomputed metric over an equally declared harmonic range.
It was chosen for Phase 4 because candidates are evaluated analytically over
continuous switching intervals, with baseline and candidate conditions held
identical. `total_thd_percent` is separately identified from sampled voltage
THD and must not be substituted silently.
