# Seven-level inverter simulation

`inverter_simulator.py` implements the PWM and RL equations supplied for this
project. The original MATLAB source file is not present in the project or
dataset archive, so this implementation follows the user's stated conventions;
it has not been compared line-by-line with MATLAB.

## Run

From the project directory:

```python
from multilevel import simulate_inverter

result = simulate_inverter(
    modulation_index=0.7,
    load_resistance_ohm=30.0,
    load_inductance_h=0.03,
    phase_angle_rad=0.0,
)
```

The inputs are a finite modulation index, positive load resistance and
inductance, and phase angle in radians. A cycle contains 400 samples at 20 kHz
for a 50 Hz reference. The three 100 V cells use 1 kHz triangular carriers
shifted by 0, 60, and 120 degrees using the supplied carrier formula.

## Outputs and gate convention

- `reference`: the input sinusoidal reference, shape `(400,)`.
- `carriers`: one carrier per cell, shape `(3, 400)`.
- `cell_states`: values -1, 0, or +1, shape `(3, 400)`.
- `switching_gates`: binary values, shape `(3, 4, 400)`, ordered by cell,
  switch `[S1, S2, S3, S4]`, then time. This matches the dataset's
  cell-major/switch-major flattened gate layout.
- `cell_voltages`: reconstructed by `100 * (S1*S4 - S2*S3)`, shape
  `(3, 400)`.
- `output_voltage`: sum of cell voltages, shape `(400,)`.
- `load_current` and `periodic_initial_current_a`: the periodic exact-ZOH RL
  recurrence specified for this dataset.
- `time_s`: sample times from 0 through 399/20,000 seconds; the repeated cycle
  endpoint is excluded.

The gate truth table is implemented exactly as specified. Same-leg complementary
switch pairs are checked for simultaneous on-state.

## Dataset comparison and scope

The supplied dataset was compared to the implementation conventions:

- Generated cell states match all 10,000 stored records.
- Generated gate arrays match all 10,000 stored records when reshaped in
  `(cell, switch, time)` order.
- Gate-derived output voltage matches `Input_Voltage.csv` exactly.
- The simulator's periodic RL recurrence matches `Load_Current.csv` within
  floating-point precision (full-dataset RMSE about `2.4e-14 A`, maximum
  absolute difference about `1.9e-13 A`).
- The target reference follows `m * sin(2*pi*50*t + phase)` scaled by 300 V in
  the dataset. This target/reference is not the simulated stepped output.

These are dataset-reproduction checks, not independent circuit validation.
This first version omits dead time, semiconductor drops, switching transients,
parasitics, thermal behavior, and hardware safety models, as instructed. It
does not interface with the CNN or PSO and must not be interpreted as physical
hardware or Simulink validation.

## Event-based harmonic evaluation

`evaluation/event_based_thd.py` locates comparator crossings on each linear
half-carrier segment, constructs the resulting piecewise-constant voltage, and
integrates its Fourier coefficients analytically over the cycle. Coefficients
are peak-scaled; their magnitude is peak harmonic voltage, and THD includes
harmonics 2 through the configured maximum (100 by default). Fundamental phase
uses the sine convention and is reported in radians.

The evaluator uses the same ideal comparator and carrier equations and defaults
to offsets 0, 60, and 120 degrees. It also reports `sampled_thd_percent` from
the existing sampled FFT helper as a secondary dataset-comparison metric. The
event-based value is the primary analytic result; neither metric models
semiconductor non-idealities or hardware behavior. The root solver supports
`0 < modulation_index <= 1`, where each comparator difference is monotonic on
each linear carrier segment. The default 2nd-through-100th band ends at 5 kHz
and can exclude switching sidebands. `thd_percent` is the finite-band
2nd-through-`max_harmonic` value. `total_thd_percent` instead uses Parseval's
theorem over the complete event waveform:

- `mean_voltage_v = sum(V_j * duration_j) / T`
- `mean_square_voltage_v2 = sum(V_j**2 * duration_j) / T`
- `ac_rms_voltage_v = sqrt(mean_square_voltage_v2 - mean_voltage_v**2)`
- `fundamental_rms_voltage_v = fundamental_peak_amplitude_v / sqrt(2)`
- `total_thd_percent = 100 * sqrt(ac_rms_voltage_v**2 - fundamental_rms_voltage_v**2) / fundamental_rms_voltage_v`

The Parseval metric includes all harmonics, unlike finite-band THD. For the
dataset's SampleID 1, finite 2nd-through-100th THD is about `2.75e-8%`,
2nd-through-200th THD is about `18.98%`, while Parseval full-spectrum THD is
about `25.36%`; the difference is energy above the 200th harmonic. Always
identify the metric and finite harmonic band, if any, and keep it identical
when comparing candidates.
