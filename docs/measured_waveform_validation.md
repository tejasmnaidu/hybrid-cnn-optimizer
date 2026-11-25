# Measured waveform validation

The live simulation workspace has an optional **Measured Waveform Validation
(CSV)** panel. It is intended for comparing a measured inverter output voltage
against the simulated output at the exact live settings that were selected in
the app. It analyzes measured and simulated sampled FFT THD over the same
harmonic range, fundamental amplitude, mean/DC offset, and harmonic amplitudes.
It does not claim a time-domain waveform error because measurement trigger
phase and switching-carrier synchronization are not established.

## Input format

Upload a UTF-8 CSV with these required columns:

```csv
time_s,voltage_v
0.00000,0.0
0.00005,12.3
```

Time must be in seconds, voltage in volts, samples must increase uniformly
(within 1% of the median time step), and the data must span at least two
complete cycles at the simulator's 50 Hz fundamental. A duplicated final
endpoint is detected and excluded from the FFT. The analyzer accepts at most
25 MB and 1,000,000 samples. It uses no window because it requires an integer
number of cycles; recordings that do not meet that condition are rejected
rather than silently reporting a leakage-contaminated THD.
The measured fundamental must match the simulator's fixed 50 Hz fundamental.

For defensible comparison, record the same output node and voltage convention
as the simulation, include the DC offset rather than silently detrending it,
and set the live controls to the corresponding measured operating point before
running the simulation. The app calculates the shared H2–N band based on the
lower Nyquist limit of the uploaded measurement and simulation. The result is
sampled FFT THD, not the event-based full-spectrum THD used by the phase
optimizers.

## Data isolation and limitations

The uploaded file is analyzed in memory in the current Streamlit session. This
feature does not add rows to the training dataset, write the file to disk,
retrain or update the CNN, or modify its checkpoint. Results are discarded
when the session state is cleared. The current simulator is ideal, fixed at
50 Hz with a 1 kHz carrier and 20 kHz sampling; agreement with measured data
does not validate device losses, dead time, parasitics, thermal behavior,
protection, or hardware safety. Treat differences as validation evidence to
investigate, not as proof that the hardware is safe or that the model is
ready for control.
