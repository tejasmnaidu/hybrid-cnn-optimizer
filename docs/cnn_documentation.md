# CNN Documentation

## What the CNN predicts

The existing `WaveformResidualCNN` predicts a 400-point target/reference voltage waveform.
It is not a gate controller, inverter-output measurement predictor, or THD predictor.

### Inputs

- Normalized input voltage: `(batch, 400)`.
- Normalized load current: `(batch, 400)`.
- Normalized operating conditions: `(batch, 4)`, in this order:
  `modulationIndex`, `loadResistance`, `loadInductance`, `phaseAngle`.

### Normalized output

- Shape: `(batch, 400)`.
- Unit: normalized target-voltage index.

### Physical output

The model output is denormalized with:

`physical voltage = normalized target * target_voltage_std + target_voltage_mean`.

The same target statistics convert `Target_Voltage.csv` ground truth to physical volts.
All CNN, analytical and trivial baseline values used in this evaluation are volts.

## Architecture

- Separate one-dimensional residual branches for input voltage and load current.
- Voltage/current branch channels: 32, then 64.
- Operating-condition MLP: 4 inputs to 32 width with GELU, then 32 width.
- Fusion: 64-channel residual blocks, then a one-channel 1x1 convolution.
- Waveform length: 400 points.
- Existing checkpoint parameter count: 201,601.

## Normalization

The training/fitted normalization stores:

- Input-voltage mean and standard deviation.
- Load-current mean and standard deviation.
- Per-feature operating-condition means and standard deviations.
- Target-voltage mean and standard deviation.

All four tables and normalization values are aligned by `SampleID`.
Normalization statistics are fitted on training IDs only.

## Loss, split and seed

- Training loss: MAE.
- Optional derivative weight: 0.0.
- Optimizer/configuration: AdamW as stored in `training/trainer.py`.
- Seed: 42.
- Split sizes: 7,000 training, 1,500 validation, 1,500 testing.
- Test records are not accepted by `Trainer.fit` and are not used for training,
  checkpoint selection or tuning.

## Leakage checks

The implementation uses `prepared.indices("training")`,
`prepared.indices("validation")`, and `prepared.indices("testing")`.
Those index sets are disjoint. The preprocessing implementation masks the
training split before calculating normalization statistics.

## Predictors compared

Every predictor is compared with the same denormalized test target:

1. **Existing CNN**: existing `best.pt`, denormalized from normalized target output.
2. **Analytical reference**:
   `300 * modulationIndex * sin(2*pi*50*n/20000 + phaseAngle)`,
   for `n = 0..399`.
3. **Trivial baseline**: pointwise mean of denormalized training `Target_Voltage`
   waveforms, reused unchanged for every test record.

The metrics are waveform MAE, MSE, RMSE and R² in volts.
Pooled metrics aggregate all 600,000 test waveform points.
Per-record metrics are also computed and averaged over the 1,500 test records.
