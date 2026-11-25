# Project Limitations

## Simulation model

- The inverter and RL-load response are ideal software simulations using the
  project's switching, sampling, and periodic-current conventions.
- The simulator uses 100 V per cell, three H-bridge cells, a 50 Hz reference,
  a 1 kHz carrier, and 20 kHz sampling. It does not model device losses,
  parasitic components, dead time, switching delays, thermal behavior,
  sensor noise, or control hardware.
- The available project archive did not include the partner's original
  MATLAB source needed for direct source-to-source validation. No physical
  hardware, MATLAB, or Simulink validation is established.
- Correct numeric signal dimensions do not by themselves establish
  equivalence to a physical gate driver or wiring implementation.

## CNN task and results

- The CNN is a waveform-regression model. It receives inverter output
  voltage, RL load current, and four operating-condition features, and
  predicts the reference target voltage waveform.
- It does not predict switching gates, inverter output voltage, or optimized
  inverter-output THD. CNN prediction accuracy must not be presented as
  evidence of harmonic reduction.
- On the audited held-out set, CNN pooled RMSE was 1.622291 V. The analytical
  reference waveform derived from known conditions had RMSE about
  0.0000179 V on the same test records, so the CNN does not improve on this
  meaningful baseline.
- The checkpoint matches the current model architecture and the current
  training-only normalization statistics. The checkpoint does not contain a
  source dataset fingerprint or source-code revision, so the exact historical
  training archive cannot be proven from checkpoint metadata alone.
- The marginal support warning detects values outside observed marginal
  feature support; it does not test joint-density coverage or guarantee
  generalization.

## Dataset and test-set policy

- The supplied dataset contains 10,000 records: 7,000 training, 1,500
  validation, and 1,500 testing records, with 400 waveform points per
  record.
- The inspected training code uses training data for gradient updates and
  validation data for checkpoint selection. It does not accept a test
  loader in `Trainer.fit`.
- The complete testing split was evaluated for the scientific validation
  audit. It must remain held out from subsequent fitting, feature/model
  selection, threshold selection, and PSO selection.
- Dataset split and duplicate checks support separation of the supplied
  records, but do not prove independent experimental acquisition or
  generalization to other inverter systems.

## Harmonic analysis

- Event-based results are analytical for the implemented piecewise-constant
  switching waveform and switching convention.
- Event finite-band THD includes harmonics 2–100. The sampled finite-band
  estimate also uses orders 2–100 but is computed from sampled signals;
  sampling and spectral representation can produce differences.
- Parseval full-spectrum THD uses exact interval energy after removing DC and
  the fundamental. It is not interchangeable with finite-band THD.
- These metrics characterize the simulated inverter output. They do not
  independently validate a physical inverter.

## PSO evidence

- The saved circular PSO experiment completed 20 runs on a fixed 30-record
  training subset, using validation for a final candidate comparison.
- Training-subset mean THD decreased by only 0.00001221 percentage points.
  The selected candidate was 0.00004996 percentage points worse on the
  validation split. The paired bootstrap 95% interval included zero.
- This result does not establish a meaningful, statistically supported, or
  physically validated harmonic reduction. It also does not establish global
  optimality.
- The CNN's waveform prediction objective and the PSO's inverter-harmonic
  objective are distinct and must not be conflated.

## Reproducibility and presentation

- The audit outputs include the checkpoint and dataset fingerprints, test
  metrics, condition summaries, and PSO validation analysis. These identify
  the audited files but cannot reconstruct missing historical provenance.
- Results depend on the software versions, supplied dataset/checkpoint, and
  ideal model assumptions. The audit is not a guarantee of other Python
  versions, topologies, sampling rates, loads, or hardware.
- The dashboard's saved PSO comparison is a validation-split result, not a
  live operating-point comparison. The exhibition simulation does not apply
  PSO-selected phases.
- Numerical results should be presented with their split, aggregation
  definition, units, harmonic range, and software-simulation scope.
