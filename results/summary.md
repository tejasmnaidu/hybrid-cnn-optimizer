# Baseline versus optimized THD experiment

## Protocol

- Fixed conditions: DC voltage 100.0 V/cell, load 30.0 ohm / 0.030 H, reference phase 0.0 degrees.
- Modulation indices: 0.55, 0.65, 0.75, 0.85, 0.95.
- Seeds: 20261124, 20261125, 20261126, 20261127, 20261128, 20261129, 20261130, 20261131, 20261132, 20261133 (10 paired seeds per modulation).
- Cell 1 is fixed at 0 degrees; the baseline is the existing canonical pair (60, 120) degrees. The baseline uses exactly the same event-based and sampled metric functions as final optimized candidates.
- Primary objective: independently recomputed event-based Parseval total voltage THD. Additional measurements are event-based voltage harmonics 2–100, sampled 20 kHz voltage/current THD harmonics 2–100, and event-Fourier RL-derived current THD harmonics 2–100. These THD definitions have different bandwidth/semantics and are reported separately.
- PSO/POA/PFO/Hybrid use the existing 8-member, 8-iteration configuration. Actual unique objective evaluations and requests are in the CSV; nominal request budget is 72. CNN-guided hybrid is included because the approved Phase 5 500-candidate score/THD Spearman correlation is 0.954079 (threshold 0.70). It uses a 64-unique-candidate ranking budget, then fully evaluates top-8 finalists and independently re-evaluates the selected candidate. Consequently its full event-THD budget and waveform-generation count differ; counts are separate in the CSV.
- Optimized phases and all reported final metrics are re-simulated independently after selection. Candidate failures are not replaced with another seed's result.

## Aggregate statistics

Primary metric is event-based Parseval total voltage THD (%). Standard deviation is the sample standard deviation (ddof=1). Paired tests compare each method's run-wise values with the baseline at the same seed and modulation. Paired mean reduction is baseline minus optimized THD, in percentage points; positive values favor the optimizer. The exact two-sided paired sign-test p-value excludes ties and is descriptive; it is not adjusted for multiple comparisons.

| Method | Mean THD (%) | Std (%) | Median (%) | Best (%) | Worst (%) | Mean reduction (%) | Improved (%) | Paired mean reduction (pp) | Paired sign-test p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 27.039059 | 5.812043 | 24.723479 | 20.689102 | 37.313788 | 0.000000 | 0.0% (0/50) | 0.000000 | 1 |
| PSO | 27.025443 | 5.812856 | 24.723479 | 20.606383 | 37.313788 | 0.053772 | 24.0% (12/50) | 0.013616 | 0.000488281 |
| POA | 27.035528 | 5.813413 | 24.723479 | 20.614369 | 37.313788 | 0.015048 | 14.0% (7/50) | 0.003531 | 0.015625 |
| PFO | 27.035979 | 5.814277 | 24.723479 | 20.650527 | 37.313788 | 0.013477 | 6.0% (3/50) | 0.003081 | 0.25 |
| HybridPOAPFO | 27.037384 | 5.813065 | 24.723479 | 20.686151 | 37.313788 | 0.007106 | 8.0% (4/50) | 0.001675 | 0.125 |
| CNN-guided Hybrid | 27.033504 | 5.809213 | 24.723479 | 20.658268 | 37.313788 | 0.019794 | 18.0% (9/50) | 0.005555 | 0.00390625 |

## Per-modulation results

| Modulation | Method | Mean THD (%) | Std (%) | Median (%) | Best (%) | Worst (%) | Mean reduction (%) | Improved (%) | Paired mean reduction (pp) | Paired sign-test p |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.55 | Baseline | 37.313788 | 0.000000 | 37.313788 | 37.313788 | 37.313788 | 0.0% (0/10) | 0.000000 | 1 |
| 0.55 | PSO | 37.294937 | 0.039742 | 37.313788 | 37.219409 | 37.313788 | 20.0% (2/10) | 0.018851 | 0.5 |
| 0.55 | POA | 37.309901 | 0.012294 | 37.313788 | 37.274913 | 37.313788 | 10.0% (1/10) | 0.003888 | 1 |
| 0.55 | PFO | 37.313788 | 0.000000 | 37.313788 | 37.313788 | 37.313788 | 0.0% (0/10) | 0.000000 | 1 |
| 0.55 | HybridPOAPFO | 37.313788 | 0.000000 | 37.313788 | 37.313788 | 37.313788 | 0.0% (0/10) | 0.000000 | 1 |
| 0.55 | CNN-guided Hybrid | 37.299870 | 0.028547 | 37.313788 | 37.239543 | 37.313788 | 30.0% (3/10) | 0.013919 | 0.25 |
| 0.65 | Baseline | 28.781283 | 0.000000 | 28.781283 | 28.781283 | 28.781283 | 0.0% (0/10) | 0.000000 | 1 |
| 0.65 | PSO | 28.780957 | 0.000686 | 28.781283 | 28.779612 | 28.781283 | 20.0% (2/10) | 0.000325 | 0.5 |
| 0.65 | POA | 28.781011 | 0.000599 | 28.781283 | 28.779555 | 28.781283 | 20.0% (2/10) | 0.000272 | 0.5 |
| 0.65 | PFO | 28.781283 | 0.000000 | 28.781283 | 28.781283 | 28.781283 | 0.0% (0/10) | 0.000000 | 1 |
| 0.65 | HybridPOAPFO | 28.781283 | 0.000000 | 28.781283 | 28.781283 | 28.781283 | 0.0% (0/10) | 0.000000 | 1 |
| 0.65 | CNN-guided Hybrid | 28.781154 | 0.000287 | 28.781283 | 28.780439 | 28.781283 | 20.0% (2/10) | 0.000128 | 0.5 |
| 0.75 | Baseline | 24.723479 | 0.000000 | 24.723479 | 24.723479 | 24.723479 | 0.0% (0/10) | 0.000000 | 1 |
| 0.75 | PSO | 24.715504 | 0.016915 | 24.723479 | 24.679658 | 24.723479 | 20.0% (2/10) | 0.007975 | 0.5 |
| 0.75 | POA | 24.723479 | 0.000000 | 24.723479 | 24.723479 | 24.723479 | 0.0% (0/10) | 0.000000 | 1 |
| 0.75 | PFO | 24.723479 | 0.000000 | 24.723479 | 24.723479 | 24.723479 | 0.0% (0/10) | 0.000000 | 1 |
| 0.75 | HybridPOAPFO | 24.723479 | 0.000000 | 24.723479 | 24.723479 | 24.723479 | 0.0% (0/10) | 0.000000 | 1 |
| 0.75 | CNN-guided Hybrid | 24.723479 | 0.000000 | 24.723479 | 24.723479 | 24.723479 | 0.0% (0/10) | 0.000000 | 1 |
| 0.85 | Baseline | 23.687643 | 0.000000 | 23.687643 | 23.687643 | 23.687643 | 0.0% (0/10) | 0.000000 | 1 |
| 0.85 | PSO | 23.666603 | 0.036515 | 23.687643 | 23.584138 | 23.687643 | 30.0% (3/10) | 0.021040 | 0.25 |
| 0.85 | POA | 23.685416 | 0.007042 | 23.687643 | 23.665374 | 23.687643 | 10.0% (1/10) | 0.002227 | 1 |
| 0.85 | PFO | 23.676098 | 0.026585 | 23.687643 | 23.607231 | 23.687643 | 20.0% (2/10) | 0.011545 | 0.5 |
| 0.85 | HybridPOAPFO | 23.679565 | 0.015772 | 23.687643 | 23.638868 | 23.687643 | 30.0% (3/10) | 0.008078 | 0.25 |
| 0.85 | CNN-guided Hybrid | 23.679245 | 0.017931 | 23.687643 | 23.639632 | 23.687643 | 20.0% (2/10) | 0.008398 | 0.5 |
| 0.95 | Baseline | 20.689102 | 0.000000 | 20.689102 | 20.689102 | 20.689102 | 0.0% (0/10) | 0.000000 | 1 |
| 0.95 | PSO | 20.669214 | 0.034632 | 20.689102 | 20.606383 | 20.689102 | 30.0% (3/10) | 0.019888 | 0.25 |
| 0.95 | POA | 20.677832 | 0.024832 | 20.689102 | 20.614369 | 20.689102 | 30.0% (3/10) | 0.011270 | 0.25 |
| 0.95 | PFO | 20.685244 | 0.012199 | 20.689102 | 20.650527 | 20.689102 | 10.0% (1/10) | 0.003858 | 1 |
| 0.95 | HybridPOAPFO | 20.688807 | 0.000933 | 20.689102 | 20.686151 | 20.689102 | 10.0% (1/10) | 0.000295 | 1 |
| 0.95 | CNN-guided Hybrid | 20.683771 | 0.011410 | 20.689102 | 20.658268 | 20.689102 | 20.0% (2/10) | 0.005331 | 0.5 |

## Honest interpretation

Reduction is calculated per run as `(baseline THD - optimized THD) / baseline THD * 100`. An improved run means a strict reduction greater than numerical tolerance (1e-12 percentage points). Flat or worse runs remain in the CSV and statistics. The per-seed result table preserves each run and reports event voltage THD as the primary independently verified result.

Across all optimizer runs, 35/250 improved, 215/250 were flat, and 0/250 were worse. No improving run was observed for: PFO at m=0.55, HybridPOAPFO at m=0.55, PFO at m=0.65, HybridPOAPFO at m=0.65, POA at m=0.75, PFO at m=0.75, HybridPOAPFO at m=0.75, CNN-guided Hybrid at m=0.75.

CNN-guided hybrid was included based on a measured ranking correlation (0.954079); this is not evidence that the CNN improves THD. The sampled and event-based harmonic metrics are not interchangeable. The CNN checkpoint was loaded read-only; the existing dataset was not read or modified.

## Figures

- [Before/after THD bars](thd_experiment_before_after.png)
- [Convergence curves](thd_experiment_convergence.png). CNN-guided is marked as a verified endpoint because its candidate ranking does not retain event-THD iteration history; extra THD calls were not added just to create a curve.
- [Representative voltage waveform and spectrum](thd_experiment_waveform_spectrum.png), at modulation 0.75 and seed 20261124.

The complete 300-row per-seed record is [`thd_experiment.csv`](thd_experiment.csv).

Run the experiment from the project root with `python -m scripts.run_thd_experiment`. It refuses to overwrite any existing output artifact.
