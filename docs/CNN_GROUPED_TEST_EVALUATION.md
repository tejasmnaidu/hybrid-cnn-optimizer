# Condition-Grouped CNN Test Evaluation

**Created (UTC):** 2026-11-24T15:09:03.029626+00:00  
**Status:** inference-only; checkpoint and dataset were not modified.  
**Testing records:** 1500; **waveform points/record:** 400  
**Checkpoint SHA-256:** `9ffea39995c4bbba15f1f3572c9aaf54d49ad6ff1b1a186c4cafffe38263636c`

## Method

The existing checkpoint was loaded strictly and evaluated once on the complete held-out testing split. It was not retrained or tuned. CNN and analytical-reference metrics below are computed against the same target records. Group boundaries for modulation are training-only quantiles; R/L categories and the training-joint-combination set are derived from training data. The test split is used only for evaluation.

Groups with fewer than **20 records** are marked **LOW N**; their estimates are descriptive and less stable. Better/equal/worse is determined from pooled RMSE using an equality tolerance of 1.0e-12 V. Metrics (MAE, RMSE, R², correlation) pool waveform points in each group; R² or correlation may be undefined for constant data.

## Results

| Group type | Group | Records | CNN MAE (V) | CNN RMSE (V) | CNN R² | CNN corr. | Analytical MAE (V) | Analytical RMSE (V) | Analytical R² | Analytical corr. | CNN vs analytical |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| modulation_band | training_quantile_band_01 | 151 | 1.23743 | 1.65378 | 0.999813 | 0.999907 | 1.07026e-05 | 1.3849e-05 | 1 | 1 | CNN worse |
| modulation_band | training_quantile_band_02 | 143 | 1.20903 | 1.60129 | 0.999848 | 0.999924 | 1.10116e-05 | 1.44022e-05 | 1 | 1 | CNN worse |
| modulation_band | training_quantile_band_03 | 148 | 1.23279 | 1.61846 | 0.999862 | 0.999931 | 1.059e-05 | 1.36559e-05 | 1 | 1 | CNN worse |
| modulation_band | training_quantile_band_04 | 126 | 1.25767 | 1.68888 | 0.999867 | 0.999933 | 1.29023e-05 | 1.69617e-05 | 1 | 1 | CNN worse |
| modulation_band | training_quantile_band_05 | 161 | 1.28044 | 1.67981 | 0.999882 | 0.999941 | 1.2588e-05 | 1.64929e-05 | 1 | 1 | CNN worse |
| modulation_band | training_quantile_band_06 | 158 | 1.10069 | 1.47506 | 0.999918 | 0.999959 | 1.4399e-05 | 1.90637e-05 | 1 | 1 | CNN worse |
| modulation_band | training_quantile_band_07 | 144 | 1.1198 | 1.47646 | 0.999926 | 0.999963 | 1.51205e-05 | 2.01322e-05 | 1 | 1 | CNN worse |
| modulation_band | training_quantile_band_08 | 160 | 1.26122 | 1.64948 | 0.999915 | 0.999958 | 1.57485e-05 | 2.05614e-05 | 1 | 1 | CNN worse |
| modulation_band | training_quantile_band_09 | 151 | 1.28848 | 1.70632 | 0.999918 | 0.999959 | 1.46548e-05 | 1.98557e-05 | 1 | 1 | CNN worse |
| modulation_band | training_quantile_band_10 | 158 | 1.25827 | 1.65748 | 0.999929 | 0.999965 | 1.60276e-05 | 2.11935e-05 | 1 | 1 | CNN worse |
| loadResistance | 10 | 379 | 1.26391 | 1.69642 | 0.999889 | 0.999945 | 1.30666e-05 | 1.73116e-05 | 1 | 1 | CNN worse |
| loadResistance | 20 | 356 | 1.18588 | 1.57241 | 0.999906 | 0.999953 | 1.39104e-05 | 1.86071e-05 | 1 | 1 | CNN worse |
| loadResistance | 30 | 393 | 1.19642 | 1.57458 | 0.999904 | 0.999952 | 1.27993e-05 | 1.69463e-05 | 1 | 1 | CNN worse |
| loadResistance | 40 | 372 | 1.252 | 1.64159 | 0.999897 | 0.999948 | 1.39606e-05 | 1.87294e-05 | 1 | 1 | CNN worse |
| loadInductance | 0.01 | 378 | 1.31458 | 1.71126 | 0.99989 | 0.999945 | 1.31163e-05 | 1.7551e-05 | 1 | 1 | CNN worse |
| loadInductance | 0.02 | 354 | 1.21554 | 1.60883 | 0.9999 | 0.99995 | 1.31223e-05 | 1.72683e-05 | 1 | 1 | CNN worse |
| loadInductance | 0.03 | 384 | 1.1934 | 1.57739 | 0.999901 | 0.999951 | 1.28141e-05 | 1.70362e-05 | 1 | 1 | CNN worse |
| loadInductance | 0.05 | 384 | 1.17619 | 1.5886 | 0.999904 | 0.999952 | 1.45937e-05 | 1.95486e-05 | 1 | 1 | CNN worse |

## Held-out joint combinations

No held-out training-band × R × L combinations occurred in the test split.

## Interpretation

These are prediction-error comparisons against the dataset target, not inverter-output THD comparisons, switch-control validation, hardware measurements, or evidence of physical performance. No optimizer or THD calculations are part of this evaluation. Any conclusion is computed from grouped metric values above; a worse CNN result is reported as **no improvement** over the analytical reference for that group.
