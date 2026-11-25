# CNN-guided hybrid optimization

## Method

The CNN predicts a target voltage waveform; it does not predict phase offsets.
For ranking, each candidate first generates a waveform with the existing
inverter simulator. The existing CNN predicts the target waveform from that
simulation, and the ranking score is RMSE in volts against the candidate's
analytical reference waveform. Lower scores rank first. CNN scores are never
reported as THD or used as phase values.

Cell 1 is fixed at 0 degrees; cell 2 and cell 3 offsets are searched over
`[0, 360)` degrees. Modulation index, DC voltage, load, and reference phase
remain fixed within each run. Candidate phases are canonicalized and wrapped
to the one-period search space. The guided proposal pool starts with the
conventional baseline and seven seeded random candidates, then alternates
POA prey-approach/surface-search proposals and PFO better-member attack/
defense proposals, using CNN RMSE only as proxy fitness. A 64-unique-candidate
budget applies to both methods. Of the guided pool, the top eight CNN-ranked
candidates are fully evaluated; the conventional baseline is reserved in
those eight if it did not rank there. Selection among finalists uses the
existing event-based total-THD objective and its 0.05% fundamental-phasor
feasibility limit.

Every selected candidate is then independently re-evaluated with the
event-based THD evaluator and re-simulated for sampled voltage and current
THD. Sampled THD uses the 20 kHz simulator output, 50 Hz fundamental, and
harmonics through 100. These sampled measurements are distinct from the
event-based Parseval objective and should not be compared numerically as if
they were the same THD definition.

## CNN-score correlation gate

The correlation experiment used 500 independently sampled uniform phase pairs
with seed `20261124`, with cell 1 fixed at zero. At modulation `0.75`, 100 V
per cell, `30 ohm`, `0.03 H`, and reference phase `0 degrees`, the tie-aware
Spearman correlation between CNN RMSE score and event-based total THD was
**0.9540794881**. The predeclared guidance threshold was `0.70`, so the
comparison proceeded. One simulator waveform generation and one event-based
THD evaluation were performed per correlation candidate. Candidate-level
values are in [`../results/cnn_guided_correlation.json`](../results/cnn_guided_correlation.json).

This correlation shows ranking association at this single operating point; it
does not establish that the CNN is an accurate waveform predictor or that the
association generalizes to other operating conditions.

## Ten-seed comparison

Both methods used the same operating point and a maximum budget of 64 unique
candidate evaluations. Seeds were `20261124` through `20261133`. Success was
defined in advance as a feasible final event-based THD no more than 0.1%
relative above the Phase 4 36-by-36 grid minimum, `24.7234791510%` at
`(60, 120) degrees`. Final event THD, sampled metrics, and candidate offsets
were independently recomputed for each run.

| Method | Mean event total THD (%) | Mean sampled voltage THD (%) | Mean sampled current THD (%) | Mean simulator waveform generations | Mean full event-THD evaluations | Mean runtime (s) | Grid successes |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hybrid POA-PFO | 24.723479 | 10.871481 | 1.482556 | 1 | 63 | 0.5225 | 10/10 |
| CNN-guided Hybrid POA-PFO | 24.723479 | 10.871481 | 1.482556 | 65 | 9 | 0.3509 | 10/10 |

The standalone optimizer used 62 unique objective candidates per run; its
67 objective requests include repeated accesses to the cached baseline and
did not cause extra THD calculations. The guided method scored 64 unique
candidates and fully evaluated eight finalists plus one independent final
re-evaluation. Waveform generations and full event-THD evaluations are
reported separately because CNN ranking requires simulator waveforms.

## Conclusion and limits

CNN guidance **did not improve THD**: both methods selected the grid-optimal
baseline phases on all ten seeds and had zero event-THD and phase-space gap
from the grid result. In this experiment guidance reduced full event-THD
evaluations (63 to 9) but increased simulator waveform generations (1 to 65).
Although measured mean runtime was lower in this one-condition, ten-seed
experiment, that observation does not establish a general efficiency gain.
No claim is made beyond this operating point and budget.

The existing checkpoint was loaded read-only. Its SHA-256 remained
`9ffea39995c4bbba15f1f3572c9aaf54d49ad6ff1b1a186c4cafffe38263636c`; hashes
of all 13 files under `data/` were unchanged. The dataset, checkpoint,
existing optimizer, and dashboard were not modified.

Per-seed results are in [`../results/cnn_guided_comparison.csv`](../results/cnn_guided_comparison.csv).
