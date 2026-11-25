# Current Project Status

**Status date:** 2026-11-24

This summary distinguishes four kinds of information:

- **Analytical simulation (inverter simulator)** — values computed by the ideal software inverter and RL-load model.
- **Dataset measurement** — values stored in the supplied project CSVs.
- **CNN prediction** — values inferred by the existing checkpoint; these are predicted target/reference waveforms, not measured inverter output or switch control.
- **Current validation** — directly checked implementation, dataset agreement, or held-out inference evidence.

No model architecture, checkpoint, dataset, inverter equations, gate logic, optimizer algorithm, or THD calculation was changed for this status review.

## Component status

| Component | Implemented | Validated against | Remains unvalidated |
|---|---|---|---|
| Seven-level ideal inverter simulator | `multilevel/inverter_simulator.py`; 3 cells, 12 gate signals, 400 samples/cycle, periodic exact-ZOH RL load, 100 V/cell default. | Supplied `Cell_States.csv`, `Switching_Gates.csv`, `Input_Voltage.csv`, and `Load_Current.csv`; 10,000 records; current validation in [SIMULATOR_VALIDATION.md](./SIMULATOR_VALIDATION.md). | Original MATLAB/Simulink source, independent hardware/bench behavior, dead time, device losses, sensor noise, thermal and parasitic effects. |
| Three H-bridge cells and gate truth table | Positive, negative and zero states generate `[1,0,0,1]`, `[0,1,1,0]`, and `[0,1,0,1]`; shoot-through and cell-voltage reconstruction are checked. | Every supplied dataset gate/state record and unit tests for state/gate/voltage summation. | Physical gate-driver implementation and authoritative device-wiring mapping. |
| Seven voltage levels | Output sums three cell voltages and is expected to form −300/−200/−100/0/+100/+200/+300 V at 100 V/cell. | Exact agreement with all supplied `Input_Voltage.csv` records. Unit tests cover 100 V and 50 V/cell scaling. | Hardware voltage levels, device drop, switching delay, and non-ideal DC-link behavior. |
| Operating-condition response | Modulation, phase, carrier shifts, DC voltage, R and L are accepted through the real simulator APIs and live workflows. | Unit tests cover bounds, phase, carrier offsets, R/L, DC scaling and determinism. | Complete operating-range validation against independent electrical data or hardware. |
| Dataset/preprocessing | The project loads, aligns and validates 10,000 records: 7,000 training, 1,500 validation and 1,500 testing. Normalization is training-only. | Dataset schemas, IDs, split counts, auxiliary value domains and split leakage checks recorded in existing validation artifacts. | Independent source acquisition provenance and all possible near-duplicate or grouped leakage mechanisms. |
| Existing CNN | `WaveformResidualCNN` and the existing checkpoint perform reference-target waveform prediction. | Strict checkpoint compatibility, architecture loading and complete 1,500-record test inference. | Physical output control, switch prediction, out-of-distribution hardware behavior and meaningful harmonic reduction. |
| Analytical reference | `300 × m × sin(2π × 50 Hz × t + phase)` defines the dataset analytical target/reference. | Directly compared with the complete test split and condition-grouped test subsets. | Independent MATLAB provenance or a newly held-out operating-condition generator. |
| CNN held-out evaluation | Existing checkpoint evaluated only on test split in inference mode. | Pooled and condition-grouped MAE, RMSE, R² and correlation in [CNN_GROUPED_TEST_EVALUATION.md](./CNN_GROUPED_TEST_EVALUATION.md). Result: CNN is worse than analytical reference in every reported group; no improvement. | Held-out joint combinations absent from training; no test groups were available for training-band × R × L combinations. |
| PSO optimization | Existing circular PSO remains implemented and separately reusable. | Internal tests and existing PSO validation artifacts cover the recorded optimizer and convergence behavior. | Independent physical candidate validation and measured post-inverter THD improvement. |
| POA/PFO/hybrid optimization | Separate POA, PFO and hybrid implementations are present in the optimization modules. | Unit tests cover bounded/reproducible optimizer behavior and live result structures. | Independent hardware validation and demonstrated harmonic improvement; no improvement is claimed unless measured. |
| Live optimization integration | Live operating points use the ideal simulator and event-based THD through existing optimization interfaces. | Unit tests and stale-result/baseline consistency checks cover the software integration. | Hardware, original-source and non-ideal circuit equivalence. |
| Event-based THD | Analytic interval integration is implemented in `evaluation/event_based_thd.py`. | Unit tests and simulator-consistency tests distinguish event-based Parseval THD from sampled FFT THD. | Hardware measurement, dead-time/switching-loss THD and independent MATLAB/Simulink THD validation. |
| Sampled FFT/THD | Existing sample-based harmonic calculation remains available. | Unit tests and live-workflow integration. | Hardware spectra and independent measurement validation. |
| Streamlit dashboard | Live workspace, playback, switching view, CNN overlay and research workflow remain present. | Streamlit AppTests and browser-oriented verification where available. | Network-disconnected launch certification and hardware-oriented visualization fidelity beyond ideal software arrays. |
| CNN overlay | The live plot checkbox is optional and defaults off. | App tests verify default-off behavior and conditional prediction display. | Any claim that an optional prediction is measured output or switch control. |

## Provenance labels

Every live or comparison result must preserve the provenance distinction:

- Use **Analytical simulation (inverter simulator)** for voltage/current arrays, gate states and metrics computed by `run_live_simulation` and `simulate_inverter`.
- Use **Dataset measurement** only for values read from the supplied dataset CSVs.
- Use **CNN prediction** for checkpoint-inferred target/reference waveforms. Never relabel it as measured inverter output or actual switch control.

## Current limitations

- The ideal simulator is not line-by-line validated against original MATLAB/Simulink source.
- No MATLAB/Simulink source or bench measurements were found in the repository or supplied archive.
- Current validation against the supplied CSV records establishes software/dataset agreement, not physical inverter behavior.
- The CNN does not control gates and showed no grouped held-out-test improvement over the analytical target calculation.
- Optimizer no-improvement results must remain reported as no improvement unless an independent re-simulation and the same operating conditions demonstrate otherwise.
