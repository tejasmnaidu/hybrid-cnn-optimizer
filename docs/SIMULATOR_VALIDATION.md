# Inverter Simulator Validation

**Result:** PASS  
**Records compared:** 10000  
**Scope:** Ideal sampled simulator compared with the supplied exported waveform CSVs. This is not hardware validation.

## Reference inventory

- Original MATLAB/Simulink or `.mat` source files: No original MATLAB/Simulink source or .mat files found.
- Source files inside supplied archives: No original MATLAB/Simulink source found in supplied ZIP archives.
- Bench/measurement data: No bench/measurement data files found.
- Exported reference waveform CSVs:
- `data/raw/Seven_Level_Inverter_Project_extracted/Cell_States.csv`
- `data/raw/Seven_Level_Inverter_Project_extracted/Configuration.csv`
- `data/raw/Seven_Level_Inverter_Project_extracted/Input_Voltage.csv`
- `data/raw/Seven_Level_Inverter_Project_extracted/Load_Current.csv`
- `data/raw/Seven_Level_Inverter_Project_extracted/Operating_Conditions.csv`
- `data/raw/Seven_Level_Inverter_Project_extracted/Sample_Splits.csv`
- `data/raw/Seven_Level_Inverter_Project_extracted/Switching_Gates.csv`
- `data/raw/Seven_Level_Inverter_Project_extracted/THD_Results.csv`
- `data/raw/Seven_Level_Inverter_Project_extracted/Target_Voltage.csv`

The CSV waveforms are available as comparison references, so validation proceeded. Their presence does not establish independent provenance or substitute for the absent original source/bench measurements.

## Comparison conditions

- Carrier shifts: [0.0, 60.0, 120.0] degrees
- DC voltage: 100.0 V/cell
- Fundamental frequency: 50.0 Hz
- Sampling rate: 20000.0 Hz; 400 samples/cycle
- Phase angle interpreted as radians, matching the simulator API.
- Carrier offsets are not recorded in the supplied operating-condition CSV; the simulator defaults were assumed.
- State and gate comparisons are exact. Voltage and current use absolute tolerance 1e-12 V/A and zero relative tolerance.

## Sample-by-sample comparison

| Signal | Mismatches | Maximum absolute error | RMS error | First mismatch |
|---|---:|---:|---:|---|
| cell_states | 0 / 12000000 | 0 — | 0 — | none |
| switching_gates | 0 / 48000000 | 0 — | 0 — | none |
| output_voltage | 0 / 4000000 | 0 V | 0 V | none |
| load_current | 0 / 4000000 | 1.82964754458e-13 A | 2.35244935467e-14 A | none |

For arrays with cell/switch dimensions, the first-mismatch index is reported in the simulator array's native axis order, followed by sample index.

## Validated

- sampled ideal cell states against supplied Cell_States.csv
- sampled ideal gate signals against supplied Switching_Gates.csv
- sampled ideal output voltage against supplied Input_Voltage.csv
- sampled ideal RL load current against supplied Load_Current.csv

## Not validated

- agreement with original MATLAB/Simulink source (not provided)
- hardware or bench behavior (no bench data provided)
- dead time, switching losses, device drops, parasitics, thermal behavior, or other non-ideal effects
- physical correctness of the supplied switching convention beyond agreement with the supplied CSV records

The validation is limited to agreement with the supplied dataset records using the stated parameters and conventions. It does not validate physical hardware behavior. The ideal switching model contains no dead time, losses, or device effects.
