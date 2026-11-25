# Exhibition Presentation Guide

## Three-minute project explanation

### 0:00–0:40 — Problem and system

“This project is a software study of a three-cell, seven-level inverter. It
simulates ideal phase-shifted PWM switching and an RL load, then analyzes the
output voltage harmonics. A trained CNN is also evaluated, but its task is to
predict the reference target waveform—not to control the inverter.”

### 0:40–1:20 — Simulation

“The live demonstration takes modulation index, load resistance, inductance,
and reference phase. The simulator generates the seven-level output voltage,
load current, cell states, and gate signals. These are ideal software-model
results, not hardware measurements.”

### 1:20–2:05 — CNN and metrics

“The CNN receives normalized inverter voltage, current, and operating
conditions. It predicts the reference target voltage. MAE and RMSE quantify
that waveform prediction error. They are separate from inverter-output THD.”

“The event-based finite-band THD uses Fourier harmonics 2–100 from exact
switching intervals. The sampled finite-band THD uses a 20 kHz sampled
waveform over the same harmonic orders. Parseval full-spectrum THD uses
interval energy and includes all harmonic energy. They are related but not
interchangeable.”

### 2:05–3:00 — Evidence and limitation

“All 1,500 held-out testing records were evaluated for the validation audit.
The CNN's pooled RMSE was about 1.622 V, while the known analytical reference
formula had an RMSE of about 0.000018 V on the same records. The saved PSO
candidate did not show a meaningful validation improvement. Therefore, the
project demonstrates simulation, regression, and harmonic-analysis tools; it
does not establish that the CNN or PSO has reduced harmonics in hardware.”

## Five-minute demonstration sequence

1. **Open the local app.** From the project root run:

   ```powershell
   .\.venv\Scripts\python.exe -B -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
   ```

   Visit `http://127.0.0.1:8501`. Confirm the header indicates that the
   dataset is validated and CNN checkpoint is loaded.

2. **Overview (about 30 seconds).** Show the selected dataset record and the
   workflow. Clarify that these overview traces are dataset records, not a
   newly run hardware experiment.

3. **Exhibition Mode (about 2 minutes).** Select the **EXHIBITION MODE** tab
   and click **RUN EXHIBITION SIMULATION**. The app runs the default valid
   condition `m=0.75`, `R=30 Ω`, `L=0.03 H`, `phase=0°`. Point out the
   inverter output and reference, CNN prediction and error, RL current, cell
   states, gate signals, and event-integrated harmonic spectrum.

4. **Explain the measurements (about 45 seconds).** Contrast inverter-output
   THD with CNN target-prediction RMSE. Explain that event and sampled
   finite-band THD cover orders 2–100, while Parseval full-spectrum THD
   includes all spectral energy.

5. **Show saved PSO validation evidence (about 30 seconds).** The exhibition
   tab displays the saved baseline and selected-candidate validation means
   and paired bootstrap interval. The observed change was negligible and the
   interval includes zero, so do not call it a demonstrated reduction.

6. **Optional technical follow-up.** In **LIVE DEMONSTRATION**, use a quick
   preset such as **Lower modulation** or **Lower inductance**, then click
   **RUN LIVE SIMULATION**. Explain that preset inputs trigger a fresh
   simulation only after the explicit run action. The marginal training
   support warning is not a joint-distribution guarantee.

7. **Optional test-set evidence.** The **CNN TEST EVALUATION** tab starts
   inference only after the user clicks its explicit button. The audit
   already evaluated the full held-out split; its fixed results are in
   `SCIENTIFIC_VALIDATION_REPORT.md` and the scientific-validation JSON/CSV
   outputs. Do not use test results to tune the model or PSO.

## Dashboard section guide

| Section | What it shows | How to describe it |
|---|---|---|
| Overview | A selected record's supplied waveforms and operating point | Recorded dataset content; not a new experiment |
| CNN Prediction | One selected record's target and predicted waveform plus pointwise error | CNN reference-waveform regression |
| CNN Test Evaluation | Explicit full-test-split batched evaluation and downloadable results | Held-out model assessment; do not tune from it |
| Data Diagnostics | Dataset splits, operating features, architecture, and saved training history | Provenance and model configuration evidence |
| Live Demonstration | User-selected simulator operating point and fresh CNN inference | Ideal simulated signals and distinct target-prediction metrics |
| Exhibition Mode | One-click live case plus saved validation PSO comparison | A streamlined view of the same model, not a separate experiment |

## Important metrics

- **MAE (V):** mean absolute difference between predicted and target
  reference-waveform points.
- **RMSE (V):** square root of the mean squared reference-waveform error;
  larger errors are weighted more heavily.
- **R²:** coefficient of determination for target-waveform regression.
- **Fundamental peak (V):** event-based peak amplitude of the inverter
  output-voltage fundamental.
- **Fundamental phasor tracking error (%):** complex fundamental mismatch
  against the target sine phasor.
- **Event THD, orders 2–100 (%):** exact event-integrated finite-band ratio.
- **Sampled THD, orders 2–100 (%):** finite-band FFT estimate from the
  sampled output.
- **Parseval full-spectrum THD (%):** AC energy excluding the fundamental,
  normalized by fundamental RMS; includes all spectral energy.
- **PSO delta (percentage points):** selected-candidate validation mean
  minus baseline validation mean; positive is worse.

## Likely faculty questions

**Does the CNN switch the inverter?**  
No. It predicts the reference target voltage waveform from voltage, current,
and operating conditions. The simulator independently generates switching
states and gates.

**Has the CNN reduced THD?**  
That is not established. The CNN's target is an analytically defined
reference waveform, and the analytical formula outperformed the CNN on the
held-out test split.

**Did PSO improve the output harmonics?**  
The saved optimization changed the training objective by a negligible
amount, while validation was slightly worse. The paired bootstrap interval
included zero; meaningful reduction is not demonstrated.

**Are these measured hardware results?**  
No. They are ideal software-model outputs. No hardware, MATLAB, or Simulink
validation is established.

**Was the testing split used to tune the system?**  
The audited training code selects checkpoints from validation and accepts
only train and validation loaders. The scientific audit evaluated the test
split for final assessment; those results must remain excluded from future
training, model selection, threshold selection, and PSO selection.

**Does an in-range operating point guarantee CNN reliability?**  
No. The UI checks marginal training ranges and categories. It does not
establish that a combination of otherwise observed conditions is represented
jointly.

