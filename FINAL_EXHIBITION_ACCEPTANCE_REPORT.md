# Final Exhibition Acceptance Report

**Project:** Hybrid CNN and PSO-Based Harmonic Reduction in a Seven-Level Multilevel Inverter  
**Acceptance run:** 20 November 2026  
**Environment:** Windows, project-local `.venv`, CPU inference  
**Application URL:** `http://127.0.0.1:8503`

## Acceptance results

| Test | Status | Evidence and limitations |
|---|---|---|
| 1. Fresh launch from outside the project directory | **PASS** | Started from `C:\Windows` using the absolute project `app.py` path. Streamlit started at `http://127.0.0.1:8503`; its health endpoint returned `ok`. The application loaded the project dataset and `colab_runs\baseline_20261120_080836\checkpoints\best.pt`. |
| 1a. Launch with the network disconnected | **NOT TESTED** | The network was not disconnected. Offline operation is therefore not certified. |
| 2. Default live inverter demonstration | **PASS** | The default point (`m=0.75`, `R=30 Ω`, `L=0.03 H`, phase `0°`) completed. The output contained seven levels, three cell states and twelve gate signals; current, reference, prediction, and error arrays were finite. See measured values below. |
| 3. Seeded PSO integration and comparison | **PASS** | Two direct runs with seed `20261120`, 8 particles and 8 iterations had identical best evaluations and convergence histories. The candidate phases were passed through the simulator and compared at the same live operating point and with the same Parseval THD method. No improvement was found or claimed. |
| 4. CSV, JSON, ZIP generation and ZIP inspection | **PASS** | The export bundle was generated and its ZIP opened and integrity-checked. Its waveform, harmonic, convergence, metadata, and plot contents were inspected. The browser automation did not independently capture a download-manager event, so the final browser-to-file handoff was not verified. |
| 5. Complete held-out CNN test evaluation | **PASS** | The dedicated evaluator completed all 1,500 testing records (400 points each) in batches on CPU. Test IDs were disjoint from training IDs. No tuning or training used the test split. |
| 6. Failure handling | **PASS** | Missing dataset/checkpoint, invalid modulation input, empty/corrupt-checkpoint error handling, and export rejection for mismatched conditions were exercised. Some file failures were simulated with test-time path/loader patches; real dataset/checkpoint files were not moved or modified. |
| Full existing unit-test suite | **PASS** | `.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v` — **60 tests passed** in 3.915 seconds. |

## 1. Launch and resource loading

The application was launched from outside the project root with:

```powershell
Set-Location C:\Windows
& "C:\Users\tejas m naidu\OneDrive\Hybrid_CNN_Inverter_Optimization-20261120T100441Z-1-001\Hybrid_CNN_Inverter_Optimization\.venv\Scripts\python.exe" -B -m streamlit run "C:\Users\tejas m naidu\OneDrive\Hybrid_CNN_Inverter_Optimization-20261120T100441Z-1-001\Hybrid_CNN_Inverter_Optimization\app.py" --server.address 127.0.0.1 --server.port 8503 --server.headless true --browser.gatherUsageStats false
```

The health endpoint was `http://127.0.0.1:8503/_stcore/health` and returned `ok`. The active checkpoint was the existing `best.pt`; the acceptance run did not overwrite it. The selected dataset was `data\raw\Seven_Level_Inverter_Project_extracted`. The temporary acceptance server was stopped afterward; no claim is made about a disconnected-network launch.

## 2. Live simulation measurements

For the default operating point, the dashboard reported:

- Fundamental peak: **225.00 V**
- Full-spectrum Parseval THD: **24.723479%**
- Event-based THD for harmonics 2–100: **1.019e-07%**
- 20 kHz sampled finite-band THD for harmonics 2–100: **10.871%**
- CNN target-waveform RMSE: **1.099 V**
- CNN target-waveform MAE: **0.848 V**
- CNN target-waveform R²: **0.99995**
- Phasor tracking error: **0.00000%** (display precision)

The simulator’s switching/state invariants and finite numerical outputs were checked. The three THD values use different measurement methods and ranges; they are not interchangeable. In particular, the event-based finite-band result must not be presented as the full-spectrum THD.

## 3. PSO repeatability and measured effect

The live action used seed **20261120**, **8 particles × 8 iterations**, a **30-record training-only subset**, and **64 unique candidate evaluations**. Repeating the seeded optimization produced identical best evaluations and convergence histories. The selected candidate remained `[0°, 60°, 120°]`.

| Measurement | Baseline | Candidate | Candidate − baseline |
|---|---:|---:|---:|
| Training-subset THD | 26.3563556% | 26.3563556% | 0.0000000 percentage points |
| Live-point Parseval THD | 24.7234792% | 24.7234792% | 0.0000000 percentage points |

The UI correctly reported no meaningful improvement at its 0.001 percentage-point reporting resolution. This run does **not** demonstrate a harmonic reduction. The held-out validation and testing splits were excluded from the live optimization subset.

## 4. Export inspection

The generated ZIP was **207,813 bytes** and passed `zipfile.testzip()` with no corrupt member. It contained:

- `waveforms.csv` — 400 waveform rows
- `summary.json` — operating conditions and baseline/candidate comparison
- `harmonic_spectrum.csv` — 200 rows (100 harmonic orders for each of baseline and candidate)
- `pso_convergence_history.csv` — 9 rows
- Two PNG plots with valid PNG signatures

Exported metadata and comparison values matched the evaluated experiment. The export builder generated CSV, JSON, and ZIP data; browser automation did not verify that a click was saved by the browser’s download manager.

## 5. Held-out CNN evaluation

The existing checkpoint was evaluated without training or weight changes on the complete testing split: **1,500 records × 400 points = 600,000 target points**. Metrics below distinguish pooled point metrics from averages over records:

| Metric | Pooled over all waveform points | Mean of per-record values |
|---|---:|---:|
| MAE | 1.2247583 V | 1.2247583 V |
| RMSE | 1.6222906 V | 1.5767528 V |
| R² | 0.9998990 | 0.9998887 |
| Correlation | 0.9999495 | 0.9999505 |

The maximum absolute point error was **13.2091217 V**. Each record has the same number of waveform points, explaining why pooled MAE equals mean per-record MAE. R² and correlation are reported separately for pooled and per-record calculations; the macro R² is not described as global R².

## 6. Failure handling and test evidence

- **Missing checkpoint:** A test-time path check was patched to simulate an absent checkpoint. The UI displayed that live CNN inference was unavailable and retained the dashboard tabs; no substitute prediction was used.
- **Missing dataset:** A test-time path was patched to a missing location. The application displayed its dataset/checkpoint loading error with the underlying missing-path detail.
- **Invalid input:** `run_live_simulation` rejected modulation index `1.2` with `ValueError: modulation_index must be in [0.55, 0.95].`
- **Empty/corrupt checkpoint:** The loader’s `EOFError` path was exercised with a test fixture. The app’s startup handler now includes `EOFError`, so the failure is surfaced rather than escaping as an unhandled startup exception.
- **Export failure:** The existing export test verified rejection when baseline and candidate operating conditions differ.

One nonfatal Streamlit log warning occurred while a widget value was being supplied through session state alongside its default. The dashboard remained operational; this warning was not a test failure and was not changed during this acceptance-only task.

## Remaining limitations

- A disconnected-network run was not performed, so offline readiness remains unverified.
- The browser download-manager handoff was not captured; generated export data and ZIP contents were verified directly.
- No screenshot was saved as a project file. A browser screenshot of the live demonstration was captured during inspection.
- CNN waveform accuracy is not evidence that the inverter’s THD was reduced. This PSO run showed no measurable improvement.
- This is a software simulation; these results do not establish hardware or physical-inverter validation.

## Files created or modified for this acceptance task

- **Created:** `FINAL_EXHIBITION_ACCEPTANCE_REPORT.md`
- **Modified:** `app.py` — added `EOFError` handling to the existing startup error handler for empty/corrupt checkpoint reporting.

The previously integrated simulator, CNN evaluation, live PSO, and export functionality was tested without redesigning the dashboard.
