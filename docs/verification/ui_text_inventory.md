# UI text inventory (Phase 1; no dashboard edits)

Inventory of visible strings in `app.py` as of the Phase 0 snapshot (commit `9c0f6be`). Streamlit display/widget search of `evaluation/` UI helpers found no matching calls. The AST inventory covers Streamlit display/widgets, chart helper labels, and Matplotlib titles/axes. Rows retain baseline source lines; repeated text may be listed separately where call-site context differs. Logic evidence is AST-based: comparisons, dictionary/subscript keys, assertions, and implicit widget identity or selector option values.

| File | Line(s) | Current text | Category | Used in logic? (search evidence) | Explicit widget key? | Action | New text |
|---|---:|---|---|---|---|---|---|
| `app.py` | 376 | Baseline | label | yes - evaluation\live_export.py:552 comparison, scripts\run_thd_experiment.py:712 comparison, scripts\run_thd_experiment.py:419 subscript key | N/A | KEEP-LOGIC | Keep unchanged |
| `app.py` | 394 | Event-integrated output-voltage harmonic spectrum (orders 1–100) | label | no - No AST logic-context match | N/A | SHORTEN | Event-integrated output-voltage harmonic |
| `app.py` | 400 | Frequency (Hz) | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 401 | Harmonic peak amplitude (V) | label | no - No AST logic-context match | N/A | SHORTEN | Harmonic Amplitude (V) |
| `app.py` | 512 | ⚡ INVERTER AI | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 513 | POWER ELECTRONICS RESEARCH | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 516 | DATASET & MODEL SETTINGS | label | yes - implicit widget identity (no explicit key) | No | KEEP-LOGIC | Keep unchanged |
| `app.py` | 517 | The dataset supplies the record browser, reference waveforms, and normalization/support checks. It does not retrain the CNN. The checkpoint supplies weights for CNN inference and must use normalization compatible with the selected dataset. | explanation | no - No AST logic-context match | N/A | DELETE | - |
| `app.py` | 523 | Dataset ZIP or extracted CSV directory | label | yes - implicit widget identity (no explicit key) | No | KEEP-LOGIC | Keep unchanged |
| `app.py` | 523 | Path to the project dataset ZIP or extracted CSV folder. Changing this selection reloads and validates the dataset. | explanation | no - No AST logic-context match | N/A | TOOLTIP | See help for the required input and its effect. |
| `app.py` | 532 | Optional checkpoint override | label | yes - implicit widget identity (no explicit key) | No | KEEP-LOGIC | Keep unchanged |
| `app.py` | 532 | Optional trained project checkpoint. When uploaded, it replaces the default checkpoint for CNN inference in this session; it does not train or modify a model. | explanation | no - No AST logic-context match | N/A | TOOLTIP | See help for the required input and its effect. |
| `app.py` | 551 | PROJECT MODULES | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 552 | 01 Dataset validation | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 553 | 02 CNN waveform prediction | label | no - No AST logic-context match | N/A | SHORTEN | CNN Prediction |
| `app.py` | 554 | 03 Inverter waveform inspection | label | no - No AST logic-context match | N/A | SHORTEN | Waveform Inspection |
| `app.py` | 555 | 04 Research diagnostics | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 557 | Seven-level multilevel inverter | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 558 | Hybrid CNN optimization research | label | no - No AST logic-context match | N/A | SHORTEN | CNN Optimization |
| `app.py` | 565 | Hybrid CNN Inverter Simulation & Harmonic Analysis | duplicate/decorative | no - No AST logic-context match | N/A | DELETE | - |
| `app.py` | 565 | INVERTER AI | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 565 | Power Electronics / AI Research | label | no - No AST logic-context match | N/A | SHORTEN | Power Electronics AI |
| `app.py` | 565 | Research and simulation environment | scope | no - No AST logic-context match | N/A | SHORTEN | Ideal software simulation, not hardware validation. |
| `app.py` | 565 | Seven-level multilevel inverter · live electrical simulation · CNN reference analysis · phase optimization | duplicate/decorative | no - No AST logic-context match | N/A | DELETE | - |
| `app.py` | 565 | ⚡ | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 587 | Loading dataset and validating project data... | status | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 613 | Loading trained CNN checkpoint... | status | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 688 | RECORD GROUP | label | yes - implicit widget identity (no explicit key) | No | KEEP-LOGIC | Keep unchanged |
| `app.py` | 702 | SAMPLE ID | label | yes - implicit widget identity (no explicit key) | No | KEEP-LOGIC | Keep unchanged |
| `app.py` | 711 | ● CNN | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 711 | ● DATASET VALIDATED | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 765 | LIVE SIMULATION WORKSPACE | label | yes - tests\test_dashboard_helpers.py:230 assertion, tab label retained; tabs must not be renamed/reordered | N/A | KEEP-LOGIC | Keep unchanged |
| `app.py` | 766 | RESEARCH WORKFLOW | label | yes - tab label retained; tabs must not be renamed/reordered | N/A | KEEP-LOGIC | Keep unchanged |
| `app.py` | 778 | OVERVIEW | label | yes - tab label retained; tabs must not be renamed/reordered | N/A | KEEP-LOGIC | Keep unchanged |
| `app.py` | 779 | CNN PREDICTION | label | yes - tab label retained; tabs must not be renamed/reordered | N/A | KEEP-LOGIC | Keep unchanged |
| `app.py` | 780 | CNN TEST EVALUATION | label | yes - tab label retained; tabs must not be renamed/reordered | N/A | KEEP-LOGIC | Keep unchanged |
| `app.py` | 781 | DATA DIAGNOSTICS | label | yes - tab label retained; tabs must not be renamed/reordered | N/A | KEEP-LOGIC | Keep unchanged |
| `app.py` | 790 | Inverter waveform monitoring | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 794 | Recorded waveform data for the selected dataset sample | duplicate/decorative | no - No AST logic-context match | N/A | DELETE | - |
| `app.py` | 839 | Selected operating point | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 876 | Record information | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 877 | Dataset split: | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 877 | One dataset waveform record | label | no - No AST logic-context match | N/A | SHORTEN | Dataset Record |
| `app.py` | 877 | SAMPLE IDENTIFIER | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 877 | WAVEFORM LENGTH | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 877 | points | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 892 | Measurement note: | explanation | no - No AST logic-context match | N/A | DELETE | - |
| `app.py` | 892 | These are waveforms from the supplied dataset. They are not a newly executed hardware experiment or a live inverter simulation. | scope | no - No AST logic-context match | N/A | DELETE | - |
| `app.py` | 892 | Project workflow | label | no - No AST logic-context match | N/A | SHORTEN | Workflow |
| `app.py` | 894 | 1 · Operating conditions | label | no - No AST logic-context match | N/A | SHORTEN | 1 Conditions |
| `app.py` | 894 | Modulation, RL load, and reference phase | explanation | no - No AST logic-context match | N/A | SHORTEN | Modulation, R, L, phase |
| `app.py` | 895 | 2 · Inverter simulation | label | no - No AST logic-context match | N/A | SHORTEN | 2 Simulation |
| `app.py` | 895 | Ideal seven-level PWM model | explanation | no - No AST logic-context match | N/A | SHORTEN | Seven-level PWM |
| `app.py` | 896 | 3 · Voltage and current | label | no - No AST logic-context match | N/A | SHORTEN | 3 Output |
| `app.py` | 896 | Sampled inverter output and RL response | explanation | no - No AST logic-context match | N/A | SHORTEN | Output voltage and RL current |
| `app.py` | 897 | 4 · CNN target prediction | label | no - No AST logic-context match | N/A | SHORTEN | 4 CNN |
| `app.py` | 897 | Reference-waveform regression only | explanation | no - No AST logic-context match | N/A | SHORTEN | Reference prediction |
| `app.py` | 898 | 5 · Prediction error | label | no - No AST logic-context match | N/A | SHORTEN | 5 Error |
| `app.py` | 898 | Compare CNN output with analytical reference | explanation | no - No AST logic-context match | N/A | SHORTEN | CNN vs. reference |
| `app.py` | 899 | 6 · PSO phase search | label | no - No AST logic-context match | N/A | SHORTEN | 6 PSO |
| `app.py` | 899 | Training-only phase selection; candidate is resimulated | explanation | no - No AST logic-context match | N/A | SHORTEN | Training-only; candidate resimulated |
| `app.py` | 900 | 7 · Harmonic analysis | label | no - No AST logic-context match | N/A | SHORTEN | 7 THD |
| `app.py` | 900 | Finite-band and Parseval THD | explanation | no - No AST logic-context match | N/A | SHORTEN | Finite-band, Parseval |
| `app.py` | 901 | 8 · Performance comparison | label | no - No AST logic-context match | N/A | SHORTEN | 8 Results |
| `app.py` | 901 | Baseline versus saved PSO validation | explanation | no - No AST logic-context match | N/A | SHORTEN | Baseline vs. PSO |
| `app.py` | 927 | The CNN predicts a reference target waveform; it does not control gates or calculate optimized inverter-output THD. Live PSO uses a fixed training-only subset, then compares baseline and candidate output THD at the same operating point; harmonic reduction is not assumed. | provenance | no - No AST logic-context match | N/A | SHORTEN | CNN predicts references; PSO candidates are simulator-verified. |
| `app.py` | 939 | CNN target waveform prediction | label | no - No AST logic-context match | N/A | SHORTEN | CNN Prediction |
| `app.py` | 943 | Model inference against the reference target for the selected sample | provenance | no - No AST logic-context match | N/A | SHORTEN | Dataset measurement. |
| `app.py` | 949 | CNN prediction is unavailable. Load the compatible trained checkpoint from the sidebar settings. | status | no - No AST logic-context match | N/A | SHORTEN | CNN unavailable. Load a compatible checkpoint. |
| `app.py` | 954 | Running CNN inference... | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 979 | MEAN ABSOLUTE ERROR | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 980 | ROOT MEAN SQUARED ERROR | label | no - No AST logic-context match | N/A | SHORTEN | ROOT MEAN SQUARED |
| `app.py` | 981 | R² SCORE | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 982 | CORRELATION | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 1004 | Detailed model metrics | label | yes - implicit widget identity (no explicit key) | No | KEEP-LOGIC | Keep unchanged |
| `app.py` | 1032 | Download prediction results (CSV) | label | yes - implicit widget identity (no explicit key) | No | KEEP-LOGIC | Keep unchanged |
| `app.py` | 1039 | Interpretation: | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 1039 | These metrics evaluate CNN prediction of the reference target waveform. They are not inverter-output THD, harmonic-reduction percentages or proof of physical performance. | explanation | no - No AST logic-context match | N/A | DELETE | - |
| `app.py` | 1055 | Held-out CNN test evaluation | label | no - No AST logic-context match | N/A | SHORTEN | CNN Test Evaluation |
| `app.py` | 1059 | Batch inference on testing records only; evaluation starts only after explicit confirmation | duplicate/decorative | no - No AST logic-context match | N/A | DELETE | - |
| `app.py` | 1063 | The CNN predicts the reference target voltage waveform from inverter voltage, load current, and operating conditions. It does not predict switching gates or optimized inverter-output THD. | explanation | no - No AST logic-context match | N/A | DELETE | - |
| `app.py` | 1088 | Evaluation complete · cached for this dataset/checkpoint in this session | status | no - No AST logic-context match | N/A | SHORTEN | Evaluation cached for this dataset/checkpoint. |
| `app.py` | 1090 | Ready · full testing split has not been evaluated in this session | status | no - No AST logic-context match | N/A | SHORTEN | Test split not evaluated. |
| `app.py` | 1092 | TEST-SET RECORDS | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 1094 | Run full held-out test evaluation | label | no - No AST logic-context match | Yes | SHORTEN | Run Test Evaluation |
| `app.py` | 1101 | Load a compatible trained checkpoint before starting evaluation. | status | no - No AST logic-context match | N/A | SHORTEN | Load a compatible checkpoint. |
| `app.py` | 1103 | The prepared dataset contains no records explicitly labeled testing. | status | no - No AST logic-context match | N/A | SHORTEN | No testing records in dataset. |
| `app.py` | 1111 | Evaluated <dynamic> of <dynamic> testing records | status | no - No AST logic-context match | N/A | SHORTEN | Evaluated {completed:,}/{total:,} records |
| `app.py` | 1127 | Completed inference for <dynamic> testing records | status | no - No AST logic-context match | N/A | SHORTEN | Evaluated {record_count:,} records |
| `app.py` | 1131 | Test-set evaluation failed: <dynamic> | status | no - No AST logic-context match | N/A | SHORTEN | Test evaluation failed: <dynamic> |
| `app.py` | 1134 | Aggregate test metrics | label | no - No AST logic-context match | N/A | SHORTEN | Test Metrics |
| `app.py` | 1135 | Micro = all concatenated waveform points. Macro = arithmetic average of per-record metrics; macro R²/correlation average only defined records. | duplicate/decorative | no - No AST logic-context match | N/A | DELETE | - |
| `app.py` | 1140 | MICRO MAE | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 1141 | MICRO RMSE | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 1143 | GLOBAL MICRO R² | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 1144 | MICRO CORRELATION | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 1149 | Additional aggregate metrics | label | yes - implicit widget identity (no explicit key) | No | KEEP-LOGIC | Keep unchanged |
| `app.py` | 1151 | MICRO MSE | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 1152 | MAXIMUM ABSOLUTE ERROR | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 1156 | MACRO MAE | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 1157 | MACRO RMSE | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 1178 | Inspect a held-out test record | label | no - No AST logic-context match | N/A | SHORTEN | Test Record |
| `app.py` | 1180 | TEST SAMPLE ID | label | no - No AST logic-context match | Yes | KEEP | Keep unchanged |
| `app.py` | 1192 | Actual target | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 1193 | CNN prediction | provenance | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 1195 | Actual vs predicted reference waveform \| Sample <dynamic> | label | no - No AST logic-context match | N/A | SHORTEN | Reference · Sample <dynamic> |
| `app.py` | 1202 | Prediction error | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 1203 | Point-wise prediction error \| Sample <dynamic> | label | no - No AST logic-context match | N/A | SHORTEN | Prediction Error · Sample <dynamic> |
| `app.py` | 1210 | Sample <dynamic> · MAE <dynamic> V · RMSE <dynamic> V · R² <dynamic> · max \|error\| <dynamic> V | status | no - No AST logic-context match | N/A | KEEP | Keep unchanged; measured values |
| `app.py` | 1226 | Pooled test-set prediction error distribution | label | no - No AST logic-context match | N/A | SHORTEN | Test Error Distribution |
| `app.py` | 1231 | Prediction error (V) | label | no - No AST logic-context match | N/A | SHORTEN | Error (V) |
| `app.py` | 1232 | Waveform points | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 1242 | Pooled error (prediction − target): mean <dynamic> V, standard deviation <dynamic> V, 1st–99th percentile [<dynamic>, <dynamic>] V. | status | no - No AST logic-context match | N/A | KEEP | Keep unchanged; measured values |
| `app.py` | 1249 | Per-sample test results | label | no - No AST logic-context match | N/A | SHORTEN | Test Results |
| `app.py` | 1268 | Download per-sample metrics (CSV) | label | yes - implicit widget identity (no explicit key) | No | KEEP-LOGIC | Keep unchanged |
| `app.py` | 1274 | Download aggregate evaluation summary (JSON) | label | yes - implicit widget identity (no explicit key) | No | KEEP-LOGIC | Keep unchanged |
| `app.py` | 1280 | These results measure prediction of the reference target waveform. They are not switching-gate predictions, inverter-output THD, proof of harmonic reduction, or physical validation. | provenance | no - No AST logic-context match | N/A | SHORTEN | CNN prediction · reference voltage only |
| `app.py` | 1296 | Dataset and model diagnostics | label | no - No AST logic-context match | N/A | SHORTEN | Model Diagnostics |
| `app.py` | 1300 | Technical metadata and dataset composition | label | no - No AST logic-context match | N/A | SHORTEN | Dataset Metadata |
| `app.py` | 1306 | TOTAL RECORDS | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 1307 | WAVEFORM POINTS | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 1308 | OPERATING FEATURES | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 1309 | COMPUTE DEVICE | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 1311 | Dataset partition | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 1326 | Dataset configuration | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 1338 | No configuration metadata is available. | label | no - No AST logic-context match | N/A | SHORTEN | No configuration metadata |
| `app.py` | 1340 | Operating-condition features | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 1350 | Model status | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 1391 | CNN architecture | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 1406 | Inputs: normalized inverter output voltage, normalized RL load current, and normalized modulation index, resistance, inductance, and phase. | duplicate/decorative | no - No AST logic-context match | N/A | DELETE | - |
| `app.py` | 1444 | This is the saved training/validation history for the default checkpoint. The held-out test split is not used for checkpoint selection. | duplicate/decorative | no - No AST logic-context match | N/A | DELETE | - |
| `app.py` | 1449 | Checkpoint-selected epoch: <dynamic> (zero-based); stored best validation loss: <dynamic>. | status | no - No AST logic-context match | N/A | KEEP | Keep unchanged; preserves measured values |
| `app.py` | 1455 | Training history is not shown for an uploaded checkpoint because no matching history artifact was supplied. | status | no - No AST logic-context match | N/A | SHORTEN | No matching training history for uploaded checkpoint. |
| `app.py` | 1465 | Live ideal-inverter simulation | label | no - No AST logic-context match | N/A | SHORTEN | Live Simulation |
| `app.py` | 1469 | Run the project simulator, trained CNN inference, and event-based harmonic analysis for one operating point | explanation | no - No AST logic-context match | N/A | DELETE | - |
| `app.py` | 1473 | Quick operating presets | label | no - No AST logic-context match | N/A | SHORTEN | Presets |
| `app.py` | 1538 | Selected preset: <dynamic> | status | no - No AST logic-context match | N/A | SHORTEN | Preset: <dynamic> |
| `app.py` | 1543 | MODULATION INDEX | label | yes - tests\test_dashboard_helpers.py:416 comparison, tests\test_dashboard_helpers.py:419 comparison, tests\test_dashboard_helpers.py:416 assertion | Yes | KEEP-LOGIC | Keep unchanged |
| `app.py` | 1552 | DC SOURCE PER CELL (V) | label | no - No AST logic-context match | Yes | SHORTEN | DC/Cell (V) |
| `app.py` | 1552 | The saved CNN was trained with 100 V per cell; other values are out-of-training-support for CNN inference. | explanation | no - No AST logic-context match | Yes | SHORTEN | CNN training used 100 V/cell; other voltages are out of support. |
| `app.py` | 1562 | LOAD RESISTANCE (Ω) | label | no - No AST logic-context match | Yes | KEEP | Keep unchanged |
| `app.py` | 1569 | LOAD INDUCTANCE (H) | label | no - No AST logic-context match | Yes | KEEP | Keep unchanged |
| `app.py` | 1576 | PHASE ANGLE (°) | label | no - No AST logic-context match | Yes | KEEP | Keep unchanged |
| `app.py` | 1586 | CARRIER 1 SHIFT (°) | label | no - No AST logic-context match | Yes | SHORTEN | C1 Shift (°) |
| `app.py` | 1595 | CARRIER 2 SHIFT (°) | label | no - No AST logic-context match | Yes | SHORTEN | C2 Shift (°) |
| `app.py` | 1604 | CARRIER 3 SHIFT (°) | label | no - No AST logic-context match | Yes | SHORTEN | C3 Shift (°) |
| `app.py` | 1612 | Run New Simulation | label | yes - tests\test_dashboard_helpers.py:231 comparison, tests\test_dashboard_helpers.py:438 comparison, implicit widget identity (no explicit key) | No | KEEP-LOGIC | Keep unchanged |
| `app.py` | 1655 | The selected live operating point is outside marginal training support. Treat CNN results as extrapolative: <dynamic> | status | no - No AST logic-context match | N/A | SHORTEN | CNN extrapolation risk: <dynamic> |
| `app.py` | 1659 | Selected inputs fall within observed marginal training support. This check does not guarantee that their joint combination is represented. | status | no - No AST logic-context match | N/A | SHORTEN | Within marginal support; joint support unknown. |
| `app.py` | 1666 | The inverter and reference use the selected DC voltage, but the saved CNN was trained at 100 V per cell. Treat its prediction as out-of-distribution. | status | no - No AST logic-context match | N/A | SHORTEN | CNN trained at 100 V/cell; other voltages are out of support. |
| `app.py` | 1669 | OFFLINE SWITCHING-PHASE RECOMMENDER | label | yes - implicit widget identity (no explicit key) | No | KEEP-LOGIC | Keep unchanged |
| `app.py` | 1670 | This is a deterministic, local Hybrid POA-PFO optimizer, not a chatbot or a trained AI switch controller. Enter conditions and select Generate; the inverter simulator then creates and verifies the gate states and THD. No network or external AI service is used. | status | no - No AST logic-context match | N/A | SHORTEN | Local POA–PFO; gates and THD are simulator-verified. |
| `app.py` | 1676 | Use this panel for a standalone recommendation. To compare optimizers against the live simulation, first run a simulation above and then use LIVE-INPUT-CONDITIONED PHASE OPTIMIZATION below. THD is reported only after simulator verification; zero modulation is rejected because its fundamental is zero. | explanation | no - No AST logic-context match | N/A | DELETE | - |
| `app.py` | 1684 | ADVISOR MODULATION | label | no - No AST logic-context match | Yes | KEEP | Keep unchanged |
| `app.py` | 1693 | REFERENCE PHASE (DEGREES) | label | no - No AST logic-context match | Yes | SHORTEN | Phase (°) |
| `app.py` | 1701 | DC VOLTAGE PER CELL (V) | label | no - No AST logic-context match | Yes | SHORTEN | DC/Cell (V) |
| `app.py` | 1708 | OPTIMIZATION OBJECTIVE | label | no - No AST logic-context match | Yes | KEEP | Keep unchanged |
| `app.py` | 1710 | current | label | yes - assistant\local_chat.py:177 comparison, assistant\switching_advisor.py:35 subscript key, assistant\switching_advisor.py:254 comparison | N/A | KEEP-LOGIC | Keep unchanged |
| `app.py` | 1710 | voltage | label | yes - app.py:1791 comparison, assistant\local_chat.py:175 comparison, assistant\switching_advisor.py:35 subscript key | N/A | KEEP-LOGIC | Keep unchanged |
| `app.py` | 1715 | LOAD RESISTANCE (Ω) | label | no - No AST logic-context match | Yes | KEEP | Keep unchanged |
| `app.py` | 1722 | LOAD INDUCTANCE (H) | label | no - No AST logic-context match | Yes | KEEP | Keep unchanged |
| `app.py` | 1730 | RANDOM SEED | label | no - No AST logic-context match | Yes | KEEP | Keep unchanged |
| `app.py` | 1738 | GENERATE VERIFIED RECOMMENDATION | label | yes - tests\test_dashboard_helpers.py:380 comparison | Yes | KEEP-LOGIC | Keep unchanged |
| `app.py` | 1756 | Optimizing carrier phases and verifying simulator gates and THD... | status | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 1773 | Switching recommendation failed: <type>: <dynamic> | status | no - No AST logic-context match | N/A | SHORTEN | Recommendation failed: <type>: <dynamic> |
| `app.py` | 1785 | Simulator-generated recommended carrier offsets: <dynamic> | status | no - No AST logic-context match | N/A | SHORTEN | Carrier offsets: <dynamic> |
| `app.py` | 1792 | BASELINE EVENT TOTAL VOLTAGE THD | label | yes - tests\test_dashboard_helpers.py:386 assertion | N/A | KEEP-LOGIC | Keep unchanged |
| `app.py` | 1794 | SIMULATOR-VERIFIED EVENT TOTAL VOLTAGE THD | label | yes - tests\test_dashboard_helpers.py:387 assertion | N/A | KEEP-LOGIC | Keep unchanged |
| `app.py` | 1797 | BASELINE EVENT CURRENT H2-100 THD | label | no - No AST logic-context match | N/A | SHORTEN | Base Event I-H2–100 |
| `app.py` | 1799 | SIMULATOR-VERIFIED EVENT CURRENT H2-100 THD | label | no - No AST logic-context match | N/A | SHORTEN | Verified Event I-H2–100 |
| `app.py` | 1810, 2299 | CHANGE FROM BASELINE | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 1815 | GATE ARRAY SHAPE | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 1823 | Sampled voltage THD: <dynamic>% · event voltage H2–100 THD: <dynamic>% · event current H2–100 THD: <dynamic>% · sampled current THD: <dynamic>% | status | no - No AST logic-context match | N/A | SHORTEN | Sample V THD: <dynamic>% · Event V H2–100: <dynamic>% · Event I H2–100: <dynamic>% · Sample I THD: <dynamic>% |
| `app.py` | 1833 | When voltage THD is selected, the objective is event-based Parseval total THD, including out-of-band switching energy. Event H2–100 values cover only those harmonic orders; sampled H2–100 values use the 20 kHz waveform. These finite-band and full-spectrum measurements are not interchangeable. | explanation | yes - tests\test_dashboard_helpers.py:396 assertion | N/A | KEEP-LOGIC | Keep unchanged |
| `app.py` | 1840 | Gate order per cell is [S1, S2, S3, S4]; rows below are cell-major, switch-major across 400 simulator samples. Binary gates and no shoot-through were verified. | status | no - No AST logic-context match | N/A | SHORTEN | Verified binary gates; no shoot-through. |
| `app.py` | 1849 | DOWNLOAD VERIFIED GATE ARRAY (CSV) | label | no - No AST logic-context match | Yes | SHORTEN | Gate CSV |
| `app.py` | 1856 | Seed <dynamic> · <dynamic> unique optimizer evaluations · <dynamic> objective requests · <dynamic> s. Predicted THD: not used. | status | no - No AST logic-context match | N/A | KEEP | Keep unchanged; includes measured values |
| `app.py` | 1864 | VALIDATED THD EXPERIMENT RESULTS | label | yes - implicit widget identity (no explicit key) | No | KEEP-LOGIC | Keep unchanged |
| `app.py` | 1865 | Primary before/after metric: event-based Parseval total voltage THD. Event-based voltage/current harmonics 2–100 and sampled 20 kHz voltage/current THD are separate measurements, not interchangeable. | explanation | yes - tests\test_local_chat.py:88 assertion | N/A | KEEP-LOGIC | Keep unchanged |
| `app.py` | 1872 | Experiment results CSV not found: <dynamic> | status | no - No AST logic-context match | N/A | KEEP | Keep unchanged; includes measured values |
| `app.py` | 1877 | Could not load THD experiment results: <dynamic>: <dynamic> | status | no - No AST logic-context match | N/A | KEEP | Keep unchanged; includes measured values |
| `app.py` | 1910 | Mean before/after results by method | label | no - No AST logic-context match | N/A | SHORTEN | Method Comparison |
| `app.py` | 1950, 1965 | All | label | yes - app.py:694 comparison, app.py:1974 comparison, app.py:1981 comparison | N/A | KEEP-LOGIC | Keep unchanged |
| `app.py` | 1959 | MODULATION | label | no - No AST logic-context match | Yes | KEEP | Keep unchanged |
| `app.py` | 1968 | METHOD | label | no - No AST logic-context match | Yes | KEEP | Keep unchanged |
| `app.py` | 1986 | Per-seed before/after measurements (phase offsets in degrees) | explanation | no - No AST logic-context match | N/A | DELETE | - |
| `app.py` | 2068 | Experiment figure not found: <dynamic> | status | no - No AST logic-context match | N/A | SHORTEN | Figure unavailable: <dynamic> |
| `app.py` | 2071 | AI PROJECT CHAT ASSISTANT (LOCAL MODEL) | label | yes - implicit widget identity (no explicit key) | No | KEEP-LOGIC | Keep unchanged |
| `app.py` | 2072 | Ask about the inverter, saved measurements, the CNN, or request a switching recommendation. Chat runs through a local Ollama model; phases and gate states are generated only by the existing optimizer and simulator. No cloud API is used. | explanation | no - No AST logic-context match | N/A | DELETE | - |
| `app.py` | 2078 | LOCAL CHAT MODEL | label | no - No AST logic-context match | Yes | KEEP | Keep unchanged |
| `app.py` | 2078 | Ollama model tag, for example qwen2.5:3b. Install it locally with `ollama pull qwen2.5:3b`. | explanation | no - No AST logic-context match | Yes | TOOLTIP | See help for the required input and its effect. |
| `app.py` | 2093 | CHECK LOCAL MODEL | label | no - No AST logic-context match | Yes | KEEP | Keep unchanged |
| `app.py` | 2101 | Local model is available: <dynamic> | status | no - No AST logic-context match | N/A | KEEP | Keep unchanged; includes measured values |
| `app.py` | 2103 | Model <dynamic> is not installed. Run `ollama pull <dynamic>`. | status | no - No AST logic-context match | N/A | KEEP | Keep unchanged; includes measured values |
| `app.py` | 2109 | CLEAR CHAT | label | no - No AST logic-context match | Yes | KEEP | Keep unchanged |
| `app.py` | 2254 | Ask about the project, CNN, THD, or request a verified phase/gate recommendation | explanation | no - No AST logic-context match | Yes | DELETE | - |
| `app.py` | 2261 | Thinking with the local model... | label | no - No AST logic-context match | N/A | SHORTEN | Thinking... |
| `app.py` | 2282 | Latest simulator-verified recommendation | explanation | no - No AST logic-context match | N/A | DELETE | - |
| `app.py` | 2291 | BASELINE <dynamic> THD | THD tag | no - No AST logic-context match | N/A | SHORTEN | Sampled THD (2-100) |
| `app.py` | 2295 | SIMULATOR-VERIFIED <dynamic> THD | THD tag | no - No AST logic-context match | N/A | SHORTEN | Sampled THD (2-100) |
| `app.py` | 2304 | Gate states were generated by the inverter simulator and validated for binary values, allowed cell states, and no shoot-through. CNN-predicted THD is not used. | provenance | no - No AST logic-context match | N/A | SHORTEN | CNN prediction; simulator-verified output. |
| `app.py` | 2322 | DOWNLOAD CHAT RECOMMENDATION GATES (CSV) | label | no - No AST logic-context match | Yes | SHORTEN | DOWNLOAD CHAT RECOMMENDATION |
| `app.py` | 2332 | RESET LIVE DEMONSTRATION | label | no - No AST logic-context match | Yes | KEEP | Keep unchanged |
| `app.py` | 2349 | Live CNN inference is unavailable because no compatible checkpoint is loaded. | status | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 2353 | CNN checkpoint: <dynamic> · Compute device: <dynamic> | duplicate/decorative | no - No AST logic-context match | N/A | DELETE | - |
| `app.py` | 2366 | Simulating inverter, evaluating harmonics, and running CNN inference... | duplicate/decorative | no - No AST logic-context match | N/A | DELETE | - |
| `app.py` | 2404 | No simulation run yet. Use Run New Simulation to create the analytical inverter output, load current and switching view. | status | no - No AST logic-context match | N/A | SHORTEN | Run New Simulation to compare measurements. |
| `app.py` | 2417 | Controls changed since this run. Click Run New Simulation to update. | duplicate/decorative | yes - tests\test_dashboard_helpers.py:433 comparison, tests\test_dashboard_helpers.py:447 comparison, tests\test_dashboard_helpers.py:433 assertion | N/A | KEEP-LOGIC | Keep unchanged |
| `app.py` | 2419 | MEASURED WAVEFORM VALIDATION (CSV) | label | yes - implicit widget identity (no explicit key) | No | KEEP-LOGIC | Keep unchanged |
| `app.py` | 2420 | Upload a measured inverter output-voltage CSV to compare its sampled harmonic spectrum with the current simulation. This is a read-only analysis: the upload is not added to the dataset, does not train or change the CNN, and is not saved by this tool. | explanation | no - No AST logic-context match | N/A | DELETE | - |
| `app.py` | 2426 | CSV format: numeric `time_s` and `voltage_v` columns, time in seconds and voltage in volts. Samples must be uniformly spaced and cover at least two complete cycles. The simulator currently uses a <dynamic> Hz fundamental; comparison uses the same H2–N harmonic band for both signals, limited by the lower Nyquist frequency. The FFT comparison is phase-independent; no time-aligned waveform error is claimed. | status | no - No AST logic-context match | N/A | KEEP | Keep unchanged; includes measured values |
| `app.py` | 2436 | Measured voltage CSV | label | no - No AST logic-context match | Yes | KEEP | Keep unchanged |
| `app.py` | 2441 | ANALYZE MEASUREMENT | label | yes - implicit widget identity (no explicit key) | No | KEEP-LOGIC | Keep unchanged |
| `app.py` | 2450 | Run a simulation with the currently selected settings before comparing a measurement. | status | no - No AST logic-context match | N/A | SHORTEN | Run a simulation before comparing measurements. |
| `app.py` | 2476 | Could not analyze measured waveform: <dynamic>: <dynamic> | status | no - No AST logic-context match | N/A | KEEP | Keep unchanged; includes measured values |
| `app.py` | 2506 | MEASURED SAMPLED VOLTAGE THD (H2–<dynamic>) | THD tag | no - No AST logic-context match | N/A | SHORTEN | Sampled THD (2-100) |
| `app.py` | 2510 | SIMULATED SAMPLED VOLTAGE THD (H2–<dynamic>) | THD tag | no - No AST logic-context match | N/A | SHORTEN | Sampled THD (2-100) |
| `app.py` | 2514 | MEASURED − SIMULATED THD | THD tag | no - No AST logic-context match | N/A | SHORTEN | Sampled THD (2-100) |
| `app.py` | 2519 | MEASURED FUNDAMENTAL AMPLITUDE | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 2523 | SIMULATED FUNDAMENTAL AMPLITUDE | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 2527 | FUNDAMENTAL AMPLITUDE CHANGE | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 2531 | <dynamic> analyzed samples · <dynamic> complete cycles · <dynamic> samples/s · measured mean/DC offset <dynamic> V (simulated <dynamic> V). THD is a sampled FFT metric for both waveforms; it is not the event-based full-spectrum optimization objective. | status | no - No AST logic-context match | N/A | KEEP | Keep unchanged; includes measured values |
| `app.py` | 2565 | LIVE-INPUT-CONDITIONED PHASE OPTIMIZATION | label | yes - implicit widget identity (no explicit key) | No | KEEP-LOGIC | Keep unchanged |
| `app.py` | 2575 | Existing PSO | label | yes - app.py:2766 dict key, selector option value; changing alters selected value | N/A | KEEP-LOGIC | Keep unchanged |
| `app.py` | 2576 | Pelican Optimization Algorithm | label | yes - app.py:2767 dict key, selector option value; changing alters selected value | N/A | KEEP-LOGIC | Keep unchanged |
| `app.py` | 2577 | Pufferfish Optimization Algorithm | label | yes - app.py:2768 dict key, selector option value; changing alters selected value | N/A | KEEP-LOGIC | Keep unchanged |
| `app.py` | 2578 | Hybrid POA-PFO | label | yes - app.py:2769 dict key, selector option value; changing alters selected value | N/A | KEEP-LOGIC | Keep unchanged |
| `app.py` | 2582 | Workflow: run the live simulation above, choose an optimizer and budget, then select RUN LIVE OPTIMIZATION. The optimizer searches carrier phases against event-based Parseval voltage THD. The CNN predicts the reference waveform; it does not select phases or guide this live search. | explanation | yes - tests\test_dashboard_helpers.py:305 assertion | N/A | KEEP-LOGIC | Keep unchanged |
| `app.py` | 2589 | Candidates use the selected modulation, DC voltage, R, L, and reference phase. Optimization is enabled only when those controls match the displayed successful baseline and carrier shifts are [0°, 60°, 120°]. The objective is event-based Parseval total THD; sampled 20 kHz 2–100 THD is a separate measurement and may move in the opposite direction. The 0.05% fundamental-tracking constraint uses the event-integrated phasor. R and L affect current, not ideal output-voltage THD. | explanation | no - No AST logic-context match | N/A | DELETE | - |
| `app.py` | 2601 | OPTIMIZER | label | no - No AST logic-context match | Yes | KEEP | Keep unchanged |
| `app.py` | 2608 | RANDOM SEED | label | no - No AST logic-context match | Yes | KEEP | Keep unchanged |
| `app.py` | 2617 | PARTICLES | label | no - No AST logic-context match | Yes | KEEP | Keep unchanged |
| `app.py` | 2626 | ITERATIONS | label | no - No AST logic-context match | Yes | KEEP | Keep unchanged |
| `app.py` | 2670 | The separate THD surrogate is unavailable; its checkpoint is not part of the existing CNN or project dataset. | explanation | no - No AST logic-context match | N/A | DELETE | - |
| `app.py` | 2675 | Surrogate screening is limited to tested modulation indices 0.55–0.95 in 0.05 steps, reference phases in 45° steps, 100 V per cell, and the [0°, 60°, 120°] baseline. Choose a tested point or use one of the exact optimizers. | duplicate/decorative | no - No AST logic-context match | N/A | DELETE | - |
| `app.py` | 2682 | The surrogate ranks 64 bounded phase candidates and sends 8 finalists (including the baseline) to exact event-based THD and electrical-feasibility evaluation. It does not replace the optimizer or final simulator. | provenance | no - No AST logic-context match | N/A | SHORTEN | Dataset measurement. |
| `app.py` | 2689 | Live phase optimization currently requires carrier shifts [0°, 60°, 120°]. The optimizer fixes cell 1 at 0°; other carrier settings cannot be compared against the displayed baseline without changing its decision-variable representation. | duplicate/decorative | no - No AST logic-context match | N/A | DELETE | - |
| `app.py` | 2708 | RUN LIVE OPTIMIZATION | label | yes - tests\test_dashboard_helpers.py:291 comparison, tests\test_dashboard_helpers.py:349 comparison | Yes | KEEP-LOGIC | Keep unchanged |
| `app.py` | 2740 | Evaluating live-conditioned inverter candidates and optimizing carrier phases... | explanation | no - No AST logic-context match | N/A | DELETE | - |
| `app.py` | 2824 | INITIAL PHASE SHIFTS | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 2828 | <dynamic>-SELECTED PHASE SHIFTS | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 2841 | Optimization objective · event-based Parseval total voltage THD | THD tag | yes - tests\test_local_chat.py:88 assertion | N/A | KEEP-LOGIC | Keep unchanged |
| `app.py` | 2845 | BASELINE OBJECTIVE | label | yes - tests\test_dashboard_helpers.py:297 assertion | N/A | KEEP-LOGIC | Keep unchanged |
| `app.py` | 2849 | <dynamic> OBJECTIVE | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 2853 | OBJECTIVE CHANGE (CANDIDATE − BASELINE) | label | yes - tests\test_dashboard_helpers.py:300 assertion | N/A | KEEP-LOGIC | Keep unchanged |
| `app.py` | 2859 | <dynamic> reduced its event-based Parseval objective at this operating point. This single-point result is not validation-set evidence. | status | no - No AST logic-context match | N/A | KEEP | Keep unchanged; includes measured values |
| `app.py` | 2865 | <dynamic> did not improve its event-based Parseval objective at this operating point. | status | no - No AST logic-context match | N/A | KEEP | Keep unchanged; includes measured values |
| `app.py` | 2870 | <dynamic> produced no material change in its event-based Parseval objective at the 0.000001 percentage-point reporting resolution. | THD tag | no - No AST logic-context match | N/A | SHORTEN | dynamic produced no |
| `app.py` | 2875 | Secondary measurement · sampled voltage THD, 20 kHz, harmonics 2–100 | explanation | no - No AST logic-context match | N/A | DELETE | - |
| `app.py` | 2882 | BASELINE SAMPLED VOLTAGE THD (2–100) | THD tag | no - No AST logic-context match | N/A | SHORTEN | Sampled THD (2-100) |
| `app.py` | 2886 | <dynamic> SAMPLED VOLTAGE THD (2–100) | THD tag | no - No AST logic-context match | N/A | SHORTEN | Sampled THD (2-100) |
| `app.py` | 2896 | <dynamic> produced lower sampled voltage THD at this operating point. This single-point result is not validation-set evidence. | explanation | no - No AST logic-context match | N/A | DELETE | - |
| `app.py` | 2902 | Sampled voltage H2–100 THD is higher by <dynamic> percentage points. This metric is not the optimizer objective and can move in the opposite direction. | explanation | no - No AST logic-context match | N/A | DELETE | - |
| `app.py` | 2909 | Sampled voltage H2–100 THD did not materially change at the 0.001 percentage-point reporting resolution. | status | no - No AST logic-context match | N/A | KEEP | Keep unchanged; includes measured values |
| `app.py` | 2914 | BASELINE FUNDAMENTAL TRACKING ERROR | label | no - No AST logic-context match | N/A | SHORTEN | Fundamental Error (%) |
| `app.py` | 2918 | <dynamic> FUNDAMENTAL TRACKING ERROR | label | no - No AST logic-context match | N/A | SHORTEN | Fundamental Error (%) |
| `app.py` | 2922 | CURRENT THD BASELINE / <dynamic> (2–100) | THD tag | no - No AST logic-context match | N/A | SHORTEN | Current THD (2-100) |
| `app.py` | 2974 | CNN-GUIDED HYBRID OPTIMIZATION | label | yes - implicit widget identity (no explicit key) | No | KEEP-LOGIC | Keep unchanged |
| `app.py` | 2975 | This separate path uses the loaded CNN only to rank 64 POA/PFO proposals; the top 8 are evaluated with event-based THD and the winner is re-simulated. CNN scores are ranking proxies, not THD measurements. The available correlation evidence (Spearman ρ = 0.954 over 500 candidates) applies only to its recorded operating point. | provenance | no - No AST logic-context match | N/A | SHORTEN | CNN prediction; simulator-verified output. |
| `app.py` | 3032 | CNN guidance is disabled because its saved ranking-correlation evidence does not meet the recorded usability threshold. | status | no - No AST logic-context match | N/A | SHORTEN | Operation status. |
| `app.py` | 3037 | CNN-guided optimization is enabled only at the validated operating point (m=0.75, 100 V/cell, 30 Ω, 0.03 H, phase 0°) with baseline carrier shifts [0°, 60°, 120°]. Other live conditions remain available to the simulator-based optimizers. | explanation | no - No AST logic-context match | N/A | DELETE | - |
| `app.py` | 3044 | Validated conditions and baseline shifts match. Recorded CNN-score/event-THD rank correlation: <dynamic>. | explanation | no - No AST logic-context match | N/A | DELETE | - |
| `app.py` | 3049 | CNN-GUIDED RANDOM SEED | label | no - No AST logic-context match | Yes | KEEP | Keep unchanged |
| `app.py` | 3077 | RUN CNN-GUIDED HYBRID | label | yes - tests\test_dashboard_helpers.py:312 comparison, tests\test_dashboard_helpers.py:427 comparison | Yes | KEEP-LOGIC | Keep unchanged |
| `app.py` | 3103 | Ranking POA/PFO proposals with the CNN and verifying the finalists with the inverter simulator... | explanation | no - No AST logic-context match | N/A | DELETE | - |
| `app.py` | 3201 | BASELINE EVENT-BASED PARSEVAL THD | THD tag | no - No AST logic-context match | N/A | SHORTEN | Parseval THD (2-infinity) |
| `app.py` | 3205 | CNN-GUIDED EVENT-BASED PARSEVAL THD | THD tag | yes - tests\test_dashboard_helpers.py:370 assertion | N/A | KEEP-LOGIC | Keep unchanged |
| `app.py` | 2890, 3209 | CANDIDATE − BASELINE | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 3215 | CNN-GUIDED SELECTED PHASE SHIFTS | label | no - No AST logic-context match | N/A | SHORTEN | CNN-GUIDED PHASE SHIFTS |
| `app.py` | 3219 | CNN-GUIDED FUNDAMENTAL TRACKING ERROR | label | yes - tests\test_dashboard_helpers.py:374 assertion | N/A | KEEP-LOGIC | Keep unchanged |
| `app.py` | 3223 | Selected shifts [0°, <dynamic>°, <dynamic>°] · seed <dynamic> · <dynamic> unique CNN-ranked proposals · 8 event-THD finalists plus independent winner re-evaluation · <dynamic> s. Final event-based THD and feasibility are independently re-simulated before display. | status | no - No AST logic-context match | N/A | KEEP | Keep unchanged; includes measured values |
| `app.py` | 3234 | BASELINE SAMPLED VOLTAGE THD (20 kHz, H2–100) | THD tag | no - No AST logic-context match | N/A | SHORTEN | Sampled THD (2-100) |
| `app.py` | 3238 | CANDIDATE SAMPLED VOLTAGE THD (20 kHz, H2–100) | THD tag | no - No AST logic-context match | N/A | SHORTEN | Sampled THD (2-100) |
| `app.py` | 3243 | The latest simulation attempt failed; displaying the last successful simulation. | status | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 3248 | Simulation completed for the currently selected operating point. | status | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 3252 | Displaying the last successful simulation using the operating conditions listed below. | status | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 3256 | #### Operating point · m = | duplicate/decorative | no - No AST logic-context match | N/A | DELETE | - |
| `app.py` | 3256 | H · phase = | label | no - No AST logic-context match | N/A | SHORTEN | H phase |
| `app.py` | 3256 | V · R = | label | no - No AST logic-context match | N/A | SHORTEN | V R |
| `app.py` | 3256 | ° | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 3256 | · Vdc/cell = | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 3256 | Ω · L = | label | no - No AST logic-context match | N/A | SHORTEN | L |
| `app.py` | 3273 | START | label | no - No AST logic-context match | Yes | KEEP | Keep unchanged |
| `app.py` | 3275 | PAUSE | label | no - No AST logic-context match | Yes | KEEP | Keep unchanged |
| `app.py` | 3277 | RESUME | label | no - No AST logic-context match | Yes | KEEP | Keep unchanged |
| `app.py` | 3279 | STOP | label | no - No AST logic-context match | Yes | KEEP | Keep unchanged |
| `app.py` | 3284 | PLAYBACK RATE | label | no - No AST logic-context match | Yes | KEEP | Keep unchanged |
| `app.py` | 3303 | Sample index | label | yes - tests\test_dashboard_helpers.py:453 comparison | Yes | KEEP-LOGIC | Keep unchanged |
| `app.py` | 3314 | SOFTWARE MONITOR STATUS | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 3315 | SIMULATION TIME | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 3316 | INSTANT OUTPUT | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 3320 | INSTANT LOAD CURRENT | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 3325 | Software playback · sample <dynamic>/<dynamic> | status | no - No AST logic-context match | N/A | SHORTEN | Playback <dynamic>/<dynamic> |
| `app.py` | 3327 | Accelerated replay of this precomputed, repeating 20 ms cycle; not a live hardware signal. | status | yes - tests\test_dashboard_helpers.py:249 assertion | N/A | KEEP-LOGIC | Keep unchanged |
| `app.py` | 3439 | Analytical simulation (inverter simulator) — instant switching snapshot | provenance | no - No AST logic-context match | N/A | KEEP | Keep concise provenance |
| `app.py` | 3449 | V1 + V2 + V3 | label | no - No AST logic-context match | N/A | SHORTEN | V1 V2 V3 |
| `app.py` | 3450 | Simulator output voltage | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 3454 | Sum check | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 3463 | PARSEVAL FULL-SPECTRUM THD | THD tag | no - No AST logic-context match | N/A | SHORTEN | Parseval THD (2-infinity) |
| `app.py` | 3463 | Uses exact interval mean-square voltage and AC energy after subtracting the fundamental. It includes all spectral energy. | explanation | no - No AST logic-context match | N/A | TOOLTIP | See help for the required input and its effect. |
| `app.py` | 3471 | Analytic event-based Fourier harmonics 2 through 100, divided by the event-based fundamental amplitude. | explanation | no - No AST logic-context match | N/A | TOOLTIP | See help for the required input and its effect. |
| `app.py` | 3471 | EVENT THD (HARMONICS 2–100) | THD tag | no - No AST logic-context match | N/A | SHORTEN | Event THD (2-100) |
| `app.py` | 3479 | 20 kHz SAMPLED THD (2–100) | THD tag | no - No AST logic-context match | N/A | SHORTEN | Sampled THD (2-100) |
| `app.py` | 3479 | Finite-band FFT estimate calculated from this simulator's 20 kHz output-voltage samples, using orders 2 through 100. | explanation | no - No AST logic-context match | N/A | TOOLTIP | See help for the required input and its effect. |
| `app.py` | 3487 | FUNDAMENTAL PEAK | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 3487 | Peak amplitude of the event-integrated 50 Hz output-voltage fundamental. | explanation | no - No AST logic-context match | N/A | TOOLTIP | See help for the required input and its effect. |
| `app.py` | 3492 | CNN TARGET RMSE | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 3492 | RMSE between the CNN prediction and analytical reference target; not output THD. | explanation | no - No AST logic-context match | N/A | TOOLTIP | See help for the required input and its effect. |
| `app.py` | 3497 | 20 kHz sampled finite-band load-current THD, orders 2–100. | THD tag | no - No AST logic-context match | N/A | SHORTEN | Current THD (2-100) |
| `app.py` | 3497 | LOAD CURRENT THD (2–100) | THD tag | no - No AST logic-context match | N/A | SHORTEN | Current THD (2-100) |
| `app.py` | 3502 | The event finite-band THD and sampled finite-band THD both cover orders 2–100 but use different waveform representations. Parseval full-spectrum THD is a separate all-energy measure; these values are not interchangeable. | explanation | no - No AST logic-context match | N/A | DELETE | - |
| `app.py` | 3508 | RECORDED HELD-OUT CNN CHECK | label | yes - implicit widget identity (no explicit key) | No | KEEP-LOGIC | Keep unchanged |
| `app.py` | 3521 | The saved held-out evaluation is for the bundled checkpoint and dataset only. Run CNN TEST EVALUATION to assess the currently selected model/data. | duplicate/decorative | no - No AST logic-context match | N/A | DELETE | - |
| `app.py` | 3596 | CNN HELD-OUT POOLED RMSE | label | yes - tests\test_dashboard_helpers.py:267 subscript key, tests\test_dashboard_helpers.py:267 assertion | N/A | KEEP-LOGIC | Keep unchanged |
| `app.py` | 3600 | ANALYTICAL HELD-OUT POOLED RMSE | label | yes - tests\test_dashboard_helpers.py:271 subscript key, tests\test_dashboard_helpers.py:271 assertion | N/A | KEEP-LOGIC | Keep unchanged |
| `app.py` | 3604 | TEST RECORDS | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 3608 | On the recorded held-out test set, the CNN prediction was less accurate than the analytical reference. These are waveform-prediction errors, not THD or optimizer measurements. Re-run CNN TEST EVALUATION after changing the dataset or checkpoint. | explanation | yes - tests\test_dashboard_helpers.py:276 assertion | N/A | KEEP-LOGIC | Keep unchanged |
| `app.py` | 3616 | Could not present the saved held-out CNN evaluation: <dynamic>: <dynamic> | status | no - No AST logic-context match | N/A | KEEP | Keep unchanged; includes measured values |
| `app.py` | 3622 | FUNDAMENTAL PHASOR TRACKING ERROR | label | no - No AST logic-context match | N/A | SHORTEN | Fundamental Error (%) |
| `app.py` | 3626 | FUNDAMENTAL PHASE | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 3630 | CNN TARGET MAE | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 3634 | CNN TARGET R² | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 3639 | Show CNN prediction overlay | label | yes - tests\test_dashboard_helpers.py:403 comparison, tests\test_dashboard_helpers.py:410 comparison | Yes | KEEP-LOGIC | Keep unchanged |
| `app.py` | 3682 | SIMULATED OUTPUT − REFERENCE RMS | label | yes - tests\test_dashboard_helpers.py:256 assertion | N/A | KEEP-LOGIC | Keep unchanged |
| `app.py` | 3686 | PEAK ABSOLUTE TRACKING ERROR | label | yes - tests\test_dashboard_helpers.py:260 assertion | N/A | KEEP-LOGIC | Keep unchanged |
| `app.py` | 3695 | <dynamic> OUTPUT − REFERENCE RMS | label | no - No AST logic-context match | N/A | SHORTEN | dynamic OUTPUT REFERENCE |
| `app.py` | 3700 | REFERENCE TYPE | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 3732 | The output is a seven-level PWM waveform; the reference is the desired sinusoid. Their visible difference is expected switching ripple and sampling error in this ideal model. This is separate from CNN prediction error and from THD. | explanation | no - No AST logic-context match | N/A | DELETE | - |
| `app.py` | 3875 | Spectrum amplitudes are event-integrated output-voltage peak amplitudes. The plotted orders 1–100 correspond to the analytic event-based Fourier coefficients. | duplicate/decorative | no - No AST logic-context match | N/A | DELETE | - |
| `app.py` | 3910 | Switching gates | label | no - No AST logic-context match | N/A | KEEP | Keep unchanged |
| `app.py` | 3946 | Scientific scope: | scope | no - No AST logic-context match | N/A | DELETE | - |
| `app.py` | 3946 | The inverter and RL load are an ideal, software simulation using this project's 20 kHz sampling and switching conventions. The CNN predicts the reference target waveform; it does not predict switching gates or optimized inverter-output THD. CNN prediction accuracy is not evidence of harmonic reduction. This is not hardware, MATLAB, or Simulink validation. | explanation | no - No AST logic-context match | N/A | DELETE | - |
| `app.py` | 3960 | The export currently contains the baseline simulation. Run the selected live-input-conditioned optimizer above to include the candidate and convergence history. | explanation | no - No AST logic-context match | N/A | DELETE | - |
| `app.py` | 4034 | DOWNLOAD WAVEFORM CSV | label | no - No AST logic-context match | Yes | KEEP | Keep unchanged |
| `app.py` | 4042 | DOWNLOAD SUMMARY JSON | label | no - No AST logic-context match | Yes | KEEP | Keep unchanged |
| `app.py` | 4050 | DOWNLOAD COMPLETE ZIP | label | no - No AST logic-context match | Yes | KEEP | Keep unchanged |
| `app.py` | 4058 | Simulation status: no result exists for the current inputs. | duplicate/decorative | no - No AST logic-context match | N/A | DELETE | - |
| `app.py` | 4071 | Application could not load the dataset/checkpoint: <dynamic> | status | no - No AST logic-context match | N/A | KEEP | Keep unchanged; includes measured values |
| `app.py` | 4072 | Technical error details | label | yes - implicit widget identity (no explicit key) | No | KEEP-LOGIC | Keep unchanged |

## Counts

- Inventory rows: 339 (342 call sites).
- Baseline display/widget/plot strings independently found: 307 distinct strings; no omissions against this inventory.
- kept: 125
- kept-logic: 57
- shortened: 108
- deleted: 41
- moved to tooltip: 8
- unsure-left-unchanged: 0

## Scope and exclusions

- `app.py` contains 14 `st.dataframe` calls. Dataframe column names and values are data/schema and will remain unchanged; this inventory covers direct display strings, widget labels/help, panel text, and plot labels.
- Unkeyed widget labels are marked `KEEP-LOGIC` and must remain unchanged unless a separately approved identity-preserving key migration is done.
- Phase 2 is in progress with user approval. The table retains Phase 0 text as its baseline; “New text” records the approved target, including changes already applied.
