# Held-out CNN evaluation

`cnn_test_evaluator.py` evaluates records selected only by
`PreparedDataset.indices("testing")` and `WaveformDataset(..., "testing")`.
The module uses ordered, non-shuffled batches and `torch.inference_mode()`.
Inputs and targets retain the training-fitted normalization in the prepared
dataset; model outputs and targets are denormalized with the checkpoint's
`NormalizationStats`. Invalid model output shapes or non-finite values raise
`CNNTestEvaluationError`.

The evaluation reports two populations explicitly:

- **Micro:** metric calculated after flattening all actual and predicted
  waveform points across the complete testing split. This is the aggregate
  MAE/MSE/RMSE/R²/correlation result.
- **Macro:** arithmetic mean of corresponding per-record metrics. Per-record
  MAE, MSE, RMSE and R² remain available individually. Undefined per-record
  R²/correlation values are excluded from their macro means; if no record has
  a defined value, the macro metric is `None`.

Error-distribution percentiles are pooled over all pointwise voltage errors
(`prediction - target`). The Streamlit test-evaluation section runs only after
the user clicks its evaluation button. Successful results are retained in the
current Streamlit session and invalidated when the dataset or checkpoint
fingerprint changes.

The trained CNN predicts the reference target voltage waveform from inverter
input voltage, load current, and operating conditions. It does not predict
switching gates or optimized inverter-output THD. Prediction metrics are not
proof of harmonic reduction, hardware behavior, MATLAB equivalence, or
Simulink validation.
