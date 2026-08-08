import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
import json
import time
import shutil
import torch

from cnn.controller import run_cnn_controller
from cnn.dataset import WINDOW_SIZE, generate_cnn_dataset
from cnn.train import train_cnn
from cnn.predict import predict_switching_state
from multilevel.eight_level_model import (
    EightLevelInverterParams,
    generate_8_level_waveform,
    save_waveform_csv,
    validate_waveform,
)
from signal_processing.fft import compute_fft
from signal_processing.thd import calculate_thd


def init_session_state() -> None:
    if "input_params" not in st.session_state:
        st.session_state.input_params = {
            "solar_irradiance": 800.0,
            "solar_temperature": 25.0,
            "wind_speed": 8.0,
            "battery_voltage": 540.0,
            "battery_soc": 75.0,
            "load_resistance": 10.0,
            "load_inductance": 5.0,
            "num_levels": 8,
            "dc_voltage": 100.0,
            "fundamental_frequency": 50.0,
            "switching_frequency": 2000.0,
            "sampling_frequency": 10000.0,
            "simulation_duration": 0.1,
            "modulation_index": 1.0,
        }
    if "dataset" not in st.session_state:
        st.session_state.dataset = None
    if "main_page" not in st.session_state:
        st.session_state.main_page = "Overview"
        st.session_state.dataset_info = {}
    if "generated_dataset" not in st.session_state:
        st.session_state.generated_dataset = None
    if "mat_file_info" not in st.session_state:
        st.session_state.mat_file_info = None
    if "cnn_training" not in st.session_state:
        st.session_state.cnn_training = None
    if "cnn_dataset_info" not in st.session_state:
        st.session_state.cnn_dataset_info = None
    if "cnn_result" not in st.session_state:
        st.session_state.cnn_result = None
    if "waveform" not in st.session_state:
        st.session_state.waveform = None
    if "waveform_summary" not in st.session_state:
        st.session_state.waveform_summary = None
    if "waveform_fft" not in st.session_state:
        st.session_state.waveform_fft = None
    if "waveform_thd" not in st.session_state:
        st.session_state.waveform_thd = None


def build_architecture_figure() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    labels = [
        "Solar PV",
        "Wind Energy",
        "Battery / DC Bus",
        "8-Level Multilevel Inverter",
        "CNN Controller",
        "HPFO Optimization",
        "Switching Pulses",
        "AC Output",
        "FFT Analysis",
        "THD Evaluation",
    ]
    positions = [
        (0.1, 0.9),
        (0.1, 0.75),
        (0.1, 0.6),
        (0.1, 0.45),
        (0.1, 0.3),
        (0.1, 0.15),
        (0.5, 0.15),
        (0.5, 0.0),
        (0.8, 0.0),
        (0.8, -0.15),
    ]

    for label, (x, y) in zip(labels, positions):
        ax.text(x, y, label, fontsize=12, ha="center", va="center", bbox={"boxstyle": "round", "facecolor": "#f0f0f0", "edgecolor": "gray"})

    arrows = [
        ((0.1, 0.84), (0.1, 0.77)),
        ((0.1, 0.69), (0.1, 0.62)),
        ((0.1, 0.54), (0.1, 0.37)),
        ((0.1, 0.27), (0.1, 0.18)),
        ((0.1, 0.12), (0.35, 0.12)),
        ((0.55, 0.12), (0.75, 0.02)),
        ((0.5, 0.0), (0.75, 0.0)),
        ((0.8, 0.0), (0.8, -0.1)),
    ]
    for start, end in arrows:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 1.2},
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.25, 1)
    return fig


def display_overview() -> None:
    st.set_page_config(page_title="AI Harmonic Optimizer", layout="wide")
    st.title("Hybrid Optimization and CNN-Based Control for Harmonic Reduction in Multilevel Inverters Applied to Microgrids")
    st.markdown("### Interactive 8-Level Inverter + CNN Prototype")

    # Top status cards
    cols = st.columns(4)
    cols[0].metric("8-Level Inverter", "READY")
    cols[1].metric("Waveform Analysis", "READY")
    cols[2].metric("FFT / THD", "READY")
    cols[3].metric("CNN Controller", "PROTOTYPE")

    st.markdown("---")

    # Live Simulation Control Panel
    st.header("Live 8-Level Inverter Simulation")
    left, right = st.columns([1, 2])
    with left:
        dc_voltage = st.number_input("DC Voltage (V)", value=st.session_state.input_params.get("dc_voltage", 100.0), step=1.0)
        fundamental_frequency = st.number_input("Fundamental Frequency (Hz)", value=st.session_state.input_params.get("fundamental_frequency", 50.0), step=0.1)
        sampling_frequency = st.number_input("Sampling Frequency (Hz)", value=st.session_state.input_params.get("sampling_frequency", 10000.0), step=100.0)
        simulation_duration = st.number_input("Simulation Duration (s)", value=st.session_state.input_params.get("simulation_duration", 0.1), step=0.01)
        modulation_index = st.slider("Modulation Index", min_value=0.1, max_value=1.0, value=st.session_state.input_params.get("modulation_index", 0.8), step=0.05)
        gen_button = st.button("⚡ GENERATE 8-LEVEL SIMULATION")

    # Use existing generator to create a default waveform if needed
    if st.session_state.get("waveform") is None or gen_button:
        try:
            st.session_state.input_params.update({
                "dc_voltage": float(dc_voltage),
                "fundamental_frequency": float(fundamental_frequency),
                "sampling_frequency": float(sampling_frequency),
                "simulation_duration": float(simulation_duration),
                "modulation_index": float(modulation_index),
            })
            params = EightLevelInverterParams(
                dc_voltage=st.session_state.input_params["dc_voltage"],
                fundamental_frequency=st.session_state.input_params["fundamental_frequency"],
                switching_frequency=st.session_state.input_params.get("switching_frequency", 2000.0),
                sampling_frequency=st.session_state.input_params["sampling_frequency"],
                simulation_duration=st.session_state.input_params["simulation_duration"],
                modulation_index=st.session_state.input_params["modulation_index"],
            )
            waveform = generate_8_level_waveform(params)
            summary = validate_waveform(waveform)
            frequencies, magnitudes = compute_fft(waveform.voltage, params.sampling_frequency)
            thd_value, fundamental_mag, harmonic_orders, harmonic_frequencies, harmonic_magnitudes = calculate_thd(
                waveform.voltage, params.sampling_frequency, params.fundamental_frequency
            )

            st.session_state.waveform = {
                "time": waveform.time,
                "voltage": waveform.voltage,
                "switching_state": waveform.switching_state_labels,
            }
            st.session_state.waveform_summary = summary
            st.session_state.waveform_fft = {"frequencies": frequencies, "magnitudes": magnitudes}
            st.session_state.waveform_thd = {
                "thd": thd_value,
                "fundamental_magnitude": fundamental_mag,
                "harmonic_orders": harmonic_orders,
                "harmonic_frequencies": harmonic_frequencies,
                "harmonic_magnitudes": harmonic_magnitudes,
            }
        except Exception as e:
            st.error(f"Failed to generate waveform: {e}")

    # Waveform display
    with right:
        st.subheader("8-Level Inverter Output Voltage")
        wf = st.session_state.waveform
        if wf is None:
            st.error("No waveform available. Try clicking ⚡ GENERATE 8-LEVEL SIMULATION.")
        else:
            time = wf["time"]
            voltage = wf["voltage"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=time, y=voltage, mode="lines", line=dict(shape="hv"), name="Voltage"))
            fig.update_layout(title="8-Level Inverter Output Voltage", xaxis_title="Time (s)", yaxis_title="Voltage (V)", height=420)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Levels and metrics
    st.subheader("8-Level Voltage States")
    df_levels = pd.DataFrame({"time": st.session_state.waveform["time"], "voltage": st.session_state.waveform["voltage"], "level": st.session_state.waveform["switching_state"]})
    unique_levels = sorted(np.unique(np.round(df_levels["voltage"], 8)))
    detected = len(unique_levels)
    cols = st.columns(8)
    for i in range(8):
        label = f"Level {i+1}"
        val = f"{unique_levels[i]:.2f} V" if i < detected else "-"
        cols[i].metric(label, val)
    st.write(f"Detected Levels: {detected}")

    st.subheader("Waveform Metrics")
    peak = float(np.max(df_levels["voltage"]))
    rms = float(np.sqrt(np.mean(df_levels["voltage"] ** 2)))
    minimum = float(np.min(df_levels["voltage"]))
    samples = int(len(df_levels))
    fundamental_freq = float(st.session_state.waveform_summary.get("fundamental_frequency", st.session_state.input_params.get("fundamental_frequency", 50.0)))
    mcols = st.columns(5)
    mcols[0].metric("Peak Voltage", f"{peak:.2f} V")
    mcols[1].metric("RMS Voltage", f"{rms:.2f} V")
    mcols[2].metric("Minimum Voltage", f"{minimum:.2f} V")
    mcols[3].metric("Samples", f"{samples}")
    mcols[4].metric("Frequency", f"{fundamental_freq:.2f} Hz")

    st.markdown("---")

    # FFT and THD
    st.header("Frequency Spectrum / FFT Analysis")
    fft_freqs = st.session_state.waveform_fft["frequencies"]
    fft_mags = st.session_state.waveform_fft["magnitudes"]
    fig_fft = go.Figure()
    fig_fft.add_trace(go.Scatter(x=fft_freqs, y=fft_mags, mode="lines", name="FFT"))
    fig_fft.update_layout(title="FFT Spectrum of 8-Level Inverter Output", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude", height=420)
    st.plotly_chart(fig_fft, use_container_width=True)

    st.subheader("Harmonic Distortion")
    thd = st.session_state.waveform_thd["thd"]
    fund_mag = st.session_state.waveform_thd["fundamental_magnitude"]
    harm_orders = st.session_state.waveform_thd["harmonic_orders"]
    harm_freqs = st.session_state.waveform_thd["harmonic_frequencies"]
    harm_mags = st.session_state.waveform_thd["harmonic_magnitudes"]
    # dominant harmonic
    dominant_idx = int(np.argmax(harm_mags)) if len(harm_mags) > 0 else 0
    dominant = int(harm_orders[dominant_idx]) if len(harm_orders) > 0 else 0
    dcols = st.columns(3)
    dcols[0].metric("Fundamental Magnitude", f"{fund_mag:.6f}")
    dcols[1].metric("THD", f"{thd * 100:.2f} %")
    dcols[2].metric("Dominant Harmonic", str(dominant))
    st.caption("THD shown here is calculated from the software-generated reference waveform.")

    # Harmonic table
    st.subheader("Harmonic Components")
    harmonic_df = pd.DataFrame({
        "Harmonic Order": harm_orders,
        "Frequency (Hz)": harm_freqs,
        "Magnitude": harm_mags,
    })
    if not harmonic_df.empty:
        harmonic_df["Relative (%)"] = 100.0 * harmonic_df["Magnitude"] / (harmonic_df["Magnitude"].iloc[0] if harmonic_df["Magnitude"].iloc[0] != 0 else 1)
        st.dataframe(harmonic_df.head(12))
    else:
        st.info("No harmonic components detected.")

    st.markdown("---")

    # Inverter Visualization
    st.header("8-Level Inverter Operating States")
    vis_fig = go.Figure()
    y_vals = unique_levels[::-1]
    for i, y in enumerate(y_vals):
        vis_fig.add_trace(go.Scatter(x=[0, 1], y=[y, y], mode="lines", line=dict(width=6), name=f"L{len(y_vals)-i}"))
        vis_fig.add_annotation(x=0.5, y=y, text=f"Level {len(y_vals)-i}: {y:.2f} V", showarrow=False, yshift=10)
    vis_fig.update_layout(xaxis=dict(visible=False), yaxis_title="Voltage (V)", height=420)
    st.plotly_chart(vis_fig, use_container_width=True)

    st.markdown("---")

    # Workflow diagram (Sankey)
    st.header("System Flow")
    try:
        nodes = [
            "Renewable / DC Source",
            "8-Level Inverter",
            "Switching State Generation",
            "Output Voltage",
            "FFT Analysis",
            "THD Calculation",
            "CNN Controller",
            "Future HPFO",
            "MATLAB/Simulink",
        ]
        link = dict(
            source=[0, 1, 2, 3, 4, 5, 6],
            target=[1, 2, 3, 4, 5, 6, 7],
            value=[1, 1, 1, 1, 1, 1, 1],
        )
        sankey = dict(node=dict(label=nodes, pad=15, thickness=20), link=link)
        fig_sankey = go.Figure(go.Sankey(sankey))
        fig_sankey.update_layout(height=360)
        st.plotly_chart(fig_sankey, use_container_width=True)
    except Exception:
        st.info("Workflow diagram unavailable.")

    st.markdown("---")

    # CNN Section
    st.header("CNN-Based Switching Controller")
    model_file = Path("models/cnn_8level_controller.pth")
    status = "TRAINED" if model_file.exists() else "READY FOR TRAINING"
    c1, c2 = st.columns([2, 1])
    with c1:
        st.write("Framework: PyTorch")
        st.write("Architecture: 1D CNN")
        st.write("Task: 8-Level Switching-State Prediction")
    with c2:
        st.metric("Model Status", status)
    if model_file.exists():
        history_path = Path("results/cnn_training_history.csv")
        if history_path.exists():
            try:
                hist = pd.read_csv(history_path)
                fig_h = go.Figure()
                fig_h.add_trace(go.Scatter(x=hist["epoch"], y=hist["val_accuracy"], mode="lines+markers", name="Val Accuracy"))
                fig_h.update_layout(title="Validation Accuracy", height=240)
                st.plotly_chart(fig_h, use_container_width=True)
            except Exception:
                pass
    else:
        st.info("CNN model not trained yet.")
    if st.button("Open CNN Controller"):
        st.session_state.main_page = "CNN Controller"
        st.experimental_rerun()

    # Quick actions
    st.markdown("---")
    st.subheader("Quick Actions")
    qa1, qa2, qa3, qa4 = st.columns(4)
    if qa1.button("⚡ Generate 8-Level Waveform"):
        st.session_state.main_page = "8-Level Inverter"
        st.experimental_rerun()
    if qa2.button("📊 Analyze FFT / THD"):
        st.session_state.main_page = "FFT & THD"
        st.experimental_rerun()
    if qa3.button("🧠 CNN Controller"):
        st.session_state.main_page = "CNN Controller"
        st.experimental_rerun()
    if qa4.button("📁 Dataset"):
        st.session_state.main_page = "Dataset"
        st.experimental_rerun()


def display_system_architecture() -> None:
    st.header("Proposed System Architecture")
    st.write(
        "This diagram presents the proposed processing flow for the 8-level inverter reference prototype. Future modules are shown separately for later integration."
    )
    st.pyplot(build_architecture_figure())

    st.subheader("Future Integration")
    st.markdown(
        "- CNN Controller (future)\n"
        "- HPFO Optimization (future)\n"
        "- MATLAB/Simulink Interaction (future)"
    )


def display_input_parameters() -> None:
    st.header("Input Parameters")
    st.write("Configure the system input parameters. These values are stored for UI state only.")
    # Simple controls for input parameters
    st.subheader("Simulation Defaults")
    st.write("These values are saved to UI state and used by the Overview demo panel.")
    col1, col2 = st.columns(2)
    with col1:
        st.number_input("Default DC Voltage", value=st.session_state.input_params.get("dc_voltage", 100.0), key="default_dc_voltage")
        st.number_input("Default Fundamental Frequency", value=st.session_state.input_params.get("fundamental_frequency", 50.0), key="default_fundamental_frequency")
    with col2:
        st.number_input("Default Sampling Frequency", value=st.session_state.input_params.get("sampling_frequency", 10000.0), key="default_sampling_frequency")
        st.number_input("Default Simulation Duration", value=st.session_state.input_params.get("simulation_duration", 0.1), key="default_simulation_duration")


def display_cnn_controller() -> None:
    st.header("Prototype CNN-Based 8-Level Switching Controller")
    st.markdown(
        "This page demonstrates a real PyTorch 1D CNN prototype for 8-level switching-state prediction. "
        "The model is trained on synthetically generated inverter waveform data and is not a final validated controller."
    )

    st.subheader("Model Summary")
    st.write("Model: 1D CNN")
    st.write("Task: 8-Level Switching-State Prediction")
    st.write("Classes: 8")
    st.write(f"Window Size: {WINDOW_SIZE}")

    dataset_path = Path("data/generated/cnn_switching_dataset.csv")
    model_path = Path("models/cnn_8level_controller.pth")
    history_path = Path("results/cnn_training_history.csv")
    confusion_path = Path("results/cnn_confusion_matrix.csv")

    trained = model_path.exists()
    # detect available trained model (prefer canonical controller model)
    controller_model = Path("models/cnn_8level_controller.pth")
    discovered_model = None
    if controller_model.exists():
        discovered_model = controller_model
    else:
        models_dir = Path("models")
        if models_dir.exists():
            pths = sorted(models_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
            if pths:
                discovered_model = pths[0]

    st.write("Status:", "TRAINED" if discovered_model is not None else "READY FOR TRAINING")
    if discovered_model is not None:
        try:
            ckpt = torch.load(discovered_model, map_location="cpu")
            total_params = 0
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                for v in ckpt["model_state_dict"].values():
                    try:
                        total_params += int(v.numel())
                    except Exception:
                        pass
            else:
                for v in ckpt.values() if isinstance(ckpt, dict) else []:
                    try:
                        total_params += int(v.numel())
                    except Exception:
                        pass
            st.write(f"Detected model: {discovered_model} — params: {total_params}")
        except Exception as e:
            st.write(f"Detected model: {discovered_model} — unable to read params: {e}")

    st.divider()
    st.subheader("Actions")
    with st.expander("Dataset Generation"):
        gen_cols = st.columns([2, 1])
        num_waveforms = gen_cols[0].number_input("Number of waveforms", value=200, min_value=10, max_value=5000, step=10)
        stride = gen_cols[1].number_input("Stride", value=8, min_value=1, max_value=WINDOW_SIZE, step=1)
        if st.button("Generate CNN Dataset"):
            with st.spinner("Generating CNN dataset..."):
                df = generate_cnn_dataset(dataset_path, num_waveforms=int(num_waveforms), stride=int(stride))
                st.session_state.cnn_dataset_info = {
                    "samples": len(df),
                    "classes": len(df["target_level"].unique()),
                    "columns": df.shape[1],
                    "window_size": WINDOW_SIZE,
                }
                st.success("CNN dataset generated.")
                st.write(df.head())

    with st.expander("Training"):
        tcol1, tcol2, tcol3 = st.columns(3)
        epochs = tcol1.number_input("Epochs", value=10, min_value=1)
        batch_size = tcol2.number_input("Batch Size", value=32, min_value=1)
        learning_rate = tcol3.number_input("Learning Rate", value=0.001, format="%.6f")

        if st.button("Train CNN Model"):
            if not dataset_path.exists():
                st.error("Generate the CNN dataset first.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                loss_chart = st.empty()
                acc_chart = st.empty()

                train_losses: list[float] = []
                val_losses: list[float] = []
                train_acc: list[float] = []
                val_acc: list[float] = []

                def _progress_cb(epoch: int, metrics: dict[str, float]) -> None:
                    # update lists
                    train_losses.append(metrics["train_loss"])
                    val_losses.append(metrics["val_loss"])
                    train_acc.append(metrics["train_accuracy"])
                    val_acc.append(metrics["val_accuracy"])

                    # update progress
                    pct = int(100 * (epoch / max(1, int(epochs))))
                    progress_bar.progress(min(pct, 100))
                    status_text.text(f"Epoch {epoch}/{epochs} — train_loss={metrics['train_loss']:.4f}, val_loss={metrics['val_loss']:.4f}, train_acc={metrics['train_accuracy']:.3f}, val_acc={metrics['val_accuracy']:.3f}")

                    # update charts
                    try:
                        hist_df = pd.DataFrame({
                            "epoch": list(range(1, epoch + 1)),
                            "train_loss": train_losses,
                            "val_loss": val_losses,
                            "train_accuracy": train_acc,
                            "val_accuracy": val_acc,
                        })
                        fig_l = go.Figure()
                        fig_l.add_trace(go.Scatter(x=hist_df["epoch"], y=hist_df["train_loss"], mode="lines+markers", name="Train Loss"))
                        fig_l.add_trace(go.Scatter(x=hist_df["epoch"], y=hist_df["val_loss"], mode="lines+markers", name="Val Loss"))
                        fig_l.update_layout(title="Loss vs Epoch", xaxis_title="Epoch", yaxis_title="Loss", height=300)
                        loss_chart.plotly_chart(fig_l, use_container_width=True)

                        fig_a = go.Figure()
                        fig_a.add_trace(go.Scatter(x=hist_df["epoch"], y=hist_df["train_accuracy"], mode="lines+markers", name="Train Acc"))
                        fig_a.add_trace(go.Scatter(x=hist_df["epoch"], y=hist_df["val_accuracy"], mode="lines+markers", name="Val Acc"))
                        fig_a.update_layout(title="Accuracy vs Epoch", xaxis_title="Epoch", yaxis_title="Accuracy", height=300)
                        acc_chart.plotly_chart(fig_a, use_container_width=True)
                    except Exception:
                        pass

                # run training
                with st.spinner("Training..."):
                    result = train_cnn(dataset_path, model_path, history_path, confusion_path, epochs=int(epochs), batch_size=int(batch_size), learning_rate=float(learning_rate), progress_callback=_progress_cb)
                progress_bar.progress(100)
                st.session_state.cnn_training = result
                st.session_state.cnn_dataset_info = {
                    "samples": result["train_samples"] + result["validation_samples"],
                    "classes": 8,
                }
                st.success("CNN model trained.")
                # Simple checkpointing: save metadata and copy model to canonical controller path if improved
                try:
                    meta = {
                        "timestamp": time.time(),
                        "dataset_size": result.get("train_samples", 0) + result.get("validation_samples", 0),
                        "epochs": result.get("epochs"),
                        "learning_rate": float(learning_rate),
                        "batch_size": int(batch_size),
                        "train_accuracy": float(result.get("train_accuracy", 0.0)),
                        "val_accuracy": float(result.get("val_accuracy", 0.0)),
                    }
                    models_dir = Path("models")
                    models_dir.mkdir(parents=True, exist_ok=True)
                    meta_path = models_dir / "cnn_8level_controller_meta.json"
                    # decide whether to replace canonical model
                    canonical = models_dir / "cnn_8level_controller.pth"
                    keep = True
                    if canonical.exists() and meta_path.exists():
                        try:
                            prev = json.loads(meta_path.read_text())
                            prev_val = float(prev.get("val_accuracy", 0.0))
                            if meta["val_accuracy"] <= prev_val:
                                keep = False
                        except Exception:
                            keep = True
                    # write metadata
                    meta_path.write_text(json.dumps(meta))
                    # copy trained model into canonical path if better or missing
                    if keep and model_path.exists():
                        shutil.copy(str(model_path), str(canonical))
                        st.write(f"Saved canonical controller model to {canonical}")
                except Exception as e:
                    st.warning(f"Checkpointing metadata failed: {e}")

    st.divider()
    st.subheader("Dataset Info")
    if st.session_state.cnn_dataset_info is not None:
        st.json(st.session_state.cnn_dataset_info)
    elif dataset_path.exists():
        st.write("Dataset file exists at data/generated/cnn_switching_dataset.csv")
    else:
        st.warning("CNN dataset is not generated yet.")

    st.divider()
    st.subheader("Training Results")
    if st.session_state.cnn_training is not None:
        training = st.session_state.cnn_training
        st.metric("Final Training Accuracy", f"{training['train_accuracy'] * 100:.2f}%")
        st.metric("Final Validation Accuracy", f"{training['val_accuracy'] * 100:.2f}%")
        st.write(f"Epochs: {training['epochs']}")
        st.write(f"Training samples: {training['train_samples']}")
        st.write(f"Validation samples: {training['validation_samples']}")
        if history_path.exists():
            history_df = pd.read_csv(history_path)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=history_df["epoch"], y=history_df["train_loss"], mode="lines+markers", name="Train Loss"))
            fig.add_trace(go.Scatter(x=history_df["epoch"], y=history_df["val_loss"], mode="lines+markers", name="Val Loss"))
            fig.update_layout(title="CNN Loss vs Epoch", xaxis_title="Epoch", yaxis_title="Loss")
            st.plotly_chart(fig, use_container_width=True)

            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(x=history_df["epoch"], y=history_df["train_accuracy"], mode="lines+markers", name="Train Accuracy"))
            fig_acc.add_trace(go.Scatter(x=history_df["epoch"], y=history_df["val_accuracy"], mode="lines+markers", name="Val Accuracy"))
            fig_acc.update_layout(title="CNN Accuracy vs Epoch", xaxis_title="Epoch", yaxis_title="Accuracy")
            st.plotly_chart(fig_acc, use_container_width=True)
        if confusion_path.exists():
            try:
                conf_matrix = pd.read_csv(confusion_path, header=None).to_numpy()
                labels = [f"L{i}" for i in range(1, conf_matrix.shape[0] + 1)]
                st.subheader("CNN Switching-State Confusion Matrix")
                hm = go.Figure(data=go.Heatmap(z=conf_matrix, x=labels, y=labels, colorscale='Viridis'))
                hm.update_layout(xaxis_title='Predicted', yaxis_title='Actual', height=480)
                st.plotly_chart(hm, use_container_width=True)

                overall_acc = np.trace(conf_matrix) / np.sum(conf_matrix) if np.sum(conf_matrix) > 0 else 0.0
                per_class_acc = np.diag(conf_matrix) / (conf_matrix.sum(axis=1) + 1e-12)
                st.write(f"Overall accuracy: {overall_acc * 100:.2f}%")
                per_df = pd.DataFrame({"class": labels, "accuracy": per_class_acc})
                st.subheader("Per-Class Accuracy")
                st.dataframe(per_df)

                poor = per_df[per_df['accuracy'] < 0.3]
                if not poor.empty:
                    st.warning(f"Poorly predicted classes: {', '.join(poor['class'].tolist())}")
            except Exception as e:
                st.error(f"Unable to read confusion matrix: {e}")
    else:
        st.info("Train the CNN model to view accuracy and confusion matrix.")

    st.divider()
    st.subheader("CNN Simulation")
    if st.button("Run CNN Simulation"):
        if not model_path.exists():
            st.error("Train the CNN model first.")
        else:
            if st.session_state.waveform is None:
                st.warning("Generate a reference 8-level waveform first on the 8-Level Inverter page.")
            else:
                params = st.session_state.input_params
                with st.spinner("Running CNN controller simulation..."):
                    cnn_result = run_cnn_controller(
                        dc_voltage=params["dc_voltage"],
                        fundamental_frequency=params["fundamental_frequency"],
                        sampling_frequency=params["sampling_frequency"],
                        duration=params["simulation_duration"],
                        modulation_index=params["modulation_index"],
                        phase_shift=0.0,
                        model_path=model_path,
                    )
                    st.session_state.cnn_result = cnn_result
                    st.success("CNN simulation completed.")

    if st.session_state.cnn_result is not None:
        result = st.session_state.cnn_result
        st.write("### Reference vs CNN Controller")
        ref_df = pd.DataFrame({"time": result.time, "reference_voltage": result.reference_voltage})
        cnn_df = pd.DataFrame({"time": result.time, "cnn_voltage": result.cnn_voltage, "cnn_level": result.cnn_levels})

        # Chart 1: Reference Voltage
        fig_ref = go.Figure()
        fig_ref.add_trace(go.Scatter(x=ref_df["time"], y=ref_df["reference_voltage"], mode="lines", name="Reference 8-Level"))
        fig_ref.update_layout(title="Reference 8-Level Output Voltage", xaxis_title="Time (s)", yaxis_title="Voltage (V)")
        st.plotly_chart(fig_ref, use_container_width=True)

        # Chart 2: CNN Predicted Switching State (map classes to percentages for visualization)
        percent_map = [-100.0, -71.43, -42.86, -14.29, 14.29, 42.86, 71.43, 100.0]
        state_percent = [percent_map[int(l)] if 0 <= int(l) < len(percent_map) else 0.0 for l in cnn_df["cnn_level"]]
        fig_state = go.Figure()
        fig_state.add_trace(go.Scatter(x=cnn_df["time"], y=state_percent, mode="lines", name="CNN Predicted State (%)"))
        fig_state.update_layout(title="CNN Predicted Switching State (percent mapping)", xaxis_title="Time (s)", yaxis_title="Percent of DC")
        st.plotly_chart(fig_state, use_container_width=True)

        # Chart 3: CNN Reconstructed Output Voltage
        fig_cnn = go.Figure()
        fig_cnn.add_trace(go.Scatter(x=cnn_df["time"], y=cnn_df["cnn_voltage"], mode="lines", name="CNN Output Voltage"))
        fig_cnn.update_layout(title="CNN Reconstructed Output Voltage", xaxis_title="Time (s)", yaxis_title="Voltage (V)")
        st.plotly_chart(fig_cnn, use_container_width=True)

        # THD comparison
        st.subheader("THD Comparison")
        ref_thd = float(result.thd_reference)
        cnn_thd = float(result.thd_cnn)
        diff = cnn_thd - ref_thd
        st.write(f"Reference THD: {ref_thd * 100:.3f}%")
        st.write(f"CNN THD: {cnn_thd * 100:.3f}%")
        st.write(f"THD Difference (CNN - Reference): {diff * 100:.3f}%")
        if cnn_thd > ref_thd:
            st.error("CNN prototype currently increases THD.")
        else:
            improvement = (ref_thd - cnn_thd) / (ref_thd + 1e-12) * 100.0
            st.success(f"CNN prototype reduces THD by {improvement:.2f}%.")

        # Harmonic comparison
        st.subheader("Harmonic Comparison")
        freqs = np.array(result.fft_reference[0])
        ref_mag = np.array(result.fft_reference[1])
        cnn_mag = np.array(result.fft_cnn[1])
        harm_df = pd.DataFrame({"frequency": freqs, "reference_magnitude": ref_mag, "cnn_magnitude": cnn_mag})
        harm_df["difference"] = harm_df["cnn_magnitude"] - harm_df["reference_magnitude"]
        st.plotly_chart(go.Figure(data=[go.Scatter(x=harm_df["frequency"], y=harm_df["reference_magnitude"], name="Reference"), go.Scatter(x=harm_df["frequency"], y=harm_df["cnn_magnitude"], name="CNN")]).update_layout(title="Harmonic Spectrum Comparison", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude"), use_container_width=True)
        # show dominant components table (top 10 by reference magnitude)
        top_ref = harm_df.sort_values(by="reference_magnitude", ascending=False).head(10)
        st.subheader("Dominant Harmonics (Reference)")
        st.dataframe(top_ref[["frequency", "reference_magnitude", "cnn_magnitude", "difference"]])
        st.write("### Prototype Note")
        st.info("Prototype CNN trained using synthetically generated 8-level inverter waveform data. This model demonstrates the ML pipeline and is not yet the final validated controller.")


def display_hpfo_page() -> None:
    st.header("HPFO Optimization")
    st.markdown("Hybrid Pelican Optimization + Pufferfish Optimization")

    st.subheader("Optimization Parameters")
    st.write("Parameter configuration UI will be connected later.")

    st.subheader("Objective Function")
    st.write("Objective definition will be added in future phases.")

    st.subheader("Population")
    st.write("Population management UI pending.")

    st.subheader("Iterations")
    st.write("Iteration tracking will be added later.")

    st.subheader("Best Solution")
    st.write("Best solution display will appear after optimization is implemented.")

    st.subheader("Convergence")
    st.write("Convergence curve support pending.")

    st.warning("Optimization implementation pending")


def display_waveform_analysis() -> None:
    st.header("Waveform Analysis")
    if st.session_state.waveform is None:
        st.warning("No waveform generated. Go to 8-Level Inverter and generate a waveform.")
        return

    st.subheader("Output Voltage")
    waveform_df = pd.DataFrame(
        {
            "time": st.session_state.waveform["time"],
            "voltage": st.session_state.waveform["voltage"],
            "level": st.session_state.waveform["switching_state"],
        }
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=waveform_df["time"], y=waveform_df["voltage"], mode="lines", name="Voltage"))
    fig.update_layout(title="Generated Output Voltage", xaxis_title="Time (s)", yaxis_title="Voltage (V)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Level Distribution")
    level_counts = waveform_df["level"].value_counts().sort_index()
    dist_fig = go.Figure()
    dist_fig.add_trace(go.Bar(x=level_counts.index, y=level_counts.values, marker_color="indianred"))
    dist_fig.update_layout(title="Switching Level Distribution", xaxis_title="Switching State", yaxis_title="Count")
    st.plotly_chart(dist_fig, use_container_width=True)

    st.subheader("Voltage Statistics")
    stats = st.session_state.waveform_summary
    st.write(f"Maximum voltage: {stats['max_voltage']:.2f} V")
    st.write(f"Minimum voltage: {stats['min_voltage']:.2f} V")
    st.write(f"Detected unique levels: {stats['detected_levels']}")
    st.write(f"Number of samples: {stats['sample_count']}")
    st.write(f"Detected fundamental frequency: {stats['fundamental_frequency']:.2f} Hz")


def display_fft_thd() -> None:
    st.header("FFT & THD")
    if st.session_state.waveform is None:
        st.warning("No waveform generated. Go to 8-Level Inverter and generate a waveform.")
        return

    voltage = st.session_state.waveform["voltage"]
    sampling_frequency = st.session_state.input_params.get("sampling_frequency", 10000.0)
    fundamental_frequency = st.session_state.input_params.get("fundamental_frequency", 50.0)

    frequencies, magnitudes = compute_fft(voltage, sampling_frequency)
    thd_value, fundamental_mag, harmonic_orders, harmonic_frequencies, harmonic_magnitudes = calculate_thd(
        voltage, sampling_frequency, fundamental_frequency
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frequencies, y=magnitudes, mode="lines", name="Magnitude"))
    fig.update_layout(
        title="FFT Spectrum of 8-Level Reference Waveform",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("THD Results")
    st.write(
        "THD is calculated from the generated reference waveform. This is a software-prototype result and is not yet the validated MATLAB/Simulink inverter result."
    )
    st.write(f"Fundamental magnitude: {fundamental_mag:.6f}")
    st.write(f"THD: {thd_value * 100:.2f} %")

    harmonic_table = pd.DataFrame(
        {
            "Harmonic Order": harmonic_orders,
            "Frequency (Hz)": harmonic_frequencies,
            "Magnitude": harmonic_magnitudes,
        }
    )
    st.subheader("Harmonic Components")
    st.dataframe(harmonic_table)


def display_results_page() -> None:
    st.header("Results")
    st.write("Final performance results will appear after simulation and model integration.")

    st.subheader("Comparison Categories")
    st.markdown(
        "- Conventional Method\n"
        "- CNN\n"
        "- CNN + POA\n"
        "- CNN + PFO\n"
        "- CNN + HPFO"
    )

    st.warning("Results will appear after simulation and model integration.")
    st.write("Metrics like THD, efficiency, voltage quality, switching loss, and power loss will be added later.")


def display_matlab_integration() -> None:
    st.header("MATLAB/Simulink Integration")
    st.warning("Not Connected")
    st.write("This interface is reserved for future MATLAB/Simulink integration.")

    csv_file = st.file_uploader("Upload CSV for MATLAB workflow", type=["csv"] )
    mat_file = st.file_uploader("Upload MAT file", type=["mat"])

    if csv_file is not None:
        st.write("CSV uploaded:")
        st.write(csv_file.name)
    if mat_file is not None:
        st.write("MAT file uploaded:")
        st.write(mat_file.name)


def display_dataset_page() -> None:
    st.header("Dataset")
    st.write("Upload a CSV dataset and inspect its basic structure.")

    uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"] )
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.dataset = df
            st.session_state.dataset_info = {
                "filename": uploaded_file.name,
                "rows": df.shape[0],
                "columns": df.shape[1],
                "column_names": list(df.columns),
                "missing_values": int(df.isna().sum().sum()),
            }
        except Exception as error:
            st.error(f"Unable to read uploaded CSV: {error}")
            st.session_state.dataset = None
            st.session_state.dataset_info = {}

    if st.session_state.dataset is None:
        st.warning("No dataset loaded.")
    else:
        st.subheader("Dataset Summary")
        st.json(st.session_state.dataset_info)
        st.subheader("Data Preview")
        st.dataframe(st.session_state.dataset.head())

    raw_path = Path("data/raw")
    raw_path.mkdir(parents=True, exist_ok=True)
    raw_files = list(raw_path.glob("*.csv"))
    if raw_files:
        st.subheader("Raw folder CSV files")
        for file in raw_files:
            st.write(file.name)
    else:
        st.info("No CSV files found in data/raw/.")


def display_training_page() -> None:
    st.header("Training")
    st.write("This page is the foundation for future CNN training workflows.")

    with st.form("training_form"):
        st.subheader("Model Training Inputs")
        dataset_selected = st.selectbox("Dataset", ["Select dataset"] + ([st.session_state.dataset_info.get("filename")] if st.session_state.dataset is not None else []))
        validation_split = st.slider("Validation Split", min_value=0.1, max_value=0.5, value=0.2)
        epochs = st.number_input("Epochs", value=10, min_value=1)
        batch_size = st.number_input("Batch Size", value=16, min_value=1)
        learning_rate = st.number_input("Learning Rate", value=0.001, format="%.4f")
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.warning("Training pipeline not implemented yet.")

    st.subheader("Model Controls")
    if st.button("Prepare Dataset"):
        st.warning("Training pipeline not implemented yet.")
    if st.button("Train Model"):
        st.warning("Training pipeline not implemented yet.")
    if st.button("Evaluate Model"):
        st.warning("Training pipeline not implemented yet.")
    if st.button("Save Model"):
        st.warning("Training pipeline not implemented yet.")


def display_project_status() -> None:
    st.header("Project Status")
    st.write("Current readiness of the interface and implementation areas.")
    status_rows = [
        ("Project Interface", "FOUNDATION READY"),
        ("System Architecture", "FOUNDATION READY"),
        ("Dataset Pipeline", "NOT STARTED"),
        ("8-Level Inverter", "FOUNDATION READY"),
        ("31-Level Inverter", "NOT STARTED"),
        ("CNN Training", "NOT STARTED"),
        ("HPFO Optimization", "NOT STARTED"),
        ("MATLAB Integration", "NOT STARTED"),
        ("FFT/THD Integration", "FOUNDATION READY"),
        ("Final Simulation", "NOT STARTED"),
    ]

    for name, status in status_rows:
        if status == "FOUNDATION READY":
            st.success(f"{name}: {status}")
        else:
            st.info(f"{name}: {status}")


def run_app() -> None:
    st.set_page_config(page_title="Hybrid Optimization and CNN Control", layout="wide")
    init_session_state()

    page = st.sidebar.selectbox(
        "Navigation",
        [
            "Overview",
            "System Architecture",
            "Input Parameters",
            "Renewable Sources",
            "8-Level Inverter",
            "CNN Controller",
            "HPFO Optimization",
            "Waveform Analysis",
            "FFT & THD",
            "Results",
            "MATLAB Integration",
            "Dataset",
            "Training",
            "Project Status",
        ],
        key="main_page",
    )

    if page == "Overview":
        display_overview()
    elif page == "System Architecture":
        display_system_architecture()
    elif page == "Input Parameters":
        display_input_parameters()
    elif page == "Renewable Sources":
        display_renewable_sources()
    elif page == "31-Level Inverter":
        display_inverter_page()
    elif page == "8-Level Inverter":
        display_8level_inverter()
    elif page == "CNN Controller":
        display_cnn_controller()
    elif page == "HPFO Optimization":
        display_hpfo_page()
    elif page == "Waveform Analysis":
        display_waveform_analysis()
    elif page == "FFT & THD":
        display_fft_thd()
    elif page == "Results":
        display_results_page()
    elif page == "MATLAB Integration":
        display_matlab_integration()
    elif page == "Dataset":
        display_dataset_page()
    elif page == "Training":
        display_training_page()
    elif page == "Project Status":
        display_project_status()


if __name__ == "__main__":
    run_app()
