
"""Professional exhibition dashboard: Hybrid CNN inverter project."""

from __future__ import annotations

import io
import hashlib
import csv
import json
import pickle
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from assistant.switching_advisor import SwitchingRecommendation, recommend_switching
from assistant.local_chat import (
    DEFAULT_CHAT_MODEL,
    LocalChatError,
    OllamaChatClient,
    chat_with_tools,
)
from evaluation.cnn_test_evaluator import CNNTestEvaluation, evaluate_cnn_test_set
from evaluation.live_export import LiveExportBundle, build_live_export
from evaluation.measured_waveform_validation import (
    MeasuredWaveformComparison,
    compare_measured_voltage_csv,
)
from evaluation.live_phase_optimization import (
    LiveConditionedPhaseOptimization,
    optimize_live_operating_point,
    optimize_live_surrogate_screening,
)
from evaluation.live_simulation import (
    ALLOWED_LOAD_INDUCTANCES_H,
    ALLOWED_LOAD_RESISTANCES_OHM,
    LiveSimulationResult,
    operating_condition_support_warnings,
    run_live_simulation,
)
from optimization.circular_phase_pso import (
    BASELINE_PHASES_DEGREES,
    TRACKING_TOLERANCE_PERCENT,
    candidate_to_carrier_offsets,
)
from evaluation.dashboard_helpers import (
    LIVE_PRESETS,
    cnn_guidance_operating_point_matches,
    live_configuration_matches,
    normalize_carrier_phase_offsets,
    optimizer_baseline_supported,
    playback_snapshot,
    synchronized_playback_index,
    thd_surrogate_operating_point_supported,
    verify_cell_voltage_sum,
)
from multilevel.inverter_simulator import (
    FUNDAMENTAL_FREQUENCY_HZ,
    SAMPLING_FREQUENCY_HZ,
)
from optimization.cnn_guided_search import (
    COMPARISON_CANDIDATE_BUDGET,
    COMPARISON_TOP_K,
    run_cnn_guided_hybrid,
)
from evaluation.metrics import calculate_metrics
from models.cnn_model import detect_device
from models.model_factory import create_model
from preprocessing.data_loader import load_dataset
from preprocessing.data_validation import validate_dataset
from preprocessing.preprocess import NormalizationStats, preprocess_dataset
from project_paths import PROJECT_ROOT, resolve_project_path


# ============================================================
# PAGE AND THEME
# ============================================================

st.set_page_config(
    page_title="INVERTER AI | Research Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def _mark_live_playback_manual_selection() -> None:
    st.session_state["live_playback_manual_selection"] = True


st.markdown(
    """
    <style>
    :root {
        --bg: #0b1018;
        --panel: #111a26;
        --panel2: #151f2d;
        --line: #263447;
        --muted: #91a1b5;
        --text: #e8eef6;
        --cyan: #43d6c5;
        --blue: #659df5;
    }

    .stApp {
        background: var(--bg);
        color: var(--text);
    }

    .block-container {
        max-width: 1580px;
        padding-top: 1.25rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background: #0f1722;
        border-right: 1px solid var(--line);
    }

    [data-testid="stHeader"] {
        background: rgba(11, 16, 24, 0.9);
    }

    .topline {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 16px;
        padding: 0.3rem 0 1rem 0;
        border-bottom: 1px solid var(--line);
        margin-bottom: 1.5rem;
    }

    .brand {
        color: #f3f7fc;
        font-size: 1.05rem;
        font-weight: 800;
        letter-spacing: .16em;
    }

    .brand span {
        color: var(--cyan);
    }

    .top-meta {
        color: var(--muted);
        font-size: .78rem;
        letter-spacing: .08em;
        text-transform: uppercase;
    }

    .hero {
        padding: 1.5rem 1.7rem;
        border: 1px solid #2b4054;
        border-radius: 15px;
        background:
            radial-gradient(circle at 85% 20%, rgba(67,214,197,.13), transparent 30%),
            linear-gradient(115deg, #142235, #111b29 65%, #142b35);
        margin-bottom: 1.15rem;
    }

    .eyebrow {
        color: var(--cyan);
        font-size: .72rem;
        font-weight: 800;
        letter-spacing: .17em;
        text-transform: uppercase;
        margin-bottom: .55rem;
    }

    .hero h1 {
        color: #f4f7fb;
        font-size: clamp(1.6rem, 3vw, 2.25rem);
        line-height: 1.15;
        font-weight: 750;
        margin: 0 0 .55rem 0;
    }

    .hero p {
        color: #aebed0;
        margin: 0;
        font-size: .95rem;
    }

    .section-heading {
        color: #e9f0f8;
        font-size: 1.05rem;
        font-weight: 700;
        margin: .4rem 0 .2rem 0;
    }

    .section-sub {
        color: var(--muted);
        font-size: .82rem;
        margin-bottom: .9rem;
    }

    .panel {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 1rem 1.1rem;
        min-height: 100%;
    }

    .panel-label {
        color: var(--muted);
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: .1em;
        text-transform: uppercase;
    }

    .panel-value {
        color: #f1f6fc;
        font-size: 1.25rem;
        font-weight: 750;
        margin-top: .35rem;
    }

    .panel-note {
        color: var(--muted);
        font-size: .76rem;
        margin-top: .2rem;
    }

    .status {
        display: inline-block;
        padding: .35rem .65rem;
        border: 1px solid #28675d;
        border-radius: 30px;
        background: #102a28;
        color: #85e3d1;
        font-size: .75rem;
        font-weight: 700;
    }

    .notice {
        background: #171f2a;
        border: 1px solid #344356;
        border-left: 3px solid #d6a75b;
        border-radius: 8px;
        padding: .75rem 1rem;
        color: #b8c5d4;
        font-size: .82rem;
        margin: .8rem 0 1rem 0;
    }

    div[data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 11px;
        padding: 13px 15px;
    }

    div[data-testid="stMetricLabel"] {
        color: var(--muted);
        font-size: .78rem;
    }

    div[data-testid="stMetricValue"] {
        color: #f2f6fb;
        font-weight: 750;
    }

    div[data-testid="stTabs"] button {
        font-weight: 650;
        font-size: .88rem;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 10px;
    }

    .stButton button, .stDownloadButton button {
        border-radius: 8px;
        font-weight: 650;
    }

    hr {
        border-color: var(--line);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

project_root = PROJECT_ROOT


@st.cache_data(show_spinner=False)
def load_prepared_dataset(source: str, fingerprint: str):
    if not fingerprint:
        raise ValueError("Dataset cache fingerprint cannot be empty.")
    tables = load_dataset(source)
    report = validate_dataset(tables, check_auxiliary=False)
    prepared = preprocess_dataset(tables)
    return prepared, report.summary


def make_chart(
    series: list[tuple[str, np.ndarray, str, str]],
    title: str,
    y_label: str,
    *,
    x_values: np.ndarray | None = None,
    x_label: str = "Sample index (–)",
    height_inches: float = 4.2,
):
    """Create a consistent dark-theme engineering waveform chart."""
    fig, ax = plt.subplots(figsize=(11, height_inches))
    fig.patch.set_facecolor("#111a26")
    ax.set_facecolor("#111a26")

    for label, values, color, style in series:
        x = (
            np.arange(len(values))
            if x_values is None
            else np.asarray(x_values, dtype=np.float64)
        )
        ax.plot(
            x,
            np.asarray(values),
            label=label,
            color=color,
            linewidth=1.8,
            linestyle=style,
        )

    ax.set_title(title, color="#edf3fa", fontsize=12, loc="left", pad=14)
    ax.set_xlabel(x_label, color="#9aabc0")
    ax.set_ylabel(y_label, color="#9aabc0")
    ax.tick_params(colors="#9aabc0", labelsize=8)
    ax.grid(True, color="#344356", alpha=.45, linestyle=":")
    ax.legend(
        facecolor="#172332",
        edgecolor="#344356",
        labelcolor="#dce6f1",
        framealpha=.95,
        fontsize=8,
    )

    for spine in ax.spines.values():
        spine.set_color("#344356")

    fig.tight_layout()
    return fig


def make_harmonic_chart(
    harmonic_result,
    fundamental_frequency_hz: float,
    candidate_harmonic_result=None,
    candidate_label: str = "Optimized candidate",
):
    """Plot paired event-integrated peak harmonic amplitudes."""
    orders = np.asarray(harmonic_result.harmonic_orders, dtype=np.int64)
    amplitudes = np.asarray(harmonic_result.harmonic_peak_amplitudes_v, dtype=np.float64)
    visible = orders <= 100
    frequencies_hz = orders[visible] * float(fundamental_frequency_hz)

    fig, ax = plt.subplots(figsize=(11, 4.2))
    fig.patch.set_facecolor("#111a26")
    ax.set_facecolor("#111a26")
    ax.plot(frequencies_hz, amplitudes[visible], color="#43d6c5", label="Baseline")
    if candidate_harmonic_result is not None:
        candidate_orders = np.asarray(
            candidate_harmonic_result.harmonic_orders, dtype=np.int64
        )
        candidate_amplitudes = np.asarray(
            candidate_harmonic_result.harmonic_peak_amplitudes_v,
            dtype=np.float64,
        )
        candidate_visible = candidate_orders <= 100
        ax.plot(
            candidate_orders[candidate_visible] * float(fundamental_frequency_hz),
            candidate_amplitudes[candidate_visible],
            color="#d889d8",
            linestyle="--",
            label=candidate_label,
        )
    ax.set_title(
        "Event-integrated output-voltage harmonic spectrum (orders 1–100)",
        color="#edf3fa",
        fontsize=12,
        loc="left",
        pad=14,
    )
    ax.set_xlabel("Frequency (Hz)", color="#9aabc0")
    ax.set_ylabel("Harmonic peak amplitude (V)", color="#9aabc0")
    ax.tick_params(colors="#9aabc0", labelsize=8)
    ax.grid(True, color="#344356", alpha=.45, linestyle=":")
    ax.legend(
        facecolor="#172332",
        edgecolor="#344356",
        labelcolor="#dce6f1",
        framealpha=.95,
        fontsize=8,
    )
    for spine in ax.spines.values():
        spine.set_color("#344356")
    fig.tight_layout()
    return fig


def show_panel(label: str, value: str, note: str = ""):
    st.markdown(
        f"""
        <div class="panel">
            <div class="panel-label">{label}</div>
            <div class="panel-value">{value}</div>
            <div class="panel-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def fmt_metric(metrics: dict, key: str, digits: int = 4) -> str:
    value = metrics.get(key)
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_percent(value: float) -> str:
    """Keep very small measured percentages distinguishable from zero."""
    numeric = float(value)
    if not np.isfinite(numeric):
        return "N/A"
    return f"{numeric:.3e}%" if abs(numeric) < 0.001 else f"{numeric:.3f}%"


def source_fingerprint(source: str | Path) -> str:
    """Build a lightweight cache identity from the selected dataset files."""
    path = resolve_project_path(source, project_root)
    if path.is_file():
        stat = path.stat()
        return f"{path}:{stat.st_size}:{stat.st_mtime_ns}"
    if not path.is_dir():
        return str(path)
    file_stats = []
    for candidate in sorted(path.glob("*.csv")):
        stat = candidate.stat()
        file_stats.append(f"{candidate.name}:{stat.st_size}:{stat.st_mtime_ns}")
    return f"{path}|" + "|".join(file_stats)


def checkpoint_fingerprint(uploaded_file, checkpoint_path: Path) -> str:
    if uploaded_file is not None:
        return hashlib.sha256(uploaded_file.getvalue()).hexdigest()
    if checkpoint_path.is_file():
        stat = checkpoint_path.stat()
        return f"{checkpoint_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    return "no-checkpoint"


def evaluation_summary_payload(result: CNNTestEvaluation) -> str:
    return json.dumps(
        {
            "task": "CNN reference-target waveform prediction on held-out testing split",
            "record_count": result.record_count,
            "waveform_points_per_record": result.waveform_points,
            "batch_size": result.batch_size,
            "micro_metrics_all_concatenated_waveform_points": asdict(result.micro_metrics),
            "macro_metrics_mean_of_per_record_metrics": asdict(result.macro_metrics),
            "error_distribution_v": asdict(result.error_distribution),
            "r_squared_definition": "1 - pooled sum of squared residuals / pooled sum of squared deviations from pooled target mean",
            "macro_r_squared_definition": "arithmetic mean of defined per-record R-squared values; undefined records excluded",
            "correlation_definition": "Pearson correlation of all concatenated waveform points",
            "scientific_limitations": [
                "The CNN predicts the reference target voltage waveform.",
                "It does not predict switching gates or optimized inverter output THD.",
                "Prediction metrics are not evidence of harmonic reduction or physical performance.",
                "This evaluation does not constitute hardware, MATLAB, or Simulink validation.",
            ],
        },
        indent=2,
    )


# ============================================================
# SIDEBAR: SETTINGS ARE NOT THE LANDING PAGE
# ============================================================

default_dataset = str(
    project_root / "data" / "raw" / "Seven_Level_Inverter_Project_extracted"
)
default_checkpoint = (
    project_root
    / "colab_runs"
    / "baseline_20261120_080836"
    / "checkpoints"
    / "best.pt"
)

with st.sidebar:
    st.markdown("## ⚡ INVERTER AI")
    st.caption("POWER ELECTRONICS RESEARCH")
    st.divider()

    with st.expander("DATASET & MODEL SETTINGS", expanded=False):
        dataset_path = st.text_input(
            "Dataset ZIP or extracted CSV directory",
            value=default_dataset,
            help=(
                "Select a dataset source; changing it reloads and validates the data."
            ),
        )

        checkpoint_file = st.file_uploader(
            "Optional checkpoint override",
            type=["pt", "pth"],
            help=(
                "Optional CNN inference weights; this does not retrain the model."
            ),
        )
        st.caption(
            "Default checkpoint: "
            + (
                str(default_checkpoint)
                if default_checkpoint.is_file()
                else "not found"
            )
        )

    st.divider()
    st.markdown("**PROJECT MODULES**")
    st.caption("01  Dataset validation")
    st.caption("02  CNN prediction")
    st.caption("03  Waveforms")
    st.caption("04  Research diagnostics")
    st.divider()
    st.caption("Seven-level multilevel inverter")
    st.caption("CNN Optimization")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="topline">
        <div class="brand"><span>⚡</span> INVERTER AI</div>
        <div class="top-meta">Power Electronics AI</div>
    </div>

    <div class="hero">
        <div class="eyebrow">Ideal software simulation, not hardware validation.</div>
        <h1>Hybrid CNN Inverter Simulation &amp; Harmonic Analysis</h1>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD DATASET AND CHECKPOINT
# ============================================================

try:
    with st.spinner("Loading dataset and validating project data..."):
        resolved_dataset_path = resolve_project_path(dataset_path, project_root)
        prepared, dataset_summary = load_prepared_dataset(
            str(resolved_dataset_path),
            source_fingerprint(resolved_dataset_path),
        )

    if checkpoint_file is not None:
        checkpoint_bytes = checkpoint_file.getvalue()
        checkpoint_source = io.BytesIO(checkpoint_bytes)
        loaded_checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    elif default_checkpoint.is_file():
        checkpoint_source = default_checkpoint
        loaded_checkpoint_sha256 = hashlib.sha256(
            default_checkpoint.read_bytes()
        ).hexdigest()
    else:
        checkpoint_source = None
        loaded_checkpoint_sha256 = None

    model = None
    stats = None
    device = detect_device()
    checkpoint_status = "Not loaded"

    if checkpoint_source is not None:
        with st.spinner("Loading trained CNN checkpoint..."):
            checkpoint = torch.load(
                checkpoint_source,
                map_location=device,
                weights_only=True,
            )

            if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
                raise ValueError("The selected file is not a project training checkpoint.")

            metadata = checkpoint.get("metadata", {})
            model_config = metadata.get("model_config")
            normalization_data = metadata.get("normalization")

            if not isinstance(model_config, dict) or not isinstance(normalization_data, dict):
                raise ValueError("Checkpoint model or normalization metadata is missing.")

            stats = NormalizationStats(**normalization_data)

            saved_features = metadata.get("operating_feature_names")
            if saved_features is not None and list(saved_features) != list(
                prepared.operating_feature_names
            ):
                raise ValueError(
                    "Checkpoint operating-condition feature order does not match the dataset."
                )

            for saved_value, current_value in (
                (stats.input_voltage_mean, prepared.stats.input_voltage_mean),
                (stats.input_voltage_std, prepared.stats.input_voltage_std),
                (stats.load_current_mean, prepared.stats.load_current_mean),
                (stats.load_current_std, prepared.stats.load_current_std),
                (stats.target_voltage_mean, prepared.stats.target_voltage_mean),
                (stats.target_voltage_std, prepared.stats.target_voltage_std),
            ):
                if not np.isclose(saved_value, current_value, rtol=1e-5, atol=1e-7):
                    raise ValueError(
                        "Checkpoint normalization does not match the selected dataset."
                    )

            if not np.allclose(
                stats.operating_mean,
                prepared.stats.operating_mean,
                rtol=1e-5,
                atol=1e-7,
            ):
                raise ValueError("Checkpoint operating-condition means do not match.")

            if not np.allclose(
                stats.operating_std,
                prepared.stats.operating_std,
                rtol=1e-5,
                atol=1e-7,
            ):
                raise ValueError("Checkpoint operating-condition scales do not match.")

            model = create_model(model_config).to(device)
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            checkpoint_status = "Loaded"

    config = dataset_summary.get("configuration", {})
    split_counts = dataset_summary.get("split_counts", {})
    parameter_count = model.parameter_count() if model is not None else 0

    # ========================================================
    # GLOBAL SAMPLE SELECTION
    # ========================================================

    all_splits = np.asarray(prepared.splits).astype(str)
    split_options = ["All"] + sorted(set(all_splits.tolist()))

    selector_col1, selector_col2, selector_col3 = st.columns([1.2, 1.2, 2.6])

    with selector_col1:
        selected_split = st.selectbox(
            "RECORD GROUP",
            split_options,
            index=0,
        )

    if selected_split == "All":
        available_indices = np.arange(len(prepared.sample_ids))
    else:
        available_indices = np.flatnonzero(all_splits == selected_split)

    available_ids = prepared.sample_ids[available_indices].astype(int).tolist()

    with selector_col2:
        selected_id = st.selectbox(
            "SAMPLE ID",
            available_ids,
            index=0,
        )

    index = int(np.flatnonzero(prepared.sample_ids == int(selected_id))[0])

    with selector_col3:
        st.markdown(
            f"""
            <div style="padding-top:1.75rem;text-align:right">
                <span class="status">● DATASET VALIDATED</span>
                &nbsp;
                <span class="status">● CNN {checkpoint_status.upper()}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ========================================================
    # TOP-LEVEL ENGINEERING INDICATORS
    # ========================================================

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        show_panel("INVERTER TOPOLOGY", "7-Level", "Three-cell multilevel structure")

    with c2:
        show_panel(
            "FUNDAMENTAL FREQUENCY",
            f"{config.get('fundamental frequency', '50')} Hz",
            "Dataset configuration",
        )

    with c3:
        show_panel(
            "SAMPLING FREQUENCY",
            f"{int(config.get('sampling frequency', 20000)) / 1000:g} kHz",
            "Dataset waveform sampling",
        )

    with c4:
        show_panel(
            "CNN MODEL",
            f"{parameter_count:,}" if model is not None else "Unavailable",
            "Trainable parameters" if model is not None else "Checkpoint not loaded",
        )

    st.write("")

    # ========================================================
    # MAIN NAVIGATION
    # ========================================================

    (
        live_demo_tab,
        research_tab,
    ) = st.tabs(
        [
            "LIVE SIMULATION WORKSPACE",
            "RESEARCH WORKFLOW",
        ]
    )

    with research_tab:
        (
            overview_tab,
            cnn_tab,
            test_evaluation_tab,
            diagnostics_tab,
        ) = st.tabs(
            [
                "OVERVIEW",
                "CNN PREDICTION",
                "CNN TEST EVALUATION",
                "DATA DIAGNOSTICS",
            ]
        )

    # --------------------------------------------------------
    # OVERVIEW
    # --------------------------------------------------------

    with overview_tab:
        st.markdown(
            '<div class="section-heading">Inverter waveform monitoring</div>',
            unsafe_allow_html=True,
        )
        left, right = st.columns([1.8, 1])

        with left:
            voltage_chart = make_chart(
                [
                    (
                        "Dataset inverter voltage",
                        prepared.input_voltage[index],
                        "#43d6c5",
                        "-",
                    ),
                    (
                        "Reference target",
                        prepared.target_voltage[index],
                        "#679df5",
                        "--",
                    ),
                ],
                "Voltage waveform comparison",
                "Voltage (V)",
            )
            st.pyplot(voltage_chart, width="stretch")
            plt.close(voltage_chart)

            current_chart = make_chart(
                [
                    (
                        "Load current",
                        prepared.load_current[index],
                        "#e7ad65",
                        "-",
                    ),
                ],
                "Load current response",
                "Current (A)",
            )
            st.pyplot(current_chart, width="stretch")
            plt.close(current_chart)

        with right:
            st.markdown("#### Selected operating point")

            operating_values = (
                prepared.operating_conditions[index].astype(np.float64)
                * np.asarray(prepared.stats.operating_std, dtype=np.float64)
                + np.asarray(prepared.stats.operating_mean, dtype=np.float64)
            )
            operating_rows = []

            unit_labels = {
                "modulationIndex": "",
                "loadResistance": "Ω",
                "loadInductance": "H",
                "phaseAngle": "rad",
            }
            for name, value in zip(
                prepared.operating_feature_names,
                operating_values,
            ):
                operating_rows.append(
                    {
                        "Operating parameter": str(name),
                        "Physical value": (
                            f"{np.degrees(float(value)):.3f}°"
                            if str(name).casefold() == "phaseangle"
                            else f"{float(value):.5g} {unit_labels.get(str(name), '')}".strip()
                        ),
                        "Source": "Dataset (inverse transformed)",
                    }
                )

            st.dataframe(
                operating_rows,
                width="stretch",
                hide_index=True,
            )

            st.markdown("#### Record information")
            st.markdown(
                f"""
                <div class="panel">
                    <div class="panel-label">SAMPLE IDENTIFIER</div>
                    <div class="panel-value">{int(selected_id)}</div>
                    <div class="panel-note">Dataset split: {all_splits[index]}</div>
                    <hr style="border-color:#263447">
                    <div class="panel-label">WAVEFORM LENGTH</div>
                    <div class="panel-value">{prepared.target_voltage.shape[1]} points</div>
                    <div class="panel-note">Dataset record</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("#### Workflow")
        workflow_steps = (
            ("1 Conditions", "Modulation, R, L, phase"),
            ("2 Simulation", "Seven-level PWM"),
            ("3 Output", "Output voltage and RL current"),
            ("4 CNN", "Reference prediction"),
            ("5 Error", "CNN vs. reference"),
            ("6 PSO", "Training-only; candidate resimulated"),
            ("7 THD", "Finite-band, Parseval"),
            ("8 Results", "Baseline vs. PSO"),
        )
        workflow_rows = (workflow_steps[:4], workflow_steps[4:])
        for row in workflow_rows:
            workflow_columns = st.columns(4)
            for column, (heading, description) in zip(workflow_columns, row):
                with column:
                    st.markdown(
                        f"""
                        <div class="panel" style="min-height:105px">
                            <div class="panel-label">{heading}</div>
                            <div class="panel-note">{description}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        st.caption(
            "CNN predicts references; PSO candidates are simulator-verified."
        )

    # --------------------------------------------------------
    # CNN PREDICTION
    # --------------------------------------------------------

    with cnn_tab:
        st.markdown(
            '<div class="section-heading">CNN prediction</div>',
            unsafe_allow_html=True,
        )

        if model is None:
            st.warning(
                "CNN unavailable. Load a compatible checkpoint."
            )
        else:
            with st.spinner("Running CNN inference..."):
                with torch.inference_mode():
                    prediction = model(
                        torch.from_numpy(
                            prepared.input_voltage[index:index + 1]
                        ).to(device),
                        torch.from_numpy(
                            prepared.load_current[index:index + 1]
                        ).to(device),
                        torch.from_numpy(
                            prepared.operating_conditions[index:index + 1]
                        ).to(device),
                    )

            predicted_volts = stats.denormalize_target(
                prediction.cpu().numpy()
            )[0]

            target_volts = stats.denormalize_target(
                prepared.target_voltage[index]
            )

            metrics = calculate_metrics(target_volts, predicted_volts)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("MEAN ABSOLUTE ERROR", fmt_metric(metrics, "mae"))
            m2.metric("RMSE (V)", fmt_metric(metrics, "rmse"))
            m3.metric("R² SCORE", fmt_metric(metrics, "r_squared", 6))
            m4.metric("CORRELATION", fmt_metric(metrics, "correlation", 6))

            prediction_chart = make_chart(
                [
                    ("Reference target", target_volts, "#43d6c5", "-"),
                    ("CNN prediction", predicted_volts, "#f0a95e", "--"),
                ],
                f"CNN vs. reference · Sample {selected_id}",
                "Voltage (V)",
            )
            st.pyplot(prediction_chart, width="stretch")
            plt.close(prediction_chart)

            error = predicted_volts - target_volts
            error_chart = make_chart(
                [("Prediction error", error, "#e58d8d", "-")],
                "Prediction error",
                "Error (V)",
            )
            st.pyplot(error_chart, width="stretch")
            plt.close(error_chart)

            with st.expander("Detailed model metrics"):
                metric_rows = [
                    {"Metric": str(key).replace("_", " ").upper(), "Value": value}
                    for key, value in metrics.items()
                ]
                st.dataframe(
                    metric_rows,
                    width="stretch",
                    hide_index=True,
                )

            export_data = np.column_stack(
                [
                    np.arange(len(target_volts)),
                    target_volts,
                    predicted_volts,
                    error,
                ]
            )
            export_buffer = io.StringIO()
            np.savetxt(
                export_buffer,
                export_data,
                delimiter=",",
                header="sample_point,reference_target_v,cnn_prediction_v,error_v",
                comments="",
            )

            st.download_button(
                "Download prediction results (CSV)",
                data=export_buffer.getvalue(),
                file_name=f"cnn_prediction_sample_{selected_id}.csv",
                mime="text/csv",
            )

            st.markdown(
                """
                <div class="notice">
                    CNN prediction; not inverter THD.
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------
    # HELD-OUT CNN TEST EVALUATION
    # --------------------------------------------------------

    with test_evaluation_tab:
        st.markdown(
            '<div class="section-heading">CNN Test Evaluation</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-sub">Batch inference on testing records only; evaluation starts only after explicit confirmation</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="notice">
                The CNN predicts the reference target voltage waveform from inverter
                voltage, load current, and operating conditions. It does not predict
                switching gates or optimized inverter-output THD.
            </div>
            """,
            unsafe_allow_html=True,
        )

        testing_count = int(np.count_nonzero(all_splits == "testing"))
        status_col, count_col = st.columns([2, 1])
        cached_result = st.session_state.get("cnn_test_evaluation_result")
        cached_key = st.session_state.get("cnn_test_evaluation_cache_key")
        current_cache_key = "|".join(
            (
                source_fingerprint(dataset_path),
                checkpoint_fingerprint(checkpoint_file, default_checkpoint),
                str(device),
            )
        )
        current_result = cached_result if cached_key == current_cache_key else None
        with status_col:
            if current_result is not None:
                st.success("Evaluation cached for this dataset/checkpoint.")
            else:
                st.info("Test split not evaluated.")
        with count_col:
            st.metric("TEST-SET RECORDS", f"{testing_count:,}")

        run_test_evaluation = st.button(
            "Run Test Evaluation",
            type="primary",
            disabled=model is None or testing_count == 0,
            key="run_cnn_test_evaluation",
        )
        if model is None:
            st.warning("Load a compatible checkpoint.")
        elif testing_count == 0:
            st.error("No testing records in dataset.")

        if run_test_evaluation and model is not None and current_result is None:
            progress_bar = st.progress(0.0)
            progress_text = st.empty()

            def update_evaluation_progress(completed: int, total: int) -> None:
                progress_bar.progress(completed / total)
                progress_text.caption(f"Evaluated {completed:,}/{total:,} records")

            try:
                evaluated = evaluate_cnn_test_set(
                    model,
                    prepared,
                    normalization=stats,
                    device=device,
                    batch_size=64,
                    progress_callback=update_evaluation_progress,
                )
                st.session_state["cnn_test_evaluation_result"] = evaluated
                st.session_state["cnn_test_evaluation_cache_key"] = current_cache_key
                current_result = evaluated
                progress_bar.progress(1.0)
                progress_text.caption(
                    f"Evaluated {evaluated.record_count:,} records"
                )
            except (OSError, ValueError, RuntimeError, TypeError) as exc:
                progress_text.empty()
                st.error(f"Test evaluation failed: {exc}")

        if current_result is not None:
            st.markdown("#### Test Metrics")
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            metric_col1.metric("MICRO MAE", f"{current_result.micro_metrics.mae:.5f} V")
            metric_col2.metric("MICRO RMSE", f"{current_result.micro_metrics.rmse:.5f} V")
            micro_r2 = current_result.micro_metrics.r_squared
            metric_col3.metric("GLOBAL MICRO R²", "N/A" if micro_r2 is None else f"{micro_r2:.6f}")
            metric_col4.metric(
                "MICRO CORRELATION",
                "N/A" if current_result.micro_metrics.correlation is None else f"{current_result.micro_metrics.correlation:.6f}",
            )

            with st.expander("Additional aggregate metrics", expanded=False):
                extra_cols = st.columns(4)
                extra_cols[0].metric("MICRO MSE", f"{current_result.micro_metrics.mse:.6f} V²")
                extra_cols[1].metric(
                    "MAXIMUM ABSOLUTE ERROR",
                    f"{current_result.micro_metrics.maximum_absolute_error:.5f} V",
                )
                extra_cols[2].metric("MACRO MAE", f"{current_result.macro_metrics.mae:.5f} V")
                extra_cols[3].metric("MACRO RMSE", f"{current_result.macro_metrics.rmse:.5f} V")
                macro_rows = [
                    {
                        "Aggregation": "Micro (all waveform points)",
                        "MAE (V)": current_result.micro_metrics.mae,
                        "MSE (V²)": current_result.micro_metrics.mse,
                        "RMSE (V)": current_result.micro_metrics.rmse,
                        "R²": current_result.micro_metrics.r_squared,
                        "Correlation": current_result.micro_metrics.correlation,
                    },
                    {
                        "Aggregation": "Macro (mean per-record scores)",
                        "MAE (V)": current_result.macro_metrics.mae,
                        "MSE (V²)": current_result.macro_metrics.mse,
                        "RMSE (V)": current_result.macro_metrics.rmse,
                        "R²": current_result.macro_metrics.r_squared,
                        "Correlation": current_result.macro_metrics.correlation,
                    },
                ]
                st.dataframe(macro_rows, width="stretch", hide_index=True)

            st.markdown("#### Test Record")
            test_ids = current_result.sample_ids.astype(int).tolist()
            selected_test_id = st.selectbox(
                "TEST SAMPLE ID",
                test_ids,
                key="cnn_test_sample_id",
            )
            test_index = int(
                np.flatnonzero(current_result.sample_ids == int(selected_test_id))[0]
            )
            test_left, test_right = st.columns(2)
            with test_left:
                waveform_figure = make_chart(
                    [
                        ("Actual target", current_result.targets_v[test_index], "#43d6c5", "-"),
                        ("CNN prediction", current_result.predictions_v[test_index], "#659df5", "--"),
                    ],
                    f"Reference · Sample {selected_test_id}",
                    "Voltage (V)",
                )
                st.pyplot(waveform_figure, width="stretch")
                plt.close(waveform_figure)
            with test_right:
                error_figure = make_chart(
                    [("Prediction error", current_result.errors_v[test_index], "#e58d8d", "-")],
                    f"Prediction Error · Sample {selected_test_id}",
                    "Error (V)",
                )
                st.pyplot(error_figure, width="stretch")
                plt.close(error_figure)
            record_metrics = current_result.per_record_metrics[test_index]
            record_r2 = "N/A" if record_metrics.r_squared is None else f"{record_metrics.r_squared:.6f}"
            st.caption(
                f"Sample {selected_test_id} · MAE {record_metrics.mae:.5f} V · "
                f"RMSE {record_metrics.rmse:.5f} V · R² {record_r2} · "
                f"max |error| {record_metrics.maximum_absolute_error:.5f} V"
            )

            distribution_figure, distribution_axis = plt.subplots(figsize=(11, 3.8))
            distribution_figure.patch.set_facecolor("#111a26")
            distribution_axis.set_facecolor("#111a26")
            distribution_axis.hist(
                current_result.errors_v.reshape(-1),
                bins=60,
                color="#43d6c5",
                edgecolor="#172332",
            )
            distribution_axis.set_title(
                "Test Error Distribution",
                color="#edf3fa",
                loc="left",
                pad=12,
            )
            distribution_axis.set_xlabel("Error (V)", color="#9aabc0")
            distribution_axis.set_ylabel("Waveform points", color="#9aabc0")
            distribution_axis.tick_params(colors="#9aabc0", labelsize=8)
            distribution_axis.grid(True, color="#344356", alpha=.45, linestyle=":")
            for spine in distribution_axis.spines.values():
                spine.set_color("#344356")
            distribution_figure.tight_layout()
            st.pyplot(distribution_figure, width="stretch")
            plt.close(distribution_figure)

            distribution = current_result.error_distribution
            st.caption(
                "Pooled error (prediction − target): "
                f"mean {distribution.mean:.5f} V, standard deviation "
                f"{distribution.standard_deviation:.5f} V, 1st–99th percentile "
                f"[{distribution.percentile_1:.5f}, {distribution.percentile_99:.5f}] V."
            )

            st.markdown("#### Test Results")
            per_sample_rows = [
                {
                    "SampleID": record.sample_id,
                    "MAE (V)": record.mae,
                    "MSE (V²)": record.mse,
                    "RMSE (V)": record.rmse,
                    "R²": record.r_squared,
                    "Max |error| (V)": record.maximum_absolute_error,
                }
                for record in current_result.per_record_metrics
            ]
            st.dataframe(per_sample_rows, width="stretch", hide_index=True)
            per_sample_buffer = io.StringIO()
            writer = csv.DictWriter(per_sample_buffer, fieldnames=list(per_sample_rows[0]))
            writer.writeheader()
            writer.writerows(per_sample_rows)
            aggregate_buffer = evaluation_summary_payload(current_result)
            download_col1, download_col2 = st.columns(2)
            download_col1.download_button(
                "Download per-sample metrics (CSV)",
                data=per_sample_buffer.getvalue(),
                file_name="cnn_test_per_sample_metrics.csv",
                mime="text/csv",
            )
            download_col2.download_button(
                "Download aggregate evaluation summary (JSON)",
                data=aggregate_buffer,
                file_name="cnn_test_aggregate_summary.json",
                mime="application/json",
            )
            st.markdown(
                """
                <div class="notice">
                    CNN prediction · reference voltage only
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------
    # DATA DIAGNOSTICS
    # --------------------------------------------------------

    with diagnostics_tab:
        st.markdown(
            '<div class="section-heading">Model Diagnostics</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-sub">Dataset Metadata</div>',
            unsafe_allow_html=True,
        )

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("TOTAL RECORDS", f"{len(prepared.sample_ids):,}")
        d2.metric("WAVEFORM POINTS", f"{prepared.target_voltage.shape[1]:,}")
        d3.metric("OPERATING FEATURES", len(prepared.operating_feature_names))
        d4.metric("COMPUTE DEVICE", str(device).upper())

        st.markdown("#### Dataset partition")
        partition_rows = [
            {
                "Partition": str(name).title(),
                "Records": int(count),
                "Percentage": f"{100 * int(count) / len(prepared.sample_ids):.1f}%",
            }
            for name, count in split_counts.items()
        ]
        st.dataframe(
            partition_rows,
            width="stretch",
            hide_index=True,
        )

        st.markdown("#### Dataset configuration")
        configuration_rows = [
            {"Parameter": str(key), "Value": str(value)}
            for key, value in config.items()
        ]
        if configuration_rows:
            st.dataframe(
                configuration_rows,
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("No configuration metadata.")

        st.markdown("#### Operating-condition features")
        st.dataframe(
            [
                {"Feature": str(name)}
                for name in prepared.operating_feature_names
            ],
            width="stretch",
            hide_index=True,
        )

        st.markdown("#### Model status")
        checkpoint_rows = [
            {"Property": "Checkpoint", "Value": checkpoint_status},
            {"Property": "Compute device", "Value": str(device).upper()},
            {"Property": "Model loaded", "Value": "Yes" if model is not None else "No"},
            {
                "Property": "Trainable parameters",
                "Value": f"{parameter_count:,}" if model is not None else "Unavailable",
            },
            {
                "Property": "Prediction target",
                "Value": "Reference target voltage waveform",
            },
        ]
        if model is not None:
            checkpoint_name = (
                checkpoint_file.name if checkpoint_file is not None else str(default_checkpoint)
            )
            checkpoint_rows.extend(
                [
                    {"Property": "Checkpoint file", "Value": checkpoint_name},
                    {
                        "Property": "Selected epoch (zero-based)",
                        "Value": str(checkpoint.get("epoch", "Unavailable")),
                    },
                    {
                        "Property": "Best validation loss",
                        "Value": (
                            f"{float(checkpoint['best_validation_loss']):.8f}"
                            if checkpoint.get("best_validation_loss") is not None
                            else "Unavailable"
                        ),
                    },
                ]
            )
        st.dataframe(
            checkpoint_rows,
            width="stretch",
            hide_index=True,
        )
        if model is not None:
            st.markdown("#### CNN architecture")
            architecture = model.config
            st.dataframe(
                [
                    {"Component": "Waveform input", "Configuration": f"{architecture.waveform_points} points"},
                    {"Component": "Operating conditions", "Configuration": f"{architecture.operating_features} normalized features"},
                    {"Component": "Voltage/current branches", "Configuration": str(architecture.branch_channels)},
                    {"Component": "Condition encoder width", "Configuration": str(architecture.condition_width)},
                    {"Component": "Fusion channels", "Configuration": str(architecture.fusion_channels)},
                    {"Component": "Convolution kernel", "Configuration": str(architecture.kernel_size)},
                    {"Component": "Output", "Configuration": "One predicted target-voltage value per waveform point"},
                ],
                width="stretch",
                hide_index=True,
            )
        history_path = (
            default_checkpoint.parent.parent / "results" / "training_history.csv"
        )
        if (
            checkpoint_file is None
            and checkpoint_source is not None
            and history_path.is_file()
        ):
            with history_path.open("r", newline="", encoding="utf-8") as history_file:
                history_rows = list(csv.DictReader(history_file))
            if history_rows:
                epochs = np.asarray(
                    [float(row["epoch"]) for row in history_rows], dtype=np.float64
                )
                training_loss = np.asarray(
                    [float(row["train_loss"]) for row in history_rows], dtype=np.float64
                )
                validation_loss = np.asarray(
                    [float(row["validation_loss"]) for row in history_rows],
                    dtype=np.float64,
                )
                history_figure = make_chart(
                    [
                        ("Training loss", training_loss, "#43d6c5", "-"),
                        ("Validation loss", validation_loss, "#659df5", "--"),
                    ],
                    "Saved training history",
                    "Loss on normalized target scale (–)",
                    x_values=epochs,
                    x_label="Epoch index (–)",
                )
                st.pyplot(history_figure, width="stretch")
                plt.close(history_figure)
                if model is not None:
                    st.caption(
                        f"Checkpoint-selected epoch: {int(checkpoint['epoch'])} "
                        f"(zero-based); stored best validation loss: "
                        f"{float(checkpoint['best_validation_loss']):.8f}."
                    )
        elif checkpoint_file is not None:
            st.caption(
                "No matching training history for uploaded checkpoint."
            )

    # --------------------------------------------------------
    # LIVE INVERTER DEMONSTRATION
    # --------------------------------------------------------

    with live_demo_tab:
        st.markdown(
            '<div class="section-heading">Live Simulation</div>',
            unsafe_allow_html=True,
        )
        st.markdown("#### Presets")
        preset_columns = st.columns(len(LIVE_PRESETS))
        for preset_column, preset in zip(preset_columns, LIVE_PRESETS):
            with preset_column:
                if st.button(
                    preset.name,
                    key=f"live_preset_{preset.name.lower().replace(' ', '_')}",
                    use_container_width=True,
                ):
                    reset_generation = int(
                        st.session_state.get("live_demo_reset_generation", 0)
                    )
                    st.session_state["live_selected_preset"] = preset.name
                    st.session_state[
                        f"live_demo_modulation_index_{reset_generation}"
                    ] = preset.modulation_index
                    st.session_state[
                        f"live_demo_dc_voltage_{reset_generation}"
                    ] = preset.dc_voltage_per_cell_v
                    st.session_state[
                        f"live_demo_load_resistance_{reset_generation}"
                    ] = preset.load_resistance_ohm
                    st.session_state[
                        f"live_demo_load_inductance_{reset_generation}"
                    ] = preset.load_inductance_h
                    st.session_state[
                        f"live_demo_phase_angle_{reset_generation}"
                    ] = preset.phase_angle_degrees
                    st.rerun()

        live_cache_key = (
            checkpoint_fingerprint(checkpoint_file, default_checkpoint),
            source_fingerprint(dataset_path),
        )
        if (
            "live_demo_result" in st.session_state
            and st.session_state.get("live_demo_cache_key") != live_cache_key
        ):
            st.session_state.pop("live_demo_result", None)
            st.session_state.pop("live_demo_cache_key", None)
            st.session_state.pop("live_demo_optimization", None)
            st.session_state.pop("live_demo_candidate", None)
            st.session_state.pop(
                "live_demo_optimization_configuration", None
            )
            st.session_state.pop("live_demo_export_bundle", None)

        reset_generation = int(st.session_state.get("live_demo_reset_generation", 0))
        default_modulation = float(
            st.session_state.get(f"live_demo_modulation_index_{reset_generation}", 0.75)
        )
        default_dc_voltage = float(
            st.session_state.get(f"live_demo_dc_voltage_{reset_generation}", 100.0)
        )
        default_resistance = float(
            st.session_state.get(f"live_demo_load_resistance_{reset_generation}", 30.0)
        )
        default_inductance = float(
            st.session_state.get(f"live_demo_load_inductance_{reset_generation}", 0.03)
        )
        default_phase = int(
            st.session_state.get(f"live_demo_phase_angle_{reset_generation}", 0)
        )
        selected_preset = st.session_state.get("live_selected_preset")
        if selected_preset:
            st.caption(f"Preset: {selected_preset}")

        with st.container():
            input_col1, input_col2, input_col3, input_col4, input_col5 = st.columns(5)
            with input_col1:
                live_modulation_index = st.slider(
                    "MODULATION INDEX",
                    min_value=0.55,
                    max_value=0.95,
                    step=0.01,
                    value=default_modulation,
                    key=f"live_demo_modulation_index_{reset_generation}",
                )
            with input_col2:
                live_dc_voltage = st.number_input(
                    "DC/Cell (V)",
                    min_value=1.0,
                    max_value=500.0,
                    value=default_dc_voltage,
                    step=5.0,
                    key=f"live_demo_dc_voltage_{reset_generation}",
                    help="CNN training used 100 V/cell; other voltages are out of support.",
                )
            with input_col3:
                live_load_resistance = st.selectbox(
                    "LOAD RESISTANCE (Ω)",
                    ALLOWED_LOAD_RESISTANCES_OHM,
                    index=ALLOWED_LOAD_RESISTANCES_OHM.index(default_resistance),
                    key=f"live_demo_load_resistance_{reset_generation}",
                )
            with input_col4:
                live_load_inductance = st.selectbox(
                    "LOAD INDUCTANCE (H)",
                    ALLOWED_LOAD_INDUCTANCES_H,
                    index=ALLOWED_LOAD_INDUCTANCES_H.index(default_inductance),
                    key=f"live_demo_load_inductance_{reset_generation}",
                )
            with input_col5:
                live_phase_angle = st.slider(
                    "PHASE ANGLE (°)",
                    min_value=0,
                    max_value=360,
                    step=1,
                    value=default_phase,
                    key=f"live_demo_phase_angle_{reset_generation}",
                )
            carrier_col1, carrier_col2, carrier_col3 = st.columns(3)
            with carrier_col1:
                live_carrier_1 = st.number_input(
                    "C1 Shift (°)",
                    min_value=0.0,
                    max_value=360.0,
                    value=0.0,
                    step=1.0,
                    key=f"live_demo_carrier_1_{reset_generation}",
                )
            with carrier_col2:
                live_carrier_2 = st.number_input(
                    "C2 Shift (°)",
                    min_value=0.0,
                    max_value=360.0,
                    value=60.0,
                    step=1.0,
                    key=f"live_demo_carrier_2_{reset_generation}",
                )
            with carrier_col3:
                live_carrier_3 = st.number_input(
                    "C3 Shift (°)",
                    min_value=0.0,
                    max_value=360.0,
                    value=120.0,
                    step=1.0,
                    key=f"live_demo_carrier_3_{reset_generation}",
                )
            run_live = st.button(
                "Run New Simulation",
                disabled=model is None or stats is None,
                type="primary",
                use_container_width=True,
            )

        training_indices = prepared.indices("training")
        normalization_for_bounds = stats if stats is not None else prepared.stats
        training_conditions = (
            prepared.operating_conditions[training_indices].astype(np.float64)
            * np.asarray(normalization_for_bounds.operating_std, dtype=np.float64)
            + np.asarray(normalization_for_bounds.operating_mean, dtype=np.float64)
        )
        selected_conditions = np.asarray(
            [
                live_modulation_index,
                live_dc_voltage,
                live_load_resistance,
                live_load_inductance,
                np.deg2rad(live_phase_angle),
            ],
            dtype=np.float64,
        )
        selected_carrier_offsets = normalize_carrier_phase_offsets(
            (live_carrier_1, live_carrier_2, live_carrier_3)
        )
        selected_model_conditions = np.asarray(
            [
                live_modulation_index,
                live_load_resistance,
                live_load_inductance,
                np.deg2rad(live_phase_angle),
            ],
            dtype=np.float64,
        )
        support_warnings = operating_condition_support_warnings(
            selected_model_conditions,
            training_conditions,
            prepared.operating_feature_names,
        )
        if support_warnings:
            st.warning(
                "CNN extrapolation risk: " + " ".join(support_warnings)
            )
        else:
            st.caption(
                "Within marginal support; joint support unknown."
            )
        if not np.isclose(live_dc_voltage, 100.0):
            st.warning(
                "CNN trained at 100 V/cell; other voltages are out of support."
            )

        with st.expander("OFFLINE SWITCHING-PHASE RECOMMENDER", expanded=False):
            st.warning(
                "Local POA–PFO; gates and THD are simulator-verified."
            )
            advisor_cols = st.columns(4)
            advisor_modulation = advisor_cols[0].number_input(
                "ADVISOR MODULATION",
                min_value=0.0,
                max_value=1.0,
                value=float(live_modulation_index),
                step=0.05,
                format="%.2f",
                key="switching_advisor_modulation",
            )
            advisor_phase = advisor_cols[1].number_input(
                "Phase (°)",
                min_value=0.0,
                max_value=360.0,
                value=float(live_phase_angle),
                step=1.0,
                key="switching_advisor_reference_phase",
            )
            advisor_dc = advisor_cols[2].number_input(
                "DC/Cell (V)",
                min_value=0.0,
                value=float(live_dc_voltage),
                step=1.0,
                key="switching_advisor_dc_voltage",
            )
            advisor_objective = advisor_cols[3].selectbox(
                "OPTIMIZATION OBJECTIVE",
                ("voltage", "current"),
                format_func=lambda value: f"{value.title()} THD",
                key="switching_advisor_objective",
            )
            advisor_cols = st.columns(4)
            advisor_resistance = advisor_cols[0].number_input(
                "LOAD RESISTANCE (Ω)",
                min_value=0.0,
                value=float(live_load_resistance),
                step=1.0,
                key="switching_advisor_resistance",
            )
            advisor_inductance = advisor_cols[1].number_input(
                "LOAD INDUCTANCE (H)",
                min_value=0.0,
                value=float(live_load_inductance),
                step=0.001,
                format="%.3f",
                key="switching_advisor_inductance",
            )
            advisor_seed = advisor_cols[2].number_input(
                "RANDOM SEED",
                min_value=0,
                max_value=2**31 - 1,
                value=42,
                step=1,
                key="switching_advisor_seed",
            )
            advisor_run = advisor_cols[3].button(
                "GENERATE VERIFIED RECOMMENDATION",
                type="primary",
                key="run_switching_advisor",
            )
            advisor_configuration = (
                float(advisor_modulation),
                float(advisor_phase),
                float(advisor_dc),
                float(advisor_resistance),
                float(advisor_inductance),
                str(advisor_objective),
                int(advisor_seed),
            )
            if advisor_run:
                st.session_state.pop("switching_advisor_result", None)
                st.session_state.pop("switching_advisor_error", None)
                try:
                    with st.spinner(
                        "Optimizing carrier phases and verifying simulator gates and THD..."
                    ):
                        st.session_state["switching_advisor_result"] = recommend_switching(
                            modulation_index=advisor_configuration[0],
                            reference_phase_degrees=advisor_configuration[1],
                            dc_voltage_v=advisor_configuration[2],
                            resistance_ohm=advisor_configuration[3],
                            inductance_h=advisor_configuration[4],
                            objective=advisor_configuration[5],
                            seed=advisor_configuration[6],
                        )
                    st.session_state["switching_advisor_configuration"] = (
                        advisor_configuration
                    )
                except (ValueError, RuntimeError) as exc:
                    st.session_state["switching_advisor_error"] = (
                        f"Recommendation failed: {type(exc).__name__}: {exc}"
                    )
            advisor_error = st.session_state.get("switching_advisor_error")
            if advisor_error:
                st.error(advisor_error)
            advisor_result = st.session_state.get("switching_advisor_result")
            if (
                isinstance(advisor_result, SwitchingRecommendation)
                and st.session_state.get("switching_advisor_configuration")
                == advisor_configuration
            ):
                st.markdown(
                    "**Carrier offsets:** "
                    + ", ".join(
                        f"{phase:.2f}°"
                        for phase in advisor_result.carrier_phase_offsets_degrees
                    )
                )
                if advisor_result.objective == "voltage":
                    baseline_metric_label = "BASELINE EVENT TOTAL VOLTAGE THD"
                    verified_metric_label = (
                        "SIMULATOR-VERIFIED EVENT TOTAL VOLTAGE THD"
                    )
                else:
                    baseline_metric_label = "Base Event I-H2–100"
                    verified_metric_label = (
                        "Verified Event I-H2–100"
                    )
                result_cols = st.columns(4)
                result_cols[0].metric(
                    baseline_metric_label,
                    f"{advisor_result.baseline_thd_percent:.6f}%",
                )
                result_cols[1].metric(
                    verified_metric_label,
                    f"{advisor_result.simulator_verified_thd_percent:.6f}%",
                )
                result_cols[2].metric(
                    "CHANGE FROM BASELINE",
                    f"{advisor_result.percent_change:+.6f}%",
                    delta_color="inverse",
                )
                result_cols[3].metric(
                    "GATE ARRAY SHAPE",
                    str(advisor_result.gate_states.shape),
                )
                if advisor_result.used_baseline_fallback:
                    st.info(advisor_result.message)
                else:
                    st.success(advisor_result.message)
                st.caption(
                    f"Sample V THD: "
                    f"{advisor_result.verified_metrics.sampled_voltage_thd_percent:.6f}% · "
                    f"Event V H2–100: "
                    f"{advisor_result.verified_metrics.event_voltage_harmonics_2_100_thd_percent:.6f}% · "
                    f"Event I H2–100: "
                    f"{advisor_result.verified_metrics.event_current_harmonics_2_100_thd_percent:.6f}% · "
                    f"Sample I THD: "
                    f"{advisor_result.verified_metrics.sampled_current_thd_percent:.6f}%"
                )
                st.caption(
                    "When voltage THD is selected, the objective is event-based "
                    "Parseval total THD, including out-of-band switching energy. "
                    "Event H2–100 values cover only those harmonic orders; sampled "
                    "H2–100 values use the 20 kHz waveform. These finite-band and "
                    "full-spectrum measurements are not interchangeable."
                )
                st.caption(
                    "Verified binary gates; no shoot-through."
                )
                gate_frame = advisor_result.gate_states.reshape(12, 400)
                st.dataframe(gate_frame, width="stretch", height=220)
                gate_csv = io.StringIO()
                np.savetxt(gate_csv, gate_frame, delimiter=",", fmt="%d")
                st.download_button(
                    "Gate CSV",
                    data=gate_csv.getvalue(),
                    file_name="simulator_verified_gate_states.csv",
                    mime="text/csv",
                    key="download_switching_advisor_gates",
                )
                st.caption(
                    f"Seed {advisor_result.seed} · "
                    f"{advisor_result.unique_candidate_evaluations} unique optimizer "
                    f"evaluations · {advisor_result.objective_requests} objective "
                    f"requests · {advisor_result.runtime_seconds:.3f} s. "
                    "Predicted THD: not used."
                )

        with st.expander("VALIDATED THD EXPERIMENT RESULTS", expanded=False):
            st.caption(
                "Primary before/after metric: event-based Parseval total voltage THD. "
                "Event-based voltage/current harmonics 2–100 and sampled 20 kHz "
                "voltage/current THD are separate measurements, not interchangeable."
            )
            thd_results_path = PROJECT_ROOT / "results" / "thd_experiment.csv"
            if not thd_results_path.is_file():
                st.error(f"Experiment results CSV not found: {thd_results_path}")
            else:
                try:
                    thd_results = pd.read_csv(thd_results_path)
                except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
                    st.error(
                        f"Could not load THD experiment results: "
                        f"{type(exc).__name__}: {exc}"
                    )
                else:
                    required_result_columns = {
                        "seed",
                        "modulation_index",
                        "method",
                        "cell_1_phase_degrees",
                        "cell_2_phase_degrees",
                        "cell_3_phase_degrees",
                        "baseline_event_voltage_total_thd_percent",
                        "optimized_event_voltage_total_thd_percent",
                        "percent_reduction_event_voltage_total_thd",
                        "improved_on_baseline",
                        "event_voltage_harmonics_2_100_thd_percent",
                        "sampled_voltage_thd_percent",
                        "event_current_harmonics_2_100_thd_percent",
                        "sampled_current_thd_percent",
                        "full_event_thd_evaluations",
                        "simulator_waveform_generations",
                        "runtime_seconds",
                    }
                    missing_result_columns = sorted(
                        required_result_columns.difference(thd_results.columns)
                    )
                    if missing_result_columns:
                        st.error(
                            "Missing columns: "
                            + ", ".join(missing_result_columns)
                        )
                    else:
                        st.markdown("**Method Comparison**")
                        method_summary = (
                            thd_results.groupby("method", as_index=False)
                            .agg(
                                runs=("seed", "count"),
                                baseline_mean_event_voltage_thd=(
                                    "baseline_event_voltage_total_thd_percent",
                                    "mean",
                                ),
                                optimized_mean_event_voltage_thd=(
                                    "optimized_event_voltage_total_thd_percent",
                                    "mean",
                                ),
                                mean_reduction_percent=(
                                    "percent_reduction_event_voltage_total_thd",
                                    "mean",
                                ),
                                improved_runs=("improved_on_baseline", "sum"),
                            )
                            .rename(
                                columns={
                                    "runs": "Runs",
                                    "method": "Method",
                                    "baseline_mean_event_voltage_thd":
                                        "Baseline event voltage total THD (%)",
                                    "optimized_mean_event_voltage_thd":
                                        "Optimized event voltage total THD (%)",
                                    "mean_reduction_percent":
                                        "Mean reduction (%)",
                                    "improved_runs": "Runs improved",
                                }
                            )
                        )
                        st.dataframe(
                            method_summary,
                            width="stretch",
                            hide_index=True,
                        )

                        modulation_options = [
                            "All",
                            *[
                                f"{value:.2f}"
                                for value in sorted(
                                    thd_results["modulation_index"].dropna().unique()
                                )
                            ],
                        ]
                        filter_columns = st.columns(2)
                        selected_result_modulation = filter_columns[0].selectbox(
                            "MODULATION",
                            modulation_options,
                            key="thd_results_modulation_filter",
                        )
                        method_options = [
                            "All",
                            *sorted(thd_results["method"].dropna().unique()),
                        ]
                        selected_result_method = filter_columns[1].selectbox(
                            "METHOD",
                            method_options,
                            key="thd_results_method_filter",
                        )
                        selected_results = thd_results.copy()
                        if selected_result_modulation != "All":
                            selected_results = selected_results.loc[
                                np.isclose(
                                    selected_results["modulation_index"],
                                    float(selected_result_modulation),
                                )
                            ]
                        if selected_result_method != "All":
                            selected_results = selected_results.loc[
                                selected_results["method"] == selected_result_method
                            ]

                        st.markdown(
                            "**Per-Seed Results**"
                        )
                        detail_columns = [
                            "seed",
                            "modulation_index",
                            "method",
                            "cell_1_phase_degrees",
                            "cell_2_phase_degrees",
                            "cell_3_phase_degrees",
                            "baseline_event_voltage_total_thd_percent",
                            "optimized_event_voltage_total_thd_percent",
                            "percent_reduction_event_voltage_total_thd",
                            "event_voltage_harmonics_2_100_thd_percent",
                            "sampled_voltage_thd_percent",
                            "event_current_harmonics_2_100_thd_percent",
                            "sampled_current_thd_percent",
                            "full_event_thd_evaluations",
                            "simulator_waveform_generations",
                            "runtime_seconds",
                        ]
                        st.dataframe(
                            selected_results[detail_columns].rename(
                                columns={
                                    "seed": "Seed",
                                    "modulation_index": "Modulation",
                                    "method": "Method",
                                    "cell_1_phase_degrees": "Cell 1 phase (°)",
                                    "cell_2_phase_degrees": "Cell 2 phase (°)",
                                    "cell_3_phase_degrees": "Cell 3 phase (°)",
                                    "baseline_event_voltage_total_thd_percent":
                                        "Baseline event voltage total THD (%)",
                                    "optimized_event_voltage_total_thd_percent":
                                        "Optimized event voltage total THD (%)",
                                    "percent_reduction_event_voltage_total_thd":
                                        "Reduction (%)",
                                    "event_voltage_harmonics_2_100_thd_percent":
                                        "Event voltage H2–100 THD (%)",
                                    "sampled_voltage_thd_percent":
                                        "Sampled voltage H2–100 THD (%)",
                                    "event_current_harmonics_2_100_thd_percent":
                                        "Event current H2–100 THD (%)",
                                    "sampled_current_thd_percent":
                                        "Sampled current H2–100 THD (%)",
                                    "full_event_thd_evaluations":
                                        "Full event-THD evaluations",
                                    "simulator_waveform_generations":
                                        "Simulator waveform generations",
                                    "runtime_seconds": "Runtime (s)",
                                }
                            ),
                            width="stretch",
                            hide_index=True,
                        )

                        figure_specs = (
                            (
                                "thd_experiment_before_after.png",
                                "Event THD",
                            ),
                            (
                                "thd_experiment_convergence.png",
                                "Convergence",
                            ),
                            (
                                "thd_experiment_waveform_spectrum.png",
                                "Waveform & Spectrum",
                            ),
                        )
                        figure_columns = st.columns(len(figure_specs))
                        for figure_column, (figure_name, caption) in zip(
                            figure_columns, figure_specs
                        ):
                            figure_path = PROJECT_ROOT / "results" / figure_name
                            if figure_path.is_file():
                                figure_column.image(
                                    str(figure_path),
                                    caption=caption,
                                )
                            else:
                                figure_column.warning(
                                    f"Figure unavailable: {figure_name}"
                                )

        with st.expander("AI PROJECT CHAT ASSISTANT (LOCAL MODEL)", expanded=True):
            chat_model = st.text_input(
                "LOCAL CHAT MODEL",
                value=st.session_state.get(
                    "project_chat_model",
                    DEFAULT_CHAT_MODEL,
                ),
                key="project_chat_model_input",
                help=(
                    "Ollama model tag; install with `ollama pull <tag>`."
                ),
            )
            st.session_state["project_chat_model"] = chat_model.strip()
            client = OllamaChatClient()
            connection_col, clear_col = st.columns([1, 1])
            if connection_col.button(
                "Check Model",
                key="check_project_chat_model",
            ):
                try:
                    available_models = client.list_models()
                    st.session_state["project_chat_available_models"] = available_models
                    if chat_model.strip() in available_models:
                        st.success(f"Local model is available: {chat_model.strip()}")
                    else:
                        st.warning(
                            f"Model {chat_model.strip()!r} is not installed. "
                            f"Run `ollama pull {chat_model.strip() or DEFAULT_CHAT_MODEL}`."
                        )
                except LocalChatError as exc:
                    st.error(str(exc))
            if clear_col.button("Clear Chat", key="clear_project_chat"):
                st.session_state["project_chat_history"] = []
                st.session_state.pop("project_chat_recommendation", None)
                st.rerun()

            conversation = st.session_state.setdefault("project_chat_history", [])
            for message in conversation:
                role = message.get("role")
                content = message.get("content")
                if role in ("user", "assistant") and isinstance(content, str) and content:
                    with st.chat_message(role):
                        st.markdown(content)

            stored_live_result = st.session_state.get("live_demo_result")
            chat_context = {
                "dataset": {
                    "validated": True,
                    "record_count": int(len(prepared.sample_ids)),
                    "split_counts": split_counts,
                    "waveform_points": int(prepared.input_voltage.shape[1]),
                },
                "cnn": {
                    "checkpoint_status": checkpoint_status,
                    "model_loaded": model is not None,
                    "purpose": "reference target waveform prediction only",
                    "current_live_prediction_metrics": (
                        stored_live_result.cnn_metrics
                        if isinstance(stored_live_result, LiveSimulationResult)
                        else None
                    ),
                },
                "current_dashboard_conditions": {
                    "modulation_index": float(live_modulation_index),
                    "reference_phase_degrees": float(live_phase_angle),
                    "dc_voltage_per_cell_v": float(live_dc_voltage),
                    "load_resistance_ohm": float(live_load_resistance),
                    "load_inductance_h": float(live_load_inductance),
                },
                "latest_live_simulation": (
                    {
                        "event_parseval_total_voltage_thd_percent":
                            stored_live_result.harmonic_result.total_thd_percent,
                        "event_voltage_h2_100_thd_percent":
                            stored_live_result.harmonic_result.thd_percent,
                        "sampled_voltage_h2_100_thd_percent":
                            stored_live_result.voltage_thd_percent,
                        "sampled_current_h2_100_thd_percent":
                            stored_live_result.current_thd_percent,
                        "carrier_phase_offsets_degrees":
                            stored_live_result.carrier_phase_offsets_degrees,
                    }
                    if isinstance(stored_live_result, LiveSimulationResult)
                    else None
                ),
                "limits": [
                    "ideal software simulation only",
                    "no hardware or real-time control",
                    "no protection system",
                    "CNN does not predict switching states or select phases",
                ],
            }

            def execute_chat_tool(
                tool_name: str,
                arguments: dict[str, object],
            ) -> dict[str, object]:
                argument_aliases = {
                    "dc_voltage_v": "dc_voltage_per_cell_v",
                    "resistance_ohm": "load_resistance_ohm",
                    "inductance_h": "load_inductance_h",
                }
                allowed_fields = {
                    "modulation_index",
                    "reference_phase_degrees",
                    "dc_voltage_per_cell_v",
                    "load_resistance_ohm",
                    "load_inductance_h",
                    "objective",
                }
                if tool_name != "run_simulator_verified_recommendation":
                    raise ValueError(f"Unsupported assistant tool: {tool_name}.")
                normalized_arguments: dict[str, object] = {}
                for field, value in arguments.items():
                    canonical_field = argument_aliases.get(field, field)
                    if canonical_field not in allowed_fields:
                        raise ValueError(
                            f"Unsupported recommendation argument: {field}."
                        )
                    if canonical_field in normalized_arguments:
                        raise ValueError(
                            f"Recommendation argument {canonical_field} was provided twice."
                        )
                    normalized_arguments[canonical_field] = value
                numeric_fields = allowed_fields - {"objective"}
                if any(
                    isinstance(value, bool) or not isinstance(value, (int, float))
                    for field, value in normalized_arguments.items()
                    if field in numeric_fields
                ):
                    raise ValueError("Recommendation numeric arguments must be numbers.")
                if "objective" in normalized_arguments and not isinstance(
                    normalized_arguments["objective"], str
                ):
                    raise ValueError("Recommendation objective must be text.")
                selected_conditions = chat_context["current_dashboard_conditions"]
                conditions = {
                    "modulation_index": normalized_arguments.get(
                        "modulation_index",
                        selected_conditions["modulation_index"],
                    ),
                    "reference_phase_degrees": normalized_arguments.get(
                        "reference_phase_degrees",
                        selected_conditions["reference_phase_degrees"],
                    ),
                    "dc_voltage_v": normalized_arguments.get(
                        "dc_voltage_per_cell_v",
                        selected_conditions["dc_voltage_per_cell_v"],
                    ),
                    "resistance_ohm": normalized_arguments.get(
                        "load_resistance_ohm",
                        selected_conditions["load_resistance_ohm"],
                    ),
                    "inductance_h": normalized_arguments.get(
                        "load_inductance_h",
                        selected_conditions["load_inductance_h"],
                    ),
                    "objective": normalized_arguments.get("objective", "voltage"),
                }
                recommendation = recommend_switching(
                    **conditions,
                )
                st.session_state["project_chat_recommendation"] = recommendation
                return {
                    "objective": recommendation.objective,
                    "carrier_phase_offsets_degrees":
                        recommendation.carrier_phase_offsets_degrees,
                    "baseline_objective_thd_percent":
                        recommendation.baseline_thd_percent,
                    "simulator_verified_objective_thd_percent":
                        recommendation.simulator_verified_thd_percent,
                    "percent_change": recommendation.percent_change,
                    "fallback_to_baseline": recommendation.used_baseline_fallback,
                    "result_message": recommendation.message,
                }

            submitted_prompt = st.chat_input(
                "Ask about the project, CNN, THD, or request a verified phase/gate recommendation",
                key="project_chat_input",
            )
            if submitted_prompt:
                conversation.append({"role": "user", "content": submitted_prompt})
                try:
                    with st.spinner("Thinking with the local model..."):
                        _, updated_conversation, _ = chat_with_tools(
                            client,
                            model=chat_model,
                            conversation=conversation,
                            project_context=chat_context,
                            tool_executor=execute_chat_tool,
                        )
                    st.session_state["project_chat_history"] = updated_conversation
                except (LocalChatError, ValueError, RuntimeError) as exc:
                    conversation.append(
                        {
                            "role": "assistant",
                            "content": f"Local assistant error: {type(exc).__name__}: {exc}",
                        }
                    )
                    st.session_state["project_chat_history"] = conversation
                st.rerun()

            recommendation = st.session_state.get("project_chat_recommendation")
            if isinstance(recommendation, SwitchingRecommendation):
                st.markdown("**Latest simulator-verified recommendation**")
                st.write(
                    "Carrier offsets: "
                    + ", ".join(
                        f"{offset:.2f}°"
                        for offset in recommendation.carrier_phase_offsets_degrees
                    )
                )
                result_cols = st.columns(3)
                result_cols[0].metric(
                    f"BASELINE {recommendation.objective.upper()} THD",
                    f"{recommendation.baseline_thd_percent:.6f}%",
                )
                result_cols[1].metric(
                    f"SIMULATOR-VERIFIED {recommendation.objective.upper()} THD",
                    f"{recommendation.simulator_verified_thd_percent:.6f}%",
                )
                result_cols[2].metric(
                    "CHANGE FROM BASELINE",
                    f"{recommendation.percent_change:+.6f}%",
                )
                st.caption(recommendation.message)
                st.caption(
                    "Gate states were generated by the inverter simulator and "
                    "validated for binary values, allowed cell states, and no "
                    "shoot-through. CNN-predicted THD is not used."
                )
                gate_table = recommendation.gate_states.reshape(12, 400)
                gate_labels = [
                    f"cell_{cell}_switch_{switch}"
                    for cell in range(1, 4)
                    for switch in range(1, 5)
                ]
                gate_frame = pd.DataFrame(
                    gate_table,
                    index=gate_labels,
                    columns=[f"sample_{index:03d}" for index in range(400)],
                )
                st.dataframe(gate_frame, width="stretch", height=220)
                gate_csv = gate_frame.to_csv(index_label="gate")
                st.download_button(
                    "DOWNLOAD CHAT RECOMMENDATION GATES (CSV)",
                    data=gate_csv,
                    file_name="chat_simulator_verified_gate_states.csv",
                    mime="text/csv",
                    key="download_project_chat_gates",
                )

        reset_col, status_col = st.columns([1, 3])
        with reset_col:
            if st.button("RESET LIVE DEMONSTRATION", key="reset_live_demo"):
                st.session_state["live_demo_reset_generation"] = reset_generation + 1
                st.session_state.pop("live_demo_result", None)
                st.session_state.pop("live_demo_cache_key", None)
                st.session_state.pop("live_demo_error", None)
                st.session_state.pop("live_demo_optimization", None)
                st.session_state.pop("live_demo_candidate", None)
                st.session_state.pop(
                    "live_demo_optimization_configuration", None
                )
                st.session_state.pop("live_demo_optimization_error", None)
                st.session_state.pop("live_demo_export_bundle", None)
                st.session_state.pop("live_demo_export_error", None)
                st.session_state.pop("live_demo_export_key", None)
                st.rerun()
        with status_col:
            if model is None:
                st.warning(
                    "Live CNN inference is unavailable because no compatible checkpoint is loaded."
                )
            else:
                st.caption(
                    f"CNN checkpoint: {checkpoint_status} · Compute device: {str(device).upper()}"
                )

        if run_live:
            st.session_state.pop("live_demo_error", None)
            if model is None or stats is None:
                st.session_state["live_demo_error"] = (
                    "Live simulation is unavailable because a compatible checkpoint "
                    "and normalization metadata are required."
                )
            else:
                try:
                    with st.spinner(
                        "Simulating inverter, evaluating harmonics, and running CNN inference..."
                    ):
                        new_live_result = run_live_simulation(
                            model,
                            stats,
                            modulation_index=live_modulation_index,
                            dc_voltage_per_cell_v=live_dc_voltage,
                            load_resistance_ohm=live_load_resistance,
                            load_inductance_h=live_load_inductance,
                            phase_angle_degrees=live_phase_angle,
                            device=device,
                            carrier_phase_offsets_degrees=selected_carrier_offsets,
                        )
                        st.session_state["live_demo_result"] = new_live_result
                        st.session_state["live_demo_cache_key"] = live_cache_key
                        st.session_state.pop("live_demo_optimization", None)
                        st.session_state.pop("live_demo_candidate", None)
                        st.session_state.pop(
                            "live_demo_optimization_configuration", None
                        )
                        st.session_state.pop("live_demo_optimization_error", None)
                        st.session_state.pop("live_demo_export_bundle", None)
                        st.session_state.pop("live_demo_export_error", None)
                        st.session_state.pop("live_demo_export_key", None)
                except (ValueError, RuntimeError, TypeError) as exc:
                    st.session_state["live_demo_error"] = (
                        f"Live simulation failed: {type(exc).__name__}: {exc}"
                    )
                    st.error(st.session_state["live_demo_error"])
                    st.exception(exc)

        live_error = st.session_state.get("live_demo_error")
        if live_error:
            st.error(live_error)

        live_result = st.session_state.get("live_demo_result")
        if not isinstance(live_result, LiveSimulationResult):
            st.info(
                "No simulation run yet. Use Run New Simulation to create the "
                "analytical inverter output, load current and switching view."
            )
        live_result_is_current = (
            isinstance(live_result, LiveSimulationResult)
            and live_configuration_matches(
                live_result,
                tuple(float(value) for value in selected_conditions),
                selected_carrier_offsets,
            )
        )
        if isinstance(live_result, LiveSimulationResult) and not live_result_is_current:
            st.info("Controls changed since this run. Click Run New Simulation to update.")
        if isinstance(live_result, LiveSimulationResult):
            with st.expander("MEASURED WAVEFORM VALIDATION (CSV)", expanded=False):
                st.info(
                    "Upload a measured inverter output-voltage CSV to compare its "
                    "sampled harmonic spectrum with the current simulation. This is "
                    "a read-only analysis: the upload is not added to the dataset, "
                    "does not train or change the CNN, and is not saved by this tool."
                )
                st.caption(
                    "CSV format: numeric `time_s` and `voltage_v` columns, time in "
                    "seconds and voltage in volts. Samples must be uniformly spaced "
                    "and cover at least two complete cycles. The simulator currently "
                    f"uses a {FUNDAMENTAL_FREQUENCY_HZ:g} Hz fundamental; comparison "
                    "uses the same H2–N harmonic band for both signals, limited by "
                    "the lower Nyquist frequency. The FFT comparison is phase-"
                    "independent; no time-aligned waveform error is claimed."
                )
                with st.form("measured_waveform_validation_form"):
                    measured_voltage_file = st.file_uploader(
                        "Measured voltage CSV",
                        type=["csv"],
                        key="measured_voltage_csv_upload",
                    )
                    analyze_measurement = st.form_submit_button(
                        "ANALYZE MEASUREMENT",
                        disabled=(
                            measured_voltage_file is None
                            or not live_result_is_current
                        ),
                        type="primary",
                    )
                if not live_result_is_current:
                    st.warning(
                        "Run a simulation with the currently selected settings "
                        "before comparing a measurement."
                    )
                if analyze_measurement and measured_voltage_file is not None:
                    measurement_key = (
                        hashlib.sha256(
                            measured_voltage_file.getvalue()
                        ).hexdigest(),
                        tuple(float(value) for value in selected_conditions),
                        tuple(float(value) for value in selected_carrier_offsets),
                        FUNDAMENTAL_FREQUENCY_HZ,
                    )
                    try:
                        comparison = compare_measured_voltage_csv(
                            measured_voltage_file.getvalue(),
                            live_result.simulation.output_voltage,
                            simulated_sampling_frequency_hz=SAMPLING_FREQUENCY_HZ,
                            fundamental_frequency_hz=FUNDAMENTAL_FREQUENCY_HZ,
                            simulated_fundamental_frequency_hz=(
                                FUNDAMENTAL_FREQUENCY_HZ
                            ),
                        )
                    except (ValueError, OSError) as exc:
                        st.session_state.pop("measured_waveform_comparison", None)
                        st.session_state.pop("measured_waveform_comparison_key", None)
                        st.error(
                            f"Could not analyze measured waveform: "
                            f"{type(exc).__name__}: {exc}"
                        )
                    else:
                        st.session_state["measured_waveform_comparison"] = comparison
                        st.session_state["measured_waveform_comparison_key"] = (
                            measurement_key
                        )

                current_measurement_key = (
                    (
                        hashlib.sha256(
                            measured_voltage_file.getvalue()
                        ).hexdigest(),
                        tuple(float(value) for value in selected_conditions),
                        tuple(float(value) for value in selected_carrier_offsets),
                        FUNDAMENTAL_FREQUENCY_HZ,
                    )
                    if measured_voltage_file is not None and live_result_is_current
                    else None
                )
                comparison = st.session_state.get("measured_waveform_comparison")
                if (
                    isinstance(comparison, MeasuredWaveformComparison)
                    and current_measurement_key is not None
                    and st.session_state.get("measured_waveform_comparison_key")
                    == current_measurement_key
                ):
                    result_columns = st.columns(3)
                    result_columns[0].metric(
                        f"MEASURED SAMPLED VOLTAGE THD (H2–{comparison.max_harmonic})",
                        f"{comparison.measured_thd_percent:.4f}%",
                    )
                    result_columns[1].metric(
                        f"SIMULATED SAMPLED VOLTAGE THD (H2–{comparison.max_harmonic})",
                        f"{comparison.simulated_thd_percent:.4f}%",
                    )
                    result_columns[2].metric(
                        "MEASURED − SIMULATED THD",
                        f"{comparison.thd_difference_percentage_points:+.4f} pp",
                    )
                    amplitude_columns = st.columns(3)
                    amplitude_columns[0].metric(
                        "MEASURED FUNDAMENTAL AMPLITUDE",
                        f"{comparison.measured_fundamental_amplitude_v:.4g} V",
                    )
                    amplitude_columns[1].metric(
                        "SIMULATED FUNDAMENTAL AMPLITUDE",
                        f"{comparison.simulated_fundamental_amplitude_v:.4g} V",
                    )
                    amplitude_columns[2].metric(
                        "FUNDAMENTAL AMPLITUDE CHANGE",
                        f"{comparison.fundamental_amplitude_change_percent:+.4f}%",
                    )
                    st.caption(
                        f"{comparison.sample_count:,} analyzed samples · "
                        f"{comparison.cycle_count} complete cycles · "
                        f"{comparison.sampling_frequency_hz:.6g} samples/s · "
                        f"measured mean/DC offset {comparison.measured_mean_voltage_v:.4g} V "
                        f"(simulated {comparison.simulated_mean_voltage_v:.4g} V). "
                        "THD is a sampled FFT metric for both waveforms; it is not "
                        "the event-based full-spectrum optimization objective."
                    )
                    measured_orders = comparison.measured_spectrum.harmonic_orders
                    spectrum_figure = make_chart(
                        [
                            (
                                "Measured voltage",
                                comparison.measured_spectrum.amplitudes,
                                "#d889d8",
                                "-",
                            ),
                            (
                                "Simulated voltage",
                                comparison.simulated_spectrum.amplitudes,
                                "#43d6c5",
                                "--",
                            ),
                        ],
                        "Measured vs simulated voltage harmonic amplitudes",
                        "Peak amplitude (V)",
                        x_values=measured_orders,
                        x_label="Harmonic order",
                    )
                    st.pyplot(spectrum_figure, width="stretch")
                    plt.close(spectrum_figure)
        candidate_live_result = None
        if isinstance(live_result, LiveSimulationResult):
            with st.expander("LIVE-INPUT-CONDITIONED PHASE OPTIMIZATION", expanded=False):
                thd_surrogate_checkpoint = (
                    PROJECT_ROOT
                    / "experiments"
                    / "thd_surrogate_v1"
                    / "model_run_modulation_holdout"
                    / "thd_surrogate.pt"
                )
                thd_surrogate_available = thd_surrogate_checkpoint.is_file()
                live_optimizer_options = [
                    "Existing PSO",
                    "Pelican Optimization Algorithm",
                    "Pufferfish Optimization Algorithm",
                    "Hybrid POA-PFO",
                ]
                if thd_surrogate_available:
                    live_optimizer_options.append("Event-THD surrogate shortlist")
                st.info(
                    "Workflow: run the live simulation above, choose an optimizer "
                    "and budget, then select RUN LIVE OPTIMIZATION. The optimizer "
                    "searches carrier phases against event-based Parseval voltage "
                    "THD. The CNN predicts the reference waveform; it does not "
                    "select phases or guide this live search."
                )
                st.caption(
                    "Candidates use the selected modulation, DC voltage, R, L, and "
                    "reference phase. Optimization is enabled only when those controls "
                    "match the displayed successful baseline and carrier shifts are "
                    "[0°, 60°, 120°]. The objective is event-based Parseval total THD; "
                    "sampled 20 kHz 2–100 THD is a separate measurement and may move "
                    "in the opposite direction. "
                    "The 0.05% fundamental-tracking constraint uses the event-integrated "
                    "phasor. R and L affect current, not ideal output-voltage THD."
                )
                pso_settings_col1, pso_settings_col2, pso_settings_col3, pso_settings_col4 = st.columns(4)
                with pso_settings_col1:
                    live_optimizer_name = st.selectbox(
                        "OPTIMIZER",
                        live_optimizer_options,
                        index=0,
                        key="live_optimizer_name",
                    )
                with pso_settings_col2:
                    live_pso_seed = st.number_input(
                        "RANDOM SEED",
                        min_value=0,
                        max_value=2**31 - 1,
                        value=20261120,
                        step=1,
                        key="live_pso_seed",
                    )
                with pso_settings_col3:
                    live_pso_particles = st.number_input(
                        "PARTICLES",
                        min_value=2,
                        max_value=12,
                        value=8,
                        step=1,
                        key="live_pso_particles",
                    )
                with pso_settings_col4:
                    live_pso_iterations = st.number_input(
                        "ITERATIONS",
                        min_value=1,
                        max_value=12,
                        value=8,
                        step=1,
                        key="live_pso_iterations",
                    )
                selected_optimizer_label = live_optimizer_name
                surrogate_checkpoint_signature = (
                    (
                        thd_surrogate_checkpoint.stat().st_mtime_ns,
                        thd_surrogate_checkpoint.stat().st_size,
                    )
                    if thd_surrogate_available
                    else None
                )
                optimization_configuration = (
                    tuple(float(value) for value in selected_conditions),
                    selected_carrier_offsets,
                    live_optimizer_name,
                    int(live_pso_seed),
                    int(live_pso_particles),
                    int(live_pso_iterations),
                    surrogate_checkpoint_signature,
                )
                baseline_offsets_supported = optimizer_baseline_supported(
                    selected_carrier_offsets,
                    BASELINE_PHASES_DEGREES,
                )
                surrogate_operating_point_supported = (
                    thd_surrogate_available
                    and thd_surrogate_operating_point_supported(
                        live_modulation_index,
                        live_dc_voltage,
                        live_phase_angle,
                        selected_carrier_offsets,
                        BASELINE_PHASES_DEGREES,
                    )
                )
                surrogate_selected = (
                    live_optimizer_name == "Event-THD surrogate shortlist"
                )
                if not thd_surrogate_available:
                    st.caption(
                        "The separate THD surrogate is unavailable; its checkpoint "
                        "is not part of the existing CNN or project dataset."
                    )
                elif surrogate_selected and not surrogate_operating_point_supported:
                    st.info(
                        "Surrogate screening is limited to tested modulation indices "
                        "0.55–0.95 in 0.05 steps, reference phases in 45° steps, "
                        "100 V per cell, and the [0°, 60°, 120°] baseline. Choose "
                        "a tested point or use one of the exact optimizers."
                    )
                elif surrogate_selected:
                    st.caption(
                        "The surrogate ranks 64 bounded phase candidates and sends "
                        "8 finalists (including the baseline) to exact event-based "
                        "THD and electrical-feasibility evaluation. It does not "
                        "replace the optimizer or final simulator."
                    )
                if not baseline_offsets_supported:
                    st.info(
                        "Live phase optimization currently requires carrier shifts "
                        "[0°, 60°, 120°]. The optimizer fixes cell 1 at 0°; other "
                        "carrier settings cannot be compared against the displayed "
                        "baseline without changing its decision-variable representation."
                    )
                stored_optimization_configuration = st.session_state.get(
                    "live_demo_optimization_configuration"
                )
                if (
                    stored_optimization_configuration is not None
                    and stored_optimization_configuration != optimization_configuration
                ):
                    st.session_state.pop("live_demo_optimization", None)
                    st.session_state.pop("live_demo_candidate", None)
                    st.session_state.pop(
                        "live_demo_optimization_configuration", None
                    )
                    st.session_state.pop("live_demo_optimization_error", None)
                run_live_pso = st.button(
                    "RUN LIVE OPTIMIZATION",
                    key="run_live_pso",
                    disabled=(
                        model is None
                        or stats is None
                        or not live_result_is_current
                        or not baseline_offsets_supported
                        or (
                            surrogate_selected
                            and not surrogate_operating_point_supported
                        )
                    ),
                    type="primary",
                )
                if run_live_pso:
                    st.session_state.pop("live_demo_optimization", None)
                    st.session_state.pop("live_demo_candidate", None)
                    st.session_state.pop("live_demo_optimization_error", None)
                    st.session_state[
                        "live_demo_optimization_configuration"
                    ] = optimization_configuration
                    st.session_state.pop("live_demo_export_bundle", None)
                    st.session_state.pop("live_demo_export_error", None)
                    st.session_state.pop("live_demo_export_key", None)
                    if model is None or stats is None:
                        st.session_state["live_demo_optimization_error"] = (
                            "Live phase optimization requires a compatible CNN checkpoint "
                            "for candidate inference."
                        )
                    else:
                        try:
                            with st.spinner(
                                "Evaluating live-conditioned inverter candidates and optimizing carrier phases..."
                            ):
                                if surrogate_selected:
                                    optimization = optimize_live_surrogate_screening(
                                        modulation_index=live_modulation_index,
                                        dc_voltage_per_cell_v=live_dc_voltage,
                                        load_resistance_ohm=live_load_resistance,
                                        load_inductance_h=live_load_inductance,
                                        phase_angle_degrees=live_phase_angle,
                                        seed=int(live_pso_seed),
                                        surrogate_checkpoint_path=(
                                            thd_surrogate_checkpoint
                                        ),
                                    )
                                else:
                                    optimization = optimize_live_operating_point(
                                        modulation_index=live_modulation_index,
                                        dc_voltage_per_cell_v=live_dc_voltage,
                                        load_resistance_ohm=live_load_resistance,
                                        load_inductance_h=live_load_inductance,
                                        phase_angle_degrees=live_phase_angle,
                                        seed=int(live_pso_seed),
                                        particles=int(live_pso_particles),
                                        iterations=int(live_pso_iterations),
                                        optimizer_name={
                                            "Existing PSO": "PSO",
                                            "Pelican Optimization Algorithm": "POA",
                                            "Pufferfish Optimization Algorithm": "PFO",
                                            "Hybrid POA-PFO": "HYBRID",
                                        }[live_optimizer_name],
                                    )
                                candidate_offsets = candidate_to_carrier_offsets(
                                    optimization.run_result.best.canonical_phases_degrees
                                )
                                candidate_result = run_live_simulation(
                                    model,
                                    stats,
                                    modulation_index=live_modulation_index,
                                    dc_voltage_per_cell_v=live_dc_voltage,
                                    load_resistance_ohm=live_load_resistance,
                                    load_inductance_h=live_load_inductance,
                                    phase_angle_degrees=live_phase_angle,
                                    device=device,
                                    carrier_phase_offsets_degrees=candidate_offsets,
                                )
                            st.session_state["live_demo_optimization"] = optimization
                            st.session_state["live_demo_candidate"] = candidate_result
                        except (
                            OSError,
                            ValueError,
                            RuntimeError,
                            TypeError,
                            pickle.UnpicklingError,
                        ) as exc:
                            st.session_state["live_demo_optimization_error"] = (
                                f"Live phase optimization failed: {type(exc).__name__}: {exc}"
                            )

                optimization_error = st.session_state.get("live_demo_optimization_error")
                if optimization_error:
                    st.error(optimization_error)
                optimization = st.session_state.get("live_demo_optimization")
                stored_candidate = st.session_state.get("live_demo_candidate")
                if (
                    isinstance(optimization, LiveConditionedPhaseOptimization)
                    and isinstance(stored_candidate, LiveSimulationResult)
                    and st.session_state.get(
                        "live_demo_optimization_configuration"
                    )
                    == optimization_configuration
                    and live_result_is_current
                ):
                    candidate_offsets = candidate_to_carrier_offsets(
                        optimization.run_result.best.canonical_phases_degrees
                    )
                    if live_configuration_matches(
                        stored_candidate,
                        tuple(float(value) for value in selected_conditions),
                        candidate_offsets,
                    ):
                        candidate_live_result = stored_candidate
                        candidate_offsets = candidate_live_result.carrier_phase_offsets_degrees
                        phase_columns = st.columns(2)
                        phase_columns[0].metric(
                            "INITIAL PHASE SHIFTS",
                            "[0°, 60°, 120°]",
                        )
                        phase_columns[1].metric(
                            f"{selected_optimizer_label.upper()}-SELECTED PHASE SHIFTS",
                            "[" + ", ".join(f"{value:.2f}°" for value in candidate_offsets) + "]",
                        )
                        baseline_full_spectrum = (
                            live_result.harmonic_result.total_thd_percent
                        )
                        candidate_full_spectrum = (
                            candidate_live_result.harmonic_result.total_thd_percent
                        )
                        objective_delta = (
                            candidate_full_spectrum - baseline_full_spectrum
                        )
                        st.markdown(
                            "**Optimization objective · event-based Parseval total voltage THD**"
                        )
                        full_spectrum_columns = st.columns(3)
                        full_spectrum_columns[0].metric(
                            "BASELINE OBJECTIVE",
                            f"{baseline_full_spectrum:.6f}%",
                        )
                        full_spectrum_columns[1].metric(
                            f"{selected_optimizer_label.upper()} OBJECTIVE",
                            f"{candidate_full_spectrum:.6f}%",
                        )
                        full_spectrum_columns[2].metric(
                            "OBJECTIVE CHANGE (CANDIDATE − BASELINE)",
                            f"{objective_delta:+.6f} percentage points",
                            delta_color="inverse",
                        )
                        if objective_delta < -1e-6:
                            st.success(
                                f"{selected_optimizer_label} reduced its event-based "
                                "Parseval objective at this operating point. This "
                                "single-point result is not validation-set evidence."
                            )
                        elif objective_delta > 1e-6:
                            st.warning(
                                f"{selected_optimizer_label} did not improve its "
                                "event-based Parseval objective at this operating point."
                            )
                        else:
                            st.info(
                                f"{selected_optimizer_label} produced no material "
                                "change in its event-based Parseval objective at the "
                                "0.000001 percentage-point reporting resolution."
                            )
                        st.markdown(
                            "**Secondary measurement · sampled voltage THD, 20 kHz, harmonics 2–100**"
                        )
                        baseline_live_thd = live_result.voltage_thd_percent
                        candidate_live_thd = candidate_live_result.voltage_thd_percent
                        live_delta = candidate_live_thd - baseline_live_thd
                        live_comparison = st.columns(3)
                        live_comparison[0].metric(
                            "BASELINE SAMPLED VOLTAGE THD (2–100)",
                            f"{baseline_live_thd:.4f}%",
                        )
                        live_comparison[1].metric(
                            f"{selected_optimizer_label.upper()} SAMPLED VOLTAGE THD (2–100)",
                            f"{candidate_live_thd:.4f}%",
                        )
                        live_comparison[2].metric(
                            "CANDIDATE − BASELINE",
                            f"{live_delta:+.6f} percentage points",
                            delta_color="inverse",
                        )
                        if live_delta < -0.001:
                            st.success(
                                f"{selected_optimizer_label} produced lower sampled "
                                "voltage THD at this operating point. This single-point "
                                "result is not validation-set evidence."
                            )
                        elif live_delta > 0.001:
                            st.warning(
                                f"Sampled voltage H2–100 THD is higher by "
                                f"{live_delta:.6f} percentage points. This metric is "
                                "not the optimizer objective and can move in the "
                                "opposite direction."
                            )
                        else:
                            st.info(
                                "Sampled voltage H2–100 THD did not materially change "
                                "at the 0.001 percentage-point reporting resolution."
                            )
                        tracking_columns = st.columns(3)
                        tracking_columns[0].metric(
                            "BASELINE FUNDAMENTAL TRACKING ERROR",
                            f"{live_result.harmonic_result.fundamental_phasor_error_percent:.6g}%",
                        )
                        tracking_columns[1].metric(
                            f"{selected_optimizer_label.upper()} FUNDAMENTAL TRACKING ERROR",
                            f"{candidate_live_result.harmonic_result.fundamental_phasor_error_percent:.6g}%",
                        )
                        tracking_columns[2].metric(
                            f"CURRENT THD BASELINE / {selected_optimizer_label.upper()} (2–100)",
                            f"{live_result.current_thd_percent:.4g}% / "
                            f"{candidate_live_result.current_thd_percent:.4g}%",
                        )
                        st.caption(
                            f"Seed {optimization.run_result.seed} · "
                            + (
                                f"{optimization.run_result.screened_candidate_count} "
                                "surrogate-ranked candidates · "
                                if surrogate_selected
                                else (
                                    f"{optimization.particles} particles × "
                                    f"{optimization.iterations} iterations · "
                                )
                            )
                            +
                            f"{optimization.run_result.unique_candidate_evaluations} "
                            "unique live-condition candidate simulations · "
                            f"{optimization.run_result.objective_requests} objective requests · "
                            f"feasible={optimization.run_result.best.feasible} · "
                            f"{optimization.run_result.runtime_seconds:.2f} s. "
                            "Voltage THD objective is independent of R and L in this ideal model."
                        )
                        convergence_figure = make_chart(
                            [
                                (
                                    "Best event-based Parseval total THD",
                                    np.asarray(
                                        [
                                            item.mean_total_thd_percent
                                            for item in optimization.run_result.convergence_history
                                        ],
                                        dtype=np.float64,
                                    ),
                                    "#43d6c5",
                                    "-",
                                )
                            ],
                            f"Live-conditioned {selected_optimizer_label} convergence",
                            "Best event-based Parseval total THD (%)",
                            x_values=np.arange(
                                len(optimization.run_result.convergence_history)
                            ),
                            x_label=(
                                "Exact finalist evaluation"
                                if surrogate_selected
                                else "Optimization iteration"
                            ),
                        )
                        st.pyplot(convergence_figure, width="stretch")
                        plt.close(convergence_figure)
            with st.expander("CNN-GUIDED HYBRID OPTIMIZATION", expanded=False):
                st.info(
                    "This separate path uses the loaded CNN only to rank 64 POA/PFO "
                    "proposals; the top 8 are evaluated with event-based THD and the "
                    "winner is re-simulated. CNN scores are ranking proxies, not THD "
                    "measurements. The available correlation evidence (Spearman "
                    "ρ = 0.954 over 500 candidates) applies only to its recorded "
                    "operating point."
                )
                guidance_validation_path = (
                    PROJECT_ROOT / "results" / "cnn_guided_correlation.json"
                )
                guidance_validation_error = None
                guidance_is_usable = False
                guidance_point_matches = False
                guidance_correlation = None
                try:
                    guidance_validation = json.loads(
                        guidance_validation_path.read_text(encoding="utf-8")
                    )
                    guidance_correlation = float(
                        guidance_validation["spearman_cnn_score_vs_event_parseval_thd"]
                    )
                    guidance_threshold = float(
                        guidance_validation["usable_threshold"]
                    )
                    guidance_is_usable = (
                        guidance_validation["guidance_usable"] is True
                        and np.isfinite(guidance_correlation)
                        and np.isfinite(guidance_threshold)
                        and guidance_correlation >= guidance_threshold
                    )
                    guidance_point_matches = cnn_guidance_operating_point_matches(
                        (
                            live_modulation_index,
                            live_dc_voltage,
                            live_load_resistance,
                            live_load_inductance,
                            live_phase_angle,
                        ),
                        guidance_validation["operating_point"],
                        selected_carrier_offsets,
                        BASELINE_PHASES_DEGREES,
                    )
                except (
                    KeyError,
                    OSError,
                    TypeError,
                    ValueError,
                    json.JSONDecodeError,
                ) as exc:
                    guidance_validation_error = (
                        f"Could not validate CNN guidance evidence: "
                        f"{type(exc).__name__}: {exc}"
                    )
                if guidance_validation_error:
                    st.error(guidance_validation_error)
                elif not guidance_is_usable:
                    st.warning(
                        "CNN guidance is disabled because its saved ranking-correlation "
                        "evidence does not meet the recorded usability threshold."
                    )
                elif not guidance_point_matches:
                    st.info(
                        "CNN-guided optimization is enabled only at the validated "
                        "operating point (m=0.75, 100 V/cell, 30 Ω, 0.03 H, phase 0°) "
                        "with baseline carrier shifts [0°, 60°, 120°]. Other live "
                        "conditions remain available to the simulator-based optimizers."
                    )
                else:
                    st.success(
                        f"Validated conditions and baseline shifts match. "
                        f"Recorded CNN-score/event-THD rank correlation: "
                        f"{guidance_correlation:.3f}."
                    )
                guided_seed = st.number_input(
                    "CNN-GUIDED RANDOM SEED",
                    min_value=0,
                    max_value=2**31 - 1,
                    value=20261120,
                    step=1,
                    key="live_cnn_guided_seed",
                )
                guided_configuration = (
                    tuple(float(value) for value in selected_conditions),
                    selected_carrier_offsets,
                    int(guided_seed),
                    COMPARISON_CANDIDATE_BUDGET,
                    COMPARISON_TOP_K,
                )
                stored_guided_configuration = st.session_state.get(
                    "live_demo_cnn_guided_configuration"
                )
                if (
                    stored_guided_configuration is not None
                    and stored_guided_configuration != guided_configuration
                ):
                    st.session_state.pop("live_demo_cnn_guided_optimization", None)
                    st.session_state.pop("live_demo_cnn_guided_candidate", None)
                    st.session_state.pop(
                        "live_demo_cnn_guided_configuration", None
                    )
                    st.session_state.pop("live_demo_cnn_guided_error", None)
                run_cnn_guided = st.button(
                    "RUN CNN-GUIDED HYBRID",
                    key="run_live_cnn_guided",
                    disabled=(
                        model is None
                        or stats is None
                        or not live_result_is_current
                        or not guidance_is_usable
                        or not guidance_point_matches
                    ),
                )
                if run_cnn_guided:
                    st.session_state.pop("live_demo_cnn_guided_optimization", None)
                    st.session_state.pop("live_demo_cnn_guided_candidate", None)
                    st.session_state.pop("live_demo_cnn_guided_error", None)
                    st.session_state[
                        "live_demo_cnn_guided_configuration"
                    ] = guided_configuration
                    try:
                        guided_operating_point = {
                            "modulation_index": float(live_modulation_index),
                            "dc_voltage_per_cell_v": float(live_dc_voltage),
                            "load_resistance_ohm": float(live_load_resistance),
                            "load_inductance_h": float(live_load_inductance),
                            "phase_angle_degrees": float(live_phase_angle),
                        }
                        with st.spinner(
                            "Ranking POA/PFO proposals with the CNN and verifying "
                            "the finalists with the inverter simulator..."
                        ):
                            guided_result = run_cnn_guided_hybrid(
                                model,
                                stats,
                                seed=int(guided_seed),
                                operating_point=guided_operating_point,
                                candidate_budget=COMPARISON_CANDIDATE_BUDGET,
                                top_k=COMPARISON_TOP_K,
                                device=device,
                            )
                            guided_phases = tuple(
                                float(value) for value in guided_result["candidate"]
                            )
                            if len(guided_phases) != 2:
                                raise RuntimeError(
                                    "CNN-guided search returned an invalid phase candidate."
                                )
                            guided_offsets = candidate_to_carrier_offsets(
                                guided_phases
                            )
                            guided_candidate = run_live_simulation(
                                model,
                                stats,
                                modulation_index=live_modulation_index,
                                dc_voltage_per_cell_v=live_dc_voltage,
                                load_resistance_ohm=live_load_resistance,
                                load_inductance_h=live_load_inductance,
                                phase_angle_degrees=live_phase_angle,
                                device=device,
                                carrier_phase_offsets_degrees=guided_offsets,
                            )
                        reported_event_thd = float(guided_result["event_thd"])
                        if not np.isclose(
                            guided_candidate.harmonic_result.total_thd_percent,
                            reported_event_thd,
                            rtol=0.0,
                            atol=1e-8,
                        ):
                            raise RuntimeError(
                                "Independent live re-simulation disagrees with the "
                                "CNN-guided event-based THD result."
                            )
                        if (
                            guided_candidate.harmonic_result.fundamental_phasor_error_percent
                            > TRACKING_TOLERANCE_PERCENT
                        ):
                            raise RuntimeError(
                                "Independent live re-simulation found a candidate "
                                "outside the fundamental-tracking tolerance."
                            )
                        st.session_state[
                            "live_demo_cnn_guided_optimization"
                        ] = guided_result
                        st.session_state[
                            "live_demo_cnn_guided_candidate"
                        ] = guided_candidate
                    except (
                        KeyError,
                        TypeError,
                        ValueError,
                        RuntimeError,
                    ) as exc:
                        st.session_state["live_demo_cnn_guided_error"] = (
                            f"CNN-guided optimization failed: "
                            f"{type(exc).__name__}: {exc}"
                        )

                guided_error = st.session_state.get("live_demo_cnn_guided_error")
                if guided_error:
                    st.error(guided_error)
                guided_result = st.session_state.get(
                    "live_demo_cnn_guided_optimization"
                )
                guided_candidate = st.session_state.get(
                    "live_demo_cnn_guided_candidate"
                )
                if (
                    isinstance(guided_result, dict)
                    and isinstance(guided_candidate, LiveSimulationResult)
                    and st.session_state.get(
                        "live_demo_cnn_guided_configuration"
                    )
                    == guided_configuration
                    and live_result_is_current
                    and guidance_is_usable
                    and guidance_point_matches
                ):
                    guided_offsets = guided_candidate.carrier_phase_offsets_degrees
                    candidate_live_result = guided_candidate
                    selected_optimizer_label = "CNN-guided POA-PFO"
                    guided_comparison = st.columns(3)
                    baseline_event_thd = live_result.harmonic_result.total_thd_percent
                    candidate_event_thd = (
                        guided_candidate.harmonic_result.total_thd_percent
                    )
                    guided_comparison[0].metric(
                        "BASELINE EVENT-BASED PARSEVAL THD",
                        f"{baseline_event_thd:.6f}%",
                    )
                    guided_comparison[1].metric(
                        "CNN-GUIDED EVENT-BASED PARSEVAL THD",
                        f"{candidate_event_thd:.6f}%",
                    )
                    guided_comparison[2].metric(
                        "CANDIDATE − BASELINE",
                        f"{candidate_event_thd - baseline_event_thd:+.6f} pp",
                        delta_color="inverse",
                    )
                    guided_detail_columns = st.columns(2)
                    guided_detail_columns[0].metric(
                        "CNN-GUIDED SELECTED PHASE SHIFTS",
                        "[" + ", ".join(f"{value:.2f}°" for value in guided_offsets) + "]",
                    )
                    guided_detail_columns[1].metric(
                        "CNN-GUIDED FUNDAMENTAL TRACKING ERROR",
                        f"{guided_candidate.harmonic_result.fundamental_phasor_error_percent:.6g}%",
                    )
                    st.caption(
                        "Selected shifts [0°, "
                        f"{guided_offsets[1]:.2f}°, {guided_offsets[2]:.2f}°] · "
                        f"seed {guided_seed} · "
                        f"{guided_result['unique_candidate_evaluations']} unique "
                        f"CNN-ranked proposals · 8 event-THD finalists plus independent "
                        f"winner re-evaluation · {guided_result['runtime_seconds']:.2f} s. "
                        "Final event-based THD and feasibility are independently "
                        "re-simulated before display."
                    )
                    sampled_columns = st.columns(2)
                    sampled_columns[0].metric(
                        "BASELINE SAMPLED VOLTAGE THD (20 kHz, H2–100)",
                        f"{live_result.voltage_thd_percent:.4f}%",
                    )
                    sampled_columns[1].metric(
                        "CANDIDATE SAMPLED VOLTAGE THD (20 kHz, H2–100)",
                        f"{guided_candidate.voltage_thd_percent:.4f}%",
                    )
            if live_error:
                st.warning(
                    "The latest simulation attempt failed; displaying the last "
                    "successful simulation."
                )
            elif live_result_is_current:
                st.success(
                    "Simulation completed for the currently selected operating point."
                )
            else:
                st.warning(
                    "Displaying the last successful simulation using the operating "
                    "conditions listed below."
                )
            st.markdown(
                f"#### Operating point · m = {live_result.modulation_index:.2f} · "
                f"Vdc/cell = {live_result.dc_voltage_per_cell_v:g} V · "
                f"R = {live_result.load_resistance_ohm:g} Ω · "
                f"L = {live_result.load_inductance_h:g} H · "
                f"phase = {live_result.phase_angle_degrees:g}°"
            )
            @st.fragment(run_every=0.25)
            def live_playback_fragment(playback_result):
                if "live_playback_status" not in st.session_state:
                    st.session_state["live_playback_status"] = "stopped"
                if "live_workspace_cursor_index" not in st.session_state:
                    st.session_state["live_workspace_cursor_index"] = int(
                        st.session_state.get("live_playback_index", 0)
                    )

                playback_cols = st.columns([1, 1, 1, 1, 3])
                if playback_cols[0].button("START", key="live_playback_start", use_container_width=True):
                    st.session_state["live_playback_status"] = "running"
                if playback_cols[1].button("PAUSE", key="live_playback_pause", use_container_width=True):
                    st.session_state["live_playback_status"] = "paused"
                if playback_cols[2].button("RESUME", key="live_playback_resume", use_container_width=True):
                    st.session_state["live_playback_status"] = "running"
                if playback_cols[3].button("STOP", key="live_playback_stop", use_container_width=True):
                    st.session_state["live_playback_status"] = "stopped"
                    st.session_state["live_workspace_cursor_index"] = 0
                    st.session_state["live_playback_index"] = 0

                playback_rate = st.selectbox(
                    "PLAYBACK RATE",
                    (("Slow", 1), ("Normal", 10), ("Fast", 25)),
                    index=1,
                    format_func=lambda rate: f"{rate[0]} · {rate[1]} samples per refresh",
                    key="live_playback_rate",
                )
                frame_count = int(playback_result.simulation.output_voltage.size)
                playback_status = st.session_state["live_playback_status"]
                manual_selection = bool(
                    st.session_state.pop("live_playback_manual_selection", False)
                )
                playback_index = synchronized_playback_index(
                    st.session_state,
                    playback_status,
                    frame_count,
                    manual_selection=manual_selection,
                    step=playback_rate[1],
                )
                playback_index = st.slider(
                    "Sample index",
                    min_value=0,
                    max_value=frame_count - 1,
                    step=1,
                    key="live_workspace_cursor_index",
                    on_change=_mark_live_playback_manual_selection,
                )
                st.session_state["live_playback_index"] = playback_index
                snapshot = playback_snapshot(playback_result.simulation, playback_index)
                monitor_cols = st.columns(4)
                monitor_cols[0].metric("SOFTWARE MONITOR STATUS", playback_status.upper())
                monitor_cols[1].metric("SIMULATION TIME", f"{snapshot.time_ms:.3f} ms")
                monitor_cols[2].metric(
                    "INSTANT OUTPUT",
                    f"{snapshot.output_voltage_v:.2f} V",
                )
                monitor_cols[3].metric(
                    "INSTANT LOAD CURRENT",
                    f"{snapshot.load_current_a:.3f} A",
                )
                playback_cols[4].caption(
                    f"Software playback · sample {playback_index + 1}/{frame_count}"
                )
                st.caption(
                    "Accelerated replay of this precomputed, repeating 20 ms cycle; "
                    "not a live hardware signal."
                )

                sync_cols = st.columns(2)
                window_size = min(frame_count, 80)
                window_indices = (
                    np.arange(playback_index - window_size + 1, playback_index + 1)
                    % frame_count
                )
                sample_interval_s = (
                    float(playback_result.simulation.time_s[1] - playback_result.simulation.time_s[0])
                    if frame_count > 1
                    else 0.0
                )
                window_time_ms = (
                    np.arange(-window_size + 1, 1, dtype=np.float64)
                    * sample_interval_s
                    * 1000.0
                )
                with sync_cols[0]:
                    sync_voltage = make_chart(
                        [
                            (
                                "Inverter output voltage",
                                playback_result.simulation.output_voltage[window_indices],
                                "#43d6c5",
                                "-",
                            ),
                        ],
                        "Rolling voltage playback · recent simulated samples",
                        "Voltage (V)",
                        x_values=window_time_ms,
                        x_label="Time before playback point (ms)",
                    )
                    sync_voltage.axes[0].axvline(
                        0.0,
                        color="#f0a95e",
                        linewidth=1.5,
                        linestyle="--",
                    )
                    sync_voltage.axes[0].scatter(
                        [0.0],
                        [snapshot.output_voltage_v],
                        color="#f0a95e",
                        s=38,
                        zorder=5,
                    )
                    st.pyplot(sync_voltage, width="stretch")
                    plt.close(sync_voltage)
                with sync_cols[1]:
                    sync_current = make_chart(
                        [
                            (
                                "RL load current",
                                playback_result.simulation.load_current[window_indices],
                                "#e7ad65",
                                "-",
                            ),
                        ],
                        "Rolling current playback · recent simulated samples",
                        "Current (A)",
                        x_values=window_time_ms,
                        x_label="Time before playback point (ms)",
                    )
                    sync_current.axes[0].axvline(
                        0.0,
                        color="#f0a95e",
                        linewidth=1.5,
                        linestyle="--",
                    )
                    sync_current.axes[0].scatter(
                        [0.0],
                        [snapshot.load_current_a],
                        color="#f0a95e",
                        s=38,
                        zorder=5,
                    )
                    st.pyplot(sync_current, width="stretch")
                    plt.close(sync_current)

                gate_state_rows = [
                    {
                        "Cell": f"C{cell_index + 1}",
                        "State": snapshot.cell_states[cell_index],
                        "Gate vector": str(
                            tuple(snapshot.switching_gates[cell_index])
                        ),
                        "S1": (
                            "ON" if snapshot.switching_gates[cell_index][0] else "OFF"
                        ),
                        "S2": (
                            "ON" if snapshot.switching_gates[cell_index][1] else "OFF"
                        ),
                        "S3": (
                            "ON" if snapshot.switching_gates[cell_index][2] else "OFF"
                        ),
                        "S4": (
                            "ON" if snapshot.switching_gates[cell_index][3] else "OFF"
                        ),
                        "Cell voltage (V)": (
                            float(playback_result.simulation.dc_voltage_per_cell_v)
                            * snapshot.cell_states[cell_index]
                        ),
                    }
                    for cell_index in range(3)
                ]
                sum_voltage, output_voltage, voltage_matches = verify_cell_voltage_sum(
                    playback_result.simulation,
                    playback_index,
                )
                st.caption(
                    "Analytical simulation (inverter simulator) — "
                    "instant switching snapshot"
                )
                st.dataframe(
                    gate_state_rows,
                    hide_index=True,
                    use_container_width=True,
                )
                check_columns = st.columns(3)
                check_columns[0].metric("V1 + V2 + V3", f"{sum_voltage:.2f} V")
                check_columns[1].metric(
                    "Simulator output voltage",
                    f"{output_voltage:.2f} V",
                )
                check_columns[2].metric(
                    "Sum check",
                    "PASS" if voltage_matches else "FAIL",
                )

            live_playback_fragment(live_result)
            simulation = live_result.simulation
            harmonic = live_result.harmonic_result
            metric_cols = st.columns(6)
            metric_cols[0].metric(
                "PARSEVAL FULL-SPECTRUM THD",
                f"{harmonic.total_thd_percent:.3f}%",
                help=(
                    "Uses exact interval mean-square voltage and AC energy after "
                    "subtracting the fundamental. It includes all spectral energy."
                ),
            )
            metric_cols[1].metric(
                "EVENT THD (HARMONICS 2–100)",
                fmt_percent(harmonic.thd_percent),
                help=(
                    "Analytic event-based Fourier harmonics 2 through 100, divided "
                    "by the event-based fundamental amplitude."
                ),
            )
            metric_cols[2].metric(
                "20 kHz SAMPLED THD (2–100)",
                f"{live_result.voltage_thd_percent:.3f}%",
                help=(
                    "Finite-band FFT estimate calculated from this simulator's "
                    "20 kHz output-voltage samples, using orders 2 through 100."
                ),
            )
            metric_cols[3].metric(
                "FUNDAMENTAL PEAK",
                f"{harmonic.fundamental_peak_amplitude_v:.2f} V",
                help="Peak amplitude of the event-integrated 50 Hz output-voltage fundamental.",
            )
            metric_cols[4].metric(
                "CNN TARGET RMSE",
                f"{live_result.cnn_metrics['rmse']:.3f} V",
                help="RMSE between the CNN prediction and analytical reference target; not output THD.",
            )
            metric_cols[5].metric(
                "LOAD CURRENT THD (2–100)",
                f"{live_result.current_thd_percent:.3f}%",
                help="20 kHz sampled finite-band load-current THD, orders 2–100.",
            )
            st.caption(
                "The event finite-band THD and sampled finite-band THD both cover "
                "orders 2–100 but use different waveform representations. Parseval "
                "full-spectrum THD is a separate all-energy measure; these values "
                "are not interchangeable."
            )
            with st.expander("RECORDED HELD-OUT CNN CHECK", expanded=False):
                held_out_report_path = (
                    PROJECT_ROOT
                    / "results"
                    / "cnn_grouped_test_eval_20261124.json"
                )
                if (
                    loaded_checkpoint_sha256 is None
                    or loaded_checkpoint_sha256
                    != "9ffea39995c4bbba15f1f3572c9aaf54d49ad6ff1b1a186c4cafffe38263636c"
                    or Path(resolved_dataset_path).resolve()
                    != Path(default_dataset).resolve()
                ):
                    st.info(
                        "The saved held-out evaluation is for the bundled checkpoint "
                        "and dataset only. Run CNN TEST EVALUATION to assess the "
                        "currently selected model/data."
                    )
                else:
                    try:
                        held_out_report = json.loads(
                            held_out_report_path.read_text(encoding="utf-8")
                        )
                        if (
                            held_out_report["checkpoint_sha256"]
                            != loaded_checkpoint_sha256
                            or held_out_report["checkpoint_loaded_strictly"] is not True
                            or held_out_report["checkpoint_modified"] is not False
                            or held_out_report["dataset_modified"] is not False
                        ):
                            raise ValueError(
                                "Saved CNN evaluation provenance does not match the "
                                "loaded checkpoint."
                            )
                        waveform_points = int(
                            held_out_report["waveform_points_per_record"]
                        )
                        test_record_count = int(
                            held_out_report["split_record_counts"]["testing"]
                        )
                        modulation_groups = [
                            group
                            for group in held_out_report["groups"]["groups"]
                            if group["category"] == "modulation_band"
                        ]
                        grouped_record_count = sum(
                            int(group["record_count"])
                            for group in modulation_groups
                        )
                        if (
                            not modulation_groups
                            or grouped_record_count != test_record_count
                            or waveform_points < 1
                        ):
                            raise ValueError(
                                "Saved held-out CNN groups do not cover the expected "
                                "test records."
                            )
                        total_points = grouped_record_count * waveform_points
                        cnn_pooled_rmse = float(
                            np.sqrt(
                                sum(
                                    float(group["cnn"]["rmse_v"]) ** 2
                                    * int(group["record_count"])
                                    * waveform_points
                                    for group in modulation_groups
                                )
                                / total_points
                            )
                        )
                        analytical_pooled_rmse = float(
                            np.sqrt(
                                sum(
                                    float(group["analytical"]["rmse_v"]) ** 2
                                    * int(group["record_count"])
                                    * waveform_points
                                    for group in modulation_groups
                                )
                                / total_points
                            )
                        )
                        if not np.isfinite(cnn_pooled_rmse) or not np.isfinite(
                            analytical_pooled_rmse
                        ):
                            raise ValueError(
                                "Saved held-out evaluation contains non-finite RMSE."
                            )
                        held_out_columns = st.columns(3)
                        held_out_columns[0].metric(
                            "CNN HELD-OUT POOLED RMSE",
                            f"{cnn_pooled_rmse:.4f} V",
                        )
                        held_out_columns[1].metric(
                            "ANALYTICAL HELD-OUT POOLED RMSE",
                            f"{analytical_pooled_rmse:.6f} V",
                        )
                        held_out_columns[2].metric(
                            "TEST RECORDS",
                            f"{test_record_count:,}",
                        )
                        st.warning(
                            "On the recorded held-out test set, the CNN prediction "
                            "was less accurate than the analytical reference. These "
                            "are waveform-prediction errors, not THD or optimizer "
                            "measurements. Re-run CNN TEST EVALUATION after changing "
                            "the dataset or checkpoint."
                        )
                    except (KeyError, OSError, TypeError, ValueError) as exc:
                        st.error(
                            "Could not present the saved held-out CNN evaluation: "
                            f"{type(exc).__name__}: {exc}"
                        )

            technical_cols = st.columns(4)
            technical_cols[0].metric(
                "FUNDAMENTAL PHASOR TRACKING ERROR",
                f"{harmonic.fundamental_phasor_error_percent:.5f}%",
            )
            technical_cols[1].metric(
                "FUNDAMENTAL PHASE",
                f"{np.degrees(harmonic.fundamental_phase_rad):.2f}°",
            )
            technical_cols[2].metric(
                "CNN TARGET MAE",
                f"{live_result.cnn_metrics['mae']:.3f} V",
            )
            technical_cols[3].metric(
                "CNN TARGET R²",
                fmt_metric(live_result.cnn_metrics, "r_squared", 5),
            )

            show_cnn_prediction_overlay = st.checkbox(
                "Show CNN prediction overlay",
                value=False,
                key="live_show_cnn_prediction_overlay",
            )

            output_series = [
                (
                    "Inverter output voltage",
                    simulation.output_voltage,
                    "#43d6c5",
                    "-",
                ),
                (
                    "Analytical reference target",
                    live_result.target_reference_voltage,
                    "#659df5",
                    "--",
                ),
            ]
            if candidate_live_result is not None:
                output_series.append(
                    (
                        f"{selected_optimizer_label} candidate output voltage",
                        candidate_live_result.simulation.output_voltage,
                        "#d889d8",
                        ":",
                    )
                )
            output_figure = make_chart(
                output_series,
                "Simulated output and reference target",
                "Voltage (V)",
                x_values=simulation.time_s * 1000.0,
                x_label="Time (ms)",
            )
            st.pyplot(output_figure, width="stretch")
            plt.close(output_figure)

            output_reference_error = (
                simulation.output_voltage - live_result.target_reference_voltage
            )
            tracking_columns = st.columns(3)
            tracking_columns[0].metric(
                "SIMULATED OUTPUT − REFERENCE RMS",
                f"{np.sqrt(np.mean(np.square(output_reference_error))):.3f} V",
            )
            tracking_columns[1].metric(
                "PEAK ABSOLUTE TRACKING ERROR",
                f"{np.max(np.abs(output_reference_error)):.3f} V",
            )
            if candidate_live_result is not None:
                candidate_reference_error = (
                    candidate_live_result.simulation.output_voltage
                    - candidate_live_result.target_reference_voltage
                )
                tracking_columns[2].metric(
                    f"{selected_optimizer_label.upper()} OUTPUT − REFERENCE RMS",
                    f"{np.sqrt(np.mean(np.square(candidate_reference_error))):.3f} V",
                )
            else:
                tracking_columns[2].metric(
                    "REFERENCE TYPE",
                    "Analytical sinusoid",
                )
            output_error_series = [
                (
                    "Simulated PWM output − analytical reference",
                    output_reference_error,
                    "#e58d8d",
                    "-",
                )
            ]
            if candidate_live_result is not None:
                output_error_series.append(
                    (
                        f"{selected_optimizer_label} output − reference",
                        candidate_live_result.simulation.output_voltage
                        - candidate_live_result.target_reference_voltage,
                        "#d889d8",
                        "--",
                    )
                )
            output_error_figure = make_chart(
                output_error_series,
                "Inverter output tracking error (not CNN prediction error)",
                "Output voltage − reference (V)",
                x_values=simulation.time_s * 1000.0,
                x_label="Time (ms)",
                height_inches=3.0,
            )
            st.pyplot(output_error_figure, width="stretch")
            plt.close(output_error_figure)
            st.caption(
                "The output is a seven-level PWM waveform; the reference is the "
                "desired sinusoid. Their visible difference is expected switching "
                "ripple and sampling error in this ideal model. This is separate "
                "from CNN prediction error and from THD."
            )

            reference_series = [
                (
                    "Analytical reference target",
                    live_result.target_reference_voltage,
                    "#43d6c5",
                    "-",
                ),
            ]
            if show_cnn_prediction_overlay:
                reference_series.append(
                    (
                        "CNN predicted target",
                        live_result.cnn_prediction_voltage,
                        "#f0a95e",
                        "--",
                    )
                )
            if show_cnn_prediction_overlay and candidate_live_result is not None:
                reference_series.append(
                    (
                        f"{selected_optimizer_label} candidate CNN target prediction",
                        candidate_live_result.cnn_prediction_voltage,
                        "#d889d8",
                        ":",
                    )
                )
            reference_figure = make_chart(
                reference_series,
                "CNN reference-target waveform prediction",
                "Voltage (V)",
                x_values=simulation.time_s * 1000.0,
                x_label="Time (ms)",
            )
            st.pyplot(reference_figure, width="stretch")
            plt.close(reference_figure)

            if show_cnn_prediction_overlay:
                prediction_error = (
                    live_result.cnn_prediction_voltage
                    - live_result.target_reference_voltage
                )
                error_series = [
                    (
                        "CNN prediction − reference target",
                        prediction_error,
                        "#e58d8d",
                        "-",
                    )
                ]
                if candidate_live_result is not None:
                    error_series.append(
                        (
                            f"{selected_optimizer_label} candidate CNN error",
                            candidate_live_result.cnn_prediction_voltage
                            - candidate_live_result.target_reference_voltage,
                            "#d889d8",
                            "--",
                        )
                    )
                error_figure = make_chart(
                    error_series,
                    "CNN target prediction error",
                    "Error (V)",
                    x_values=simulation.time_s * 1000.0,
                    x_label="Time (ms)",
                )
                st.pyplot(error_figure, width="stretch")
                plt.close(error_figure)

            current_series = [("Baseline RL load current", simulation.load_current, "#e7ad65", "-")]
            if candidate_live_result is not None:
                current_series.append(
                    (
                        f"{selected_optimizer_label} candidate RL load current",
                        candidate_live_result.simulation.load_current,
                        "#d889d8",
                        "--",
                    )
                )
            current_figure = make_chart(
                current_series,
                "Simulated RL load current",
                "Current (A)",
                x_values=simulation.time_s * 1000.0,
                x_label="Time (ms)",
            )
            st.pyplot(current_figure, width="stretch")
            plt.close(current_figure)

            state_series = [
                    (
                        f"Baseline cell {cell_index + 1}",
                        simulation.cell_states[cell_index],
                        color,
                        "-",
                    )
                    for cell_index, color in enumerate(
                        ("#43d6c5", "#659df5", "#e7ad65")
                    )
                ]
            if candidate_live_result is not None:
                state_series.extend(
                    (
                        f"{selected_optimizer_label} cell {cell_index + 1}",
                        candidate_live_result.simulation.cell_states[cell_index],
                        color,
                        "--",
                    )
                    for cell_index, color in enumerate(
                        ("#43d6c5", "#659df5", "#e7ad65")
                    )
                )
            state_figure = make_chart(
                state_series,
                f"Baseline and {selected_optimizer_label if candidate_live_result is not None else 'baseline'} switching states",
                "Cell state",
                x_values=simulation.time_s * 1000.0,
                x_label="Time (ms)",
            )
            st.pyplot(state_figure, width="stretch")
            plt.close(state_figure)

            spectrum_figure = make_harmonic_chart(
                harmonic,
                FUNDAMENTAL_FREQUENCY_HZ,
                candidate_live_result.harmonic_result
                if candidate_live_result is not None
                else None,
                (
                    f"{selected_optimizer_label} candidate"
                    if candidate_live_result is not None
                    else "Optimized candidate"
                ),
            )
            st.pyplot(spectrum_figure, width="stretch")
            plt.close(spectrum_figure)
            st.caption(
                "Spectrum amplitudes are event-integrated output-voltage peak "
                "amplitudes. The plotted orders 1–100 correspond to the analytic "
                "event-based Fourier coefficients."
            )

            current_spectrum_series = [
                (
                    "Baseline load current",
                    live_result.current_harmonic_spectrum.amplitudes,
                    "#e7ad65",
                    "-",
                )
            ]
            if candidate_live_result is not None:
                current_spectrum_series.append(
                    (
                        f"{selected_optimizer_label} load current",
                        candidate_live_result.current_harmonic_spectrum.amplitudes,
                        "#d889d8",
                        "--",
                    )
                )
            current_spectrum_figure = make_chart(
                current_spectrum_series,
                "Load-current harmonic spectrum (sampled FFT)",
                "Peak amplitude (A)",
                x_values=(
                    live_result.current_harmonic_spectrum.frequencies_hz
                ),
                x_label="Frequency (Hz)",
            )
            st.pyplot(current_spectrum_figure, width="stretch")
            plt.close(current_spectrum_figure)

            st.markdown("#### Switching gates")
            gate_columns = st.columns(3)
            gate_colors = ("#43d6c5", "#659df5", "#e7ad65", "#d889d8")
            for cell_index, gate_column in enumerate(gate_columns):
                with gate_column:
                    gate_series = [
                            (
                                f"Baseline S{switch_index + 1}",
                                simulation.switching_gates[cell_index, switch_index],
                                gate_colors[switch_index],
                                "-",
                            )
                            for switch_index in range(4)
                        ]
                    if candidate_live_result is not None:
                        gate_series.extend(
                            (
                                f"{selected_optimizer_label} S{switch_index + 1}",
                                candidate_live_result.simulation.switching_gates[
                                    cell_index, switch_index
                                ],
                                gate_colors[switch_index],
                                "--",
                            )
                            for switch_index in range(4)
                        )
                    gate_figure = make_chart(
                        gate_series,
                        f"Cell {cell_index + 1} baseline and {selected_optimizer_label} gate signals",
                        "Gate state",
                        x_values=simulation.time_s * 1000.0,
                        x_label="Time (ms)",
                    )
                    st.pyplot(gate_figure, width="stretch")
                    plt.close(gate_figure)

            st.markdown(
                """
                <div class="notice">
                    <b>Scientific scope:</b> The inverter and RL load are an ideal,
                    software simulation using this project's 20 kHz sampling and
                    switching conventions. The CNN predicts the reference target
                    waveform; it does not predict switching gates or optimized
                    inverter-output THD. CNN prediction accuracy is not evidence of
                    harmonic reduction. This is not hardware, MATLAB, or Simulink validation.
                </div>
                """,
                unsafe_allow_html=True,
            )
            if candidate_live_result is None:
                st.caption(
                    "The export currently contains the baseline simulation. "
                    "Run the selected live-input-conditioned optimizer above to include the candidate and convergence history."
                )
            export_key = (
                live_cache_key,
                tuple(float(value) for value in selected_conditions),
                (
                    tuple(candidate_live_result.carrier_phase_offsets_degrees)
                    if candidate_live_result is not None
                    else None
                ),
                (
                    optimization.run_result.seed,
                    optimization.particles,
                    optimization.iterations,
                )
                if isinstance(
                    (optimization := st.session_state.get("live_demo_optimization")),
                    LiveConditionedPhaseOptimization,
                )
                and candidate_live_result is not None
                else None,
            )
            if st.session_state.get("live_demo_export_key") != export_key:
                st.session_state.pop("live_demo_export_bundle", None)
                st.session_state.pop("live_demo_export_error", None)
                try:
                    checkpoint_bytes = (
                        checkpoint_file.getvalue()
                        if checkpoint_file is not None
                        else default_checkpoint.read_bytes()
                    )
                    run_metadata = {
                        "checkpoint_filename": (
                            checkpoint_file.name
                            if checkpoint_file is not None
                            else default_checkpoint.name
                        ),
                        "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
                        "dataset_source_name": resolved_dataset_path.name,
                        "dataset_cache_identity_sha256": hashlib.sha256(
                            source_fingerprint(resolved_dataset_path).encode("utf-8")
                        ).hexdigest(),
                        "dataset_usage": (
                            "Live phase optimization evaluates candidates at the explicitly recorded "
                            "user-selected operating conditions; no dataset record lookup is used."
                        ),
                    }
                    st.session_state["live_demo_export_bundle"] = build_live_export(
                        live_result,
                        candidate=candidate_live_result,
                        optimization=(
                            optimization
                            if isinstance(
                                optimization, LiveConditionedPhaseOptimization
                            )
                            and candidate_live_result is not None
                            else None
                        ),
                        run_metadata=run_metadata,
                    )
                    st.session_state["live_demo_export_key"] = export_key
                except (OSError, RuntimeError, TypeError, ValueError) as exc:
                    st.session_state["live_demo_export_error"] = (
                        f"Could not prepare live result exports: {type(exc).__name__}: {exc}"
                    )
            export_error = st.session_state.get("live_demo_export_error")
            if export_error:
                st.error(export_error)
            export_bundle = st.session_state.get("live_demo_export_bundle")
            if isinstance(export_bundle, LiveExportBundle):
                download_cols = st.columns(3)
                with download_cols[0]:
                    st.download_button(
                        "DOWNLOAD WAVEFORM CSV",
                        data=export_bundle.waveform_csv,
                        file_name="live_inverter_waveforms.csv",
                        mime="text/csv",
                        key="download_live_waveforms",
                    )
                with download_cols[1]:
                    st.download_button(
                        "DOWNLOAD SUMMARY JSON",
                        data=export_bundle.summary_json,
                        file_name="live_inverter_summary.json",
                        mime="application/json",
                        key="download_live_summary",
                    )
                with download_cols[2]:
                    st.download_button(
                        "DOWNLOAD COMPLETE ZIP",
                        data=export_bundle.experiment_zip,
                        file_name="live_inverter_experiment.zip",
                        mime="application/zip",
                        key="download_live_zip",
                    )
        else:
            st.caption("Simulation status: no result exists for the current inputs.")

    # --------------------------------------------------------
except (
    OSError,
    ValueError,
    KeyError,
    RuntimeError,
    TypeError,
    IndexError,
    EOFError,
    pickle.UnpicklingError,
) as exc:
    st.error(f"Application could not load the dataset/checkpoint: {exc}")
    with st.expander("Technical error details"):
        st.exception(exc)
