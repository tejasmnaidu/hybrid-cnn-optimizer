"""Reproducible baseline-versus-optimizer THD experiment."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import math
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from optimization.cnn_guided_search import (
    CORRELATION_OUTPUT,
    COMPARISON_CANDIDATE_BUDGET,
    COMPARISON_TOP_K,
    run_cnn_guided_hybrid,
)
from evaluation.event_based_thd import evaluate_event_based_thd
from evaluation.live_phase_optimization import optimize_live_operating_point
from evaluation.thd_analysis import calculate_thd
from models.model_factory import create_model
from multilevel.inverter_simulator import (
    FUNDAMENTAL_FREQUENCY_HZ,
    SAMPLING_FREQUENCY_HZ,
    simulate_inverter,
)
from optimization.circular_phase_pso import (
    BASELINE_PHASES_DEGREES,
    TRACKING_TOLERANCE_PERCENT,
    candidate_to_carrier_offsets,
    canonicalize_candidate,
)
from preprocessing.preprocess import NormalizationStats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIRECTORY = PROJECT_ROOT / "results"
CSV_OUTPUT = RESULTS_DIRECTORY / "thd_experiment.csv"
SUMMARY_OUTPUT = RESULTS_DIRECTORY / "summary.md"
BEFORE_AFTER_FIGURE = RESULTS_DIRECTORY / "thd_experiment_before_after.png"
CONVERGENCE_FIGURE = RESULTS_DIRECTORY / "thd_experiment_convergence.png"
WAVEFORM_SPECTRUM_FIGURE = RESULTS_DIRECTORY / "thd_experiment_waveform_spectrum.png"
CHECKPOINT = (
    PROJECT_ROOT
    / "colab_runs"
    / "baseline_20261120_080836"
    / "checkpoints"
    / "best.pt"
)

MODULATION_INDICES = (0.55, 0.65, 0.75, 0.85, 0.95)
SEEDS = tuple(range(20261124, 20261134))
OPERATING_CONDITIONS = {
    "dc_voltage_per_cell_v": 100.0,
    "load_resistance_ohm": 30.0,
    "load_inductance_h": 0.03,
    "phase_angle_degrees": 0.0,
}
POPULATION_SIZE = 8
ITERATIONS = 8
CNN_GUIDED_CORRELATION_THRESHOLD = 0.70
GRID_BASELINE_THD_PERCENT = 24.723479151037804
GRID_BASELINE_PHASES = (60.0, 120.0)
METHODS = ("Baseline", "PSO", "POA", "PFO", "HybridPOAPFO", "CNN-guided Hybrid")
OPTIMIZER_NAMES = {
    "PSO": "PSO",
    "POA": "POA",
    "PFO": "PFO",
    "HybridPOAPFO": "HYBRID",
}

CSV_FIELDS = (
    "seed",
    "modulation_index",
    "method",
    "cell_1_phase_degrees",
    "cell_2_phase_degrees",
    "cell_3_phase_degrees",
    "baseline_event_voltage_total_thd_percent",
    "optimized_event_voltage_total_thd_percent",
    "event_voltage_harmonics_2_100_thd_percent",
    "sampled_voltage_thd_percent",
    "event_current_harmonics_2_100_thd_percent",
    "sampled_current_thd_percent",
    "percent_reduction_event_voltage_total_thd",
    "improved_on_baseline",
    "unique_optimizer_candidate_evaluations",
    "optimizer_objective_requests",
    "full_event_thd_evaluations",
    "simulator_waveform_generations",
    "evaluation_budget",
    "runtime_seconds",
    "feasible",
)


def _read_cnn() -> tuple[torch.nn.Module, NormalizationStats]:
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Existing read-only CNN checkpoint not found: {CHECKPOINT}")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    normalization = NormalizationStats(**metadata["normalization"])
    model = create_model(metadata["model_config"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, normalization


def cnn_guidance_is_justified() -> tuple[bool, float]:
    if not CORRELATION_OUTPUT.is_file():
        raise FileNotFoundError(
            f"Phase 5 correlation report not found: {CORRELATION_OUTPUT}"
        )
    report = json.loads(CORRELATION_OUTPUT.read_text(encoding="utf-8"))
    correlation = float(report["spearman_cnn_score_vs_event_parseval_thd"])
    if not np.isfinite(correlation):
        raise ValueError("Phase 5 CNN correlation must be finite.")
    return correlation >= CNN_GUIDED_CORRELATION_THRESHOLD, correlation


def _event_current_thd_percent(
    event_result: Any,
    *,
    resistance_ohm: float,
    inductance_h: float,
) -> float:
    """Derive event harmonics 2..100 of load current through the RL impedance."""
    orders = np.asarray(event_result.harmonic_orders, dtype=np.int64)
    voltage_amplitudes = np.asarray(
        event_result.harmonic_peak_amplitudes_v, dtype=np.float64
    )
    if (
        orders.ndim != 1
        or voltage_amplitudes.shape != orders.shape
        or orders.size < 2
        or orders[0] != 1
        or not np.isfinite(voltage_amplitudes).all()
    ):
        raise ValueError("Event Fourier data is not a finite harmonic series beginning at 1.")
    angular_frequency = 2.0 * np.pi * FUNDAMENTAL_FREQUENCY_HZ * orders
    impedance = np.hypot(resistance_ohm, angular_frequency * inductance_h)
    current_amplitudes = voltage_amplitudes / impedance
    fundamental = float(current_amplitudes[0])
    if fundamental <= np.finfo(np.float64).eps:
        raise ValueError("Event-based current THD is undefined for zero fundamental current.")
    return float(np.sqrt(np.sum(np.square(current_amplitudes[1:]))) / fundamental * 100.0)


def _metrics(
    modulation_index: float,
    phases: tuple[float, float],
) -> dict[str, float | bool]:
    conditions = OPERATING_CONDITIONS
    offsets = candidate_to_carrier_offsets(phases)
    reference_phase_rad = np.deg2rad(conditions["phase_angle_degrees"])
    simulation = simulate_inverter(
        modulation_index,
        conditions["load_resistance_ohm"],
        conditions["load_inductance_h"],
        reference_phase_rad,
        carrier_phase_offsets_degrees=offsets,
        dc_voltage_per_cell_v=conditions["dc_voltage_per_cell_v"],
    )
    event = evaluate_event_based_thd(
        modulation_index,
        reference_phase_rad,
        carrier_phase_offsets_degrees=offsets,
        dc_voltage_per_cell_v=conditions["dc_voltage_per_cell_v"],
    )
    if not np.isfinite(simulation.output_voltage).all() or not np.isfinite(
        simulation.load_current
    ).all():
        raise RuntimeError("Independent final inverter simulation produced non-finite data.")
    return {
        "event_voltage_total": float(event.total_thd_percent),
        "event_voltage_2_100": float(event.thd_percent),
        "sampled_voltage": float(
            calculate_thd(
                simulation.output_voltage,
                sampling_frequency_hz=SAMPLING_FREQUENCY_HZ,
                fundamental_frequency_hz=FUNDAMENTAL_FREQUENCY_HZ,
                max_harmonic=100,
            )
        ),
        "feasible": bool(
            event.fundamental_phasor_error_percent
            <= TRACKING_TOLERANCE_PERCENT
        ),
        "event_current_2_100": _event_current_thd_percent(
            event,
            resistance_ohm=conditions["load_resistance_ohm"],
            inductance_h=conditions["load_inductance_h"],
        ),
        "sampled_current": float(
            calculate_thd(
                simulation.load_current,
                sampling_frequency_hz=SAMPLING_FREQUENCY_HZ,
                fundamental_frequency_hz=FUNDAMENTAL_FREQUENCY_HZ,
                max_harmonic=100,
            )
        ),
    }


def _baseline_row(
    *,
    seed: int,
    modulation_index: float,
) -> tuple[dict[str, Any], dict[str, float | bool]]:
    started = time.perf_counter()
    phases = canonicalize_candidate(BASELINE_PHASES_DEGREES[1:])
    metrics = _metrics(modulation_index, phases)
    elapsed = time.perf_counter() - started
    return (
        _format_row(
            seed=seed,
            modulation_index=modulation_index,
            method="Baseline",
            phases=phases,
            baseline_thd=metrics["event_voltage_total"],
            metrics=metrics,
            unique_evaluations=1,
            objective_requests=1,
            full_event_evaluations=1,
            waveform_generations=1,
            evaluation_budget=1,
            runtime=elapsed,
        ),
        metrics,
    )


def _format_row(
    *,
    seed: int,
    modulation_index: float,
    method: str,
    phases: tuple[float, float],
    baseline_thd: float,
    metrics: dict[str, float | bool],
    unique_evaluations: int,
    objective_requests: int | str,
    full_event_evaluations: int,
    waveform_generations: int,
    evaluation_budget: int,
    runtime: float,
) -> dict[str, Any]:
    reduction = (
        (baseline_thd - metrics["event_voltage_total"]) / baseline_thd * 100.0
        if baseline_thd > 0
        else 0.0
    )
    return {
        "seed": seed,
        "modulation_index": modulation_index,
        "method": method,
        "cell_1_phase_degrees": 0.0,
        "cell_2_phase_degrees": phases[0],
        "cell_3_phase_degrees": phases[1],
        "baseline_event_voltage_total_thd_percent": baseline_thd,
        "optimized_event_voltage_total_thd_percent": metrics["event_voltage_total"],
        "event_voltage_harmonics_2_100_thd_percent": metrics["event_voltage_2_100"],
        "sampled_voltage_thd_percent": metrics["sampled_voltage"],
        "event_current_harmonics_2_100_thd_percent": metrics["event_current_2_100"],
        "sampled_current_thd_percent": metrics["sampled_current"],
        "percent_reduction_event_voltage_total_thd": reduction,
        "improved_on_baseline": reduction > 1e-12,
        "unique_optimizer_candidate_evaluations": unique_evaluations,
        "optimizer_objective_requests": objective_requests,
        "full_event_thd_evaluations": full_event_evaluations,
        "simulator_waveform_generations": waveform_generations,
        "evaluation_budget": evaluation_budget,
        "runtime_seconds": runtime,
        "feasible": bool(metrics["feasible"]),
    }


def _optimizer_run(
    *,
    method: str,
    seed: int,
    modulation_index: float,
    model: torch.nn.Module | None,
    normalization: NormalizationStats | None,
) -> tuple[dict[str, Any], list[float], dict[str, float | bool]]:
    conditions = {
        **OPERATING_CONDITIONS,
        "modulation_index": modulation_index,
    }
    started = time.perf_counter()
    if method in OPTIMIZER_NAMES:
        result = optimize_live_operating_point(
            **conditions,
            seed=seed,
            particles=POPULATION_SIZE,
            iterations=ITERATIONS,
            optimizer_name=OPTIMIZER_NAMES[method],
        )
        run = result.run_result
        candidate = canonicalize_candidate(run.best.canonical_phases_degrees)
        unique_evaluations = int(run.unique_candidate_evaluations)
        objective_requests = int(run.objective_requests)
        full_event_evaluations = unique_evaluations + 2
        waveform_generations = 1
        budget = POPULATION_SIZE * (ITERATIONS + 1)
        history = [
            float(item.mean_total_thd_percent)
            for item in run.convergence_history
        ]
    elif method == "CNN-guided Hybrid":
        if model is None or normalization is None:
            raise RuntimeError("The CNN-guided method requires the approved checkpoint.")
        result = run_cnn_guided_hybrid(
            model,
            normalization,
            seed=seed,
            operating_point=conditions,
            candidate_budget=COMPARISON_CANDIDATE_BUDGET,
            top_k=COMPARISON_TOP_K,
            device="cpu",
        )
        candidate = canonicalize_candidate(result["candidate"])
        unique_evaluations = int(result["unique_candidate_evaluations"])
        objective_requests = ""
        full_event_evaluations = int(result["full_thd_evaluations"]) + 1
        waveform_generations = int(result["waveform_generations"]) + 1
        budget = COMPARISON_CANDIDATE_BUDGET
        history = []
    else:
        raise ValueError(f"Unknown optimizer method: {method}")
    if method in OPTIMIZER_NAMES and not result.run_result.best.feasible:
        raise RuntimeError(f"{method} returned an infeasible candidate.")
    metrics = _metrics(modulation_index, candidate)
    elapsed = time.perf_counter() - started
    if metrics["event_voltage_total"] < 0:
        raise RuntimeError("Independent final THD must be non-negative.")
    row = _format_row(
        seed=seed,
        modulation_index=modulation_index,
        method=method,
        phases=candidate,
        baseline_thd=0.0,
        metrics=metrics,
        unique_evaluations=unique_evaluations,
        objective_requests=objective_requests,
        full_event_evaluations=full_event_evaluations,
        waveform_generations=waveform_generations,
        evaluation_budget=budget,
        runtime=elapsed,
    )
    return row, history, metrics


def run_experiment() -> list[dict[str, Any]]:
    """Run all modulation/seed/method combinations and create deliverables."""
    outputs = (
        CSV_OUTPUT,
        SUMMARY_OUTPUT,
        BEFORE_AFTER_FIGURE,
        CONVERGENCE_FIGURE,
        WAVEFORM_SPECTRUM_FIGURE,
    )
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing experiment outputs: " + ", ".join(existing)
        )
    guidance_usable, correlation = cnn_guidance_is_justified()
    methods = METHODS if guidance_usable else METHODS[:-1]
    model, normalization = _read_cnn() if guidance_usable else (None, None)

    rows: list[dict[str, Any]] = []
    convergence: dict[tuple[float, str], list[list[float]]] = {}
    representative: dict[
        str, tuple[tuple[float, float], dict[str, float | bool]]
    ] = {}
    for modulation_index in MODULATION_INDICES:
        for seed in SEEDS:
            baseline_row, baseline_metrics = _baseline_row(
                seed=seed,
                modulation_index=modulation_index,
            )
            rows.append(baseline_row)
            for method in methods[1:]:
                row, history, metrics = _optimizer_run(
                    method=method,
                    seed=seed,
                    modulation_index=modulation_index,
                    model=model,
                    normalization=normalization,
                )
                row["baseline_event_voltage_total_thd_percent"] = (
                    baseline_row["baseline_event_voltage_total_thd_percent"]
                )
                row["percent_reduction_event_voltage_total_thd"] = (
                    (
                        baseline_row["baseline_event_voltage_total_thd_percent"]
                        - row["optimized_event_voltage_total_thd_percent"]
                    )
                    / baseline_row["baseline_event_voltage_total_thd_percent"]
                    * 100.0
                )
                row["improved_on_baseline"] = (
                    row["percent_reduction_event_voltage_total_thd"] > 1e-12
                )
                rows.append(row)
                if history:
                    convergence.setdefault((modulation_index, method), []).append(history)
                if (
                    modulation_index == 0.75
                    and seed == SEEDS[0]
                ):
                    representative[method] = (row_phases(row), metrics)
            if modulation_index == 0.75 and seed == SEEDS[0]:
                representative["Baseline"] = (
                    canonicalize_candidate(BASELINE_PHASES_DEGREES[1:]),
                    baseline_metrics,
                )

    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_csv(rows)
    _write_before_after_figure(rows, methods)
    _write_convergence_figure(rows, convergence, methods)
    _write_waveform_spectrum_figure(representative, methods)
    _write_summary(rows, methods, correlation, guidance_usable)
    return rows


def row_phases(row: dict[str, Any]) -> tuple[float, float]:
    return (
        float(row["cell_2_phase_degrees"]),
        float(row["cell_3_phase_degrees"]),
    )


def _write_csv(rows: list[dict[str, Any]]) -> None:
    with CSV_OUTPUT.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _group_rows(rows: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["method"] == method]


def _write_before_after_figure(
    rows: list[dict[str, Any]],
    methods: tuple[str, ...],
) -> None:
    figure, axes = plt.subplots(1, len(MODULATION_INDICES), figsize=(20, 5), sharey=True)
    for axis, modulation in zip(axes, MODULATION_INDICES):
        selected = [
            row for row in rows
            if np.isclose(row["modulation_index"], modulation)
        ]
        baseline = _group_rows(selected, "Baseline")
        base_mean = float(np.mean([
            row["baseline_event_voltage_total_thd_percent"] for row in baseline
        ]))
        after_means = [
            float(np.mean([
                row["optimized_event_voltage_total_thd_percent"]
                for row in _group_rows(selected, method)
            ]))
            for method in methods[1:]
        ]
        x = np.arange(len(methods))
        axis.bar(x[0], base_mean, color="#777777", label="Baseline")
        axis.bar(x[1:], after_means, color="#2878B5", label="Optimized")
        axis.set_xticks(x, ["Base"] + [name.replace(" Hybrid", "\nHybrid") for name in methods[1:]], rotation=30, ha="right")
        axis.set_title(f"m = {modulation:.2f}")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Event-based Parseval voltage THD (%)")
    figure.suptitle("Baseline and independently verified optimized THD (mean of 10 seeds)")
    figure.tight_layout()
    figure.savefig(BEFORE_AFTER_FIGURE, dpi=160)
    plt.close(figure)


def _write_convergence_figure(
    rows: list[dict[str, Any]],
    histories: dict[tuple[float, str], list[list[float]]],
    methods: tuple[str, ...],
) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    colors = {
        "PSO": "#1f77b4",
        "POA": "#ff7f0e",
        "PFO": "#2ca02c",
        "HybridPOAPFO": "#9467bd",
    }
    all_histories = [
        history
        for (modulation, _method), per_seed in histories.items()
        for history in per_seed
    ]
    max_length = max((len(history) for history in all_histories), default=1)
    baseline_records = _group_rows(rows, "Baseline")
    axis.axhline(100.0, color="#555555", linestyle="--", label="Baseline = 100%")
    for method in methods[1:]:
        if method not in colors:
            continue
        normalized_histories: list[np.ndarray] = []
        for modulation in MODULATION_INDICES:
            base_values = [
                row["baseline_event_voltage_total_thd_percent"]
                for row in baseline_records
                if np.isclose(row["modulation_index"], modulation)
            ]
            baseline = float(np.mean(base_values))
            for history in histories.get((modulation, method), []):
                if not history or baseline <= 0:
                    continue
                values = np.asarray(history, dtype=np.float64)
                padded = np.full(max_length, values[-1], dtype=np.float64)
                padded[: min(max_length, values.size)] = values[:max_length]
                normalized_histories.append(padded / baseline * 100.0)
        if normalized_histories:
            matrix = np.vstack(normalized_histories)
            mean = np.mean(matrix, axis=0)
            std = np.std(matrix, axis=0, ddof=1) if matrix.shape[0] > 1 else np.zeros_like(mean)
            x = np.arange(max_length)
            axis.plot(x, mean, label=method, color=colors[method])
            axis.fill_between(x, mean - std, mean + std, color=colors[method], alpha=0.15)
    guided_rows = _group_rows(rows, "CNN-guided Hybrid")
    if guided_rows:
        ratios = np.asarray([
            row["optimized_event_voltage_total_thd_percent"]
            / row["baseline_event_voltage_total_thd_percent"] * 100.0
            for row in guided_rows
        ])
        axis.errorbar(
            [max_length - 1],
            [float(np.mean(ratios))],
            yerr=[float(np.std(ratios, ddof=1))],
            fmt="D",
            color="#d62728",
            capsize=4,
            label="CNN-guided verified endpoint (no event-THD history)",
        )
    axis.set_xlabel("Optimizer iteration (curves normalized to baseline = 100%)")
    axis.set_ylabel("Best event-based THD (% of paired baseline)")
    axis.set_title("Mean convergence ± 1 standard deviation")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(CONVERGENCE_FIGURE, dpi=160)
    plt.close(figure)


def _write_waveform_spectrum_figure(
    representative: dict[
        str, tuple[tuple[float, float], dict[str, float | bool]]
    ],
    methods: tuple[str, ...],
) -> None:
    figure, (wave_axis, spectrum_axis) = plt.subplots(2, 1, figsize=(12, 8))
    conditions = OPERATING_CONDITIONS
    phase_rad = np.deg2rad(conditions["phase_angle_degrees"])
    simulation = simulate_inverter(
        0.75,
        conditions["load_resistance_ohm"],
        conditions["load_inductance_h"],
        phase_rad,
        carrier_phase_offsets_degrees=(0.0, *representative["Baseline"][0]),
        dc_voltage_per_cell_v=conditions["dc_voltage_per_cell_v"],
    )
    time_ms = simulation.time_s * 1000.0
    wave_axis.plot(time_ms, simulation.output_voltage, color="#222222", label="Baseline")
    voltage_spectrum = None
    _, voltage_spectrum = calculate_thd(
        simulation.output_voltage,
        sampling_frequency_hz=SAMPLING_FREQUENCY_HZ,
        fundamental_frequency_hz=FUNDAMENTAL_FREQUENCY_HZ,
        max_harmonic=100,
        return_spectrum=True,
    )
    spectrum_axis.stem(
        voltage_spectrum.harmonic_orders,
        voltage_spectrum.amplitudes,
        linefmt="#222222",
        markerfmt=" ",
        basefmt=" ",
        label="Baseline",
    )
    for method in methods[1:]:
        candidate, _ = representative[method]
        result = simulate_inverter(
            0.75,
            conditions["load_resistance_ohm"],
            conditions["load_inductance_h"],
            phase_rad,
            carrier_phase_offsets_degrees=(0.0, *candidate),
            dc_voltage_per_cell_v=conditions["dc_voltage_per_cell_v"],
        )
        wave_axis.plot(time_ms, result.output_voltage, alpha=0.8, label=method)
        _, spectrum = calculate_thd(
            result.output_voltage,
            sampling_frequency_hz=SAMPLING_FREQUENCY_HZ,
            fundamental_frequency_hz=FUNDAMENTAL_FREQUENCY_HZ,
            max_harmonic=100,
            return_spectrum=True,
        )
        spectrum_axis.plot(
            spectrum.harmonic_orders,
            spectrum.amplitudes,
            marker=".",
            linewidth=1.0,
            label=method,
        )
    wave_axis.set_ylabel("Output voltage (V)")
    wave_axis.set_xlabel("Time (ms)")
    wave_axis.set_title("Measured simulator waveforms (m=0.75, seed 20261124)")
    wave_axis.legend(ncol=3, fontsize=8)
    wave_axis.grid(alpha=0.2)
    spectrum_axis.set_xlabel("Harmonic order")
    spectrum_axis.set_ylabel("Sampled voltage amplitude (V peak)")
    spectrum_axis.set_xlim(1, 100)
    spectrum_axis.set_title("Sampled 20 kHz voltage spectra (harmonics 1–100)")
    spectrum_axis.legend(ncol=3, fontsize=8)
    spectrum_axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(WAVEFORM_SPECTRUM_FIGURE, dpi=160)
    plt.close(figure)


def _paired_sign_test(
    baseline_rows: list[dict[str, Any]],
    method_rows: list[dict[str, Any]],
) -> tuple[float, float]:
    baseline_by_key = {
        (int(row["seed"]), float(row.get("modulation_index", 0.0))): row
        for row in baseline_rows
    }
    optimized_by_key = {
        (int(row["seed"]), float(row.get("modulation_index", 0.0))): row
        for row in method_rows
    }
    shared = sorted(set(baseline_by_key) & set(optimized_by_key))
    baseline = np.asarray([
        baseline_by_key[key]["optimized_event_voltage_total_thd_percent"]
        for key in shared
    ], dtype=np.float64)
    optimized = np.asarray([
        optimized_by_key[key]["optimized_event_voltage_total_thd_percent"]
        for key in shared
    ], dtype=np.float64)
    differences = baseline - optimized
    if differences.size == 0:
        return float("nan"), float("nan")
    non_tied = differences[np.abs(differences) > 1e-12]
    if non_tied.size == 0:
        return 0.0, 1.0
    positive = int(np.sum(non_tied > 0.0))
    negative = int(np.sum(non_tied < 0.0))
    smaller_tail = min(positive, negative)
    one_sided_probability = sum(
        math.comb(non_tied.size, index)
        for index in range(smaller_tail + 1)
    ) / (2**non_tied.size)
    return float(np.mean(differences)), float(min(1.0, 2.0 * one_sided_probability))


def _statistics_table(rows: list[dict[str, Any]], methods: tuple[str, ...]) -> str:
    lines = [
        "| Modulation | Method | Mean THD (%) | Std (%) | Median (%) | Best (%) | Worst (%) | Mean reduction (%) | Improved (%) | Paired mean reduction (pp) | Paired sign-test p |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for modulation in MODULATION_INDICES:
        subset = [
            row for row in rows
            if np.isclose(row["modulation_index"], modulation)
        ]
        baseline = _group_rows(subset, "Baseline")
        for method in methods:
            method_rows = _group_rows(subset, method)
            values = np.asarray([
                row["optimized_event_voltage_total_thd_percent"]
                for row in method_rows
            ], dtype=np.float64)
            reductions = np.asarray([
                row["percent_reduction_event_voltage_total_thd"]
                for row in method_rows
            ], dtype=np.float64)
            mean_difference, p_value = _paired_sign_test(baseline, method_rows)
            std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            improved = int(np.sum(reductions > 1e-12))
            lines.append(
                f"| {modulation:.2f} | {method} | {np.mean(values):.6f} | {std:.6f} | "
                f"{np.median(values):.6f} | {np.min(values):.6f} | {np.max(values):.6f} | "
                f"{(improved / values.size * 100.0):.1f}% ({improved}/{values.size}) | "
                f"{mean_difference:.6f} | {p_value:.6g} |"
            )
    return "\n".join(lines)


def _write_summary(
    rows: list[dict[str, Any]],
    methods: tuple[str, ...],
    correlation: float,
    guidance_usable: bool,
) -> None:
    overall_lines = [
        "| Method | Mean THD (%) | Std (%) | Median (%) | Best (%) | Worst (%) | Mean reduction (%) | Improved (%) | Paired mean reduction (pp) | Paired sign-test p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    optimized_rows = [row for row in rows if row["method"] != "Baseline"]
    total_improved = sum(
        float(row["percent_reduction_event_voltage_total_thd"]) > 1e-12
        for row in optimized_rows
    )
    total_worse = sum(
        float(row["percent_reduction_event_voltage_total_thd"]) < -1e-12
        for row in optimized_rows
    )
    no_improvement_conditions: list[str] = []
    for modulation in MODULATION_INDICES:
        subset = [
            row for row in optimized_rows
            if np.isclose(float(row["modulation_index"]), modulation)
        ]
        for method in methods[1:]:
            method_rows = _group_rows(subset, method)
            if not any(
                float(row["percent_reduction_event_voltage_total_thd"]) > 1e-12
                for row in method_rows
            ):
                no_improvement_conditions.append(
                    f"{method} at m={modulation:.2f}"
                )
    baseline_rows = _group_rows(rows, "Baseline")
    for method in methods:
        method_rows = _group_rows(rows, method)
        values = np.asarray([
            row["optimized_event_voltage_total_thd_percent"]
            for row in method_rows
        ], dtype=np.float64)
        reductions = np.asarray([
            row["percent_reduction_event_voltage_total_thd"]
            for row in method_rows
        ], dtype=np.float64)
        baseline_by_key = {
            (int(row["seed"]), float(row["modulation_index"])): row
            for row in baseline_rows
        }
        paired = np.asarray([
            baseline_by_key[(int(row["seed"]), float(row["modulation_index"]))][
                "optimized_event_voltage_total_thd_percent"
            ] - row["optimized_event_voltage_total_thd_percent"]
            for row in method_rows
        ], dtype=np.float64)

        _, p_value = _paired_sign_test(baseline_rows, method_rows)
        std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        improved = int(np.sum(reductions > 1e-12))
        overall_lines.append(
            f"| {method} | {np.mean(values):.6f} | {std:.6f} | {np.median(values):.6f} | "
            f"{np.min(values):.6f} | {np.max(values):.6f} | {np.mean(reductions):.6f} | "
            f"{(improved / values.size * 100.0):.1f}% ({improved}/{values.size}) | "
            f"{np.mean(paired):.6f} | {p_value:.6g} |"
        )

    content = f"""# Baseline versus optimized THD experiment

## Protocol

- Fixed conditions: DC voltage {OPERATING_CONDITIONS['dc_voltage_per_cell_v']:.1f} V/cell, load {OPERATING_CONDITIONS['load_resistance_ohm']:.1f} ohm / {OPERATING_CONDITIONS['load_inductance_h']:.3f} H, reference phase {OPERATING_CONDITIONS['phase_angle_degrees']:.1f} degrees.
- Modulation indices: {', '.join(f'{value:.2f}' for value in MODULATION_INDICES)}.
- Seeds: {', '.join(str(seed) for seed in SEEDS)} (10 paired seeds per modulation).
- Cell 1 is fixed at 0 degrees; the baseline is the existing canonical pair (60, 120) degrees. The baseline uses exactly the same event-based and sampled metric functions as final optimized candidates.
- Primary objective: independently recomputed event-based Parseval total voltage THD. Additional measurements are event-based voltage harmonics 2–100, sampled 20 kHz voltage/current THD harmonics 2–100, and event-Fourier RL-derived current THD harmonics 2–100. These THD definitions have different bandwidth/semantics and are reported separately.
- PSO/POA/PFO/Hybrid use the existing 8-member, 8-iteration configuration. Actual unique objective evaluations and requests are in the CSV; nominal request budget is 72. CNN-guided hybrid is included because the approved Phase 5 500-candidate score/THD Spearman correlation is {correlation:.6f} (threshold {CNN_GUIDED_CORRELATION_THRESHOLD:.2f}). It uses a 64-unique-candidate ranking budget, then fully evaluates top-{COMPARISON_TOP_K} finalists and independently re-evaluates the selected candidate. Consequently its full event-THD budget and waveform-generation count differ; counts are separate in the CSV.
- Optimized phases and all reported final metrics are re-simulated independently after selection. Candidate failures are not replaced with another seed's result.

## Aggregate statistics

Primary metric is event-based Parseval total voltage THD (%). Standard deviation is the sample standard deviation (ddof=1). Paired tests compare each method's run-wise values with the baseline at the same seed and modulation. Paired mean reduction is baseline minus optimized THD, in percentage points; positive values favor the optimizer. The exact two-sided paired sign-test p-value excludes ties and is descriptive; it is not adjusted for multiple comparisons.

{chr(10).join(overall_lines)}

## Per-modulation results

{_statistics_table(rows, methods)}

## Honest interpretation

Reduction is calculated per run as `(baseline THD - optimized THD) / baseline THD * 100`. An improved run means a strict reduction greater than numerical tolerance (1e-12 percentage points). Flat or worse runs remain in the CSV and statistics. The per-seed result table preserves each run and reports event voltage THD as the primary independently verified result.

Across all optimizer runs, {total_improved}/{len(optimized_rows)} improved, {len(optimized_rows) - total_improved - total_worse}/{len(optimized_rows)} were flat, and {total_worse}/{len(optimized_rows)} were worse. No improving run was observed for: {', '.join(no_improvement_conditions)}.

{"CNN-guided hybrid was included" if guidance_usable else "CNN-guided hybrid was omitted"} based on a measured ranking correlation ({correlation:.6f}); this is not evidence that the CNN improves THD. The sampled and event-based harmonic metrics are not interchangeable. The CNN checkpoint was loaded read-only; the existing dataset was not read or modified.

## Figures

- [Before/after THD bars](thd_experiment_before_after.png)
- [Convergence curves](thd_experiment_convergence.png). CNN-guided is marked as a verified endpoint because its candidate ranking does not retain event-THD iteration history; extra THD calls were not added just to create a curve.
- [Representative voltage waveform and spectrum](thd_experiment_waveform_spectrum.png), at modulation 0.75 and seed {SEEDS[0]}.

The complete 300-row per-seed record is [`thd_experiment.csv`](thd_experiment.csv).

Run the experiment from the project root with `python -m scripts.run_thd_experiment`. It refuses to overwrite any existing output artifact.
"""
    SUMMARY_OUTPUT.write_text(content, encoding="utf-8")


def main() -> None:
    rows = run_experiment()
    print(f"Wrote {len(rows)} rows to {CSV_OUTPUT}")
    print(f"Wrote summary to {SUMMARY_OUTPUT}")


if __name__ == "__main__":
    main()
