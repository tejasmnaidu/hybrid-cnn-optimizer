"""Build complete, local downloadable artifacts for a live simulation run."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import io
import json
import math
import uuid
import zipfile

import matplotlib.pyplot as plt
import numpy as np

from evaluation.live_phase_optimization import (
    LiveConditionedPhaseOptimization,
    LivePhaseOptimization,
)
from evaluation.live_simulation import LiveSimulationResult
from multilevel.inverter_simulator import (
    CARRIER_FREQUENCY_HZ,
    CELL_COUNT,
    FUNDAMENTAL_FREQUENCY_HZ,
    SAMPLING_FREQUENCY_HZ,
)
from optimization.thd_surrogate_search import SurrogateScreeningRunResult


THD_REPORTING_RESOLUTION_PERCENTAGE_POINTS = 0.001


@dataclass(frozen=True)
class LiveExportBundle:
    waveform_csv: bytes
    summary_json: bytes
    experiment_zip: bytes


def _validate_live_result(result: LiveSimulationResult) -> int:
    simulation = result.simulation
    sample_count = int(np.asarray(simulation.time_s).size)
    expected_shapes = {
        "reference": (sample_count,),
        "cell_states": (CELL_COUNT, sample_count),
        "switching_gates": (CELL_COUNT, 4, sample_count),
        "cell_voltages": (CELL_COUNT, sample_count),
        "output_voltage": (sample_count,),
        "load_current": (sample_count,),
        "target_reference_voltage": (sample_count,),
        "cnn_prediction_voltage": (sample_count,),
    }
    if sample_count == 0:
        raise ValueError("Cannot export an empty live simulation.")
    for name, shape in expected_shapes.items():
        values = np.asarray(
            getattr(simulation, name)
            if hasattr(simulation, name)
            else getattr(result, name),
        )
        if values.shape != shape:
            raise ValueError(f"{name} has shape {values.shape}; expected {shape}.")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values.")
    if not np.isfinite(np.asarray(simulation.time_s, dtype=np.float64)).all():
        raise ValueError("time_s contains non-finite values.")
    if not np.isfinite(np.asarray(result.harmonic_result.harmonic_peak_amplitudes_v)).all():
        raise ValueError("Harmonic amplitudes contain non-finite values.")
    return sample_count


def _csv_bytes(
    baseline: LiveSimulationResult,
    candidate: LiveSimulationResult | None,
) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    headers = ["time_s"]
    for label, result in (("baseline", baseline), ("candidate", candidate)):
        if result is None:
            continue
        headers.extend(
            [
                f"{label}_reference_modulation",
                f"{label}_reference_target_v",
                f"{label}_cnn_predicted_target_v",
                f"{label}_cnn_prediction_error_v",
                f"{label}_output_voltage_v",
                f"{label}_load_current_a",
            ]
        )
        headers.extend(
            f"{label}_dc_source_voltage_cell{cell + 1}_v"
            for cell in range(CELL_COUNT)
        )
        headers.extend(
            f"{label}_cell_voltage_cell{cell + 1}_v"
            for cell in range(CELL_COUNT)
        )
        headers.extend(
            f"{label}_cell_state_cell{cell + 1}"
            for cell in range(CELL_COUNT)
        )
        headers.extend(
            f"{label}_gate_cell{cell + 1}_s{switch + 1}"
            for cell in range(CELL_COUNT)
            for switch in range(4)
        )
    writer.writerow(headers)
    sample_count = _validate_live_result(baseline)
    if candidate is not None and _validate_live_result(candidate) != sample_count:
        raise ValueError("Baseline and candidate waveform lengths do not match.")

    for index in range(sample_count):
        row: list[float | int] = [float(baseline.simulation.time_s[index])]
        for result in (baseline, candidate):
            if result is None:
                continue
            simulation = result.simulation
            row.extend(
                [
                    float(simulation.reference[index]),
                    float(result.target_reference_voltage[index]),
                    float(result.cnn_prediction_voltage[index]),
                    float(
                        result.cnn_prediction_voltage[index]
                        - result.target_reference_voltage[index]
                    ),
                    float(simulation.output_voltage[index]),
                    float(simulation.load_current[index]),
                ]
            )
            row.extend([simulation.dc_voltage_per_cell_v] * CELL_COUNT)
            row.extend(float(simulation.cell_voltages[cell, index]) for cell in range(CELL_COUNT))
            row.extend(int(simulation.cell_states[cell, index]) for cell in range(CELL_COUNT))
            row.extend(
                int(simulation.switching_gates[cell, switch, index])
                for cell in range(CELL_COUNT)
                for switch in range(4)
            )
        writer.writerow(row)
    return output.getvalue().encode("utf-8")


def _harmonic_payload(result: LiveSimulationResult) -> dict[str, object]:
    harmonic = result.harmonic_result
    return {
        "parseval_full_spectrum_thd_percent": float(harmonic.total_thd_percent),
        "event_finite_band_thd_orders_2_to_100_percent": float(harmonic.thd_percent),
        "sampled_finite_band_thd_orders_2_to_100_percent": float(
            harmonic.sampled_thd_percent
        ),
        "sampled_load_current_thd_orders_2_to_100_percent": float(
            result.current_thd_percent
        ),
        "fundamental_peak_amplitude_v": float(harmonic.fundamental_peak_amplitude_v),
        "fundamental_phase_degrees": float(math.degrees(harmonic.fundamental_phase_rad)),
        "target_fundamental_peak_amplitude_v": float(
            harmonic.target_fundamental_peak_amplitude_v
        ),
        "target_fundamental_phase_degrees": float(
            math.degrees(harmonic.target_fundamental_phase_rad)
        ),
        "fundamental_phasor_tracking_error_percent": float(
            harmonic.fundamental_phasor_error_percent
        ),
        "harmonic_spectrum": [
            {
                "order": int(order),
                "frequency_hz": float(order * FUNDAMENTAL_FREQUENCY_HZ),
                "peak_amplitude_v": float(amplitude),
                "phase_degrees": float(math.degrees(phase)),
                "fourier_coefficient_real_v": float(coefficient.real),
                "fourier_coefficient_imaginary_v": float(coefficient.imag),
            }
            for order, amplitude, phase, coefficient in zip(
                harmonic.harmonic_orders,
                harmonic.harmonic_peak_amplitudes_v,
                harmonic.harmonic_phase_rad,
                harmonic.fourier_coefficients,
            )
        ],
        "load_current_harmonic_spectrum": [
            {
                "order": int(order),
                "frequency_hz": float(frequency),
                "peak_amplitude_a": float(amplitude),
            }
            for order, frequency, amplitude in zip(
                result.current_harmonic_spectrum.harmonic_orders,
                result.current_harmonic_spectrum.frequencies_hz,
                result.current_harmonic_spectrum.amplitudes,
            )
        ],
    }


def _simulation_payload(result: LiveSimulationResult) -> dict[str, object]:
    return {
        "operating_conditions": {
            "modulation_index": float(result.modulation_index),
            "dc_voltage_per_cell_v": float(result.dc_voltage_per_cell_v),
            "load_resistance_ohm": float(result.load_resistance_ohm),
            "load_inductance_h": float(result.load_inductance_h),
            "reference_phase_degrees": float(result.phase_angle_degrees),
        },
        "carrier_phase_offsets_degrees": [
            float(value) for value in result.carrier_phase_offsets_degrees
        ],
        "cnn_reference_target_metrics": {
            key: float(value) for key, value in result.cnn_metrics.items()
        },
        "harmonics": _harmonic_payload(result),
        "periodic_initial_load_current_a": float(
            result.simulation.periodic_initial_current_a
        ),
        "waveform_sample_count": int(result.simulation.time_s.size),
    }


def _comparison_payload(
    baseline: LiveSimulationResult,
    candidate: LiveSimulationResult | None,
) -> dict[str, object] | None:
    if candidate is None:
        return None
    baseline_thd = float(baseline.harmonic_result.total_thd_percent)
    candidate_thd = float(candidate.harmonic_result.total_thd_percent)
    delta = candidate_thd - baseline_thd
    baseline_current_thd = float(baseline.current_thd_percent)
    candidate_current_thd = float(candidate.current_thd_percent)
    if delta < -THD_REPORTING_RESOLUTION_PERCENTAGE_POINTS:
        interpretation = (
            "Lower measured Parseval THD at this one live operating point; "
            "this is descriptive and does not establish validation-set improvement."
        )
    elif delta > THD_REPORTING_RESOLUTION_PERCENTAGE_POINTS:
        interpretation = "Higher measured Parseval THD at this one live operating point."
    else:
        interpretation = (
            "No meaningful change at the declared 0.001 percentage-point "
            "reporting resolution."
        )
    return {
        "thd_definition": (
            "Parseval full-spectrum THD from the event-integrated switching waveform"
        ),
        "baseline_thd_percent": baseline_thd,
        "candidate_thd_percent": candidate_thd,
        "candidate_minus_baseline_percentage_points": delta,
        "current_thd_definition": (
            "20 kHz sampled finite-band load-current THD, harmonic orders 2 through 100"
        ),
        "baseline_current_thd_percent": baseline_current_thd,
        "candidate_current_thd_percent": candidate_current_thd,
        "candidate_minus_baseline_current_thd_percentage_points": (
            candidate_current_thd - baseline_current_thd
        ),
        "relative_change_percent": (
            delta / baseline_thd * 100.0 if baseline_thd > 0 else None
        ),
        "reporting_resolution_percentage_points": (
            THD_REPORTING_RESOLUTION_PERCENTAGE_POINTS
        ),
        "interpretation": interpretation,
        "baseline_live_tracking_feasible_at_0_05_percent": bool(
            baseline.harmonic_result.fundamental_phasor_error_percent <= 0.05
        ),
        "candidate_live_tracking_feasible_at_0_05_percent": bool(
            candidate.harmonic_result.fundamental_phasor_error_percent <= 0.05
        ),
    }


def _optimization_payload(
    optimization: LivePhaseOptimization | LiveConditionedPhaseOptimization | None,
) -> dict[str, object] | None:
    if optimization is None:
        return None
    result = optimization.run_result
    if isinstance(optimization, LiveConditionedPhaseOptimization):
        is_surrogate_screening = isinstance(result, SurrogateScreeningRunResult)
        payload = {
            "objective": (
                "Single live operating point: event-based full-spectrum Parseval voltage THD "
                "with 0.05% fundamental phasor tracking constraint"
            ),
            "optimizer": optimization.optimizer_name,
            "live_operating_conditions": {
                "modulation_index": optimization.modulation_index,
                "dc_voltage_per_cell_v": optimization.dc_voltage_per_cell_v,
                "load_resistance_ohm": optimization.load_resistance_ohm,
                "load_inductance_h": optimization.load_inductance_h,
                "phase_angle_degrees": optimization.phase_angle_degrees,
            },
            "hard_tracking_constraint_percent": 0.05,
            "particles": None if is_surrogate_screening else optimization.particles,
            "iterations": None if is_surrogate_screening else optimization.iterations,
            "seed": result.seed,
            "best_candidate_phase_offsets_degrees": [
                0.0,
                *[float(value) for value in result.best.canonical_phases_degrees],
            ],
            "best_live_parseval_voltage_thd_percent": float(
                result.best.mean_total_thd_percent
            ),
            "baseline_live_parseval_voltage_thd_percent": float(
                optimization.baseline_evaluation.mean_total_thd_percent
            ),
            "best_live_tracking_error_percent": float(
                result.best.maximum_tracking_error_percent
            ),
            "objective_requests": result.objective_requests,
            "unique_candidate_simulations": result.unique_candidate_evaluations,
            "runtime_seconds": float(result.runtime_seconds),
            "convergence_history": [
                {
                    "iteration": index,
                    "phase_cell2_degrees": float(item.canonical_phases_degrees[0]),
                    "phase_cell3_degrees": float(item.canonical_phases_degrees[1]),
                    "best_parseval_voltage_thd_percent": float(
                        item.mean_total_thd_percent
                    ),
                    "tracking_error_percent": float(
                        item.maximum_tracking_error_percent
                    ),
                    "feasible": bool(item.feasible),
                }
                for index, item in enumerate(result.convergence_history)
            ],
        }
        if is_surrogate_screening:
            payload["screened_candidate_count"] = result.screened_candidate_count
            payload["exact_finalist_count"] = result.objective_requests
            payload["surrogate_checkpoint_sha256"] = result.checkpoint_sha256
            payload["surrogate_dataset_sha256"] = result.dataset_sha256
        return payload
    return {
        "objective": (
            "Mean Parseval full-spectrum THD on the deterministic stratified "
            "30-record training subset"
        ),
        "hard_tracking_constraint_percent": 0.05,
        "training_sample_ids": list(optimization.training_sample_ids),
        "particles": optimization.particles,
        "iterations": optimization.iterations,
        "seed": result.seed,
        "best_candidate_phase_offsets_degrees": [
            0.0,
            *[float(value) for value in result.best.canonical_phases_degrees],
        ],
        "best_training_mean_parseval_thd_percent": float(
            result.best.mean_total_thd_percent
        ),
        "best_training_maximum_tracking_error_percent": float(
            result.best.maximum_tracking_error_percent
        ),
        "best_training_feasible": bool(result.best.feasible),
        "baseline_training_mean_parseval_thd_percent": float(
            optimization.baseline_training_evaluation.mean_total_thd_percent
        ),
        "objective_requests": result.objective_requests,
        "unique_candidate_evaluations": result.unique_candidate_evaluations,
        "record_evaluations": result.record_evaluations,
        "cache_hits": result.cache_hits,
        "runtime_seconds": float(result.runtime_seconds),
        "final_particles_feasible": result.final_particles_feasible,
        "convergence_history": [
            {
                "iteration": index,
                "best_phase_cell2_degrees": float(
                    evaluation.canonical_phases_degrees[0]
                ),
                "best_phase_cell3_degrees": float(
                    evaluation.canonical_phases_degrees[1]
                ),
                "best_mean_parseval_thd_percent": float(
                    evaluation.mean_total_thd_percent
                ),
                "maximum_tracking_error_percent": float(
                    evaluation.maximum_tracking_error_percent
                ),
                "feasible": bool(evaluation.feasible),
                "feasible_particles": int(
                    optimization.run_result.feasibility_history[index]
                ),
            }
            for index, evaluation in enumerate(result.convergence_history)
        ],
    }


def _summary_json(
    baseline: LiveSimulationResult,
    candidate: LiveSimulationResult | None,
    optimization: LivePhaseOptimization | LiveConditionedPhaseOptimization | None,
    run_metadata: dict[str, str] | None,
) -> bytes:
    payload = {
        "schema_version": 1,
        "run_metadata": {
            "run_id": uuid.uuid4().hex,
            "created_at_local": datetime.now().astimezone().isoformat(),
            **(run_metadata or {}),
        },
        "task": "Live seven-level inverter software simulation and CNN target prediction",
        "simulator": {
            "cell_count": CELL_COUNT,
            "baseline_dc_voltage_per_cell_v": float(
                baseline.simulation.dc_voltage_per_cell_v
            ),
            "fundamental_frequency_hz": FUNDAMENTAL_FREQUENCY_HZ,
            "carrier_frequency_hz": CARRIER_FREQUENCY_HZ,
            "sampling_frequency_hz": SAMPLING_FREQUENCY_HZ,
            "time_step_s": 1.0 / SAMPLING_FREQUENCY_HZ,
        },
        "baseline": _simulation_payload(baseline),
        "candidate": _simulation_payload(candidate) if candidate is not None else None,
        "comparison": _comparison_payload(baseline, candidate),
        "optimization": _optimization_payload(optimization),
        "scientific_limitations": [
            "The CNN predicts the analytical reference target waveform; it does not predict switching gates.",
            "CNN prediction accuracy is not evidence of harmonic reduction.",
            "Live baseline/candidate THD is a single-operating-point software comparison, not a validation-set claim.",
            "The inverter is an ideal model; this export is not hardware, MATLAB, or Simulink validation.",
            "Event-integrated, sampled finite-band, and Parseval full-spectrum THD are distinct metrics.",
        ],
    }
    return json.dumps(payload, indent=2, allow_nan=False).encode("utf-8")


def _harmonic_csv(
    baseline: LiveSimulationResult,
    candidate: LiveSimulationResult | None,
) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(
        [
            "simulation",
            "harmonic_order",
            "frequency_hz",
            "peak_amplitude_v",
            "phase_degrees",
            "fourier_coefficient_real_v",
            "fourier_coefficient_imaginary_v",
        ]
    )
    for label, result in (("baseline", baseline), ("candidate", candidate)):
        if result is None:
            continue
        harmonic = result.harmonic_result
        for order, amplitude, phase, coefficient in zip(
            harmonic.harmonic_orders,
            harmonic.harmonic_peak_amplitudes_v,
            harmonic.harmonic_phase_rad,
            harmonic.fourier_coefficients,
        ):
            writer.writerow(
                [
                    label,
                    int(order),
                    float(order * FUNDAMENTAL_FREQUENCY_HZ),
                    float(amplitude),
                    math.degrees(float(phase)),
                    float(coefficient.real),
                    float(coefficient.imag),
                ]
            )
    return output.getvalue().encode("utf-8")


def _current_harmonic_csv(
    baseline: LiveSimulationResult,
    candidate: LiveSimulationResult | None,
) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["simulation", "harmonic_order", "frequency_hz", "peak_amplitude_a"])
    for label, result in (("baseline", baseline), ("candidate", candidate)):
        if result is None:
            continue
        spectrum = result.current_harmonic_spectrum
        for order, frequency, amplitude in zip(
            spectrum.harmonic_orders,
            spectrum.frequencies_hz,
            spectrum.amplitudes,
        ):
            writer.writerow(
                [label, int(order), float(frequency), float(amplitude)]
            )
    return output.getvalue().encode("utf-8")


def _convergence_csv(
    optimization: LivePhaseOptimization | LiveConditionedPhaseOptimization | None,
) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(
        [
            "iteration",
            "best_phase_cell2_degrees",
            "best_phase_cell3_degrees",
            "best_mean_parseval_thd_percent",
            "maximum_tracking_error_percent",
            "feasible",
            "feasible_particles",
        ]
    )
    if optimization is not None:
        for iteration, evaluation in enumerate(
            optimization.run_result.convergence_history
        ):
            writer.writerow(
                [
                    iteration,
                    *evaluation.canonical_phases_degrees,
                    evaluation.mean_total_thd_percent,
                    evaluation.maximum_tracking_error_percent,
                    evaluation.feasible,
                    optimization.run_result.feasibility_history[iteration],
                ]
            )
    return output.getvalue().encode("utf-8")


def _plot_png(
    baseline: LiveSimulationResult,
    candidate: LiveSimulationResult | None,
    *,
    harmonics: bool,
) -> bytes:
    figure, axis = plt.subplots(figsize=(10, 4.5))
    figure.patch.set_facecolor("#111a26")
    axis.set_facecolor("#111a26")
    if harmonics:
        series = [
            ("Baseline", baseline.harmonic_result.harmonic_peak_amplitudes_v),
        ]
        if candidate is not None:
            series.append(
                ("Candidate", candidate.harmonic_result.harmonic_peak_amplitudes_v)
            )
        orders = baseline.harmonic_result.harmonic_orders
        for label, amplitudes in series:
            axis.stem(
                orders,
                amplitudes,
                label=label,
                linefmt="#43d6c5" if label == "Baseline" else "#659df5",
                markerfmt="o",
                basefmt="#344356",
            )
        axis.set_xlabel("Harmonic order")
        axis.set_ylabel("Peak amplitude (V)")
        axis.set_title("Event-integrated output-voltage harmonic spectrum")
    else:
        axis.plot(
            baseline.simulation.time_s * 1000.0,
            baseline.simulation.output_voltage,
            label="Baseline inverter output",
            color="#43d6c5",
        )
        axis.plot(
            baseline.simulation.time_s * 1000.0,
            baseline.target_reference_voltage,
            label="Analytical target reference",
            color="#e7ad65",
            linestyle="--",
        )
        axis.plot(
            baseline.simulation.time_s * 1000.0,
            baseline.cnn_prediction_voltage,
            label="Baseline CNN target prediction",
            color="#659df5",
            linestyle=":",
        )
        if candidate is not None:
            axis.plot(
                candidate.simulation.time_s * 1000.0,
                candidate.simulation.output_voltage,
                label="Candidate inverter output",
                color="#d889d8",
                alpha=0.8,
            )
        axis.set_xlabel("Time (ms)")
        axis.set_ylabel("Voltage (V)")
        axis.set_title("Live inverter and reference-target waveforms")
    axis.tick_params(colors="#9aabc0")
    axis.grid(True, color="#344356", alpha=0.45, linestyle=":")
    axis.legend(facecolor="#172332", edgecolor="#344356", labelcolor="#dce6f1")
    for spine in axis.spines.values():
        spine.set_color("#344356")
    figure.tight_layout()
    image = io.BytesIO()
    figure.savefig(image, format="png", dpi=140)
    plt.close(figure)
    return image.getvalue()


def _current_harmonic_plot_png(
    baseline: LiveSimulationResult,
    candidate: LiveSimulationResult | None,
) -> bytes:
    figure, axis = plt.subplots(figsize=(10, 4.5))
    figure.patch.set_facecolor("#111a26")
    axis.set_facecolor("#111a26")
    for label, result, color, style in (
        ("Baseline load current", baseline, "#43d6c5", "-"),
        ("PSO load current", candidate, "#d889d8", "--"),
    ):
        if result is None:
            continue
        spectrum = result.current_harmonic_spectrum
        axis.plot(
            spectrum.frequencies_hz,
            spectrum.amplitudes,
            label=label,
            color=color,
            linestyle=style,
        )
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Harmonic peak amplitude (A)")
    axis.set_title("Sampled load-current harmonic spectrum (orders 1–100)")
    axis.tick_params(colors="#9aabc0")
    axis.grid(True, color="#344356", alpha=0.45, linestyle=":")
    axis.legend(facecolor="#172332", edgecolor="#344356", labelcolor="#dce6f1")
    for spine in axis.spines.values():
        spine.set_color("#344356")
    figure.tight_layout()
    image = io.BytesIO()
    figure.savefig(image, format="png", dpi=140)
    plt.close(figure)
    return image.getvalue()


def build_live_export(
    baseline: LiveSimulationResult,
    *,
    candidate: LiveSimulationResult | None = None,
    optimization: LivePhaseOptimization | LiveConditionedPhaseOptimization | None = None,
    run_metadata: dict[str, str] | None = None,
) -> LiveExportBundle:
    """Create waveform CSV, complete JSON summary, and a ZIP with plots/history."""
    _validate_live_result(baseline)
    if candidate is not None:
        _validate_live_result(candidate)
        baseline_conditions = (
            baseline.modulation_index,
            baseline.dc_voltage_per_cell_v,
            baseline.load_resistance_ohm,
            baseline.load_inductance_h,
            baseline.phase_angle_degrees,
        )
        candidate_conditions = (
            candidate.modulation_index,
            candidate.dc_voltage_per_cell_v,
            candidate.load_resistance_ohm,
            candidate.load_inductance_h,
            candidate.phase_angle_degrees,
        )
        if not np.allclose(baseline_conditions, candidate_conditions, rtol=0.0, atol=1e-12):
            raise ValueError("Baseline and candidate must use identical operating conditions.")

    waveform_csv = _csv_bytes(baseline, candidate)
    summary_json = _summary_json(baseline, candidate, optimization, run_metadata)
    harmonics_csv = _harmonic_csv(baseline, candidate)
    current_harmonics_csv = _current_harmonic_csv(baseline, candidate)
    convergence_csv = _convergence_csv(optimization)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as package:
        package.writestr("waveforms.csv", waveform_csv)
        package.writestr("summary.json", summary_json)
        package.writestr("harmonic_spectrum.csv", harmonics_csv)
        package.writestr("current_harmonic_spectrum.csv", current_harmonics_csv)
        package.writestr("pso_convergence_history.csv", convergence_csv)
        package.writestr(
            "plots/live_waveforms.png",
            _plot_png(baseline, candidate, harmonics=False),
        )
        package.writestr(
            "plots/harmonic_spectrum.png",
            _plot_png(baseline, candidate, harmonics=True),
        )
        package.writestr(
            "plots/current_harmonic_spectrum.png",
            _current_harmonic_plot_png(baseline, candidate),
        )
    return LiveExportBundle(
        waveform_csv=waveform_csv,
        summary_json=summary_json,
        experiment_zip=archive.getvalue(),
    )
