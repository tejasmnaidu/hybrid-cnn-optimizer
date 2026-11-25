"""Run one ideal inverter simulation and the trained CNN reference prediction."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn

from evaluation.event_based_thd import EventBasedHarmonicResult, evaluate_event_based_thd
from evaluation.harmonic_analysis import HarmonicSpectrum
from evaluation.metrics import mae, r_squared, rmse
from evaluation.thd_analysis import calculate_thd
from multilevel.inverter_simulator import (
    CELL_COUNT,
    DC_VOLTAGE_PER_CELL_V,
    InverterSimulation,
    simulate_inverter,
)
from preprocessing.preprocess import NormalizationStats


ALLOWED_LOAD_RESISTANCES_OHM = (10.0, 20.0, 30.0, 40.0)
ALLOWED_LOAD_INDUCTANCES_H = (0.01, 0.02, 0.03, 0.05)


@dataclass(frozen=True)
class LiveSimulationResult:
    modulation_index: float
    dc_voltage_per_cell_v: float
    load_resistance_ohm: float
    load_inductance_h: float
    phase_angle_degrees: float
    simulation: InverterSimulation
    harmonic_result: EventBasedHarmonicResult
    target_reference_voltage: np.ndarray
    cnn_prediction_voltage: np.ndarray
    cnn_metrics: dict[str, float]
    voltage_thd_percent: float
    current_thd_percent: float
    current_harmonic_spectrum: HarmonicSpectrum
    carrier_phase_offsets_degrees: tuple[float, float, float] = (0.0, 60.0, 120.0)


def operating_condition_support_warnings(
    operating_conditions: np.ndarray,
    training_conditions: np.ndarray,
    feature_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Report marginal operating inputs outside observed training support.

    Phase is treated circularly in radians; a full-circle phase range is not
    incorrectly flagged at the equivalent 0/360-degree boundary. This does
    not detect unseen combinations of individually supported features.
    """
    candidate = np.asarray(operating_conditions, dtype=np.float64)
    training = np.asarray(training_conditions, dtype=np.float64)
    if candidate.shape != (len(feature_names),):
        raise ValueError("operating_conditions must contain one value per named feature.")
    if training.ndim != 2 or training.shape[1] != len(feature_names) or training.shape[0] == 0:
        raise ValueError("training_conditions must be a non-empty matrix matching feature_names.")
    if not np.isfinite(candidate).all() or not np.isfinite(training).all():
        raise ValueError("Operating conditions used for support checks must be finite.")

    warnings: list[str] = []
    for name in ("modulationIndex",):
        if name not in feature_names:
            continue
        index = feature_names.index(name)
        lower = float(np.min(training[:, index]))
        upper = float(np.max(training[:, index]))
        tolerance = max(1e-6, 1e-6 * (upper - lower))
        value = float(candidate[index])
        if value < lower - tolerance or value > upper + tolerance:
            warnings.append(
                f"{name}={value:g} is outside the training range [{lower:g}, {upper:g}]."
            )

    for name in ("loadResistance", "loadInductance"):
        if name not in feature_names:
            continue
        index = feature_names.index(name)
        observed = np.unique(training[:, index])
        value = float(candidate[index])
        if not np.any(np.isclose(observed, value, rtol=0.0, atol=1e-6)):
            warnings.append(
                f"{name}={value:g} was not an observed training category "
                f"({', '.join(f'{item:g}' for item in observed)})."
            )

    phase_name = "phaseAngle"
    if phase_name in feature_names:
        index = feature_names.index(phase_name)
        period = 2.0 * math.pi
        observed = np.sort(np.mod(training[:, index], period))
        gaps = np.diff(np.concatenate((observed, [observed[0] + period])))
        largest_gap_index = int(np.argmax(gaps))
        largest_gap = float(gaps[largest_gap_index])
        phase = float(np.mod(candidate[index], period))
        gap_start = float(observed[largest_gap_index])
        gap_end = float(np.mod(gap_start + largest_gap, period))
        is_inside_largest_gap = (
            gap_start < phase < gap_end
            if gap_start < gap_end
            else phase > gap_start or phase < gap_end
        )
        if largest_gap > math.pi and is_inside_largest_gap:
            warnings.append(
                "phaseAngle falls in a large unobserved circular training sector "
                f"({math.degrees(largest_gap):.1f} degrees)."
            )
    return tuple(warnings)


def _validate_operating_inputs(
    modulation_index: float,
    dc_voltage_per_cell_v: float,
    load_resistance_ohm: float,
    load_inductance_h: float,
    phase_angle_degrees: float,
) -> tuple[float, float, float, float]:
    try:
        modulation, dc_voltage, resistance, inductance, phase_degrees = (
            float(modulation_index),
            float(dc_voltage_per_cell_v),
            float(load_resistance_ohm),
            float(load_inductance_h),
            float(phase_angle_degrees),
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Operating conditions must be finite numeric scalars.") from exc
    if not all(math.isfinite(value) for value in (modulation, dc_voltage, resistance, inductance, phase_degrees)):
        raise ValueError("Operating conditions must be finite numeric scalars.")
    if not 0.55 <= modulation <= 0.95:
        raise ValueError("modulation_index must be in [0.55, 0.95].")
    if dc_voltage <= 0.0:
        raise ValueError("dc_voltage_per_cell_v must be positive.")
    if resistance not in ALLOWED_LOAD_RESISTANCES_OHM:
        raise ValueError(f"load_resistance_ohm must be one of {ALLOWED_LOAD_RESISTANCES_OHM}.")
    if inductance not in ALLOWED_LOAD_INDUCTANCES_H:
        raise ValueError(f"load_inductance_h must be one of {ALLOWED_LOAD_INDUCTANCES_H}.")
    if not 0.0 <= phase_degrees <= 360.0:
        raise ValueError("phase_angle_degrees must be in [0, 360].")
    return modulation, dc_voltage, resistance, inductance, phase_degrees


def run_live_simulation(
    model: nn.Module,
    normalization: NormalizationStats,
    *,
    modulation_index: float,
    dc_voltage_per_cell_v: float = DC_VOLTAGE_PER_CELL_V,
    load_resistance_ohm: float,
    load_inductance_h: float,
    phase_angle_degrees: float,
    device: torch.device | str,
    carrier_phase_offsets_degrees: tuple[float, ...] = (0.0, 60.0, 120.0),
) -> LiveSimulationResult:
    """Simulate the inverter, evaluate event THD, and infer target waveform.

    The CNN receives normalized simulated output voltage/current and raw
    operating conditions normalized with the checkpoint's training statistics.
    CNN prediction metrics are against the analytical reference target, not
    inverter-output harmonic performance.
    """
    modulation, dc_voltage, resistance, inductance, phase_degrees = _validate_operating_inputs(
        modulation_index,
        dc_voltage_per_cell_v,
        load_resistance_ohm,
        load_inductance_h,
        phase_angle_degrees,
    )
    phase_radians = math.radians(phase_degrees)
    simulation = simulate_inverter(
        modulation,
        resistance,
        inductance,
        phase_radians,
        carrier_phase_offsets_degrees=carrier_phase_offsets_degrees,
        dc_voltage_per_cell_v=dc_voltage,
    )
    normalized_phase_offsets = tuple(
        float(offset) % 360.0 for offset in carrier_phase_offsets_degrees
    )
    harmonic_result = evaluate_event_based_thd(
        modulation,
        phase_radians,
        carrier_phase_offsets_degrees=normalized_phase_offsets,
        dc_voltage_per_cell_v=dc_voltage,
    )
    target_reference = simulation.reference * CELL_COUNT * dc_voltage

    raw_conditions = np.asarray(
        [[modulation, resistance, inductance, phase_radians]],
        dtype=np.float64,
    )
    normalized_voltage, normalized_current, normalized_conditions = normalization.normalize_inputs(
        simulation.output_voltage[np.newaxis, :],
        simulation.load_current[np.newaxis, :],
        raw_conditions,
    )
    tensors = (
        torch.as_tensor(np.asarray(normalized_voltage, dtype=np.float32), device=device),
        torch.as_tensor(np.asarray(normalized_current, dtype=np.float32), device=device),
        torch.as_tensor(np.asarray(normalized_conditions, dtype=np.float32), device=device),
    )
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            prediction = model(*tensors)
    finally:
        model.train(was_training)
    if not isinstance(prediction, torch.Tensor) or prediction.shape != (1, simulation.output_voltage.size):
        output_shape = (
            tuple(prediction.shape)
            if isinstance(prediction, torch.Tensor)
            else type(prediction).__name__
        )
        raise ValueError(
            f"CNN output {output_shape} does not match expected "
            f"(1, {simulation.output_voltage.size})."
        )
    if not torch.isfinite(prediction).all():
        raise ValueError("CNN returned non-finite target-waveform predictions.")
    prediction_voltage = np.asarray(
        normalization.denormalize_target(prediction.detach().cpu().numpy())[0],
        dtype=np.float64,
    )
    if not np.isfinite(prediction_voltage).all():
        raise ValueError("Denormalized CNN waveform contains non-finite values.")
    cnn_metrics = {
        "mae": mae(target_reference, prediction_voltage),
        "rmse": rmse(target_reference, prediction_voltage),
        "r_squared": r_squared(target_reference, prediction_voltage),
    }
    voltage_thd = calculate_thd(simulation.output_voltage)
    current_thd, current_spectrum = calculate_thd(
        simulation.load_current,
        return_spectrum=True,
    )
    return LiveSimulationResult(
        modulation_index=modulation,
        dc_voltage_per_cell_v=dc_voltage,
        load_resistance_ohm=resistance,
        load_inductance_h=inductance,
        phase_angle_degrees=phase_degrees,
        carrier_phase_offsets_degrees=normalized_phase_offsets,
        simulation=simulation,
        harmonic_result=harmonic_result,
        target_reference_voltage=target_reference,
        cnn_prediction_voltage=prediction_voltage,
        cnn_metrics=cnn_metrics,
        voltage_thd_percent=float(voltage_thd),
        current_thd_percent=float(current_thd),
        current_harmonic_spectrum=current_spectrum,
    )


__all__ = [
    "ALLOWED_LOAD_INDUCTANCES_H",
    "ALLOWED_LOAD_RESISTANCES_OHM",
    "LiveSimulationResult",
    "operating_condition_support_warnings",
    "run_live_simulation",
]
