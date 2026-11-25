"""Reproducible sampled-versus-event THD sweep for the ideal inverter."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from evaluation.event_based_thd import evaluate_event_based_thd
from evaluation.thd_analysis import calculate_thd
from multilevel.inverter_simulator import (
    FUNDAMENTAL_FREQUENCY_HZ,
    SAMPLING_FREQUENCY_HZ,
    simulate_inverter,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "results" / "thd_method_comparison.csv"
MODULATIONS = np.arange(0.55, 0.951, 0.05, dtype=np.float64)
R_L_CASES = (
    (10.0, 0.01),
    (20.0, 0.02),
    (30.0, 0.03),
    (40.0, 0.05),
)


def run_sweep(output_path: str | Path = OUTPUT_PATH) -> Path:
    """Write modulation/R/L THD values without changing existing evaluators."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing sweep output: {destination}")

    fieldnames = (
        "modulation_index",
        "load_resistance_ohm",
        "load_inductance_h",
        "carrier_phase_offsets_degrees",
        "event_parseval_total_thd_percent",
        "event_harmonic_2_to_100_thd_percent",
        "sampled_voltage_thd_2_to_100_percent",
        "sampled_current_thd_2_to_100_percent",
        "methods_difference_percentage_points",
    )
    with destination.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for modulation in MODULATIONS:
            modulation = float(modulation)
            for resistance, inductance in R_L_CASES:
                simulation = simulate_inverter(
                    modulation,
                    resistance,
                    inductance,
                    0.0,
                    carrier_phase_offsets_degrees=(0.0, 60.0, 120.0),
                )
                event = evaluate_event_based_thd(
                    modulation,
                    0.0,
                    carrier_phase_offsets_degrees=(0.0, 60.0, 120.0),
                    max_harmonic=100,
                    sampled_frequency_hz=SAMPLING_FREQUENCY_HZ,
                )
                voltage_thd = calculate_thd(
                    simulation.output_voltage,
                    sampling_frequency_hz=SAMPLING_FREQUENCY_HZ,
                    fundamental_frequency_hz=FUNDAMENTAL_FREQUENCY_HZ,
                    max_harmonic=100,
                )
                current_thd = calculate_thd(
                    simulation.load_current,
                    sampling_frequency_hz=SAMPLING_FREQUENCY_HZ,
                    fundamental_frequency_hz=FUNDAMENTAL_FREQUENCY_HZ,
                    max_harmonic=100,
                )
                writer.writerow(
                    {
                        "modulation_index": f"{modulation:.10g}",
                        "load_resistance_ohm": f"{resistance:.10g}",
                        "load_inductance_h": f"{inductance:.10g}",
                        "carrier_phase_offsets_degrees": "[0,60,120]",
                        "event_parseval_total_thd_percent": (
                            f"{event.total_thd_percent:.12g}"
                        ),
                        "event_harmonic_2_to_100_thd_percent": (
                            f"{event.thd_percent:.12g}"
                        ),
                        "sampled_voltage_thd_2_to_100_percent": (
                            f"{voltage_thd:.12g}"
                        ),
                        "sampled_current_thd_2_to_100_percent": (
                            f"{current_thd:.12g}"
                        ),
                        "methods_difference_percentage_points": (
                            f"{voltage_thd - event.thd_percent:.12g}"
                        ),
                    }
                )
    return destination


if __name__ == "__main__":
    print(run_sweep())
