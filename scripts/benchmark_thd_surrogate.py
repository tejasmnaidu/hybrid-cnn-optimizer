"""Compare surrogate screening with exact event-THD evaluations."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from evaluation.event_based_thd import evaluate_event_based_thd
from optimization.circular_phase_pso import TRACKING_TOLERANCE_PERCENT
from optimization.thd_surrogate import (
    EventTHDSurrogate,
    encode_thd_surrogate_features,
    spearman_rank_correlation,
)


def _read_model(checkpoint_path: Path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, dict) or checkpoint.get(
        "model_type"
    ) != "EventTHDSurrogate":
        raise ValueError("Checkpoint is not an event-based THD surrogate.")
    model = EventTHDSurrogate(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, np.asarray(checkpoint["target_mean"]), np.asarray(
        checkpoint["target_scale"]
    )


def _make_candidate_pool(seed: int, count: int) -> list[tuple[float, float]]:
    rng = np.random.default_rng(seed)
    candidates = {(60.0, 120.0)}
    while len(candidates) < count:
        pair = tuple(sorted(rng.uniform(0.0, 360.0, size=2).tolist()))
        candidates.add(pair)
    return sorted(candidates)


def benchmark_surrogate(
    *,
    checkpoint_path: Path,
    seeds: tuple[int, ...] = (20261125, 20261126, 20261127),
    modulation_indices: tuple[float, ...] = (
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
    ),
    reference_phases_degrees: tuple[float, ...] = (
        0.0,
        45.0,
        90.0,
        135.0,
        180.0,
        225.0,
        270.0,
        315.0,
    ),
    candidate_count: int = 64,
    finalist_count: int = 8,
) -> dict[str, object]:
    """Measure in-range modulation ranking and exact-finalist regret."""
    if not seeds or not modulation_indices or not reference_phases_degrees:
        raise ValueError(
            "At least one seed, modulation index, and reference phase are required."
        )
    if candidate_count < 2 or not 1 <= finalist_count <= candidate_count:
        raise ValueError("Invalid candidate or finalist count.")
    if any(not 0.55 <= modulation <= 0.95 for modulation in modulation_indices):
        raise ValueError("Benchmark modulation indices must be in [0.55, 0.95].")
    if any(not 0.0 <= phase <= 360.0 for phase in reference_phases_degrees):
        raise ValueError("Reference phases must be in [0, 360] degrees.")
    model, target_mean, target_scale = _read_model(checkpoint_path)
    rows: list[dict[str, object]] = []
    for modulation in modulation_indices:
        for reference_phase in reference_phases_degrees:
            for seed in seeds:
                candidates = _make_candidate_pool(seed, candidate_count)
                candidate_matrix = np.asarray(
                    [
                        (modulation, reference_phase, phase_2, phase_3)
                        for phase_2, phase_3 in candidates
                    ],
                    dtype=np.float64,
                )
                features = encode_thd_surrogate_features(candidate_matrix)
                started = time.perf_counter()
                with torch.inference_mode():
                    standardized_predictions = model(
                        torch.from_numpy(features)
                    ).numpy()
                model_runtime = time.perf_counter() - started
                predicted_thd = (
                    standardized_predictions[:, 0] * target_scale[0]
                    + target_mean[0]
                )

                exact_values = np.empty((candidate_count, 2), dtype=np.float64)
                exact_started = time.perf_counter()
                for index, (phase_2, phase_3) in enumerate(candidates):
                    result = evaluate_event_based_thd(
                        modulation,
                        np.deg2rad(reference_phase),
                        carrier_phase_offsets_degrees=(0.0, phase_2, phase_3),
                    )
                    exact_values[index] = (
                        result.total_thd_percent,
                        result.fundamental_phasor_error_percent,
                    )
                full_exact_runtime = time.perf_counter() - exact_started

                baseline_index = candidates.index((60.0, 120.0))
                predicted_order = np.argsort(predicted_thd, kind="stable")
                shortlist_indices = list(predicted_order[:finalist_count])
                if baseline_index not in shortlist_indices:
                    shortlist_indices[-1] = baseline_index
                finalist_started = time.perf_counter()
                finalist_values: dict[int, tuple[float, float]] = {}
                for index in shortlist_indices:
                    phase_2, phase_3 = candidates[index]
                    exact_result = evaluate_event_based_thd(
                        modulation,
                        np.deg2rad(reference_phase),
                        carrier_phase_offsets_degrees=(0.0, phase_2, phase_3),
                    )
                    finalist_values[index] = (
                        exact_result.total_thd_percent,
                        exact_result.fundamental_phasor_error_percent,
                    )
                feasible_finalists = [
                    index
                    for index, (_, tracking_error) in finalist_values.items()
                    if tracking_error <= TRACKING_TOLERANCE_PERCENT
                ]
                if not feasible_finalists:
                    raise RuntimeError(
                        "No surrogate finalist satisfied exact tracking feasibility."
                    )
                selected_index = min(
                    feasible_finalists,
                    key=lambda index: finalist_values[index][0],
                )
                shortlist_exact_runtime = time.perf_counter() - finalist_started
                feasible_full_indices = np.flatnonzero(
                    exact_values[:, 1] <= TRACKING_TOLERANCE_PERCENT
                )
                if feasible_full_indices.size == 0:
                    raise RuntimeError(
                        "The exact full candidate pool has no feasible point."
                    )
                exact_best_index = int(
                    feasible_full_indices[
                        np.argmin(exact_values[feasible_full_indices, 0])
                    ]
                )
                rows.append(
                    {
                        "modulation_index": modulation,
                        "reference_phase_degrees": reference_phase,
                        "seed": seed,
                        "spearman_predicted_vs_exact": spearman_rank_correlation(
                            predicted_thd,
                            exact_values[:, 0],
                        ),
                        "exact_candidate_evaluations_full_search": candidate_count,
                        "exact_candidate_evaluations_shortlist": len(shortlist_indices),
                        "exact_feasible_candidates_full_search": int(
                            feasible_full_indices.size
                        ),
                        "baseline_event_thd_percent": float(
                            exact_values[baseline_index, 0]
                        ),
                        "full_pool_best_event_thd_percent": float(
                            exact_values[exact_best_index, 0]
                        ),
                        "surrogate_shortlist_selected_phases_degrees": list(
                            candidates[selected_index]
                        ),
                        "surrogate_shortlist_selected_event_thd_percent": float(
                            finalist_values[selected_index][0]
                        ),
                        "shortlist_regret_vs_full_pool_pp": float(
                            finalist_values[selected_index][0]
                            - exact_values[exact_best_index, 0]
                        ),
                        "shortlist_change_vs_baseline_pp": float(
                            finalist_values[selected_index][0]
                            - exact_values[baseline_index, 0]
                        ),
                        "surrogate_inference_seconds": model_runtime,
                        "full_exact_search_seconds": full_exact_runtime,
                        "shortlist_exact_seconds": shortlist_exact_runtime,
                    }
                )
    return {
        "operating_point": {
            "modulation_indices": list(modulation_indices),
            "reference_phase_degrees": list(reference_phases_degrees),
            "dc_voltage_per_cell_v": 100.0,
            "load_resistance_ohm": 30.0,
            "load_inductance_h": 0.03,
        },
        "objective": "exact event-based Parseval total voltage THD",
        "candidate_phase_domain": "cell 1 fixed at 0 degrees; cell 2 and 3 sorted in [0,360)",
        "feasibility": (
            "fundamental phasor tracking error <= "
            f"{TRACKING_TOLERANCE_PERCENT}% checked using exact event evaluator"
        ),
        "candidate_count": candidate_count,
        "finalist_count_including_reserved_baseline": finalist_count,
        "seeds": list(seeds),
        "results": rows,
        "mean_shortlist_regret_pp": float(
            np.mean([row["shortlist_regret_vs_full_pool_pp"] for row in rows])
        ),
        "maximum_shortlist_regret_pp": float(
            np.max([row["shortlist_regret_vs_full_pool_pp"] for row in rows])
        ),
        "mean_shortlist_change_vs_baseline_pp": float(
            np.mean([row["shortlist_change_vs_baseline_pp"] for row in rows])
        ),
        "note": (
            "Offline benchmark only. Every candidate in each pool was evaluated "
            "exactly to measure ranking/regret; the reported shortlist represents "
            "the exact evaluations a deployment would perform after inference. "
            "This is not evidence of performance outside this operating point."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "experiments/thd_surrogate_v1/model_run_modulation_holdout/"
            "thd_surrogate.pt"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "experiments/thd_surrogate_v1/model_run_modulation_holdout/"
            "ranking_benchmark_operating_grid.json"
        ),
    )
    parser.add_argument("--candidate-count", type=int, default=64)
    parser.add_argument("--finalist-count", type=int, default=8)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark output: {arguments.output}")
    result = benchmark_surrogate(
        checkpoint_path=arguments.checkpoint,
        candidate_count=arguments.candidate_count,
        finalist_count=arguments.finalist_count,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
