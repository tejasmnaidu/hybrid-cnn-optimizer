"""Use a trained THD surrogate to shortlist candidates for exact evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import pickle
from pathlib import Path
import time

import numpy as np
import torch

from optimization.circular_phase_pso import (
    BASELINE_PHASES_DEGREES,
    CandidateEvaluation,
    PhaseOptimizationProblem,
    canonicalize_candidate,
)
from optimization.thd_surrogate import (
    EventTHDSurrogate,
    THD_SURROGATE_FEATURE_NAMES,
    THD_SURROGATE_TARGET_NAMES,
    encode_thd_surrogate_features,
)


@dataclass(frozen=True)
class EventTHDSurrogateArtifact:
    model: EventTHDSurrogate
    target_mean: float
    target_scale: float
    dataset_sha256: str
    checkpoint_sha256: str
    split_policy: str

    def predict(self, candidates: np.ndarray) -> np.ndarray:
        values = np.asarray(candidates, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 4 or values.shape[0] == 0:
            raise ValueError("candidates must be a non-empty four-column matrix.")
        features = encode_thd_surrogate_features(values)
        self.model.eval()
        with torch.inference_mode():
            standardized = self.model(torch.from_numpy(features)).cpu().numpy()
        predictions = standardized[:, 0] * self.target_scale + self.target_mean
        if not np.isfinite(predictions).all():
            raise RuntimeError("THD surrogate returned non-finite predictions.")
        if np.any(predictions < 0.0):
            raise RuntimeError("THD surrogate returned physically invalid negative THD.")
        return predictions.astype(np.float64, copy=False)


@dataclass(frozen=True)
class SurrogateScreeningRunResult:
    seed: int
    best: CandidateEvaluation
    convergence_history: tuple[CandidateEvaluation, ...]
    feasibility_history: tuple[int, ...]
    objective_requests: int
    unique_candidate_evaluations: int
    cache_hits: int
    screened_candidate_count: int
    dataset_sha256: str
    checkpoint_sha256: str
    runtime_seconds: float


def load_event_thd_surrogate(
    checkpoint_path: str | Path,
) -> EventTHDSurrogateArtifact:
    """Load a compatible experimental surrogate from its separate checkpoint."""
    path = Path(checkpoint_path)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        checkpoint_bytes = path.read_bytes()
    except (OSError, EOFError, RuntimeError, pickle.UnpicklingError) as exc:
        raise ValueError(f"Could not load THD surrogate checkpoint {path}: {exc}") from exc
    if not isinstance(checkpoint, dict):
        raise ValueError("THD surrogate checkpoint must contain a mapping.")
    if checkpoint.get("model_type") != "EventTHDSurrogate":
        raise ValueError("Checkpoint is not a supported event-THD surrogate.")
    if tuple(checkpoint.get("feature_names", ())) != THD_SURROGATE_FEATURE_NAMES:
        raise ValueError("THD surrogate feature schema does not match this application.")
    if tuple(checkpoint.get("target_names", ())) != THD_SURROGATE_TARGET_NAMES:
        raise ValueError("THD surrogate target schema does not match this application.")
    split_policy = checkpoint.get("split_policy")
    if not isinstance(split_policy, str) or not split_policy.startswith(
        "Modulation-group holdout"
    ):
        raise ValueError("THD surrogate lacks modulation-group holdout validation.")
    dataset_sha256 = checkpoint.get("dataset_sha256")
    if (
        not isinstance(dataset_sha256, str)
        or len(dataset_sha256) != 64
        or any(character not in "0123456789abcdef" for character in dataset_sha256)
    ):
        raise ValueError("THD surrogate checkpoint has invalid dataset provenance.")
    try:
        target_mean_values = np.asarray(checkpoint["target_mean"], dtype=np.float64)
        target_scale_values = np.asarray(checkpoint["target_scale"], dtype=np.float64)
        model_config = checkpoint["model_config"]
        model = EventTHDSurrogate(**model_config)
        model.load_state_dict(checkpoint["model_state"])
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"THD surrogate checkpoint is incompatible: {exc}") from exc
    if (
        target_mean_values.shape != (1,)
        or target_scale_values.shape != (1,)
        or not np.isfinite(target_mean_values).all()
        or not np.isfinite(target_scale_values).all()
        or target_scale_values[0] <= 0.0
    ):
        raise ValueError("THD surrogate checkpoint has invalid target normalization.")
    model.eval()
    return EventTHDSurrogateArtifact(
        model=model,
        target_mean=float(target_mean_values[0]),
        target_scale=float(target_scale_values[0]),
        dataset_sha256=dataset_sha256,
        checkpoint_sha256=hashlib.sha256(checkpoint_bytes).hexdigest(),
        split_policy=split_policy,
    )


def screen_phase_candidates(
    problem: PhaseOptimizationProblem,
    artifact: EventTHDSurrogateArtifact,
    *,
    seed: int,
    baseline_evaluation: CandidateEvaluation,
    candidate_count: int = 64,
    finalist_count: int = 8,
) -> SurrogateScreeningRunResult:
    """Rank random phase candidates cheaply; use exact THD for final selection."""
    if not 0 <= seed <= 2**31 - 1:
        raise ValueError("seed must be in [0, 2**31 - 1].")
    if candidate_count < 2 or not 1 <= finalist_count <= candidate_count:
        raise ValueError("Invalid surrogate candidate or finalist count.")
    if len(problem.operating_points) != 1:
        raise ValueError("Surrogate screening currently supports one live operating point.")
    baseline = canonicalize_candidate(BASELINE_PHASES_DEGREES[1:])
    if baseline_evaluation.canonical_phases_degrees != baseline:
        raise ValueError("baseline_evaluation must describe conventional [0, 60, 120] phases.")
    if not baseline_evaluation.feasible:
        raise ValueError("The conventional baseline must be feasible before screening.")

    started = time.perf_counter()
    point = problem.operating_points[0]
    candidate_set = {baseline}
    generator = np.random.default_rng(seed)
    maximum_draws = candidate_count * 100
    for _ in range(maximum_draws):
        if len(candidate_set) == candidate_count:
            break
        candidate_set.add(
            canonicalize_candidate(generator.uniform(0.0, 360.0, size=2))
        )
    if len(candidate_set) != candidate_count:
        raise RuntimeError("Could not generate the requested number of unique candidates.")
    candidates = sorted(candidate_set)
    candidate_inputs = np.asarray(
        [
            (
                point.modulation_index,
                math.degrees(point.phase_angle_rad),
                candidate[0],
                candidate[1],
            )
            for candidate in candidates
        ],
        dtype=np.float64,
    )
    predicted_thd = artifact.predict(candidate_inputs)
    finalist_indices = list(
        np.argsort(predicted_thd, kind="stable")[:finalist_count]
    )
    baseline_index = candidates.index(baseline)
    if baseline_index not in finalist_indices:
        finalist_indices[-1] = baseline_index

    exact_finalists = [baseline_evaluation]
    best = baseline_evaluation
    convergence_history = [best]
    feasibility_history = [int(best.feasible)]
    for candidate_index in finalist_indices:
        candidate = candidates[candidate_index]
        if candidate == baseline:
            continue
        evaluation = problem.evaluate(candidate)
        exact_finalists.append(evaluation)
        if evaluation.rank(problem.tracking_tolerance_percent) < best.rank(
            problem.tracking_tolerance_percent
        ):
            best = evaluation
        convergence_history.append(best)
        feasibility_history.append(
            sum(int(item.feasible) for item in exact_finalists)
        )

    if not best.feasible:
        raise RuntimeError("Exact THD finalist selection returned an infeasible result.")
    return SurrogateScreeningRunResult(
        seed=seed,
        best=best,
        convergence_history=tuple(convergence_history),
        feasibility_history=tuple(feasibility_history),
        objective_requests=len(exact_finalists),
        unique_candidate_evaluations=len(exact_finalists),
        cache_hits=0,
        screened_candidate_count=len(candidates),
        dataset_sha256=artifact.dataset_sha256,
        checkpoint_sha256=artifact.checkpoint_sha256,
        runtime_seconds=time.perf_counter() - started,
    )
