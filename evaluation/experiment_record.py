"""Non-overwriting JSON records for reproducible optimizer experiments."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import numpy as np


def _json_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("Experiment records cannot contain non-finite numbers.")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported experiment record value: {type(value).__name__}.")


def record_experiment(
    *,
    output_directory: str | Path,
    optimizer_name: str,
    operating_point: Mapping[str, Any],
    carrier_bounds: Mapping[str, Any],
    evaluation_budget: int,
    seed: int,
    feasibility_rules: Mapping[str, Any],
    thd_method: str,
    harmonic_range: Mapping[str, Any],
    final_independent_resimulation: Mapping[str, Any],
    baseline: Mapping[str, Any],
    optimizer_run: Mapping[str, Any] | None = None,
) -> Path:
    """Persist all conditions needed to interpret an optimizer comparison.

    This helper only writes experiment metadata/results supplied by a caller.
    New standalone experiment scripts may call it; the live dashboard path does not.
    """
    if not optimizer_name.strip():
        raise ValueError("optimizer_name must not be empty.")
    if not thd_method.strip():
        raise ValueError("thd_method must not be empty.")
    if evaluation_budget < 1:
        raise ValueError("evaluation_budget must be positive.")
    if not 0 <= seed <= 2**31 - 1:
        raise ValueError("seed must be in [0, 2**31 - 1].")
    required = {
        "operating_point": operating_point,
        "carrier_bounds": carrier_bounds,
        "feasibility_rules": feasibility_rules,
        "harmonic_range": harmonic_range,
        "final_independent_resimulation": final_independent_resimulation,
        "baseline": baseline,
    }
    for name, value in required.items():
        if not isinstance(value, Mapping) or not value:
            raise ValueError(f"{name} must be a non-empty mapping.")
    if "candidate" not in final_independent_resimulation:
        raise ValueError("Final independent re-simulation must include its candidate.")
    if "metrics" not in final_independent_resimulation:
        raise ValueError("Final independent re-simulation must include metrics.")
    if "metrics" not in baseline:
        raise ValueError("Baseline record must include metrics.")

    record = {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "optimizer_name": optimizer_name.strip(),
        "operating_point": operating_point,
        "carrier_bounds": carrier_bounds,
        "evaluation_budget": int(evaluation_budget),
        "seed": int(seed),
        "feasibility_rules": feasibility_rules,
        "thd_method": thd_method.strip(),
        "harmonic_range": harmonic_range,
        "optimizer_run": optimizer_run or {},
        "final_independent_resimulation": final_independent_resimulation,
        "baseline": baseline,
    }
    serialized = _json_value(record)
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    for _ in range(5):
        destination = directory / (
            f"optimizer_experiment_{timestamp}_{uuid4().hex[:8]}.json"
        )
        try:
            with destination.open("x", encoding="utf-8") as file:
                json.dump(serialized, file, indent=2, allow_nan=False)
                file.write("\n")
            return destination
        except FileExistsError:
            continue
    raise FileExistsError("Could not allocate a unique optimizer experiment record path.")
