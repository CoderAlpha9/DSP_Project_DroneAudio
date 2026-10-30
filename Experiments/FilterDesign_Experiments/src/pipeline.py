from __future__ import annotations

from dataclasses import dataclass
import json
import re
import time
from typing import Any

import numpy as np

from .methods import DSPState, METHODS, METHOD_INFO


@dataclass(frozen=True)
class StageSpec:
    name: str
    params: dict[str, Any]


def _parse_scalar(text: str) -> Any:
    s = text.strip()
    low = s.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"none", "null"}:
        return None
    try:
        if re.fullmatch(r"[-+]?\d+", s):
            return int(s)
        return float(s)
    except ValueError:
        return s


def parse_pipeline(spec: str | list[dict[str, Any]] | list[StageSpec]) -> list[StageSpec]:
    if isinstance(spec, list):
        out = []
        for item in spec:
            if isinstance(item, StageSpec):
                out.append(item)
            elif isinstance(item, dict):
                out.append(StageSpec(str(item["name"]), dict(item.get("params") or {})))
            else:
                raise TypeError("Pipeline list items must be StageSpec or dict")
        return out
    text = str(spec).strip()
    if not text:
        return [StageSpec("identity", {})]
    # JSON is supported for maximum reproducibility.
    if text.startswith("["):
        return parse_pipeline(json.loads(text))
    stages: list[StageSpec] = []
    for raw in text.split("|"):
        raw = raw.strip()
        if not raw:
            continue
        if ":" in raw:
            name, param_text = raw.split(":", 1)
        else:
            name, param_text = raw, ""
        name = name.strip()
        if name not in METHODS:
            raise ValueError(f"Unknown method {name!r}. Available: {', '.join(sorted(METHODS))}")
        params: dict[str, Any] = {}
        if param_text.strip():
            for token in param_text.split(","):
                if not token.strip():
                    continue
                if "=" not in token:
                    raise ValueError(f"Expected key=value in {token!r}")
                k, v = token.split("=", 1)
                params[k.strip()] = _parse_scalar(v)
        stages.append(StageSpec(name, params))
    if not stages:
        stages = [StageSpec("identity", {})]
    return stages


def canonical_pipeline(stages: list[StageSpec]) -> str:
    parts = []
    for s in stages:
        if s.params:
            params = ",".join(f"{k}={s.params[k]}" for k in sorted(s.params))
            parts.append(f"{s.name}:{params}")
        else:
            parts.append(s.name)
    return "|".join(parts)


def pipeline_metadata(stages: list[StageSpec]) -> dict[str, Any]:
    infos = [METHOD_INFO[s.name] for s in stages]
    return {
        "uses_oracle_noise": any(i.requires_oracle_noise for i in infos),
        "requires_rpm": any(i.requires_rpm for i in infos),
        "requires_imu": any(i.requires_imu for i in infos),
        "families": sorted({i.family for i in infos}),
        "labels": [i.label for i in infos],
    }


def run_pipeline(state: DSPState, stages: list[StageSpec]) -> tuple[DSPState, list[dict[str, Any]]]:
    current = state
    audit: list[dict[str, Any]] = []
    for index, stage in enumerate(stages, 1):
        info = METHOD_INFO[stage.name]
        if info.requires_oracle_noise and current.noise_ref is None:
            raise ValueError(f"Stage {stage.name} requires a noise reference, but it is unavailable after an earlier nonlinear stage")
        t0 = time.perf_counter()
        before = current.signal
        current = METHODS[stage.name](current, dict(stage.params))
        elapsed_ms = 1000.0 * (time.perf_counter() - t0)
        if len(current.signal) != len(before):
            raise RuntimeError(f"Stage {stage.name} changed signal length")
        if not np.all(np.isfinite(current.signal)):
            raise RuntimeError(f"Stage {stage.name} produced NaN/Inf")
        peak = float(np.max(np.abs(current.signal))) if len(current.signal) else 0.0
        audit.append({
            "stage": index,
            "name": stage.name,
            "label": info.label,
            "runtime_ms": elapsed_ms,
            "peak_abs": peak,
            "decomposition_exact": bool(current.decomposition_exact),
        })
    return current, audit
