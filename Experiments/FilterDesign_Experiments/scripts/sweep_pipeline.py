from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.benchmark import evaluate_pipeline  # noqa: E402
from src.dataset import available_conditions, load_catalog, run_path  # noqa: E402
from src.experiment import load_condition  # noqa: E402
from src.metrics import speech_mask  # noqa: E402
from src.methods import DSPState  # noqa: E402
from src.pipeline import canonical_pipeline, parse_pipeline, pipeline_metadata, run_pipeline  # noqa: E402


def parse_value(s: str):
    lo = s.lower()
    if lo in {"true", "false"}:
        return lo == "true"
    try:
        if any(c in s for c in ".eE"):
            return float(s)
        return int(s)
    except ValueError:
        return s


def parse_grid(tokens: list[str]) -> dict[str, list[object]]:
    grid = {}
    for tok in tokens:
        if "=" not in tok:
            raise ValueError(f"Grid token must be name=v1,v2,..., got {tok!r}")
        name, values = tok.split("=", 1)
        vals = [parse_value(v.strip()) for v in values.split(",") if v.strip()]
        if not vals:
            raise ValueError(f"Grid {name!r} has no values")
        grid[name.strip()] = vals
    return grid


def main():
    ap = argparse.ArgumentParser(description="Generic parameter sweep for any single or combined DSP pipeline")
    ap.add_argument("--template", required=True, help='Example: "rpm_order:harmonics={h}|wiener:alpha={a}"')
    ap.add_argument("--grid", nargs="+", required=True, help='Example: a=0.9,0.95,0.98 h=4,8')
    ap.add_argument("--split", choices=["development", "validation", "test"], default="development")
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--snrs", nargs="*", default=None)
    ap.add_argument("--max-runs", type=int, default=0)
    ap.add_argument("--output", default=str(ROOT / "outputs" / "pipeline_sweep.csv"))
    args = ap.parse_args()

    grid = parse_grid(args.grid)
    keys = list(grid)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]
    catalog = load_catalog()
    use = catalog[(catalog["split"] == args.split) & (catalog["speech_valid_for_metrics"] == True)].copy()  # noqa: E712
    if args.scenarios:
        use = use[use["scenario"].astype(str).isin(args.scenarios)]
    if args.max_runs > 0:
        use = use.head(args.max_runs)
    if use.empty:
        raise RuntimeError("No runs match the requested split/scenario filters")

    rows = []
    for _, row in use.iterrows():
        rdir = run_path(row)
        manifest = json.loads((rdir / "manifest.json").read_text())
        active = ((manifest.get("config") or {}).get("experiment") or {}).get("active_interval_seconds")
        active_interval = tuple(map(float, active)) if isinstance(active, list) and len(active) == 2 else None
        conditions = available_conditions(rdir)
        if args.snrs:
            conditions = [c for c in conditions if c["label"] in set(args.snrs)]
        for condition in conditions:
            sr, clean, noise, noisy = load_condition(rdir, condition)
            mask = speech_mask(clean, threshold_fraction=0.015, sr=sr, active_interval_s=active_interval)
            for combo in combos:
                spec = args.template.format(**combo)
                stages = parse_pipeline(spec)
                state = DSPState(noisy, sr, clean, noise, rdir, manifest, True, {})
                t0 = time.perf_counter()
                final, audit = run_pipeline(state, stages)
                met = evaluate_pipeline(clean, noise, final.signal, sr, final.clean_ref, final.noise_ref, final.decomposition_exact, mask)
                rows.append({
                    **combo,
                    "pipeline": canonical_pipeline(stages),
                    "run_id": str(row["run_id"]),
                    "scenario": str(row["scenario"]),
                    "split": str(row["split"]),
                    "condition": condition["label"],
                    "target_snr_db": float(condition["target_snr_db"]),
                    "wall_runtime_ms": 1000.0 * (time.perf_counter() - t0),
                    "stage_runtime_ms": sum(a["runtime_ms"] for a in audit),
                    **pipeline_metadata(stages),
                    **met,
                })
                print(f"{row['scenario']} {condition['label']} {combo}: ok")
    df = pd.DataFrame(rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    group_cols = keys + ["pipeline"]
    summary = df.groupby(group_cols, dropna=False).agg(
        median_ref_snr_gain_db=("reference_snr_improvement_db", "median"),
        median_si_sdr_db=("si_sdr_db", "median"),
        median_lsd_db=("log_spectral_distance_db", "median"),
        median_runtime_ms=("stage_runtime_ms", "median"),
        cases=("run_id", "count"),
    ).reset_index()
    summary.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
