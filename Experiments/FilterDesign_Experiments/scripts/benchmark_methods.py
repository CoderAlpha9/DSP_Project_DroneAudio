from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.benchmark import evaluate_pipeline, add_quality_score  # noqa: E402
from src.dataset import available_conditions, load_catalog, run_path  # noqa: E402
from src.experiment import load_condition  # noqa: E402
from src.metrics import speech_mask  # noqa: E402
from src.methods import DSPState  # noqa: E402
from src.pipeline import canonical_pipeline, parse_pipeline, pipeline_metadata, run_pipeline  # noqa: E402


def load_presets(names: list[str] | None) -> dict[str, str]:
    presets = json.loads((ROOT / "configs" / "methods.json").read_text())
    if not names:
        return presets
    missing = [n for n in names if n not in presets]
    if missing:
        raise ValueError(f"Unknown presets: {missing}. Available: {list(presets)}")
    return {n: presets[n] for n in names}


def main():
    ap = argparse.ArgumentParser(description="Benchmark independent DSP methods and arbitrary method chains")
    ap.add_argument("--split", choices=["development", "validation", "test"], default="development")
    ap.add_argument("--presets", nargs="*", help="Names from configs/methods.json. Omit to run all 11 methods.")
    ap.add_argument("--max-runs", type=int, default=0, help="0 means all runs in the selected split")
    ap.add_argument("--scenarios", nargs="*", default=None, help="Optional scenario filter, e.g. S3 S6")
    ap.add_argument("--pipeline", action="append", default=[], help="Add custom preset as NAME=PIPELINE. Can be repeated.")
    ap.add_argument("--snrs", nargs="*", default=None, help='Example: --snrs "-5 dB" "0 dB"')
    ap.add_argument("--output", default=str(ROOT / "outputs" / "method_benchmark.csv"))
    args = ap.parse_args()

    presets = load_presets(args.presets)
    for item in args.pipeline:
        if "=" not in item:
            raise ValueError("--pipeline must be NAME=PIPELINE")
        name, spec = item.split("=", 1)
        presets[name.strip()] = spec.strip()
    catalog = load_catalog()
    use = catalog[(catalog["split"] == args.split) & (catalog["speech_valid_for_metrics"] == True)].copy()  # noqa: E712
    if args.scenarios:
        use = use[use["scenario"].astype(str).isin(args.scenarios)]
    if args.max_runs > 0:
        use = use.head(args.max_runs)
    if use.empty:
        raise RuntimeError("No runs available for the selected split")

    rows = []
    total = len(use) * len(presets)
    done = 0
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
            for preset_name, spec in presets.items():
                done += 1
                stages = parse_pipeline(spec)
                state = DSPState(noisy, sr, clean, noise, rdir, manifest, True, {})
                t0 = time.perf_counter()
                try:
                    final, audit = run_pipeline(state, stages)
                    metrics = evaluate_pipeline(clean, noise, final.signal, sr, final.clean_ref, final.noise_ref, final.decomposition_exact, mask)
                    meta = pipeline_metadata(stages)
                    status, error = "ok", ""
                except Exception as exc:
                    metrics = {}
                    meta = pipeline_metadata(stages)
                    audit = []
                    status, error = "error", f"{type(exc).__name__}: {exc}"
                elapsed = 1000.0 * (time.perf_counter() - t0)
                rows.append({
                    "preset": preset_name,
                    "pipeline": canonical_pipeline(stages),
                    "run_id": str(row["run_id"]),
                    "scenario": str(row["scenario"]),
                    "seed": int(row["seed"]),
                    "split": str(row["split"]),
                    "condition": condition["label"],
                    "target_snr_db": float(condition["target_snr_db"]),
                    "status": status,
                    "error": error,
                    "wall_runtime_ms": elapsed,
                    "stage_runtime_ms": sum(a.get("runtime_ms", 0.0) for a in audit),
                    **meta,
                    **metrics,
                })
                print(f"[{done}] {row['scenario']} {condition['label']} {preset_name}: {status}")

    df = pd.DataFrame(rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    okdf = df[df["status"] == "ok"].copy()
    summary = (
        okdf.groupby(["preset", "pipeline", "uses_oracle_noise"], dropna=False)
        .agg(
            reference_snr_improvement_db=("reference_snr_improvement_db", "median"),
            si_sdr_db=("si_sdr_db", "median"),
            correlation=("correlation", "median"),
            log_spectral_distance_db=("log_spectral_distance_db", "median"),
            median_runtime_ms=("wall_runtime_ms", "median"),
            cases=("run_id", "count"),
        )
        .reset_index()
    )
    summary = add_quality_score(summary).sort_values("quality_score", ascending=False)
    summary_path = out.with_name(out.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved {out}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
