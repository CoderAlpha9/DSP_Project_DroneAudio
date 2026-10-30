from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audioio import write_wav  # noqa: E402
from src.benchmark import evaluate_pipeline  # noqa: E402
from src.dataset import available_conditions, load_catalog, run_path  # noqa: E402
from src.experiment import load_condition  # noqa: E402
from src.metrics import speech_mask  # noqa: E402
from src.methods import DSPState  # noqa: E402
from src.pipeline import canonical_pipeline, parse_pipeline, pipeline_metadata, run_pipeline  # noqa: E402


def _choose_row(catalog: pd.DataFrame, run_id: str | None, scenario: str | None, split: str):
    view = catalog[catalog["split"] == split].copy()
    if scenario:
        view = view[view["scenario"].astype(str) == scenario]
    if run_id:
        view = view[view["run_id"].astype(str) == run_id]
    if view.empty:
        raise ValueError("No matching run. Use --list-runs to inspect available runs.")
    return view.iloc[0]


def main():
    ap = argparse.ArgumentParser(description="Run any DSP method or method chain on one prepared condition")
    ap.add_argument("--pipeline", default="wiener:alpha=0.95", help="Example: rpm_order:harmonics=6|wiener:alpha=0.97")
    ap.add_argument("--run-id")
    ap.add_argument("--scenario")
    ap.add_argument("--split", choices=["development", "validation", "test"], default="development")
    ap.add_argument("--snr", default="-5 dB", help='Condition label such as "-5 dB"')
    ap.add_argument("--output", default=str(ROOT / "outputs" / "pipeline_output.wav"))
    ap.add_argument("--metrics", default=str(ROOT / "outputs" / "pipeline_metrics.json"))
    ap.add_argument("--list-runs", action="store_true")
    args = ap.parse_args()

    catalog = load_catalog()
    if args.list_runs:
        print(catalog[["run_id", "scenario", "seed", "split"]].to_string(index=False))
        return

    row = _choose_row(catalog, args.run_id, args.scenario, args.split)
    rdir = run_path(row)
    conditions = available_conditions(rdir)
    condition = next((c for c in conditions if c["label"] == args.snr), None)
    if condition is None:
        raise ValueError(f"Condition {args.snr!r} unavailable. Choices: {[c['label'] for c in conditions]}")
    sr, clean, noise, noisy = load_condition(rdir, condition)
    manifest = json.loads((rdir / "manifest.json").read_text())
    active = ((manifest.get("config") or {}).get("experiment") or {}).get("active_interval_seconds")
    active_interval = tuple(map(float, active)) if isinstance(active, list) and len(active) == 2 else None
    mask = speech_mask(clean, threshold_fraction=0.015, sr=sr, active_interval_s=active_interval)

    stages = parse_pipeline(args.pipeline)
    state = DSPState(noisy, sr, clean, noise, rdir, manifest, True, {})
    final, audit = run_pipeline(state, stages)
    metrics = evaluate_pipeline(clean, noise, final.signal, sr, final.clean_ref, final.noise_ref, final.decomposition_exact, mask)
    metrics.update({
        "run_id": str(row["run_id"]),
        "scenario": str(row["scenario"]),
        "split": str(row["split"]),
        "condition": condition["label"],
        "pipeline": canonical_pipeline(stages),
        **pipeline_metadata(stages),
        "total_runtime_ms": float(sum(s["runtime_ms"] for s in audit)),
        "stage_audit": audit,
        "diagnostics": final.diagnostics,
    })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_wav(out, sr, final.signal)
    mp = Path(args.metrics)
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(metrics, indent=2, allow_nan=True))
    print(f"Pipeline: {metrics['pipeline']}")
    print(f"Reference SNR improvement: {metrics['reference_snr_improvement_db']:+.3f} dB")
    print(f"SI-SDR: {metrics['si_sdr_db']:.3f} dB")
    print(f"LSD: {metrics['log_spectral_distance_db']:.3f} dB")
    print(f"Runtime: {metrics['total_runtime_ms']:.1f} ms")
    print(f"Oracle-assisted: {metrics['uses_oracle_noise']}")
    print(f"Exact component decomposition: {metrics['decomposition_exact']}")
    print(f"Saved audio: {out}")
    print(f"Saved metrics: {mp}")


if __name__ == "__main__":
    main()
