from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import os
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    DEFAULT_GAIN_FLOOR,
    DEFAULT_HOP,
    DEFAULT_MAX_SPEECH_ATTENUATION_DB,
    DEFAULT_N_FFT,
    DEFAULT_NOISE_SCALE,
    DEFAULT_POLES,
    OUTPUT_DIR,
)
from src.dataset import available_conditions, load_catalog, run_path  # noqa: E402
from src.experiment import evaluate_pole  # noqa: E402


def _task(args):
    row_dict, condition, poles, n_fft, hop, noise_scale, gain_floor = args
    row = pd.Series(row_dict)
    out = []
    for alpha in poles:
        metrics, _, _ = evaluate_pole(
            row,
            condition,
            alpha,
            n_fft=n_fft,
            hop=hop,
            noise_scale=noise_scale,
            gain_floor=gain_floor,
        )
        out.append(metrics)
    return out


def _conditions_for_row(row):
    return available_conditions(run_path(row))


def run_phase(phase: str, workers: int, poles: list[float], n_fft: int, hop: int, noise_scale: float, gain_floor: float):
    catalog = load_catalog()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if phase == "tune":
        use = catalog[catalog["split"].isin(["development", "validation"])].copy()
        phase_poles = poles
        outfile = OUTPUT_DIR / "tuning_metrics.csv"
    elif phase == "test":
        selected_path = OUTPUT_DIR / "selected_pole.json"
        if not selected_path.exists():
            raise FileNotFoundError("Run the tune phase first. outputs/selected_pole.json is missing.")
        selected = json.loads(selected_path.read_text())
        use = catalog[catalog["split"] == "test"].copy()
        phase_poles = [float(selected["alpha"])]
        outfile = OUTPUT_DIR / "test_metrics.csv"
    else:
        raise ValueError("phase must be tune or test")

    tasks = []
    for _, row in use.iterrows():
        for cond in _conditions_for_row(row):
            tasks.append((row.to_dict(), cond, phase_poles, n_fft, hop, noise_scale, gain_floor))

    rows = []
    workers = max(1, workers)
    if workers == 1:
        for i, task in enumerate(tasks, 1):
            rows.extend(_task(task))
            print(f"[{i}/{len(tasks)}] complete")
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_task, t) for t in tasks]
            for i, f in enumerate(as_completed(futures), 1):
                rows.extend(f.result())
                print(f"[{i}/{len(tasks)}] complete")

    df = pd.DataFrame(rows).sort_values(["split", "scenario", "run_id", "target_snr_db", "alpha"])
    df.to_csv(outfile, index=False)
    print(f"Saved {outfile}")
    return df


def select_pole(df: pd.DataFrame, max_speech_attenuation_db: float):
    val = df[df["split"] == "validation"].copy()
    if val.empty:
        raise RuntimeError("No validation rows were produced.")
    agg = (
        val.groupby("alpha")
        .agg(
            median_snr_improvement_db=("snr_improvement_db", "median"),
            median_speech_attenuation_db=("speech_attenuation_db", "median"),
            median_si_sdr_db=("si_sdr_db", "median"),
            worst_speech_attenuation_db=("speech_attenuation_db", "min"),
        )
        .reset_index()
    )
    feasible = agg[agg["median_speech_attenuation_db"] >= -abs(max_speech_attenuation_db)].copy()
    if feasible.empty:
        agg["fallback_score"] = agg["median_snr_improvement_db"] + 0.5 * agg["median_speech_attenuation_db"]
        best = agg.sort_values(["fallback_score", "median_si_sdr_db"], ascending=False).iloc[0]
        rule = "fallback: max median SNR gain + 0.5 * median speech attenuation"
    else:
        best = feasible.sort_values(["median_snr_improvement_db", "median_si_sdr_db"], ascending=False).iloc[0]
        rule = f"max median validation SNR gain with median speech attenuation >= -{abs(max_speech_attenuation_db):g} dB"
    payload = {k: float(best[k]) for k in best.index if k != "alpha" and pd.notna(best[k])}
    payload["alpha"] = float(best["alpha"])
    payload["selection_rule"] = rule
    path = OUTPUT_DIR / "selected_pole.json"
    path.write_text(json.dumps(payload, indent=2))
    agg.to_csv(OUTPUT_DIR / "validation_summary_by_pole.csv", index=False)
    print(f"Selected alpha={payload['alpha']:.6f}")
    print(rule)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Leakage-aware Wiener pole sweep")
    parser.add_argument("phase", choices=["tune", "test"])
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--n-fft", type=int, default=DEFAULT_N_FFT)
    parser.add_argument("--hop", type=int, default=DEFAULT_HOP)
    parser.add_argument("--noise-scale", type=float, default=DEFAULT_NOISE_SCALE)
    parser.add_argument("--gain-floor", type=float, default=DEFAULT_GAIN_FLOOR)
    parser.add_argument("--max-speech-attenuation-db", type=float, default=DEFAULT_MAX_SPEECH_ATTENUATION_DB)
    parser.add_argument("--poles", type=float, nargs="*", default=DEFAULT_POLES)
    args = parser.parse_args()

    if any(a < 0.0 or a >= 1.0 for a in args.poles):
        parser.error("Every pole must satisfy 0 <= alpha < 1")
    if args.hop <= 0 or args.hop > args.n_fft:
        parser.error("Require 0 < hop <= n_fft")

    df = run_phase(args.phase, args.workers, args.poles, args.n_fft, args.hop, args.noise_scale, args.gain_floor)
    if args.phase == "tune":
        select_pole(df, args.max_speech_attenuation_db)
    else:
        summary = df.agg(
            {
                "input_snr_db": "median",
                "output_snr_db": "median",
                "snr_improvement_db": "median",
                "speech_attenuation_db": "median",
                "noise_attenuation_db": "median",
                "si_sdr_db": "median",
                "correlation": "median",
            }
        ).to_dict()
        (OUTPUT_DIR / "test_summary.json").write_text(json.dumps({k: float(v) for k, v in summary.items()}, indent=2))
        print("Final test summary saved to outputs/test_summary.json")


if __name__ == "__main__":
    main()
