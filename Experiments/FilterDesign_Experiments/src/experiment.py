from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .audioio import read_wav
from .config import DEFAULT_GAIN_FLOOR, DEFAULT_HOP, DEFAULT_N_FFT, DEFAULT_NOISE_SCALE, EXTRACTED_DIR
from .dataset import available_conditions, run_path
from .metrics import evaluate, speech_mask
from .wiener import apply_wiener_oracle_noise, pole_time_constant_seconds, settling_time_seconds


def load_condition(run_dir: Path, condition: dict) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    audio_dir = Path(condition["path"])
    required = [audio_dir / "clean_speech.wav", audio_dir / "drone_noise.wav", audio_dir / "mic_noisy.wav"]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")
    sr_s, clean = read_wav(required[0])
    sr_n, noise = read_wav(required[1])
    sr_y, noisy = read_wav(required[2])
    if not (sr_s == sr_n == sr_y):
        raise ValueError(f"Sample-rate mismatch: clean={sr_s}, noise={sr_n}, noisy={sr_y}")
    n = min(len(clean), len(noise), len(noisy))
    clean, noise, noisy = clean[:n], noise[:n], noisy[:n]
    mix_error = float(np.max(np.abs((clean + noise) - noisy)))
    if mix_error > 2.5 / 32768.0:
        raise ValueError(f"Mixture consistency check failed, max abs error={mix_error:.3e}")
    return sr_s, clean, noise, noisy


def evaluate_pole(
    row: pd.Series,
    condition: dict,
    alpha: float,
    n_fft: int = DEFAULT_N_FFT,
    hop: int = DEFAULT_HOP,
    noise_scale: float = DEFAULT_NOISE_SCALE,
    gain_floor: float = DEFAULT_GAIN_FLOOR,
):
    rdir = run_path(row)
    sr, clean, noise, noisy = load_condition(rdir, condition)
    result = apply_wiener_oracle_noise(
        clean=clean,
        noise=noise,
        sr=sr,
        alpha=alpha,
        n_fft=n_fft,
        hop=hop,
        noise_scale=noise_scale,
        gain_floor=gain_floor,
    )
    manifest = json.loads((rdir / "manifest.json").read_text())
    active = ((manifest.get("config") or {}).get("experiment") or {}).get("active_interval_seconds")
    active_interval = tuple(map(float, active)) if isinstance(active, list) and len(active) == 2 else None
    mask = speech_mask(clean, threshold_fraction=0.015, sr=sr, active_interval_s=active_interval)
    metrics = evaluate(clean, noise, result.filtered, result.filtered_clean, result.filtered_noise, mask=mask)
    metrics.update(
        {
            "run_id": str(row["run_id"]),
            "scenario": str(row["scenario"]),
            "seed": int(row["seed"]),
            "split": str(row["split"]),
            "condition": str(condition["label"]),
            "target_snr_db": float(condition["target_snr_db"]),
            "alpha": float(alpha),
            "tau_ms": 1000.0 * pole_time_constant_seconds(alpha, hop, sr),
            "settling_95_ms": 1000.0 * settling_time_seconds(alpha, hop, sr),
            "sample_rate_hz": sr,
            "n_fft": n_fft,
            "hop": hop,
            "gain_floor": gain_floor,
            "noise_scale": noise_scale,
        }
    )
    return metrics, result, (sr, clean, noise, noisy)
