from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal

from .metrics import power_db_ratio, si_sdr, correlation, speech_mask


def log_spectral_distance(reference: np.ndarray, estimate: np.ndarray, sr: int, n_fft: int = 512, hop: int = 128, mask_db: float = -60.0) -> float:
    _, _, R = signal.stft(reference, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft-hop, nfft=n_fft, boundary="zeros", padded=True)
    _, _, E = signal.stft(estimate, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft-hop, nfft=n_fft, boundary="zeros", padded=True)
    rdb = 20.0 * np.log10(np.maximum(np.abs(R), 1e-10))
    edb = 20.0 * np.log10(np.maximum(np.abs(E), 1e-10))
    active = rdb >= (float(np.max(rdb)) + mask_db)
    if not np.any(active):
        active = np.ones_like(rdb, dtype=bool)
    return float(np.sqrt(np.mean((rdb[active] - edb[active]) ** 2)))


def evaluate_pipeline(
    clean: np.ndarray,
    noise: np.ndarray,
    output: np.ndarray,
    sr: int,
    clean_component: np.ndarray | None = None,
    noise_component: np.ndarray | None = None,
    decomposition_exact: bool = False,
    mask: np.ndarray | None = None,
) -> dict[str, float | bool]:
    if mask is None:
        mask = speech_mask(clean)
    inp_snr = power_db_ratio(clean, noise, mask)
    residual = output - clean
    reconstruction_snr = power_db_ratio(clean, residual, mask)
    out = {
        "input_snr_db": inp_snr,
        "reference_reconstruction_snr_db": reconstruction_snr,
        "reference_snr_improvement_db": reconstruction_snr - inp_snr,
        "si_sdr_db": si_sdr(clean[mask], output[mask]),
        "correlation": correlation(clean[mask], output[mask]),
        "log_spectral_distance_db": log_spectral_distance(clean, output, sr),
        "output_peak_abs": float(np.max(np.abs(output))) if len(output) else 0.0,
        "output_rms": float(np.sqrt(np.mean(output**2))) if len(output) else 0.0,
        "decomposition_exact": bool(decomposition_exact and clean_component is not None and noise_component is not None),
    }
    if out["decomposition_exact"]:
        comp_snr = power_db_ratio(clean_component, noise_component, mask)
        out.update({
            "component_output_snr_db": comp_snr,
            "component_snr_improvement_db": comp_snr - inp_snr,
            "speech_attenuation_db": power_db_ratio(clean_component, clean, mask),
            "noise_attenuation_db": power_db_ratio(noise, noise_component, mask),
        })
    else:
        out.update({
            "component_output_snr_db": float("nan"),
            "component_snr_improvement_db": float("nan"),
            "speech_attenuation_db": float("nan"),
            "noise_attenuation_db": float("nan"),
        })
    return out


def add_quality_score(df):
    """Add a transparent 0..100 reference-quality score for comparable rows.

    The score uses only quality metrics and never runtime, so a fast but poor
    denoiser cannot win on speed alone. Higher is better for SNR gain, SI-SDR and
    correlation; lower is better for log-spectral distance. Rank-percentiles are
    used to avoid mixing incompatible metric units.
    """
    out = df.copy()
    if len(out) == 0:
        out["quality_score"] = []
        return out
    needed = [
        "reference_snr_improvement_db",
        "si_sdr_db",
        "correlation",
        "log_spectral_distance_db",
    ]
    for c in needed:
        if c not in out:
            out[c] = np.nan
    valid = np.ones(len(out), dtype=bool)
    for c in needed:
        valid &= np.isfinite(pd.to_numeric(out[c], errors="coerce").to_numpy(float))
    score = np.full(len(out), np.nan, dtype=float)
    if np.any(valid):
        v = out.loc[valid, needed].apply(pd.to_numeric, errors="coerce")
        snr_rank = v["reference_snr_improvement_db"].rank(pct=True, method="average", ascending=True)
        sisdr_rank = v["si_sdr_db"].rank(pct=True, method="average", ascending=True)
        corr_rank = v["correlation"].rank(pct=True, method="average", ascending=True)
        lsd_rank = v["log_spectral_distance_db"].rank(pct=True, method="average", ascending=False)
        score[np.flatnonzero(valid)] = 100.0 * (
            0.35 * snr_rank.to_numpy()
            + 0.35 * sisdr_rank.to_numpy()
            + 0.15 * corr_rank.to_numpy()
            + 0.15 * lsd_rank.to_numpy()
        )
    out["quality_score"] = score
    return out
