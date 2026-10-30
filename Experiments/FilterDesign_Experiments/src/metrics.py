from __future__ import annotations

import numpy as np


def speech_mask(
    clean: np.ndarray,
    threshold_fraction: float = 0.015,
    sr: int | None = None,
    active_interval_s: tuple[float, float] | None = None,
) -> np.ndarray:
    clean = np.asarray(clean, dtype=np.float64)
    peak = float(np.max(np.abs(clean))) if clean.size else 0.0
    if peak <= 0.0:
        return np.ones_like(clean, dtype=bool)
    mask = np.abs(clean) > threshold_fraction * peak
    if active_interval_s is not None:
        if sr is None or sr <= 0:
            raise ValueError("sr must be provided when active_interval_s is used")
        lo = max(0, int(round(active_interval_s[0] * sr)))
        hi = min(len(clean), int(round(active_interval_s[1] * sr)))
        interval = np.zeros_like(mask, dtype=bool)
        interval[lo:hi] = True
        mask &= interval
    if np.count_nonzero(mask) < max(32, int(0.01 * len(clean))):
        return np.ones_like(clean, dtype=bool)
    return mask


def power_db_ratio(num: np.ndarray, den: np.ndarray, mask: np.ndarray | None = None, eps: float = 1e-15) -> float:
    a = np.asarray(num, dtype=np.float64)
    b = np.asarray(den, dtype=np.float64)
    if mask is not None:
        a = a[mask]
        b = b[mask]
    p_num = float(np.sum(a * a))
    p_den = float(np.sum(b * b))
    return 10.0 * np.log10((p_num + eps) / (p_den + eps))


def si_sdr(reference: np.ndarray, estimate: np.ndarray, eps: float = 1e-15) -> float:
    s = np.asarray(reference, dtype=np.float64)
    x = np.asarray(estimate, dtype=np.float64)
    s = s - np.mean(s)
    x = x - np.mean(x)
    scale = float(np.dot(x, s) / (np.dot(s, s) + eps))
    target = scale * s
    residual = x - target
    return power_db_ratio(target, residual, eps=eps)


def correlation(reference: np.ndarray, estimate: np.ndarray, eps: float = 1e-15) -> float:
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(estimate, dtype=np.float64)
    a = a - np.mean(a)
    b = b - np.mean(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))


def evaluate(
    clean: np.ndarray,
    noise: np.ndarray,
    filtered: np.ndarray,
    filtered_clean: np.ndarray,
    filtered_noise: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    if mask is None:
        mask = speech_mask(clean)
    input_snr = power_db_ratio(clean, noise, mask)
    output_snr = power_db_ratio(filtered_clean, filtered_noise, mask)
    speech_attenuation_db = power_db_ratio(filtered_clean, clean, mask)
    noise_attenuation_db = power_db_ratio(noise, filtered_noise, mask)
    reconstruction_snr_db = power_db_ratio(clean, filtered - clean, mask)
    return {
        "input_snr_db": input_snr,
        "output_snr_db": output_snr,
        "snr_improvement_db": output_snr - input_snr,
        "speech_attenuation_db": speech_attenuation_db,
        "noise_attenuation_db": noise_attenuation_db,
        "reconstruction_snr_db": reconstruction_snr_db,
        "si_sdr_db": si_sdr(clean[mask], filtered[mask]),
        "correlation": correlation(clean[mask], filtered[mask]),
    }
