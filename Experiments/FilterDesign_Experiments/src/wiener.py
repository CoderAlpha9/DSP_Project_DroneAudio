from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy import signal


@dataclass
class WienerResult:
    filtered: np.ndarray
    filtered_clean: np.ndarray
    filtered_noise: np.ndarray
    gain: np.ndarray
    noise_psd_instant: np.ndarray
    noise_psd_smooth: np.ndarray
    freqs_hz: np.ndarray
    times_s: np.ndarray
    stft_mixture: np.ndarray
    stft_clean: np.ndarray
    stft_noise: np.ndarray


def smooth_one_pole(x: np.ndarray, alpha: float) -> np.ndarray:
    if not (0.0 <= alpha < 1.0):
        raise ValueError("alpha must satisfy 0 <= alpha < 1")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("x must have shape [frequency, frame]")
    y = np.empty_like(x)
    if x.shape[1] == 0:
        return y
    y[:, 0] = x[:, 0]
    one_minus = 1.0 - alpha
    for m in range(1, x.shape[1]):
        y[:, m] = alpha * y[:, m - 1] + one_minus * x[:, m]
    return y


def pole_time_constant_seconds(alpha: float, hop: int, sr: int) -> float:
    if alpha <= 0.0:
        return 0.0
    return -(hop / sr) / math.log(alpha)


def settling_time_seconds(alpha: float, hop: int, sr: int, fraction_remaining: float = 0.05) -> float:
    if alpha <= 0.0:
        return 0.0
    return math.log(fraction_remaining) / math.log(alpha) * (hop / sr)


def smoother_frequency_response(alpha: float, points: int = 512) -> tuple[np.ndarray, np.ndarray]:
    w = np.linspace(0.0, np.pi, points)
    z_inv = np.exp(-1j * w)
    h = (1.0 - alpha) / (1.0 - alpha * z_inv)
    return w / np.pi, np.abs(h)


def _stft(x: np.ndarray, sr: int, n_fft: int, hop: int):
    noverlap = n_fft - hop
    if hop <= 0 or noverlap < 0:
        raise ValueError("Require 0 < hop <= n_fft")
    return signal.stft(
        x,
        fs=sr,
        window="hann",
        nperseg=n_fft,
        noverlap=noverlap,
        nfft=n_fft,
        boundary="zeros",
        padded=True,
    )


def _istft(z: np.ndarray, sr: int, n_fft: int, hop: int, length: int) -> np.ndarray:
    _, x = signal.istft(
        z,
        fs=sr,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop,
        nfft=n_fft,
        input_onesided=True,
        boundary=True,
    )
    if len(x) < length:
        x = np.pad(x, (0, length - len(x)))
    return np.asarray(x[:length], dtype=np.float64)


def apply_wiener_oracle_noise(
    clean: np.ndarray,
    noise: np.ndarray,
    sr: int,
    alpha: float,
    n_fft: int = 512,
    hop: int = 128,
    noise_scale: float = 1.0,
    gain_floor: float = 0.02,
    eps: float = 1e-12,
) -> WienerResult:
    clean = np.asarray(clean, dtype=np.float64)
    noise = np.asarray(noise, dtype=np.float64)
    if clean.shape != noise.shape:
        raise ValueError("clean and noise must have identical shape")
    if clean.ndim != 1:
        raise ValueError("Only mono audio is supported")
    if not (0.0 <= gain_floor <= 1.0):
        raise ValueError("gain_floor must be within [0, 1]")
    if noise_scale <= 0.0:
        raise ValueError("noise_scale must be positive")

    mixture = clean + noise
    f, t, Y = _stft(mixture, sr, n_fft, hop)
    _, _, S = _stft(clean, sr, n_fft, hop)
    _, _, N = _stft(noise, sr, n_fft, hop)

    p_noise_inst = np.abs(N) ** 2
    p_noise = smooth_one_pole(p_noise_inst, alpha)
    p_y = np.abs(Y) ** 2
    p_speech = np.maximum(p_y - noise_scale * p_noise, 0.0)
    gain = p_speech / (p_speech + noise_scale * p_noise + eps)
    gain = np.clip(gain, gain_floor, 1.0)

    Z = gain * Y
    Zs = gain * S
    Zn = gain * N

    filtered = _istft(Z, sr, n_fft, hop, len(clean))
    filtered_clean = _istft(Zs, sr, n_fft, hop, len(clean))
    filtered_noise = _istft(Zn, sr, n_fft, hop, len(clean))

    linearity_error = np.max(np.abs(filtered - (filtered_clean + filtered_noise)))
    if not np.isfinite(linearity_error) or linearity_error > 1e-7:
        raise RuntimeError(f"Linearity reconstruction check failed: {linearity_error}")

    return WienerResult(
        filtered=filtered,
        filtered_clean=filtered_clean,
        filtered_noise=filtered_noise,
        gain=gain,
        noise_psd_instant=p_noise_inst,
        noise_psd_smooth=p_noise,
        freqs_hz=f,
        times_s=t,
        stft_mixture=Y,
        stft_clean=S,
        stft_noise=N,
    )
