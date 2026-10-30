from __future__ import annotations

import numpy as np
from scipy import signal


def stft(x: np.ndarray, sr: int, n_fft: int = 512, hop: int = 128):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("Audio must be mono")
    if n_fft < 16 or hop <= 0 or hop > n_fft:
        raise ValueError("Require n_fft >= 16 and 0 < hop <= n_fft")
    return signal.stft(
        x,
        fs=sr,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop,
        nfft=n_fft,
        boundary="zeros",
        padded=True,
    )


def istft(z: np.ndarray, sr: int, n_fft: int, hop: int, length: int) -> np.ndarray:
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


def one_pole(x: np.ndarray, alpha: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("one_pole expects [frequency, frame]")
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must satisfy 0 <= alpha < 1")
    if x.shape[1] == 0:
        return np.empty_like(x)
    y = np.empty_like(x)
    y[:, 0] = x[:, 0]
    for m in range(1, x.shape[1]):
        y[:, m] = alpha * y[:, m - 1] + (1.0 - alpha) * x[:, m]
    return y


def apply_tf_gain(
    x: np.ndarray,
    gain: np.ndarray,
    sr: int,
    n_fft: int,
    hop: int,
) -> np.ndarray:
    _, _, X = stft(x, sr, n_fft, hop)
    if X.shape != gain.shape:
        raise ValueError(f"Gain shape {gain.shape} does not match STFT shape {X.shape}")
    return istft(gain * X, sr, n_fft, hop, len(x))


def stable_db(x: np.ndarray, floor_db: float = -100.0) -> np.ndarray:
    x = np.asarray(x)
    floor = 10.0 ** (floor_db / 20.0)
    return np.maximum(20.0 * np.log10(np.maximum(np.abs(x), floor)), floor_db)
