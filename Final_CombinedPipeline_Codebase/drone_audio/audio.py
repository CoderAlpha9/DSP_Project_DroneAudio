"""Explicit conversion, floating WAV export, and no hidden normalization."""
from pathlib import Path
from math import gcd
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


def read_audio(path, *, allow_resample=False, downmix=False):
    sr, raw = wavfile.read(path)
    original_shape = list(raw.shape)
    if raw.dtype.kind == "i": x = raw.astype(float)/float(2**(raw.dtype.itemsize*8-1))
    elif raw.dtype == np.uint8: x = (raw.astype(float)-128)/128
    elif raw.dtype.kind == "f": x = raw.astype(float)
    else: raise ValueError(f"Unsupported WAV dtype {raw.dtype}")
    if not np.isfinite(x).all(): raise ValueError("Audio contains NaN/Inf")
    if x.ndim == 2:
        if not downmix: raise ValueError("Mono required; use --downmix explicitly")
        x = x.mean(axis=1)
    if x.ndim != 1: raise ValueError("Audio must have one or two dimensions")
    if sr != 16000:
        if not allow_resample: raise ValueError(f"16000 Hz required, got {sr}; use --resample explicitly")
        g = gcd(int(sr), 16000); x = resample_poly(x, 16000//g, int(sr)//g)
    return x, {"original_rate_hz": int(sr), "internal_rate_hz": 16000, "original_shape": original_shape,
               "resampled": sr != 16000, "downmixed": len(original_shape) == 2, "normalization": "PCM dtype scaling only"}


def write_audio(path, samples):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(samples, float)
    if x.ndim != 1 or not np.isfinite(x).all(): raise ValueError("Cannot export invalid audio")
    wavfile.write(p, 16000, x.astype(np.float32))
