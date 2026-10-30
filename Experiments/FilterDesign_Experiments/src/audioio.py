from __future__ import annotations

import io
import wave
import warnings
from pathlib import Path

import numpy as np
from scipy.io import wavfile
try:
    from scipy.io.wavfile import WavFileWarning
except ImportError:
    WavFileWarning = Warning


def _pcm_to_float(x: np.ndarray) -> np.ndarray:
    if np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float64, copy=False)
    elif x.dtype == np.int16:
        y = x.astype(np.float64) / 32768.0
    elif x.dtype == np.int32:
        y = x.astype(np.float64) / 2147483648.0
    elif x.dtype == np.uint8:
        y = (x.astype(np.float64) - 128.0) / 128.0
    else:
        info = np.iinfo(x.dtype)
        y = x.astype(np.float64) / max(abs(info.min), abs(info.max))
    if y.ndim == 2:
        y = np.mean(y, axis=1)
    return np.ascontiguousarray(y, dtype=np.float64)


def read_wav(path: Path | str) -> tuple[int, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", WavFileWarning)
        sr, x = wavfile.read(str(path))
    return int(sr), _pcm_to_float(x)


def wav_bytes(x: np.ndarray, sr: int) -> bytes:
    x = np.asarray(x, dtype=np.float64)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1.0:
        x = x / peak
    pcm = np.clip(x, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm.tobytes())
    return bio.getvalue()


def write_wav(path, sr: int, x: np.ndarray) -> None:
    from scipy.io import wavfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.asarray(x, dtype=np.float64)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1.0:
        y = y / peak
    wavfile.write(str(path), int(sr), (np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16))
