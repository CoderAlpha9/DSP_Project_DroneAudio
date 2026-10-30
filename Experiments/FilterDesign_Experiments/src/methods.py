from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
import math
import os
import shutil
import subprocess
import tempfile


import numpy as np
import pandas as pd
from scipy import signal
from scipy.special import exp1

from .dsp_core import stft, istft, one_pole, apply_tf_gain


@dataclass
class DSPState:
    signal: np.ndarray
    sr: int
    clean_ref: np.ndarray | None = None
    noise_ref: np.ndarray | None = None
    run_dir: Path | None = None
    manifest: dict[str, Any] | None = None
    decomposition_exact: bool = True
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MethodInfo:
    name: str
    label: str
    family: str
    requires_oracle_noise: bool = False
    requires_rpm: bool = False
    requires_imu: bool = False
    preserves_exact_decomposition: bool = True
    description: str = ""


METHOD_INFO: dict[str, MethodInfo] = {
    "identity": MethodInfo("identity", "No processing", "baseline", description="Pass-through control."),
    "spectral_sub": MethodInfo("spectral_sub", "Spectral subtraction", "spectral", True, description="Oracle-noise spectral subtraction baseline."),
    "wiener": MethodInfo("wiener", "Wiener", "statistical", True, description="One-pole noise PSD Wiener filter."),
    "mmse_lsa": MethodInfo("mmse_lsa", "MMSE-LSA", "statistical", True, description="Decision-directed log-spectral amplitude estimator."),
    "spectral_kalman": MethodInfo("spectral_kalman", "Spectral Kalman", "state-space", True, description="Per-bin random-walk Kalman smoother."),
    "rpm_order": MethodInfo("rpm_order", "RPM order suppressor", "physics-informed", False, True, description="Time-varying STFT attenuation around RPM-derived blade-passing orders."),
    "imu_nlms": MethodInfo("imu_nlms", "IMU-NLMS", "adaptive", False, False, True, description="Adaptive cancellation using low-rate inertial motion as an external reference."),
    "imu_rls": MethodInfo("imu_rls", "IMU block-RLS", "adaptive", False, False, True, description="Regularized block recursive least-squares cancellation using low-rate inertial motion as an external reference."),
    "adaptive_comb": MethodInfo("adaptive_comb", "Adaptive harmonic / comb", "physics-informed", False, True, description="Aggregate-RPM harmonic comb suppressor with temporally smoothed blade-passing frequency."),
    "haar": MethodInfo("haar", "Wavelet denoising (Haar)", "transform", False, False, False, False, description="Dependency-free nonlinear Haar shrinkage baseline."),
    "nmf": MethodInfo("nmf", "NMF denoising", "decomposition", True, False, False, False, description="Semi-supervised non-negative matrix factorization using clean/noise reference dictionaries for controlled benchmarking."),
    "neural": MethodInfo("neural", "DeepFilterNet / RNNoise", "neural", False, False, False, False, description="Optional pretrained neural denoiser. Auto-selects DeepFilterNet or a local RNNoise executable when available."),
}


def _float(p: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(p.get(key, default))
    except (TypeError, ValueError) as e:
        raise ValueError(f"Parameter {key!r} must be numeric") from e


def _int(p: dict[str, Any], key: str, default: int) -> int:
    try:
        return int(p.get(key, default))
    except (TypeError, ValueError) as e:
        raise ValueError(f"Parameter {key!r} must be an integer") from e


def _tf_common(state: DSPState, p: dict[str, Any]):
    if state.noise_ref is None:
        raise ValueError("This method requires a tracked/oracle noise reference")
    n_fft = _int(p, "n_fft", 512)
    hop = _int(p, "hop", n_fft // 4)
    _, _, Y = stft(state.signal, state.sr, n_fft, hop)
    f, t, N = stft(state.noise_ref, state.sr, n_fft, hop)
    return n_fft, hop, f, t, Y, N


def _apply_gain_state(state: DSPState, gain: np.ndarray, n_fft: int, hop: int, tag: str, diagnostics: dict[str, Any] | None = None) -> DSPState:
    out = apply_tf_gain(state.signal, gain, state.sr, n_fft, hop)
    clean = apply_tf_gain(state.clean_ref, gain, state.sr, n_fft, hop) if state.clean_ref is not None else None
    noise = apply_tf_gain(state.noise_ref, gain, state.sr, n_fft, hop) if state.noise_ref is not None else None
    exact = state.decomposition_exact and clean is not None and noise is not None
    d = dict(state.diagnostics)
    d[tag] = diagnostics or {}
    if exact:
        err = float(np.max(np.abs(out - (clean + noise))))
        d[tag]["component_linearity_error"] = err
        if err > 2e-6:
            exact = False
    return DSPState(out, state.sr, clean, noise, state.run_dir, state.manifest, exact, d)


def method_identity(state: DSPState, p: dict[str, Any]) -> DSPState:
    return DSPState(state.signal.copy(), state.sr,
                    None if state.clean_ref is None else state.clean_ref.copy(),
                    None if state.noise_ref is None else state.noise_ref.copy(),
                    state.run_dir, state.manifest, state.decomposition_exact, dict(state.diagnostics))


def method_spectral_sub(state: DSPState, p: dict[str, Any]) -> DSPState:
    n_fft, hop, _, _, Y, N = _tf_common(state, p)
    oversub = _float(p, "oversub", 1.0)
    floor = _float(p, "floor", 0.02)
    if oversub < 0 or not 0 <= floor <= 1:
        raise ValueError("Require oversub >= 0 and 0 <= floor <= 1")
    mag_y = np.abs(Y)
    mag_n = np.abs(N)
    target = np.maximum(mag_y - oversub * mag_n, floor * mag_y)
    gain = target / np.maximum(mag_y, 1e-12)
    return _apply_gain_state(state, gain, n_fft, hop, "spectral_sub", {"oversub": oversub, "floor": floor})


def method_wiener(state: DSPState, p: dict[str, Any]) -> DSPState:
    n_fft, hop, _, _, Y, N = _tf_common(state, p)
    alpha = _float(p, "alpha", 0.95)
    beta = _float(p, "beta", 1.0)
    floor = _float(p, "floor", 0.02)
    if not 0 <= alpha < 1 or beta <= 0 or not 0 <= floor <= 1:
        raise ValueError("Wiener requires 0<=alpha<1, beta>0, 0<=floor<=1")
    pn = one_pole(np.abs(N) ** 2, alpha)
    py = np.abs(Y) ** 2
    ps = np.maximum(py - beta * pn, 0.0)
    gain = np.clip(ps / (ps + beta * pn + 1e-12), floor, 1.0)
    return _apply_gain_state(state, gain, n_fft, hop, "wiener", {"alpha": alpha, "beta": beta, "floor": floor})


def method_mmse_lsa(state: DSPState, p: dict[str, Any]) -> DSPState:
    n_fft, hop, _, _, Y, N = _tf_common(state, p)
    alpha_noise = _float(p, "alpha_noise", 0.95)
    dd = _float(p, "dd", 0.98)
    floor = _float(p, "floor", 0.02)
    if not 0 <= alpha_noise < 1 or not 0 <= dd < 1 or not 0 <= floor <= 1:
        raise ValueError("MMSE-LSA requires smoothing parameters in [0,1) and floor in [0,1]")
    pn = one_pole(np.abs(N) ** 2, alpha_noise)
    gamma = np.abs(Y) ** 2 / np.maximum(pn, 1e-12)
    gain = np.empty_like(gamma)
    prev_g2_gamma = np.ones(gamma.shape[0])
    for m in range(gamma.shape[1]):
        post_minus = np.maximum(gamma[:, m] - 1.0, 0.0)
        xi = dd * prev_g2_gamma + (1.0 - dd) * post_minus
        xi = np.maximum(xi, 1e-8)
        v = np.maximum(gamma[:, m] * xi / (1.0 + xi), 1e-12)
        g = (xi / (1.0 + xi)) * np.exp(0.5 * exp1(v))
        g = np.clip(g, floor, 1.0)
        gain[:, m] = g
        prev_g2_gamma = (g * g) * gamma[:, m]
    return _apply_gain_state(state, gain, n_fft, hop, "mmse_lsa", {"alpha_noise": alpha_noise, "dd": dd, "floor": floor})


def method_spectral_kalman(state: DSPState, p: dict[str, Any]) -> DSPState:
    n_fft, hop, _, _, Y, N = _tf_common(state, p)
    alpha_noise = _float(p, "alpha_noise", 0.95)
    q_scale = _float(p, "q_scale", 0.03)
    floor = _float(p, "floor", 0.02)
    if q_scale < 0 or not 0 <= floor <= 1:
        raise ValueError("spectral_kalman requires q_scale >= 0 and floor in [0,1]")
    r = one_pole(np.abs(N) ** 2, alpha_noise)
    power_y = np.abs(Y) ** 2
    q = q_scale * np.maximum(one_pole(power_y, 0.8), 1e-12)
    P = np.maximum(power_y[:, 0], 1e-12)
    gain = np.empty_like(power_y)
    for m in range(power_y.shape[1]):
        pred = P + q[:, m]
        k = pred / (pred + r[:, m] + 1e-12)
        k = np.clip(k, floor, 1.0)
        gain[:, m] = k
        P = (1.0 - k) * pred
    return _apply_gain_state(state, gain, n_fft, hop, "spectral_kalman", {"alpha_noise": alpha_noise, "q_scale": q_scale, "floor": floor})


def _load_rpm(state: DSPState, times: np.ndarray) -> np.ndarray:
    if state.run_dir is None:
        raise ValueError("rpm_order requires run_dir")
    path = state.run_dir / "rpm_aligned.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing RPM telemetry: {path}")
    df = pd.read_csv(path)
    rpm_cols = [c for c in ["rpm_0", "rpm_1", "rpm_2", "rpm_3"] if c in df.columns]
    if not rpm_cols:
        raise ValueError("rpm_aligned.csv contains no rpm_0..rpm_3 columns")
    if "sim_time_ns" in df.columns and state.manifest:
        t0 = float(state.manifest.get("audio_t0_sim_ns", 0.0))
        tt = (df["sim_time_ns"].to_numpy(float) - t0) * 1e-9
    elif "sim_time_s" in df.columns:
        tt = df["sim_time_s"].to_numpy(float)
        tt -= tt[0]
    else:
        raise ValueError("RPM telemetry needs sim_time_ns or sim_time_s")
    vals = df[rpm_cols].to_numpy(float)
    good = np.isfinite(tt) & np.all(np.isfinite(vals), axis=1)
    tt, vals = tt[good], vals[good]
    if len(tt) < 2:
        raise ValueError("Not enough finite RPM telemetry samples")
    order = np.argsort(tt)
    tt, vals = tt[order], vals[order]
    return np.vstack([np.interp(times, tt, vals[:, i], left=vals[0, i], right=vals[-1, i]) for i in range(vals.shape[1])])


def method_rpm_order(state: DSPState, p: dict[str, Any]) -> DSPState:
    n_fft = _int(p, "n_fft", 512)
    hop = _int(p, "hop", n_fft // 4)
    blades = _int(p, "blades", 2)
    harmonics = _int(p, "harmonics", 8)
    width_hz = _float(p, "width_hz", 18.0)
    attenuation = _float(p, "attenuation", 0.15)
    if blades < 1 or harmonics < 1 or width_hz <= 0 or not 0 <= attenuation <= 1:
        raise ValueError("Invalid rpm_order parameters")
    f, t, _ = stft(state.signal, state.sr, n_fft, hop)
    rpms = _load_rpm(state, t)
    gain = np.ones((len(f), len(t)), dtype=np.float64)
    for m in range(len(t)):
        centers = []
        for rpm in rpms[:, m]:
            bpf = blades * rpm / 60.0
            centers.extend([h * bpf for h in range(1, harmonics + 1)])
        for fc in centers:
            if fc <= 0 or fc > f[-1] + width_hz:
                continue
            d = np.abs(f - fc)
            local = np.ones_like(f)
            inside = d <= width_hz
            if np.any(inside):
                # Raised-cosine notch, attenuation at the exact order and unity at its edge.
                local[inside] = attenuation + (1.0 - attenuation) * 0.5 * (1.0 - np.cos(np.pi * d[inside] / width_hz))
                gain[:, m] = np.minimum(gain[:, m], local)
    return _apply_gain_state(state, gain, n_fft, hop, "rpm_order", {"blades": blades, "harmonics": harmonics, "width_hz": width_hz, "attenuation": attenuation})


def _imu_reference(state: DSPState, length: int, mode: str = "accel_mag") -> np.ndarray:
    if state.run_dir is None:
        raise ValueError("IMU adaptive filtering requires run_dir")
    path = state.run_dir / "imu_aligned.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing IMU telemetry: {path}. Re-run prepare_data.py with the updated repository.")
    df = pd.read_csv(path)
    acc_cols = ["accel_x_m_s2", "accel_y_m_s2", "accel_z_m_s2"]
    gyro_cols = ["gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s"]
    if mode == "gyro_mag":
        cols = gyro_cols
    else:
        cols = acc_cols
    if not all(c in df.columns for c in cols):
        raise ValueError(f"IMU file missing required columns for {mode}: {cols}")
    vals = df[cols].to_numpy(float)
    ref = np.linalg.norm(vals, axis=1)
    ref = ref - np.nanmedian(ref)
    if "sim_time_ns" in df.columns and state.manifest:
        t0 = float(state.manifest.get("audio_t0_sim_ns", 0.0))
        tt = (df["sim_time_ns"].to_numpy(float) - t0) * 1e-9
    elif "sim_time_s" in df.columns:
        tt = df["sim_time_s"].to_numpy(float)
        tt -= tt[0]
    else:
        raise ValueError("IMU telemetry needs sim_time_ns or sim_time_s")
    good = np.isfinite(tt) & np.isfinite(ref)
    tt, ref = tt[good], ref[good]
    if len(tt) < 2:
        raise ValueError("Not enough finite IMU telemetry")
    order = np.argsort(tt)
    tt, ref = tt[order], ref[order]
    ta = np.arange(length, dtype=float) / state.sr
    up = np.interp(ta, tt, ref, left=ref[0], right=ref[-1])
    std = float(np.std(up))
    if std < 1e-10:
        return np.zeros_like(up)
    up = up / std
    # IMU is low rate. Restrict reference to low-frequency motion content rather than inventing high-frequency vibration.
    cutoff = min(20.0, 0.45 * state.sr)
    if cutoff > 0 and state.sr > 2 * cutoff:
        sos = signal.butter(2, cutoff, btype="lowpass", fs=state.sr, output="sos")
        up = signal.sosfiltfilt(sos, up)
    return up


def _adaptive_subtract_nlms(d: np.ndarray, x: np.ndarray, taps: int, mu: float, eps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if taps < 1 or not 0 < mu <= 2:
        raise ValueError("NLMS requires taps>=1 and 0<mu<=2")
    w = np.zeros(taps, dtype=float)
    yhat = np.zeros_like(d, dtype=float)
    e = np.zeros_like(d, dtype=float)
    buf = np.zeros(taps, dtype=float)
    for n in range(len(d)):
        buf[1:] = buf[:-1]
        buf[0] = x[n]
        yh = float(np.dot(w, buf))
        err = d[n] - yh
        norm = float(np.dot(buf, buf)) + eps
        w += (mu / norm) * err * buf
        yhat[n] = yh
        e[n] = err
    return e, yhat, w


def method_imu_nlms(state: DSPState, p: dict[str, Any]) -> DSPState:
    taps = _int(p, "taps", 64)
    mu = _float(p, "mu", 0.08)
    eps = _float(p, "eps", 1e-6)
    mode = str(p.get("reference", "accel_mag"))
    ref = _imu_reference(state, len(state.signal), mode)
    out, est, w = _adaptive_subtract_nlms(state.signal, ref, taps, mu, eps)
    clean = state.clean_ref.copy() if state.clean_ref is not None else None
    noise = state.noise_ref - est if state.noise_ref is not None else None
    exact = state.decomposition_exact and clean is not None and noise is not None
    d = dict(state.diagnostics)
    d["imu_nlms"] = {"taps": taps, "mu": mu, "reference": mode, "estimated_reference_noise_rms": float(np.sqrt(np.mean(est**2))), "final_weight_norm": float(np.linalg.norm(w))}
    return DSPState(out, state.sr, clean, noise, state.run_dir, state.manifest, exact, d)


def _adaptive_subtract_rls(d: np.ndarray, x: np.ndarray, taps: int, lam: float, delta: float, block: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numerically stable block recursive least-squares cancellation.

    The sufficient statistics are exponentially updated once per block, then a
    regularized least-squares system is solved. This is slower-adapting than
    sample RLS but much more robust for highly correlated, low-rate IMU signals
    after interpolation to audio rate.
    """
    if taps < 1 or not 0.9 <= lam <= 1.0 or delta <= 0 or block < taps:
        raise ValueError("RLS requires taps>=1, 0.9<=lambda<=1, delta>0 and block>=taps")
    w = np.zeros(taps, dtype=float)
    R = delta * np.eye(taps, dtype=float)
    pvec = np.zeros(taps, dtype=float)
    yhat = np.zeros_like(d, dtype=float)
    e = np.zeros_like(d, dtype=float)
    hist = np.zeros(taps - 1, dtype=float)
    eye = np.eye(taps, dtype=float)
    for lo in range(0, len(d), block):
        hi = min(len(d), lo + block)
        xb = x[lo:hi]
        db = d[lo:hi]
        joined = np.concatenate([hist, xb])
        X = np.lib.stride_tricks.sliding_window_view(joined, taps)[:, ::-1]
        X = X[-len(xb):]
        pred = X @ w
        yhat[lo:hi] = pred
        e[lo:hi] = db - pred
        scale = max(1, len(db))
        R = lam * R + (X.T @ X) / scale
        pvec = lam * pvec + (X.T @ db) / scale
        try:
            w = np.linalg.solve(R + delta * 1e-3 * eye, pvec)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(R + delta * 1e-3 * eye, pvec, rcond=None)[0]
        if not np.all(np.isfinite(w)):
            raise FloatingPointError("RLS coefficient solve became non-finite")
        hist = joined[-(taps - 1):] if taps > 1 else np.empty(0)
    return e, yhat, w


def method_imu_rls(state: DSPState, p: dict[str, Any]) -> DSPState:
    taps = _int(p, "taps", 16)
    lam = _float(p, "lambda", 0.995)
    delta = _float(p, "delta", 1.0)
    block = _int(p, "block", 256)
    mode = str(p.get("reference", "accel_mag"))
    ref = _imu_reference(state, len(state.signal), mode)
    out, est, w = _adaptive_subtract_rls(state.signal, ref, taps, lam, delta, block)
    clean = state.clean_ref.copy() if state.clean_ref is not None else None
    noise = state.noise_ref - est if state.noise_ref is not None else None
    exact = state.decomposition_exact and clean is not None and noise is not None
    d = dict(state.diagnostics)
    d["imu_rls"] = {"taps": taps, "lambda": lam, "delta": delta, "block": block, "reference": mode, "estimated_reference_noise_rms": float(np.sqrt(np.mean(est**2))), "final_weight_norm": float(np.linalg.norm(w))}
    return DSPState(out, state.sr, clean, noise, state.run_dir, state.manifest, exact, d)



def _haar_forward(x: np.ndarray, levels: int) -> tuple[np.ndarray, list[int]]:
    a = np.asarray(x, dtype=float).copy()
    original = len(a)
    n_pow = 1 << int(math.ceil(math.log2(max(2, original))))
    a = np.pad(a, (0, n_pow - original))
    sizes = []
    n = len(a)
    root2 = math.sqrt(2.0)
    for _ in range(levels):
        if n < 2:
            break
        even, odd = a[:n:2], a[1:n:2]
        approx = (even + odd) / root2
        detail = (even - odd) / root2
        a[: n // 2] = approx
        a[n // 2 : n] = detail
        sizes.append(n)
        n //= 2
    return a, sizes


def _haar_inverse(c: np.ndarray, sizes: list[int], original: int) -> np.ndarray:
    a = c.copy()
    root2 = math.sqrt(2.0)
    for n in reversed(sizes):
        half = n // 2
        approx = a[:half].copy()
        detail = a[half:n].copy()
        a[:n:2] = (approx + detail) / root2
        a[1:n:2] = (approx - detail) / root2
    return a[:original]

def method_haar(state: DSPState, p: dict[str, Any]) -> DSPState:
    levels = _int(p, "levels", 6)
    threshold_scale = _float(p, "threshold_scale", 1.0)
    if levels < 1 or threshold_scale < 0:
        raise ValueError("Haar requires levels>=1 and threshold_scale>=0")
    c, sizes = _haar_forward(state.signal, levels)
    if not sizes:
        return method_identity(state, {})
    # Finest-level details occupy the upper half after first transform.
    n0 = sizes[0]
    finest = c[n0 // 2 : n0]
    sigma = float(np.median(np.abs(finest - np.median(finest))) / 0.6745) if len(finest) else 0.0
    threshold = threshold_scale * sigma * math.sqrt(2.0 * math.log(max(2, len(state.signal))))
    cc = c.copy()
    smallest_approx = sizes[-1] // 2
    d = cc[smallest_approx:]
    cc[smallest_approx:] = np.sign(d) * np.maximum(np.abs(d) - threshold, 0.0)
    out = _haar_inverse(cc, sizes, len(state.signal))
    diag = dict(state.diagnostics)
    diag["haar"] = {"levels": len(sizes), "threshold": threshold, "sigma_est": sigma}
    # Nonlinear shrinkage destroys exact additive component tracking.
    proxy_noise = out - state.clean_ref if state.clean_ref is not None else None
    return DSPState(out, state.sr, state.clean_ref, proxy_noise, state.run_dir, state.manifest, False, diag)


def method_adaptive_comb(state: DSPState, p: dict[str, Any]) -> DSPState:
    """Time-varying harmonic comb based on the aggregate rotor speed.

    This intentionally differs from rpm_order: rpm_order tracks every rotor order
    independently, whereas this method estimates one smoothed aggregate blade-
    passing frequency and applies a harmonic comb around that track.
    """
    n_fft = _int(p, "n_fft", 512)
    hop = _int(p, "hop", n_fft // 4)
    blades = _int(p, "blades", 2)
    harmonics = _int(p, "harmonics", 10)
    width_hz = _float(p, "width_hz", 14.0)
    attenuation = _float(p, "attenuation", 0.12)
    rpm_alpha = _float(p, "rpm_alpha", 0.90)
    if blades < 1 or harmonics < 1 or width_hz <= 0 or not 0 <= attenuation <= 1 or not 0 <= rpm_alpha < 1:
        raise ValueError("Invalid adaptive_comb parameters")
    f, t, _ = stft(state.signal, state.sr, n_fft, hop)
    rpms = _load_rpm(state, t)
    aggregate = np.median(rpms, axis=0)
    smooth = np.empty_like(aggregate)
    smooth[0] = aggregate[0]
    for i in range(1, len(aggregate)):
        smooth[i] = rpm_alpha * smooth[i-1] + (1-rpm_alpha) * aggregate[i]
    gain = np.ones((len(f), len(t)), dtype=float)
    for m, rpm in enumerate(smooth):
        bpf = blades * rpm / 60.0
        for h in range(1, harmonics + 1):
            fc = h * bpf
            if fc <= 0 or fc > f[-1] + width_hz:
                continue
            d = np.abs(f - fc)
            inside = d <= width_hz
            if np.any(inside):
                # Smooth raised-cosine harmonic notch.
                local = np.ones_like(f)
                local[inside] = attenuation + (1.0 - attenuation) * 0.5 * (1.0 - np.cos(np.pi * d[inside] / width_hz))
                gain[:, m] = np.minimum(gain[:, m], local)
    return _apply_gain_state(state, gain, n_fft, hop, "adaptive_comb", {
        "blades": blades, "harmonics": harmonics, "width_hz": width_hz,
        "attenuation": attenuation, "rpm_alpha": rpm_alpha,
    })


def _nmf_mu(V: np.ndarray, rank: int, iterations: int, rng: np.random.Generator, W0: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    eps = 1e-10
    F, T = V.shape
    if W0 is None:
        W = rng.random((F, rank)) + 0.1
    else:
        W = np.maximum(np.asarray(W0, dtype=float).copy(), eps)
        rank = W.shape[1]
    H = rng.random((rank, T)) + 0.1
    W /= np.maximum(np.sum(W, axis=0, keepdims=True), eps)
    for _ in range(iterations):
        H *= (W.T @ V) / np.maximum(W.T @ W @ H, eps)
        if W0 is None:
            W *= (V @ H.T) / np.maximum(W @ H @ H.T, eps)
            W /= np.maximum(np.sum(W, axis=0, keepdims=True), eps)
    return W, H


def method_nmf(state: DSPState, p: dict[str, Any]) -> DSPState:
    """Semi-supervised NMF controlled baseline.

    Dictionaries are learned independently from the supplied clean and noise
    references, then activations are fit to the mixture. Because those references
    are unavailable in deployment this method is explicitly marked oracle-assisted.
    """
    if state.clean_ref is None or state.noise_ref is None:
        raise ValueError("NMF controlled baseline requires clean and noise references")
    n_fft = _int(p, "n_fft", 512)
    hop = _int(p, "hop", n_fft // 4)
    rank_s = _int(p, "rank_s", 8)
    rank_n = _int(p, "rank_n", 8)
    dict_iter = _int(p, "dict_iter", 24)
    mix_iter = _int(p, "mix_iter", 36)
    floor = _float(p, "floor", 0.02)
    seed = _int(p, "seed", 0)
    if min(rank_s, rank_n, dict_iter, mix_iter) < 1 or not 0 <= floor <= 1:
        raise ValueError("Invalid NMF parameters")
    _, _, Y = stft(state.signal, state.sr, n_fft, hop)
    _, _, S = stft(state.clean_ref, state.sr, n_fft, hop)
    _, _, N = stft(state.noise_ref, state.sr, n_fft, hop)
    V, Vs, Vn = np.abs(Y) + 1e-10, np.abs(S) + 1e-10, np.abs(N) + 1e-10
    rng = np.random.default_rng(seed)
    Ws, _ = _nmf_mu(Vs, rank_s, dict_iter, rng)
    Wn, _ = _nmf_mu(Vn, rank_n, dict_iter, rng)
    W = np.concatenate([Ws, Wn], axis=1)
    W /= np.maximum(np.sum(W, axis=0, keepdims=True), 1e-10)
    _, H = _nmf_mu(V, W.shape[1], mix_iter, rng, W0=W)
    Vshat = Ws @ H[:rank_s]
    Vnhat = Wn @ H[rank_s:]
    gain = np.clip(Vshat / np.maximum(Vshat + Vnhat, 1e-10), floor, 1.0)
    out = apply_tf_gain(state.signal, gain, state.sr, n_fft, hop)
    d = dict(state.diagnostics)
    d["nmf"] = {"rank_s": rank_s, "rank_n": rank_n, "dict_iter": dict_iter, "mix_iter": mix_iter, "floor": floor}
    # NMF masking depends nonlinearly on the observed mixture.
    return DSPState(out, state.sr, state.clean_ref, out - state.clean_ref, state.run_dir, state.manifest, False, d)


_DF_CACHE: dict[str, Any] = {}

def _deepfilternet_available() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec("df") is not None and importlib.util.find_spec("torch") is not None
    except Exception:
        return False

def _run_deepfilternet(x: np.ndarray, sr: int, atten_lim_db: float | None = None) -> np.ndarray:
    try:
        import torch
        from df.enhance import enhance, init_df
    except Exception as exc:
        raise RuntimeError("DeepFilterNet is not installed. Optional one-time install: pip install deepfilternet") from exc
    if "model" not in _DF_CACHE:
        model, df_state, _ = init_df()
        _DF_CACHE["model"] = model
        _DF_CACHE["state"] = df_state
    model, df_state = _DF_CACHE["model"], _DF_CACHE["state"]
    target_sr = int(df_state.sr())
    xin = np.asarray(x, dtype=np.float32)
    if sr != target_sr:
        from fractions import Fraction
        frac = Fraction(target_sr, sr).limit_denominator(1000)
        xin = signal.resample_poly(xin, frac.numerator, frac.denominator).astype(np.float32)
    audio = torch.from_numpy(xin).reshape(1, -1)
    y = enhance(model, df_state, audio, atten_lim_db=atten_lim_db).detach().cpu().numpy().reshape(-1)
    if sr != target_sr:
        from fractions import Fraction
        frac = Fraction(sr, target_sr).limit_denominator(1000)
        y = signal.resample_poly(y, frac.numerator, frac.denominator)
    if len(y) < len(x):
        y = np.pad(y, (0, len(x)-len(y)))
    return np.asarray(y[:len(x)], dtype=float)

def _find_rnnoise_bin(explicit: str | None = None) -> str | None:
    candidates = [explicit, os.environ.get("RNNOISE_BIN"), shutil.which("rnnoise_demo")]
    if explicit:
        candidates.append(str(Path(explicit).expanduser()))
    for c in candidates:
        if c and Path(c).exists():
            return str(Path(c))
    return None

def _run_rnnoise(x: np.ndarray, sr: int, executable: str | None = None) -> np.ndarray:
    exe = _find_rnnoise_bin(executable)
    if not exe:
        raise RuntimeError("RNNoise executable not found. Set RNNOISE_BIN to the compiled rnnoise_demo path, or use DeepFilterNet.")
    from fractions import Fraction
    xin = np.asarray(x, dtype=float)
    if sr != 48000:
        frac = Fraction(48000, sr).limit_denominator(1000)
        xin = signal.resample_poly(xin, frac.numerator, frac.denominator)
    scale = max(1.0, float(np.max(np.abs(xin))) / 0.98)
    pcm = np.clip(xin / scale, -1, 1)
    pcm = (pcm * 32767.0).astype(np.int16)
    with tempfile.TemporaryDirectory(prefix="rnnoise_") as td:
        inp = Path(td) / "in.raw"
        outp = Path(td) / "out.raw"
        pcm.tofile(inp)
        proc = subprocess.run([exe, str(inp), str(outp)], capture_output=True, text=True)
        if proc.returncode != 0 or not outp.exists():
            raise RuntimeError(f"RNNoise failed: {proc.stderr.strip() or proc.stdout.strip()}")
        y = np.fromfile(outp, dtype=np.int16).astype(float) / 32767.0 * scale
    if sr != 48000:
        frac = Fraction(sr, 48000).limit_denominator(1000)
        y = signal.resample_poly(y, frac.numerator, frac.denominator)
    if len(y) < len(x):
        y = np.pad(y, (0, len(x)-len(y)))
    return y[:len(x)]

def method_neural(state: DSPState, p: dict[str, Any]) -> DSPState:
    engine = str(p.get("engine", "auto")).lower()
    atten = p.get("atten_lim_db", None)
    atten_lim = None if atten in (None, "none", "null") else float(atten)
    chosen = engine
    if engine == "auto":
        chosen = "deepfilternet" if _deepfilternet_available() else ("rnnoise" if _find_rnnoise_bin(None) else "none")
    if chosen in {"deepfilternet", "deepfilter", "df"}:
        out = _run_deepfilternet(state.signal, state.sr, atten_lim)
        chosen = "deepfilternet"
    elif chosen == "rnnoise":
        out = _run_rnnoise(state.signal, state.sr, str(p.get("executable")) if p.get("executable") else None)
    else:
        raise RuntimeError("No neural engine is available. Install DeepFilterNet with `pip install deepfilternet`, or provide RNNoise via RNNOISE_BIN.")
    d = dict(state.diagnostics)
    d["neural"] = {"engine": chosen, "atten_lim_db": atten_lim}
    return DSPState(out, state.sr, state.clean_ref, out - state.clean_ref if state.clean_ref is not None else None, state.run_dir, state.manifest, False, d)


METHODS: dict[str, Callable[[DSPState, dict[str, Any]], DSPState]] = {
    "identity": method_identity,
    "spectral_sub": method_spectral_sub,
    "wiener": method_wiener,
    "mmse_lsa": method_mmse_lsa,
    "spectral_kalman": method_spectral_kalman,
    "rpm_order": method_rpm_order,
    "imu_nlms": method_imu_nlms,
    "imu_rls": method_imu_rls,
    "adaptive_comb": method_adaptive_comb,
    "haar": method_haar,
    "nmf": method_nmf,
    "neural": method_neural,
}
