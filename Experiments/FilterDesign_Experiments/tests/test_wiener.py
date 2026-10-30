import numpy as np

from src.metrics import evaluate
from src.wiener import apply_wiener_oracle_noise, smooth_one_pole


def test_alpha_zero_is_identity_smoother():
    rng = np.random.default_rng(0)
    x = rng.random((12, 20))
    y = smooth_one_pole(x, 0.0)
    assert np.allclose(x, y)


def test_gain_bounds_and_linearity():
    sr = 8000
    t = np.arange(sr) / sr
    clean = 0.2 * np.sin(2 * np.pi * 440 * t)
    noise = 0.1 * np.sin(2 * np.pi * 150 * t)
    result = apply_wiener_oracle_noise(clean, noise, sr, alpha=0.95, n_fft=256, hop=64)
    assert np.all(result.gain >= 0.0)
    assert np.all(result.gain <= 1.0)
    assert np.max(np.abs(result.filtered - result.filtered_clean - result.filtered_noise)) < 1e-7


def test_metrics_are_finite():
    sr = 8000
    t = np.arange(sr) / sr
    clean = 0.2 * np.sin(2 * np.pi * 500 * t)
    noise = 0.08 * np.sin(2 * np.pi * 120 * t)
    result = apply_wiener_oracle_noise(clean, noise, sr, alpha=0.9, n_fft=256, hop=64)
    values = evaluate(clean, noise, result.filtered, result.filtered_clean, result.filtered_noise)
    assert all(np.isfinite(v) for v in values.values())
