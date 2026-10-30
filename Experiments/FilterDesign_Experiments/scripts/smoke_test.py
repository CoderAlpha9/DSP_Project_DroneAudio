from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import evaluate  # noqa: E402
from src.wiener import apply_wiener_oracle_noise  # noqa: E402


if __name__ == "__main__":
    sr = 16000
    t = np.arange(sr * 2) / sr
    clean = 0.25 * np.sin(2 * np.pi * 300 * t) + 0.1 * np.sin(2 * np.pi * 900 * t)
    rng = np.random.default_rng(7)
    noise = 0.20 * np.sin(2 * np.pi * 160 * t) + 0.07 * rng.standard_normal(len(t))
    result = apply_wiener_oracle_noise(clean, noise, sr, alpha=0.95)
    m = evaluate(clean, noise, result.filtered, result.filtered_clean, result.filtered_noise)
    assert np.all(np.isfinite(list(m.values())))
    assert np.min(result.gain) >= 0.0 and np.max(result.gain) <= 1.0
    print("Smoke test passed.")
    for k, v in m.items():
        print(f"{k}: {v:.4f}")
