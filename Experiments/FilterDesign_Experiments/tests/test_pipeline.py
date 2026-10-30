import numpy as np

from src.methods import DSPState
from src.pipeline import parse_pipeline, canonical_pipeline, run_pipeline


def test_parse_and_canonicalize():
    s = parse_pipeline("spectral_sub:oversub=1.2,floor=0.03|wiener:alpha=0.95")
    assert [x.name for x in s] == ["spectral_sub", "wiener"]
    assert s[0].params["oversub"] == 1.2
    assert "wiener:alpha=0.95" in canonical_pipeline(s)


def test_oracle_chain_finite_and_length_preserved():
    sr = 8000
    t = np.arange(sr) / sr
    clean = 0.1 * np.sin(2*np.pi*440*t)
    noise = 0.05 * np.sin(2*np.pi*120*t) + 0.01*np.random.default_rng(0).normal(size=len(t))
    state = DSPState(clean + noise, sr, clean, noise)
    stages = parse_pipeline("spectral_sub:oversub=0.8|wiener:alpha=0.9")
    final, audit = run_pipeline(state, stages)
    assert len(final.signal) == len(clean)
    assert np.all(np.isfinite(final.signal))
    assert len(audit) == 2
    assert final.decomposition_exact


def test_nmf_pipeline_parses():
    s = parse_pipeline("nmf:rank_s=4,rank_n=4")
    assert s[0].name == "nmf"
