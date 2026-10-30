import numpy as np

from src.methods import DSPState, METHODS


def synthetic_state():
    sr = 8000
    t = np.arange(sr) / sr
    clean = 0.08*np.sin(2*np.pi*700*t)
    noise = 0.04*np.sin(2*np.pi*150*t)
    return DSPState(clean+noise, sr, clean, noise)


def test_core_oracle_methods_run():
    for name in ["spectral_sub", "wiener", "mmse_lsa", "spectral_kalman"]:
        out = METHODS[name](synthetic_state(), {})
        assert len(out.signal) == 8000
        assert np.all(np.isfinite(out.signal))


def test_haar_runs_and_marks_nonexact():
    out = METHODS["haar"](synthetic_state(), {"levels": 4})
    assert np.all(np.isfinite(out.signal))
    assert not out.decomposition_exact
    assert out.noise_ref is not None


def test_nmf_runs_finite_and_nonexact():
    out = METHODS["nmf"](synthetic_state(), {"rank_s": 3, "rank_n": 3, "dict_iter": 4, "mix_iter": 5})
    assert len(out.signal) == 8000
    assert np.all(np.isfinite(out.signal))
    assert not out.decomposition_exact
