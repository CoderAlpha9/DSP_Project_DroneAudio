import pandas as pd
from src.benchmark import add_quality_score


def test_quality_score_prefers_better_quality():
    df = pd.DataFrame([
        {"method":"bad", "reference_snr_improvement_db":0.0, "si_sdr_db":0.0, "correlation":0.3, "log_spectral_distance_db":20.0},
        {"method":"good", "reference_snr_improvement_db":8.0, "si_sdr_db":7.0, "correlation":0.9, "log_spectral_distance_db":4.0},
    ])
    out = add_quality_score(df).set_index("method")
    assert out.loc["good", "quality_score"] > out.loc["bad", "quality_score"]
