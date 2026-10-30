from src.dataset import RunRecord, make_split_file


def test_80_10_10_counts(tmp_path, monkeypatch):
    records = [
        RunRecord(
            run_id=f"r{i}", scenario=f"S{(i % 7) + 1}", seed=[101, 202, 303][i % 3],
            sample_rate_hz=16000, duration_seconds=15.0, speech_valid_for_metrics=True,
            flight_success_verified=True, acoustic_variant="nominal", fault_type="none",
            source_zip="x.zip", run_dir=f"runs/S1/r{i}"
        )
        for i in range(20)
    ]
    import src.dataset as ds
    monkeypatch.setattr(ds, "SPLIT_FILE", tmp_path / "splits.csv")
    df = make_split_file(records)
    counts = df["split"].value_counts().to_dict()
    assert counts["development"] == 16
    assert counts["validation"] == 2
    assert counts["test"] == 2
