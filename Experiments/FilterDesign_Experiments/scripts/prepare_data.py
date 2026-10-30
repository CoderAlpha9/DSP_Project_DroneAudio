from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import prepare_from_zips  # noqa: E402


if __name__ == "__main__":
    records = prepare_from_zips()
    valid = sum(r.speech_valid_for_metrics and r.flight_success_verified for r in records)
    print(f"Prepared {len(records)} runs. Metric-valid runs: {valid}.")
    print("Catalog: data/catalog.csv")
    print("Split file: data/splits.csv")
