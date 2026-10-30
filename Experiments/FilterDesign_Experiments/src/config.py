from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
EXTRACTED_DIR = ROOT / "data" / "extracted"
OUTPUT_DIR = ROOT / "outputs"
SPLIT_FILE = ROOT / "data" / "splits.csv"
CATALOG_FILE = ROOT / "data" / "catalog.csv"

DEFAULT_N_FFT = 512
DEFAULT_HOP = 128
DEFAULT_WINDOW = "hann"
DEFAULT_POLES = [
    0.0, 0.20, 0.40, 0.60, 0.70, 0.80, 0.85, 0.90,
    0.93, 0.95, 0.97, 0.98, 0.985, 0.99, 0.995, 0.998,
]
DEFAULT_GAIN_FLOOR = 0.02
DEFAULT_NOISE_SCALE = 1.0
DEFAULT_MAX_SPEECH_ATTENUATION_DB = 3.0
SPLIT_SEED = 20261029
