from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import CATALOG_FILE, EXTRACTED_DIR, RAW_DIR, SPLIT_FILE, SPLIT_SEED


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    scenario: str
    seed: int
    sample_rate_hz: int
    duration_seconds: float
    speech_valid_for_metrics: bool
    flight_success_verified: bool
    acoustic_variant: str
    fault_type: str
    source_zip: str
    run_dir: str


def _wanted_member(relative: str) -> bool:
    if relative in {"manifest.json", "qc.json", "rpm_aligned.csv", "imu_aligned.csv"}:
        return True
    if relative.startswith("audio/") and relative.endswith(".wav"):
        return True
    if relative.startswith("audio_snr_sweep/") and relative.endswith(".wav"):
        return True
    return False


def prepare_from_zips(raw_dir: Path = RAW_DIR, extracted_dir: Path = EXTRACTED_DIR) -> list[RunRecord]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)
    zip_paths = sorted(raw_dir.glob("*.zip"))
    if not zip_paths:
        raise FileNotFoundError(
            f"No ZIP files found in {raw_dir}. Copy S3.zip and S6.zip there first."
        )

    records: list[RunRecord] = []
    seen_run_ids: set[str] = set()

    for zpath in zip_paths:
        with zipfile.ZipFile(zpath) as zf:
            manifest_members = sorted(n for n in zf.namelist() if n.endswith("/manifest.json"))
            for manifest_member in manifest_members:
                manifest = json.loads(zf.read(manifest_member))
                run_prefix = manifest_member.rsplit("/", 1)[0]
                run_id = run_prefix.split("/")[-1]
                scenario = str(manifest.get("scenario", "UNKNOWN"))
                if run_id in seen_run_ids:
                    suffix = hashlib.sha1(f"{zpath.name}:{run_prefix}".encode()).hexdigest()[:8]
                    run_id = f"{run_id}_{suffix}"
                seen_run_ids.add(run_id)

                target = extracted_dir / "runs" / scenario / run_id
                target.mkdir(parents=True, exist_ok=True)

                for member in zf.namelist():
                    if not member.startswith(run_prefix + "/") or member.endswith("/"):
                        continue
                    rel = member[len(run_prefix) + 1 :]
                    rel_path = Path(rel)
                    if rel_path.is_absolute() or ".." in rel_path.parts:
                        raise ValueError(f"Unsafe ZIP member path: {member}")
                    if not _wanted_member(rel):
                        continue
                    dest = target / rel_path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as src, open(dest, "wb") as out:
                        shutil.copyfileobj(src, out)

                fault = manifest.get("fault") or {}
                rec = RunRecord(
                    run_id=run_id,
                    scenario=scenario,
                    seed=int(manifest.get("seed", -1)),
                    sample_rate_hz=int(manifest.get("sample_rate_hz", 0)),
                    duration_seconds=float(manifest.get("duration_seconds", 0.0)),
                    speech_valid_for_metrics=bool(manifest.get("speech_valid_for_metrics", False)),
                    flight_success_verified=bool(manifest.get("flight_success_verified", False)),
                    acoustic_variant=str(fault.get("acoustic_variant") or "unknown"),
                    fault_type=str(manifest.get("fault_type") or fault.get("fault_provenance") or "none"),
                    source_zip=zpath.name,
                    run_dir=str(target.relative_to(extracted_dir)),
                )
                records.append(rec)

    df = pd.DataFrame([asdict(r) for r in records]).sort_values(["scenario", "run_id"])
    CATALOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CATALOG_FILE, index=False)
    make_split_file(records)
    return records


def _split_counts(n: int) -> tuple[int, int, int]:
    if n < 10:
        raise ValueError("At least 10 metric-valid independent runs are required for an 80/10/10 run-level split.")
    n_test = max(1, int(round(0.10 * n)))
    n_val = max(1, int(round(0.10 * n)))
    n_dev = n - n_val - n_test
    return n_dev, n_val, n_test


def _pick_diverse(items: list[RunRecord], count: int, rng: random.Random) -> list[RunRecord]:
    pool = items[:]
    rng.shuffle(pool)
    selected: list[RunRecord] = []
    used_scenarios: set[str] = set()
    used_seeds: set[int] = set()

    while pool and len(selected) < count:
        best_i = max(
            range(len(pool)),
            key=lambda i: (
                int(pool[i].scenario not in used_scenarios) + int(pool[i].seed not in used_seeds),
                rng.random(),
            ),
        )
        item = pool.pop(best_i)
        selected.append(item)
        used_scenarios.add(item.scenario)
        used_seeds.add(item.seed)
    return selected


def make_split_file(records: Iterable[RunRecord], split_seed: int = SPLIT_SEED) -> pd.DataFrame:
    valid = [r for r in records if r.speech_valid_for_metrics and r.flight_success_verified]
    n_dev, n_val, n_test = _split_counts(len(valid))
    rng = random.Random(split_seed)

    test = _pick_diverse(valid, n_test, rng)
    test_ids = {r.run_id for r in test}
    remaining = [r for r in valid if r.run_id not in test_ids]
    val = _pick_diverse(remaining, n_val, rng)
    val_ids = {r.run_id for r in val}

    rows = []
    for r in records:
        if not (r.speech_valid_for_metrics and r.flight_success_verified):
            split = "excluded"
        elif r.run_id in test_ids:
            split = "test"
        elif r.run_id in val_ids:
            split = "validation"
        else:
            split = "development"
        rows.append({"run_id": r.run_id, "scenario": r.scenario, "seed": r.seed, "split": split})

    df = pd.DataFrame(rows).sort_values(["split", "scenario", "run_id"])
    df.to_csv(SPLIT_FILE, index=False)
    counts = df[df["split"] != "excluded"]["split"].value_counts().to_dict()
    if counts.get("development", 0) != n_dev or counts.get("validation", 0) != n_val or counts.get("test", 0) != n_test:
        raise RuntimeError("Split construction failed its internal count check.")
    return df


def load_catalog() -> pd.DataFrame:
    if not CATALOG_FILE.exists():
        raise FileNotFoundError("Catalog not found. Run: python scripts/prepare_data.py")
    df = pd.read_csv(CATALOG_FILE)
    splits = pd.read_csv(SPLIT_FILE)
    return df.merge(splits[["run_id", "split"]], on="run_id", how="left")


def run_path(row: pd.Series) -> Path:
    return EXTRACTED_DIR / str(row["run_dir"])


def available_conditions(run_dir: Path) -> list[dict]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    base_snr = float(manifest.get("snr", {}).get("target_snr_db", -5.0))
    conditions = [
        {
            "label": f"{base_snr:g} dB",
            "key": "base",
            "target_snr_db": base_snr,
            "path": run_dir / "audio",
        }
    ]
    sweep = manifest.get("paired_snr_sweep") or {}
    for key, info in sweep.items():
        target = float((info.get("snr") or {}).get("target_snr_db", 0.0))
        p = run_dir / str(info.get("path", f"audio_snr_sweep/{key}"))
        conditions.append({"label": f"{target:g} dB", "key": key, "target_snr_db": target, "path": p})
    conditions.sort(key=lambda x: x["target_snr_db"])
    return conditions
