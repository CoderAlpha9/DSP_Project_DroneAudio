"""Manifest adapters. Inference input objects contain no evaluation paths."""
from dataclasses import dataclass, field, asdict
from pathlib import Path
import csv
import hashlib
import json
from .telemetry import ObservedTelemetry, read_observed_csv


@dataclass(frozen=True)
class InferenceInput:
    microphone: Path
    telemetry_schema: str
    observed_csv: Path | None = None
    observed_imu_csv: Path | None = None
    time_origin_s: float = 0.
    sample_rate: int = 16000
    source_position_enu_m: tuple | None = None
    source_associated: bool = False
    speech_free_prefix_s: float | None = None

    def load_telemetry(self):
        if self.observed_csv is None: result = ObservedTelemetry()
        else: result = read_observed_csv(self.observed_csv, self.telemetry_schema, self.time_origin_s)
        if self.observed_imu_csv is not None:
            result.imu = read_observed_csv(self.observed_imu_csv, "batch2_imu", self.time_origin_s).imu
        result.__post_init__()
        return result


@dataclass(frozen=True)
class EvaluationInput:
    target_at_microphone: Path
    source_reference: Path | None = None
    noise: Path | None = None
    true_trajectory: Path | None = None
    source_offset_samples: int = 0
    fixed_reference_m: float | None = None
    active_interval_s: tuple | None = None
    event_labels: Path | None = None


@dataclass
class Sample:
    id: str
    batch: str
    parent_id: str
    source_id: str
    split: str
    inference: InferenceInput
    evaluation: EvaluationInput
    condition: str = "base"
    metadata: dict = field(default_factory=dict)

    def record(self):
        return json.loads(json.dumps(asdict(self), default=str))


def _csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f: return list(csv.DictReader(f))


def discover(data_root, *, include_variants=True, include_event_probes=False):
    root = Path(data_root).expanduser().resolve(); samples = []; warnings = []
    b1 = root/"Data_Batch1"
    if (b1/"manifest.csv").exists():
        for row in _csv(b1/"manifest.csv"):
            p = b1/row["relative_path"]
            meta = json.loads((p/"metadata.json").read_text())
            source = row["source_group"]
            split = "test" if row["split"] == "test" else ("validation" if source == "c414202e8f9c" else "development")
            samples.append(Sample(row["scenario_id"], "batch1", row["scenario_id"], source, split,
                InferenceInput(p/"mixture.wav", "batch1", p/"telemetry_observed.csv", source_position_enu_m=tuple(meta["source_position_enu_m"]), source_associated=True),
                EvaluationInput(p/"target_at_drone.wav", p/"reference.wav", p/"noise.wav", p/"ground_truth_trajectory.csv",
                                int(row["source_offset_samples"]), float(meta["reference_distance_m"])),
                metadata={"profile": row["profile"], "sensor_condition": row["sensor_condition"], "original_split": row["split"], "reused_test": True}))
    b2 = root/"Data_Batch2"; manifests = sorted((b2/"runs").glob("*/*/manifest.json")); selected = {}
    for path in manifests:
        m = json.loads(path.read_text())
        if m.get("speech_valid_for_metrics") and m.get("flight_success_verified"):
            selected[m["scenario"], int(m["seed"])] = (path, m)
    if (b2/"index.csv").exists() and len(_csv(b2/"index.csv")) != len(manifests):
        warnings.append(f"Batch2 index has {len(_csv(b2/'index.csv'))} rows but {len(manifests)} manifests; manifests used, latest valid per scenario/seed")
    for (scenario, seed), (path, m) in sorted(selected.items()):
        p = path.parent; exp = m["config"]["experiment"]
        split = "test" if scenario == "S7" or seed == 303 else ("development" if seed == 101 else "validation")
        conditions = [("base", p/"audio")]
        if include_variants:
            conditions += [(name, p/info["path"]) for name, info in sorted(m.get("paired_snr_sweep", {}).items())]
        for condition, audio in conditions:
            if not (audio/"mic_noisy.wav").exists():
                warnings.append(f"Missing microphone condition: {audio}"); continue
            observed = p/"rpm_sensor_observed.csv"
            samples.append(Sample(p.name+"__"+condition, "batch2", p.name, m.get("speech_sha256", m.get("speech_speaker_id", "unknown")), split,
                InferenceInput(audio/"mic_noisy.wav", "batch2_rpm", observed if observed.exists() else None,
                               p/"imu_aligned.csv" if (p/"imu_aligned.csv").exists() else None,
                               float(m["audio_t0_sim_ns"])/1e9, int(m["sample_rate_hz"]),
                               speech_free_prefix_s=min(.8, exp["silence_start_seconds"]) if exp.get("silence_start_seconds", 0) >= .8 else None),
                EvaluationInput(audio/"clean_speech.wav", noise=audio/"drone_noise.wav" if (audio/"drone_noise.wav").exists() else None,
                                active_interval_s=tuple(exp.get("active_interval_seconds", (2.,14.)))), condition,
                {"scenario": scenario, "seed": seed, "reused_test": True, "pose": "unavailable", "source_shared_across_partitions": True,
                 "manifest": str(path), "provenance": m.get("provenance_class"), "capture_sha256": m.get("sim_capture_sha256")}))
    if include_event_probes:
        event_root = root.parent/"Experiments/EventDetection_Experiments/data"
        manifest = event_root/"manifest.csv"
        if manifest.exists():
            parents = {s.parent_id:s for s in samples if s.batch == "batch1"}
            for row in _csv(manifest):
                if row["kind"] == "speech": continue  # Original mixture is already included.
                parent = parents[row["base_id"]]; p = event_root/"cases"/row["case_id"]
                # See source generation: input.wav is the declared mixed observation.
                samples.append(Sample(row["case_id"], "event_probe", parent.parent_id, parent.source_id, parent.split,
                    InferenceInput(p/"input.wav", "batch1", parent.inference.observed_csv),
                    EvaluationInput(p/"truth.npz", event_labels=p/"truth.npz"), row["kind"],
                    {"kind": row["kind"], "synthetic_descendant": True}))
    audit = split_audit(samples)
    if audit["conflicting_parent_splits"]: raise ValueError(f"Conflicting parent splits: {audit['conflicting_parent_splits']}")
    return samples, {"warnings": warnings, **audit, "excluded_batch2_manifests": len(manifests)-len(selected)}


def split_audit(samples):
    parents = {}; sources = {}; captures = {}
    for sample in samples:
        parents.setdefault(sample.parent_id,set()).add(sample.split)
        sources.setdefault(sample.source_id,set()).add(sample.split)
        cap = sample.metadata.get("capture_sha256")
        if cap: captures.setdefault(cap,set()).add(sample.split)
    return {"conflicting_parent_splits": {k:sorted(v) for k,v in parents.items() if len(v)>1},
            "overlapping_source_splits": {k:sorted(v) for k,v in sources.items() if len(v)>1},
            "conflicting_capture_splits": {k:sorted(v) for k,v in captures.items() if len(v)>1},
            "source_overlap_note": "Batch2 reuses one speech file across capture-level partitions; this is not speaker-disjoint validation. Batch1 rec4/rec5 share one source ID."}


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024), b""): h.update(block)
    return h.hexdigest()


def input_overlap_audit(samples):
    """Explicit audit only; microphone bytes, not reference files."""
    hashes = {}; conflicts = []
    for s in samples:
        digest = hash_file(s.inference.microphone)
        hashes.setdefault(digest, []).append({"id":s.id,"parent":s.parent_id,"split":s.split})
    duplicates = {k:v for k,v in hashes.items() if len(v)>1}
    for h, group in duplicates.items():
        if len({x['split'] for x in group})>1: conflicts.append(h)
    return {"duplicate_microphones":duplicates,"conflicting_microphone_splits":conflicts}
