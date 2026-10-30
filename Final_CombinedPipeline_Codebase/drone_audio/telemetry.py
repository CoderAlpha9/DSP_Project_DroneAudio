"""Observed sensor data only. All lookups are causal zero-order holds."""
from dataclasses import dataclass, field
import csv
from pathlib import Path
import numpy as np


def flag(value):
    return str(value).strip().lower() in {"1", "1.0", "true", "valid", "good"}


@dataclass
class SensorSeries:
    times: np.ndarray
    values: np.ndarray
    valid: np.ndarray | None = None
    units: str = ""
    frame: str = ""

    def __post_init__(self):
        self.times = np.asarray(self.times, float)
        self.values = np.asarray(self.values, float)
        if self.values.ndim == 1: self.values = self.values[:, None]
        if self.times.ndim != 1 or self.values.ndim != 2 or len(self.values) != len(self.times):
            raise ValueError("Sensor timestamps/values must be aligned arrays")
        if not np.isfinite(self.times).all() or np.any(np.diff(self.times) <= 0):
            raise ValueError("Sensor timestamps must be finite and strictly increasing")
        self.valid = np.ones(len(self.times), bool) if self.valid is None else np.asarray(self.valid, bool)
        if self.valid.shape != self.times.shape: raise ValueError("Sensor validity length mismatch")
        self.valid = self.valid & np.isfinite(self.values).all(axis=1)

    def at(self, time_s, max_age_s):
        i = int(np.searchsorted(self.times, time_s + 1e-12, side="right") - 1)
        if i < 0: return None, "leading_gap", None
        age = float(time_s-self.times[i])
        if age > max_age_s: return None, "stale", age
        if not self.valid[i]: return None, "invalid_observation", age
        return self.values[i].copy(), "valid", age


@dataclass
class ObservedTelemetry:
    rpm: SensorSeries | None = None
    pose: SensorSeries | None = None
    imu: SensorSeries | None = None
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        for name, units, width in [("rpm", "RPM", None), ("pose", "m", 3), ("imu", "m/s2", 3)]:
            s = getattr(self, name)
            if s is not None:
                if s.units != units: raise ValueError(f"{name} units must be explicitly {units}")
                if width is not None and s.values.shape[1] != width: raise ValueError(f"{name} needs {width} axes")
                if name == "pose" and s.frame != "ENU": raise ValueError("Pose must be explicitly ENU; NED conversion is not implicit")

    def at(self, time_s, max_age_s=.2):
        out = {}
        for name in ("rpm", "pose", "imu"):
            s = getattr(self, name)
            value, reason, age = s.at(time_s, max_age_s) if s is not None else (None, "missing", None)
            out[name] = value; out[name+"_status"] = reason; out[name+"_age_s"] = age
        return out


def _number(row, key):
    try: return float(row.get(key, "nan"))
    except (ValueError, TypeError): return float("nan")


def read_observed_csv(path, schema="batch1", time_origin_s=0.):
    """Only explicitly named observed schemas; no discovery of latent files."""
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows: return ObservedTelemetry(provenance={"path": str(path), "empty": True})
    columns = set(rows[0]); result = ObservedTelemetry(provenance={"path": str(Path(path).resolve()), "schema": schema, "time_origin_s": time_origin_s})
    if schema == "batch1":
        if "time_s" not in columns: raise ValueError("Observed CSV requires time_s")
        ts = np.array([_number(r, "time_s") for r in rows])-time_origin_s
        definitions = [
            ("pose", ["pose_enu_x_m", "pose_enu_y_m", "pose_enu_z_m"], "pose_valid", "m", "ENU"),
            ("imu", ["accel_body_x_m_s2", "accel_body_y_m_s2", "accel_body_z_m_s2"], "imu_valid", "m/s2", "body"),
        ]
        for name, keys, validity, units, frame in definitions:
            if set(keys).issubset(columns):
                if validity not in columns: raise ValueError(f"Observed {name} needs {validity}")
                setattr(result, name, SensorSeries(ts, [[_number(r,k) for k in keys] for r in rows], [flag(r[validity]) for r in rows], units, frame))
        rpm_keys = sorted(k for k in columns if k.startswith("rpm_") and k[4:].isdigit())
        if not rpm_keys and "rpm" in columns: rpm_keys = ["rpm"]
        if rpm_keys: result.rpm = SensorSeries(ts, [[_number(r,k) for k in rpm_keys] for r in rows], units="RPM")
    elif schema in {"batch2_rpm", "batch2_imu"}:
        if "sim_time_s" not in columns: raise ValueError("Batch2 observations require sim_time_s")
        ts = np.array([_number(r, "sim_time_s") for r in rows])-time_origin_s
        if schema.endswith("rpm"):
            keys = ["rpm_0", "rpm_1", "rpm_2", "rpm_3"]
            if not set(keys).issubset(columns): raise ValueError("RPM schema needs all four named motors")
            result.rpm = SensorSeries(ts, [[_number(r,k) for k in keys] for r in rows], units="RPM")
        else:
            keys = ["accel_x_m_s2", "accel_y_m_s2", "accel_z_m_s2"]
            if not set(keys).issubset(columns): raise ValueError("IMU schema needs three acceleration axes")
            result.imu = SensorSeries(ts, [[_number(r,k) for k in keys] for r in rows], units="m/s2", frame="body")
    else: raise ValueError(f"Unknown observed telemetry schema {schema}")
    result.__post_init__()
    return result


class ReliabilityMonitor:
    """Validity and freeze suspicion, not an actuator-fault classifier."""
    def __init__(self, config):
        self.c = config; self.last = None; self.changed_s = None

    def inspect(self, snapshot, time_s):
        rpm = snapshot["rpm"]
        reason = snapshot["rpm_status"]
        if rpm is None: return None, reason
        if np.any(rpm <= self.c.rpm_min): return None, "stopped_or_below_range"
        if np.any(rpm > self.c.rpm_max): return None, "above_range"
        if self.last is None or self.last.shape != rpm.shape or np.max(np.abs(rpm-self.last)) > .001:
            self.changed_s = time_s
        self.last = rpm.copy()
        if time_s-self.changed_s >= self.c.freeze_after_s: return None, "constant_value_suspected_freeze"
        return rpm, "valid"
