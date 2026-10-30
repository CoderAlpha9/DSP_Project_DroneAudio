"""Validated application settings. No references or ground truth are accepted."""
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import math
import yaml


def _finite(value, name):
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass
class EnhancementConfig:
    enabled: bool = True
    mode: str = "audio_only"
    n_fft: int = 1024
    hop: int = 256
    noise_alpha: float = .90
    dd_alpha: float = .96
    beta: float = 1.0
    floor: float = .04
    rpm_weight: float = .1
    rpm_power: float = 2.
    psd_cap: float = 4.
    blades: int = 2
    harmonics: int = 8
    order_width_hz: float = 18.
    order_attenuation: float = .15
    startup_s: float = .5
    motion_enabled: bool = False
    motion_mu: float = .01
    motion_cutoff_hz: float = 20.

    def validate(self):
        if self.mode not in {"audio_only", "fixed_rpm", "bounded_rpm", "correlation", "ridge", "order"}:
            raise ValueError(f"Unknown enhancement mode: {self.mode}")
        if self.n_fft not in (512, 1024) or self.hop not in (128, 256) or self.n_fft % self.hop:
            raise ValueError("Use a 512/1024-point transform with a dividing 128/256 hop")
        for name in ("noise_alpha", "dd_alpha"):
            value = getattr(self, name); _finite(value, name)
            if not 0 <= value < 1: raise ValueError(f"{name} must satisfy 0 <= alpha < 1")
        for name in ("beta", "psd_cap", "rpm_power", "startup_s", "order_width_hz", "motion_mu", "motion_cutoff_hz"):
            _finite(getattr(self, name), name)
        if self.beta <= 0 or self.psd_cap < 1 or self.startup_s < 0 or not 0 <= self.rpm_power <= 6:
            raise ValueError("Invalid enhancement scale/cap/startup/power")
        if not 0 < self.floor <= 1 or not 0 <= self.rpm_weight <= 1:
            raise ValueError("Invalid gain floor or RPM weight")
        if not 0 < self.order_attenuation <= 1 or self.blades < 1 or self.harmonics < 1 or self.order_width_hz <= 0:
            raise ValueError("Invalid harmonic suppressor settings")
        if not 0 < self.motion_mu <= .1 or not 1 <= self.motion_cutoff_hz <= 20:
            raise ValueError("Motion adaptation must be bounded and restricted to <=20 Hz")


@dataclass
class AmplitudeConfig:
    enabled: bool = True
    max_gain_db: float = 18.
    smooth_s: float = .04
    max_slew_db_s: float = 36.
    hold_s: float = 3.5
    peak_ceiling: float = .95
    reference_mode: str = "session"
    reference_distance_m: float | None = None
    source_position_enu_m: tuple[float, float, float] | None = None
    source_associated: bool = False
    speech_only: bool = True
    event_control: bool = True
    snr_guard_enabled: bool = True
    snr_guard_low_db: float = 0.
    snr_guard_high_db: float = 10.
    max_pose_speed_m_s: float = 14.

    def validate(self):
        for name in ("max_gain_db", "smooth_s", "max_slew_db_s", "hold_s", "peak_ceiling", "snr_guard_low_db", "snr_guard_high_db", "max_pose_speed_m_s"):
            _finite(getattr(self, name), name)
        if self.max_gain_db < 0 or self.smooth_s <= 0 or self.max_slew_db_s <= 0 or self.hold_s < 0 or self.max_pose_speed_m_s <= 0:
            raise ValueError("Invalid amplitude bounds/time constants")
        if not 0 < self.peak_ceiling < 1: raise ValueError("peak_ceiling must be in (0,1)")
        if self.reference_mode not in {"session", "fixed"}: raise ValueError("reference_mode must be session or fixed")
        if self.reference_distance_m is not None:
            _finite(self.reference_distance_m, "reference_distance_m")
            if self.reference_distance_m <= 0: raise ValueError("reference distance must be positive")
        if self.reference_mode == "fixed" and self.reference_distance_m is None:
            raise ValueError("Fixed amplitude reference requires explicit calibrated distance")
        if self.snr_guard_high_db <= self.snr_guard_low_db: raise ValueError("Invalid SNR guard interval")
        if self.source_position_enu_m is not None:
            if len(self.source_position_enu_m) != 3: raise ValueError("Source position must have three ENU coordinates")
            for v in self.source_position_enu_m: _finite(v, "source coordinate")


@dataclass
class PipelineConfig:
    sample_rate: int = 16000
    detector_enabled: bool = True
    telemetry_max_age_s: float = .2
    rpm_min: float = 100.
    rpm_max: float = 20000.
    freeze_after_s: float = 2.
    # Only an application/protocol may declare this. None means unknown startup.
    speech_free_prefix_s: float | None = None
    enhancement: EnhancementConfig = field(default_factory=EnhancementConfig)
    amplitude: AmplitudeConfig = field(default_factory=AmplitudeConfig)

    def validate(self):
        if self.sample_rate != 16000: raise ValueError("Internal rate is 16000 Hz; explicitly resample at file input")
        for name in ("telemetry_max_age_s", "rpm_min", "rpm_max", "freeze_after_s"):
            _finite(getattr(self, name), name)
        if self.telemetry_max_age_s <= 0 or not 0 <= self.rpm_min < self.rpm_max or self.freeze_after_s <= 0:
            raise ValueError("Invalid sensor age/range/freeze settings")
        if self.speech_free_prefix_s is not None:
            _finite(self.speech_free_prefix_s, "speech_free_prefix_s")
            if self.speech_free_prefix_s < self.enhancement.n_fft/self.sample_rate:
                raise ValueError("Declared prefix must include at least one complete analysis window")
        self.enhancement.validate(); self.amplitude.validate()
        return self

    def to_dict(self): return asdict(self)

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_dict(cls, value):
        d = dict(value)
        d["enhancement"] = EnhancementConfig(**d.get("enhancement", {}))
        d["amplitude"] = AmplitudeConfig(**d.get("amplitude", {}))
        return cls(**d).validate()


def load_config(path=None):
    if path is None: return PipelineConfig().validate()
    p = Path(path).expanduser().resolve()
    return PipelineConfig.from_dict(yaml.safe_load(p.read_text(encoding="utf-8")) or {})


def save_config(config, path):
    Path(path).write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")
