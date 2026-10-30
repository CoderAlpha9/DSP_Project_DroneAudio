# Target-only amplitude experiments and outputs

**Project:** Sensor-Assisted Adaptive Drone Noise suppression with Motion Artefact mitigation and Actuator Fault awareness.

## What was selected

Internal development selection chose:

- Gain cap **18 dB**.
- Geometry weight **1.0** when pose is accepted.
- Gain smoothing **0.04 s**, nominal slew **36 dB/s**.
- Hold the last accepted geometry request for up to **3.5 s** during missing/implausible pose; subsequently return toward unity.
- Sample-peak ceiling **0.95**.
- One **20 ms** buffered block at **16 kHz mono**.

The search included energy/geometry fusion, but the selected setting is effectively **pose-guided**, with hold fallback. Audio-only energy control remains a comparison/explicit alternative, not the selected fallback. If geometry was never available, the hold policy has no valid request to retain and uses unity gain. Raw IMU is not double-integrated into an assumed source distance.

In this controlled target-only protocol, every ten-second case is treated as one event. Ordinary word pauses do not reset it. In the reusable API, an upstream event state can close/reset the reference, or a simple RMS/hangover detector can be used instead.

## Test-only performance

The overall split is the same 60 development / 15 test (80/20). Internally, 40 calibration cases from rec1/rec3 rank 54 fixed configurations; 20 validation cases from rec2 choose among three finalists. **Development ranking values and performance tables are not exported.** Numeric dataset performance tables contain only the rec4/rec5 unique test partition.

The selected setting was frozen before this protocol's test scoring. The test recording had been used in the earlier noisy study, so this is a reused held-out benchmark, not a new blind independent validation.

On the 15 unique test cases:

| Restoration target | No gain: active level MAE | Selected: active level MAE |
|---|---:|---:|
| Closest-so-far event reference | 4.24 dB | 1.23 dB |
| Known fixed 2 m reference | 15.42 dB | 3.92 dB |

Primary waveform RMSE falls from approximately 0.0050 to 0.0018. These are conditional results on one independent test recording, not population or field-performance claims. Sensor failures and missing calibration still limit recovery.

The PDF report covers derivations, methods, baselines, selection, metrics, frequency/time analysis, ablations, inference, runtime and limitations. It has 14 single-column pages including references. The original project title is retained.

## Package layout

```text
report/target_only_amplitude_report.pdf   Detailed report
pipeline/core.py                         Final causal target-only gain controller
pipeline/__main__.py                     WAV/observed-telemetry CLI
configs/selected.json                    Frozen selected configuration
configs/selection_protocol.json          Grid/partition/selection procedure, no development scores
configs/freeze_record.json               Selected-config and controller hashes
experiments/run.py                       Development selection, test-only comparisons, all-case export
experiments/analyze.py                   Test-only summaries, frame metrics, figures and audit
experiments/report.py                    Rebuild PDF from recorded results
tests/test_pipeline.py                   Automated functional/physics checks
metrics/test_per_case.csv                270 method/policy/test-case comparisons
metrics/test_frame_metrics.csv           15,000 selected-output test frame records
metrics/test_summary_mean.csv            Macro test means
metrics/test_summary_std.csv             Across-condition spread, not speaker uncertainty
metrics/test_sensor_probes.csv           Frozen-setting sensor perturbations on test inputs only
metrics/test_runtime_precision.json      Runtime/precision on a named test case only
analysis/filter_analysis.csv             Analytic envelope-filter characterization
figures/                                Diagnostic scientific figures
data/cases/                             Only target/reference audio, observed telemetry and true trajectory
data/manifest.csv                        75 unique cases and source labels
data/source_groups.json                  Original rec labels and duplicate grouping
data/input_integrity.csv                 Bit-exact input hashes
listening/index.html                     Local all-case audio player
listening/audio_index.csv                All rec/case/input/output mapping
listening/rec1 ... rec5/                  Operated audio for both reference policies
audio_logs/                             Controller traces and gain arrays for every unique case
```

All-case logs contain control state for reproduction/listening alignment, not development performance-error tables. `analysis/filter_analysis.csv` characterizes analytic filters rather than scoring development recordings.

## Install and reproduce

Use Python 3.10 or newer. FFmpeg, a GPU, downloaded models and network access are **not required** once Python dependencies are installed. Everything needed to rerun the audio experiment is in this ZIP.

Windows PowerShell, from the extracted `amplitude_target_only` folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m unittest discover -s tests -v
python experiments/run.py --resume
python experiments/analyze.py
python experiments/report.py
```

On Linux/macOS activate with `source .venv/bin/activate`. `--resume` uses the delivered frozen configuration. Omit it to repeat the internal development selection, still exporting performance tables only on test. Reruns intentionally overwrite generated outputs/metrics; retain a copy if comparing versions. Dependency versions are in `requirements-lock.txt`.

## Process another target-only WAV

Event-relative output using available observed geometry:

```powershell
python -m pipeline --input target_at_drone.wav --output restored_event.wav --telemetry telemetry_observed.csv --source-position 0 0 0 --trace trace.csv
```

Optional known-distance calibration:

```powershell
python -m pipeline --input target_at_drone.wav --output restored_2m.wav --telemetry telemetry_observed.csv --source-position 0 0 0 --fixed-reference-m 2
```

Use the source origin only if it is actually known in the telemetry frame. Input must be finite mono PCM16 or float WAV at 16 kHz; resample explicitly beforehand. The CLI invalidates observations older than 0.2 s and uses no future telemetry sample. Pose CSV fields follow the supplied dataset: `time_s, pose_valid, imu_valid, pose_enu_x_m, pose_enu_y_m, pose_enu_z_m, accel_body_x_m_s2, accel_body_y_m_s2, accel_body_z_m_s2`. Times must strictly increase.

The core can also be integrated directly:

```python
import json
from pipeline import Config, AmplitudePipeline

cfg = Config(**json.load(open("configs/selected.json")))
controller = AmplitudePipeline(cfg)
output, sample_gain, log = controller.process_block(
    target_audio_block,             # 320 samples at 16 kHz
    pose=observed_position_enu_m,
    pose_valid=position_is_available,
    source_position=known_source_enu_m,
    imu_force=observed_specific_force_body_m_s2,
    imu_valid=imu_is_available,
    event_active=True,              # or upstream event state
)
```

Load `configs/selected.json` for the selected configuration; bare `Config()` supplies generic defaults. A complete dataset clip is one event in this experiment. Event closure resets the event reference without pretending to reset any unrelated rotor/noise model.

## Methods, baselines and caveats

The physical law is pressure amplitude proportional to **1/r**, not 1/r². The primary reference is the minimum observed range so far, while the optional absolute mode uses the generator's known 2 m reference. Their targets differ. Audio-only input cannot identify absolute source level or distinguish intentional quiet speech from greater distance without additional assumptions.

Two post-2015 non-neural gain-control cores are compared: Tisserand/Berviller feedback AGC (2016, DOI 10.1049/el.2016.1398) and a one-band power-law/envelope specialization from Corey/Singer (2021, DOI 10.1121/10.0005314). Common gain/peak bounds and task-specific level settings are documented adaptations; original-paper benchmark results, multiband hearing-aid systems and hardware are not reproduced. Neither baseline adds or suppresses noise.

Reconstruction SNR is waveform-error fidelity, not a measured target/drone-noise ratio. It is capped at 80 dB for summaries; raw values are retained. Exact no-gain controls can otherwise produce arbitrary huge numerical SNRs, so level error and RMSE are the main comparisons. No SNR gate is evaluated in this target-only study. It should be reintroduced and validated separately when integrating with noisy recordings.

The true trajectory and reference waveform are evaluation data only; the deployed algorithm uses observed pose and a declared source location. The test activity mask comes from reference RMS for scoring only. Pose validity does not reveal drift or freezing. Holding a last gain is continuity, not proof of unchanged range. Large fades can exceed the gain cap; nothing here recovers missing spectral information.

The emergency peak guard protects sample amplitudes but does not measure intersample true peak, and may create abrupt boundaries under overload. The original speech may contain room noise or recording defects. No independent listening panel, intelligibility study, new real-drone field validation or general guarantee of recovery is claimed. Original source permission constraints remain unchanged.
