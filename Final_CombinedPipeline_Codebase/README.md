# Combined drone-audio DSP pipeline

An implemented CPU pipeline for **Sensor-Assisted Adaptive Drone Noise suppression with Motion Artefact mitigation and Actuator Fault awareness**. It analyzes the original microphone signal for acoustic activity and speech, enhances continuous audio, then applies bounded pose-guided gain when a source is explicitly associated. Sensor anomaly traces are not actuator-fault diagnoses.

## Windows setup

Run from this folder, with Python 3.10 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[analysis,test]"
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python -m drone_audio doctor
python -m pytest -q --basetemp=runs/test_temp
```

If activation is unavailable, invoke `.\.venv\Scripts\python.exe` directly. The actual build was verified on Windows/Python 3.13.1. `[analysis]` supplies plotting and STOI/ESTOI; inference alone needs NumPy, SciPy and PyYAML. Install with `pip install -e .` for inference only. Exact executed versions are saved in run environment records. Historical neural executables and PESQ are not installed.

## Discover and process

```powershell
python -m drone_audio discover --data-root ../Data --output runs/discovery_new --hash-audit --event-probes
python -m drone_audio process --input "../Data/Data_Batch1/test/test_01_00_recede_slow/mixture.wav" --config configs/default.yaml --output runs/single_new
python -m drone_audio batch --data-root ../Data --config configs/default.yaml --split test --limit 2 --output runs/batch_new
python -m drone_audio evaluate --data-root ../Data --config configs/frozen.yaml --output runs/evaluation_new --plots
```

Every run directory must be new. Relative paths resolve against the current working directory. After editable installation, commands also work elsewhere with absolute `--input`, `--config`, `--data-root`, and `--output` paths; quote paths containing spaces. CLI-generated outputs must remain under this package directory. Data and historical experiments are read-only and are not copied.

The simple single-file example intentionally has no geometry and therefore no source-specific amplification. To declare an associated source and observed ENU pose:

```powershell
python -m drone_audio process --input "../Data/Data_Batch1/test/test_01_00_recede_slow/mixture.wav" --telemetry "../Data/Data_Batch1/test/test_01_00_recede_slow/telemetry_observed.csv" --source-position 0 0 0 --source-associated --config configs/frozen.yaml --clips --pre-roll-s 0.1 --post-roll-s 0.1 --output runs/associated_new
```

Use an origin source only when it is actually known in the pose coordinate system. `--fixed-reference-m 2` is explicit calibration, not a universal assumption. `--speech-free-prefix-s 0.8` may be supplied only for a justified noise-only prefix. Without that declaration, startup uses a conservative microphone estimate and limits initial attenuation; relative-RPM template modes abstain without a template calibration. `--resample` and `--downmix` explicitly authorize recorded audio conversion; the program otherwise rejects unexpected rates/channels.

Dataset adapters supply only declared application metadata: Batch1's source position; Batch2's documented noise-only prefix. Batch2 has no source pose/geometry, so it does not receive range compensation. Only S7 has observed per-motor RPM files; other captures' simulator-derived RPM is never substituted.

## Outputs

Each processed case has `enhanced.wav` (before amplitude), `final.wav`, `detector.csv`, `enhancement.csv`, `amplitude.csv`, `events.json`, and `execution.json`. WAVs are float32 at their original scale; no peak normalization is applied. The enabled amplitude stage uses its traced sample-peak limiter. Optional clips contain original microphone samples from an explicit file buffer, not denoised or retroactively detected audio.

Evaluation adds `per_case.csv`, `per_parent.csv`, `summary.csv`, `paired_vs_audio.csv`, a frozen configuration, environment, split audit and execution ledger. Metrics distinguish pre-amplitude fidelity to the microphone target from post-amplitude fidelity to the declared range-restoration goal. No exact component-SNR claim is made. Figures are generated from the new run, not copied from historical results.

## Python API

```python
import numpy as np
from drone_audio import DronePipeline, load_config
from drone_audio.telemetry import read_observed_csv

cfg = load_config("configs/frozen.yaml")
telemetry = read_observed_csv("telemetry_observed.csv", "batch1")
cfg.amplitude.source_position_enu_m = (0., 0., 0.)  # only if known
cfg.amplitude.source_associated = True
processor = DronePipeline(cfg, telemetry=telemetry)
ready = []
for chunk in microphone_chunks:  # arbitrary finite mono chunks at 16 kHz
    ready.append(processor.process_block(chunk))
ready.append(processor.flush())
output = np.concatenate(ready)  # exactly the input sample count
# processor.new_source_session(...) at an emitted 20 ms boundary changes
# only the source anchor; processor.reset(...) resets the entire recording.
```

The `process_audio` wrapper returns the processor, including `audio`, `enhanced`, `sample_gain`, logs and timing. The object keeps recording traces/audio for export; it is stateful and online-compatible but not a fixed-memory embedded implementation. It never opens dataset references. Stream owners can supply their own `ObservedTelemetry` / `SensorSeries` objects; all lookups hold past samples without future interpolation.
