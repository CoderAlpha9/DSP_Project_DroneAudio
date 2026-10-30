# Drone Audio DSP

**Group 4:** ED24B051, ED24B034, ED24B027, ED24B011

This project combines acoustic event detection, drone-noise suppression and pose-guided amplitude restoration. It processes microphone audio with optional rotor RPM, IMU and position telemetry.

## Processing pipeline

1. **Event detection:** Identifies acoustic activity and provides a separate speech flag from the microphone signal.
2. **Noise enhancement:** Uses recursive noise estimation and decision-directed Wiener filtering, with configurable RPM-based suppression and telemetry reliability controls.
3. **Amplitude restoration:** Applies bounded gain using observed position and a known source location, controlled by speech activity and source association.

The default configuration uses audio-based enhancement. RPM-assisted modes and motion-reference cancellation are available through separate configurations. Processing preserves continuous audio and exports event intervals and diagnostic traces.

## Repository structure

| Folder | Contents |
|---|---|
| `Final_CombinedPipeline_Codebase/` | Combined Python package, configurations, tests and processing outputs |
| `Experiments/` | Individual studies for event detection, filter design, amplitude equalization and rotor-sensor reliability |
| `Data/` | Audio and telemetry used by the experiments and combined pipeline |

## Setup and usage

Use Python 3.10 or newer. From the repository root, run these commands in PowerShell:

```powershell
cd Final_CombinedPipeline_Codebase
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[analysis,test]"
python -m drone_audio doctor
```

Process a mono WAV recording at 16 kHz:

```powershell
python -m drone_audio process --input "path/to/recording.wav" --config configs/default.yaml --output runs/my_recording
```

Each run saves enhanced audio, final audio, event intervals and time-aligned detector, enhancement and gain traces. Use a new output directory for each run.

Run the tests from the same folder:

```powershell
python -m pytest -q --basetemp=runs/test_temp
```

See the [combined pipeline README](Final_CombinedPipeline_Codebase/README.md) for telemetry inputs, configuration options, batch processing and the Python API.
