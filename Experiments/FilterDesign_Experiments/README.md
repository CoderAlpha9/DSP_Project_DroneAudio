# Drone Audio DSP Experiments

This package contains the experiments, DSP algorithms, evaluation code, parameter sweeps, and reproducibility tests for the drone audio noise suppression study. It intentionally contains no website or web server code.

## Objective

The experiment suite compares multiple methods for recovering a desired audio signal from drone generated acoustic and vibration noise. Every method is evaluated on the same prepared conditions so that SNR improvement, SI-SDR, correlation, spectral distortion, and runtime can be compared consistently.

The suite also supports arbitrary cascades of methods, for example RPM order tracking followed by Wiener filtering.

## Methods included

| # | Method | Pipeline name | Main information used |
|---|---|---|---|
| 1 | Spectral Subtraction | `spectral_sub` | Known noise stem in controlled experiment |
| 2 | Wiener | `wiener` | Known noise stem in controlled experiment |
| 3 | MMSE-LSA | `mmse_lsa` | Known noise stem in controlled experiment |
| 4 | Spectral Kalman | `spectral_kalman` | Known noise stem in controlled experiment |
| 5 | NLMS | `imu_nlms` | IMU reference |
| 6 | RLS | `imu_rls` | IMU reference |
| 7 | Adaptive Harmonic / Comb | `adaptive_comb` | RPM telemetry |
| 8 | RPM Order Tracking | `rpm_order` | Per-rotor RPM telemetry |
| 9 | Wavelet Denoising | `haar` | No auxiliary sensor required |
| 10 | NMF | `nmf` | Semi-supervised spectral decomposition |
| 11 | DeepFilterNet / RNNoise | `neural` | Optional external neural runtime |

The first four controlled spectral baselines can use the known drone-noise stem. They are therefore treated as oracle-assisted baselines. Sensor-based and blind methods are tagged separately during benchmarking.

## Dataset placement

Raw dataset ZIP chunks are intentionally not bundled with this code submission.

Place the provided dataset ZIP files in:

```text
data/raw/
    S3.zip
    S6.zip
```

The ZIP filenames are not treated as scenario labels. Run metadata inside the archives is used to discover the actual runs and scenarios.

Do not manually extract the ZIP files.

## Environment setup

Recommended Python version: Python 3.11.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Or run:

```powershell
.\setup_windows.ps1
```

### Optional neural baseline

The classical and sensor-based methods do not require this dependency.

```powershell
pip install -r requirements-optional.txt
```

If the neural runtime is not installed, DeepFilterNet / RNNoise is reported as unavailable rather than assigned a fabricated result.

## Prepare the dataset

```powershell
python scripts/prepare_data.py
```

This creates:

```text
data/catalog.csv
data/splits.csv
data/extracted/
```

The split is performed at the run level so that derived SNR conditions from the same underlying run remain in one partition. Metric-invalid runs are excluded from quantitative tuning and final metrics.

## Validate the code

```powershell
python scripts/smoke_test.py
pytest -q
```

On Windows the same checks can be launched with:

```powershell
.\run_validation_windows.ps1
```

## Experiment 1: Wiener pole study

Tune the Wiener recursive noise PSD pole on development and validation data:

```powershell
python scripts/run_sweep.py tune
```

Evaluate the frozen selected pole on the held-out test split:

```powershell
python scripts/run_sweep.py test
```

Important outputs include:

```text
outputs/tuning_metrics.csv
outputs/validation_summary_by_pole.csv
outputs/selected_pole.json
outputs/test_metrics.csv
```

## Experiment 2: Run one DSP method

Example Wiener run:

```powershell
python scripts/run_pipeline.py --scenario S3 --snr "-5 dB" --pipeline "wiener:alpha=0.97"
```

Example RPM order tracking run:

```powershell
python scripts/run_pipeline.py --scenario S3 --snr "-5 dB" --pipeline "rpm_order:harmonics=8"
```

The script writes processed audio and metric JSON files to `outputs/` unless custom paths are supplied.

## Experiment 3: Compare all methods

Quick check on a small subset:

```powershell
python scripts/benchmark_methods.py --split development --max-runs 2 --snrs "-5 dB"
```

Full development benchmark:

```powershell
python scripts/benchmark_methods.py --split development
```

Benchmark selected methods only:

```powershell
python scripts/benchmark_methods.py --split development --presets wiener mmse_lsa rpm_order_tracking nmf
```

The default benchmark evaluates all eleven configured methods. Unavailable optional methods are reported safely.

## Experiment 4: Combine methods

The general pipeline syntax is:

```text
method:param=value,param=value|method:param=value
```

Examples:

```powershell
python scripts/run_pipeline.py --scenario S3 --snr "-5 dB" --pipeline "rpm_order:harmonics=8|wiener:alpha=0.97"
```

```powershell
python scripts/run_pipeline.py --scenario S3 --snr "-5 dB" --pipeline "imu_nlms:mu=0.08|wiener:alpha=0.97"
```

```powershell
python scripts/run_pipeline.py --scenario S3 --snr "-5 dB" --pipeline "rpm_order:harmonics=8|imu_nlms:mu=0.08|wiener:alpha=0.97"
```

The order of the stages is significant and can be changed freely.

## Experiment 5: Joint parameter sweep

Any pipeline can be converted into a parameter grid.

Example: sweep RPM harmonic count and Wiener pole together.

```powershell
python scripts/sweep_pipeline.py --template "rpm_order:harmonics={h},width_hz=18|wiener:alpha={a}" --grid h=4,6,8 a=0.90,0.95,0.98 --scenarios S3 S6 --snrs "-5 dB"
```

This evaluates all combinations of the supplied grid values and writes a CSV result table.

## Metrics

The benchmark code reports metrics suitable for comparing algorithms with different internal structures. These include:

- input SNR and reference reconstruction SNR
- SNR improvement
- SI-SDR
- correlation with clean reference
- log spectral distance
- runtime
- metadata indicating oracle-noise, RPM, and IMU dependence

Runtime is reported separately from the quality score.

## Fair comparison protocol

1. Use the same prepared run and SNR condition for every method in a comparison.
2. Do not use the held-out test split to select method parameters.
3. Keep all derived conditions of one run in the same split.
4. Report oracle-assisted and deployable methods separately where appropriate.
5. Do not assume a cascade is better simply because it contains more stages.
6. Tune on development and validation data, then freeze the configuration before final test evaluation.

See `EXPERIMENT_PROTOCOL.md` for the recommended experiment order.

## Project structure

```text
configs/                 Default single-method and combined pipelines
data/raw/                Place raw ZIP chunks here
data/extracted/          Generated prepared run data
outputs/                 Generated metrics and processed audio
scripts/prepare_data.py  Dataset discovery and preparation
scripts/run_sweep.py     Wiener pole tuning and held-out test
scripts/run_pipeline.py  Run any single or combined DSP pipeline
scripts/benchmark_methods.py
                         Cross-method benchmark
scripts/sweep_pipeline.py
                         Generic multi-parameter sweep
src/                     DSP methods, metrics, dataset and pipeline engine
tests/                   Reproducibility and numerical sanity tests
```

## Reproducibility

Default method definitions are stored in `configs/methods.json` and reusable cascade presets are stored in `configs/pipelines.json`. Commands, parameters, and output files can therefore be preserved with the reported results.

## Submission note

This ZIP contains code only. Large raw audio and telemetry files are kept outside the code archive to avoid duplicating the dataset and to keep the submission package portable.
