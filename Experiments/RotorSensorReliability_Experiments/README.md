# Rotor telemetry reliability: selection-committee code submission

This package contains the archived experimental implementations for this paper, made portable, with frozen configurations, row-level evidence, paper source, tests, and reproduction commands. It excludes unrelated amplitude restoration and general drone-audio projects.

**Validation boundary:** the fresh-acoustics study is reproducible with the included speech recording. The original 19-capture benchmark needs three original raw archives, which are not bundled. Archived original results are clearly separated from new execution outputs. Read `VALIDATION.md` before submission.

## Installation

Use Python 3.12 and run commands from this folder. No ROS, Gazebo, GPU, Streamlit, or microphone device is needed to replay the offline experiments.

```bash
python -m venv .venv
```

Activate on Linux/macOS:

```bash
source .venv/bin/activate
```

Activate on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the fresh-study and plotting dependencies:

```bash
python -m pip install -r requirements-core.txt
python scripts/verify_package.py
python run.py doctor
python run.py test
python run.py audit
```

Install the full requirements for original STOI, ESTOI and PESQ scoring:

```bash
python -m pip install -r requirements.txt
```

PESQ may require a C/C++ build toolchain. The neural comparison uses the paper's Linux x86_64 executable; run that stage under Linux or WSL2. Exact package versions match the archived environment. `run.py` limits BLAS/OpenMP threads to one in its child processes.

## Fresh-acoustics experiments: runnable immediately

Use the paper's frozen configurations:

```bash
python run.py fresh --output runs/fresh_replay
```

This executes 48 cases x six telemetry conditions x 17 variants = **4,896 spectral evaluations**, reconstructs **1,344 sealed-test waveforms** for scoring, and runs the additional **960 waveform evaluations** in the harmonic-ridge audit. Waveforms are reconstructed in memory; row-level scores, statistics, figures, logs and an execution ledger are saved. Clean/noise stems are available only for evaluation and explicitly marked oracle methods.

To reproduce development tuning before test evaluation:

```bash
python run.py fresh --retune --output runs/fresh_development_replay
```

Development uses cases 0-15 (seeds 51000-51015); test uses cases 16-47. The correlation grid has 24 candidates and the ridge grid has 40 candidates. The selection objective and ridge near-optimal tie-break are preserved. Tuning uses complex STFT-SDR on development only. The test set is an archived fixed split; rerunning it does not create a new untouched holdout.

Each command requires a new output directory, protecting archived results and previous runs. `archive_comparison.json` checks row-level values against the archived results to an absolute tolerance of 1e-7, excluding machine-dependent timing.

## Original PX4/Gazebo controlled benchmark

Place the three **exact original archives** together in an input folder:

| Original archive | SHA-256 |
|---|---|
| `S3(1).zip` | `417298946520b3e792de01b06bdfe1df522d3227c995df4e61023fab00b0a1ff` |
| `S6(1).zip` | `7993a254c4647327a81ece16122660bc46803ff8b2079d5da9192a5c3b369ec5` |
| `data.zip` | `99460c7ea84c37761f2f8f8441d7dbe836a4ca98e06c76303afa13db4a987d52` |

Filenames may differ: import matches SHA-256 bytes. A differently compressed ZIP may contain equivalent data but will fail exact-archive verification; investigate rather than bypassing the check. The available related codebase archives are not replacements for these captured data.

```bash
python run.py prepare /absolute/path/to/input_archives
```

The importer checks all three hashes before extraction, validates ZIP CRCs, rejects traversal paths, and checks 19 selected captures and 76 mixtures. Selected captures are the latest valid run for each scenario/seed. Manifests for earlier invalid captures are retained for provenance.

For the paper's DeepFilterNet3 comparator, obtain the official v0.5.6 Linux x86_64 binary:

https://github.com/Rikorose/DeepFilterNet/releases/download/v0.5.6/deep-filter-0.5.6-x86_64-unknown-linux-musl

Save as `tools/deep-filter`, then on Linux:

```bash
chmod +x tools/deep-filter
```

The script enforces SHA-256 `70775e251eee44c0f2451a1e833326cf8bcbbe304d3e7cd12851e6fce72ef7da`. Third-party binary and model are not redistributed. It uses bundled DeepFilterNet3 ONNX, compensated delay (`-D`), default attenuation, no optional postfilter, and the same 2-14 s scoring interval as classical methods.

```bash
python run.py doctor
python run.py original --output runs/original_replay
```

This runs the 16-system benchmark (1,216 evaluations including neural/oracle), baseline sensitivity evaluation, telemetry/calibration interventions, matched ablations, fault windows, speech/noise decomposition, isolated timing, statistics, figures and verification. Frozen settings are used by default. The archived development CSV is copied as selection provenance, not presented as a new tuning execution.

```bash
python run.py original --retune --output runs/original_development_replay
```

This additionally reruns the 340 initial configuration evaluations and 300 expanded baseline evaluations: **640 development evaluations** in total. The selected RPM method is not retuned in the baseline-only sweep. Original raw acoustics generation source was not supplied with the research artifact; this package replays captured/synthesized recordings, not a new PX4 flight or a replacement acoustic generator.

## Read evidence or regenerate plots without raw capture data

```bash
python run.py audit
python run.py figures --output runs/archived_figures
```

The original plot/statistics script is `original/code/analyze.py`; it also plots RPM and spectrogram traces and therefore requires the original capture data. Original paper figure assets are supplied in `paper/figures/`. The fresh plot script regenerates the data charts from CSV evidence; visual styling may differ from the manuscript assets.

## Layout

| Folder/file | Purpose |
|---|---|
| `run.py` | Install checks, data import, experiments, audits and plots |
| `original/code/` | Original benchmark, baselines, interventions, ablations and neural comparator |
| `fresh/` | Fresh generator, audio/RPM fusion, correlation and ridge selectors |
| `configs/` | Published development-selected settings |
| `evidence/original/` | Archived original row-level data, statistics and provenance |
| `evidence/fresh/` | Archived fresh-study grids, rows and statistics |
| `original/reference/` | Original speech and source/usage record |
| `original/input_metadata/` | Capture manifests and recorded simulator metadata |
| `tests/` | Scientific invariant and reference-separation checks |
| `runs/` | Actual rerun outputs with logs and comparisons |
| `validation/` | Package audit and test records |
| `EXPERIMENT_MAP.md` | Paper-to-code-to-evidence mapping |
| `paper/` | User's report, LaTeX, bibliography and manuscript figures |

To build the supplied paper, install pdfLaTeX and Biber, then run `bash build.sh` from `paper/`. The original manuscript remains anonymized. Add authorship and any required AI-use disclosure according to the committee's rules.

## Scientific and data-use boundaries

These are hybrid simulation and synthetic stress studies, not physical onboard recordings. Fresh cases reuse one Open Speech Repository speech file; they are not speaker-disjoint. Nominal SNR labels and fixed-region scoring SNR are distinct. The original and fresh studies use different STFT/gain designs and cannot be compared by absolute SI-SDR.

Speech attribution: Open Speech Repository, `OSR_us_000_0010_8k.wav`, https://www.voiptroubleshooter.com/open_speech/american.html . See `original/reference/data/clean_speech/SOURCE_AND_LICENSE.json` for the archived source/usage record. Retain attribution; no blanket license is invented for third-party audio, models or user-supplied code.

The negative results are intentionally retained: correlation and harmonic support do not establish useful sensor reliance. Oracle masks, oracle PSD, and reference-selected fixed weights are diagnostics, not deployable systems. See `CLAIM_LEDGER.md` for claim boundaries.
