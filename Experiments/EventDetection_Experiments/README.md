# Mixed-audio acoustic events, with a separate speech output

This revision **uses the original `mixture.wav` (target_at_drone + synthetic drone background)** for speech experiments. It is no longer tuned on target-only audio. General acoustic activity and speech flags are distinct. It does not attach semantic event labels or identify speakers.

## Deliverables

- `pipeline/detector.py`: final causal acoustic-event detector and speech head.
- `pipeline/integrated.py`, `pipeline/amplitude.py`: continuous mixed-audio amplitude demonstration, with explicit session reset.
- `experiments/`: data generation, calibration/validation selection, test comparisons, analysis and PDF creation.
- `configs/`: all five fitted feature-family models, selected model, training protocol, frozen hashes and original visual speech labels.
- `report/mixed_event_report.pdf`: separate detailed report for this revision only.
- `metrics/`: test-only frame/confusion, event, feature comparison, noise-sweep, literature-comparator and mixture-amplitude data.

The 75 base cases keep the **60 development / 15 test split**. All five descendants inherit their base split: original speech mixture, background-only control, chirp, tone, impact/texture. Therefore there are 300 development and 75 test derived cases, not 375 independent recordings. Calibration uses rec1/rec3; validation uses rec2; test speech uses rec4/rec5's one unique recording. Test recording reuse from earlier studies is disclosed.

General-event probes use new randomized foreground waveforms over the existing time-varying backgrounds. They are synthetic and cover three non-speech families, not every possible real event. Speech intervals remain approximate visual annotations, with 80 ms boundary collars excluded. Generic-event support is known from construction. Background-only variation tests false alarms against moving rotor/wind noise rather than assuming a perfectly constant background.

## Algorithm

A causal 25 ms analysis window with 10 ms hops at 16 kHz feeds energy/background contrast, spectral flux, Jensen-Shannon spectral change, complex phase-prediction residual, wrapped phase deviation, regularized signal group-delay change, flatness, autocorrelation periodicity and centroid. Weak-bin masking and denominator regularization limit phase instability. Five feature families are calibrated on development data with non-neural L2 logistic readouts. Development selection decides whether phase or group delay actually helps. Read the measured comparison before claiming an improvement.

The group-delay feature measures signal structure, **not propagation delay or the filter's group delay**. The original amplitude-only generator contains no propagation delay/Doppler. No physical delay can be reconstructed from those missing effects.

Outputs:

- `activity_active`: sustained general foreground evidence.
- `event_active`: grouped activity including an additional release tail.
- `speech_active`: separate speech score/state while a general event is present.
- `event_opened`, `event_closed`, `event_id`, feature/score trace and availability timestamps.
- `reset_reference`: always zero automatically; source-session changes require application input.

This is not a denoiser. Applying amplitude gain to a mixture multiplies foreground and background together. `amplitude_on_mixture.wav` is a clearly labelled demonstration using inherited amplitude parameters, not clean recovered speech. Actual integration with noise suppression requires independent validation and a low-SNR gain policy.

## Run

Python 3.10+; tested package versions are in `requirements-lock.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m unittest discover -s tests -v
python experiments/run.py --resume
python experiments/analyze_report.py
```

Use `source .venv/bin/activate` on Linux/macOS. Omit `--resume` to repeat training/selection on packaged data. Reruns overwrite outputs. No GPU or downloaded models are required. Inference requires only NumPy and SciPy; scikit-learn is used to fit coefficients during experiments.

Process new mono 16 kHz mixed PCM16 audio:

```powershell
python -m pipeline --input mixed_audio.wav --output-dir detected
```

The CLI exports mixed event clips with 100 ms pre-roll and a feature/decision trace. It does not resample silently. For continuous streaming:

```python
from pipeline import Detector, MixedPipeline

# Standalone detector: 160 new samples per hop.
detector = Detector()
result = detector.process_hop(audio_160_samples)

# Amplitude demo: 320 new samples per block, known source at origin by default.
processor = MixedPipeline()
y, gain, detector_logs, amplitude_log = processor.process_block(
    audio_320_samples, pose=observed_position, pose_valid=True
)
processor.new_session()  # only when the application knows the source/session changed
```

Event clips contain the original mixed samples; no hard gate changes the continuous waveform. The selected detector is causal; literature speech baselines are explicitly offline mathematical adaptations, not author-code parity reproductions.

## Limitations and attribution

Speech evaluation uses one held-out source recording; speaker independence is unverified. Synthetic non-speech waveforms use disjoint seeds but do not prove transfer to real alarms, claps or environmental events. A semi-synthetic open-speech probe is supplied from LibriSpeech SLR12 (CC BY 4.0); it is not real drone-flight validation. Full limitations, failed comparisons and Moodle coverage are in the PDF. References and baseline adaptations are documented there and in `experiments/baselines.py`.
