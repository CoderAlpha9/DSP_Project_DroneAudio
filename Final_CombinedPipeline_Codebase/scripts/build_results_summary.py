"""Build implementation validation notes from executed tables, not archived scores."""
from pathlib import Path
import csv,json

ROOT=Path(__file__).resolve().parents[1]


def rows(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def number(value,digits=3):
    try:return f'{float(value):.{digits}f}'
    except (ValueError,TypeError):return 'unavailable'


def main():
    summary=rows(ROOT/'runs/combined_test/summary.csv')
    probes=rows(ROOT/'runs/event_probes_test/summary.csv')
    sensors=rows(ROOT/'runs/sensor_degradation/per_case.csv')
    selection=json.loads((ROOT/'runs/development_selection_with_order/selection.json').read_text())
    checks=json.loads((ROOT/'runs/delivery_validation.json').read_text())
    lines=['# Executed validation and results','',
        'These tables come from this combined codebase. Historical module-level results are discussed separately in SOURCE_MAP.md and are not imported into new metric tables.',
        '',f"Executed {checks['tests_passed']} automated tests, 47 matched test conditions × eight method labels = 376 rows, 60 additional synthetic non-speech test descendants, and 84 sensor-study rows. Both dataset adapters were exercised. All run ledgers completed without failed cases.",
        '', '## Frozen selection','',
        f"The final development/validation winner was `{selection['winner']}`. Nine bounded candidates were compared over 27 existing development/validation cases (243 evaluations total in two documented rounds). The first round ranked eight candidates on development and chose among the best three on validation; the second added calibration-free order suppression after missing observed RPM in Batch2 development was confirmed. No test data selected the frozen parameters.",
        '', 'The final configuration uses the inherited mixed-audio spectral detector; 1024/256 SPP + decision-directed Wiener enhancement with noise alpha .90, DD alpha .96, beta 1 and floor .04; and the inherited bounded geometry gain with the newly integrated source/speech controls and SNR guard. RPM and motion alternatives remain available, but neither is forced into the default.',
        '', '## New matched test means','',
        'Means are formed over SNR variants within a capture, then over parents. Batch1 has 15 test parents from one unique source recording; Batch2 has eight test captures and four SNR conditions each, reusing one speech file across capture splits. These are reused tests, not new independent field validation.',
        '', '| Batch | Method | Pre SI-SDR dB | Pre STOI | Post level MAE dB | Mean gain dB | Compute RTF |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['batch']} | {r['method']} | {number(r['pre_si_sdr_db'])} | {number(r['pre_stoi'])} | {number(r['post_active_level_mae_db'])} | {number(r['mean_applied_gain_db'])} | {number(r['real_time_factor'])} |")
    lines += ['', 'Pre-amplitude SI-SDR/reconstruction SNR compare with the target at the microphone. Post-amplitude level error compares with the closest-so-far true-range goal for Batch1, and the microphone target for Batch2 because geometry is unavailable. Spectral distortion and STOI/ESTOI use the full aligned clip; Batch2 SI-SDR/SNR use 2–14 s. Detailed SNR, RMSE, LSD, ESTOI, clipping, timing, fallback and paired parent values are in the CSVs.',
        '', 'The amplitude stage makes only small corrections on many noisy cases because its new estimated-SNR guard abstains and its source/speech control limits requests. Denoising can also lower target level, which geometric compensation does not undo. The combined level errors therefore do not reproduce the target-only amplitude study and can be worse than unprocessed level fidelity. No claim of joint optimality or universally improved intelligibility/level restoration is made. The default was not retuned after inspecting test behavior.',
        '', 'The audio-only and complete-without-amplitude labels intentionally coincide for the selected configuration. Their identical per-case configuration hashes are explicit, and `inference_cache_reused` marks reused deterministic inference. Method labels are comparisons, not extra independent observations. Timing fields for cached aliases describe the shared inference; metric/export costs are separately recorded.',
        '', '## Events and speech','',
        '| Test set | Activity F1 | Speech F1 | Matched events / parent | Unmatched reference / parent | Unmatched predicted / parent |',
        '|---|---:|---:|---:|---:|---:|']
    for r in [r for r in summary if r['batch']=='batch1' and r['method']=='complete']+probes:
        lines.append(f"| {r['batch']} | {number(r.get('activity_f1'))} | {number(r.get('speech_f1'))} | {number(r.get('matched_events'))} | {number(r.get('unmatched_reference_events'))} | {number(r.get('unmatched_predicted_events'))} |")
    lines += ['', 'Original speech labels are approximate visual regions with 80 ms collars. Non-speech support is synthetic, also using the supplied valid-frame mask. Endpoint errors are conditional on overlapping matched events and are exported alongside unmatched counts. Grouped event intervals may cover multiple short visual speech regions. Batch2 has no compatible event annotation, so its event metrics are explicitly unavailable. Logistic scores are not asserted to be calibrated probabilities.',
        '', '## Sensor interventions','',
        'Two representative reused test inputs were exercised across nominal, missing, 200 ms delayed, two-second held, −20% drifting, constant per-motor scaled and stopped RPM. Each has six explicit enhancement modes (audio, fixed .1, bounded .75, correlation, ridge, order): 2 × 7 × 6 = 84 runs. No new physical fault labels were introduced.',
        '', '| Batch / representative | Condition | Mode | Pre SI-SDR dB | Mean RPM weight | Structural coverage |',
        '|---|---|---|---:|---:|---:|']
    for r in sensors:
        if r['sensor_condition'] in ('nominal','missing','drift_m20') and r['method'] in ('audio_only','bounded_rpm','ridge'):
            lines.append(f"| {r['batch']} | {r['sensor_condition']} | {r['method']} | {number(r['pre_si_sdr_db'])} | {number(r['mean_rpm_weight'])} | {number(r['rpm_valid_coverage'])} |")
    lines += ['', 'Observed RPM is missing from non-S7 Batch2 captures and is not replaced with simulator truth. Batch1 relative-template modes have no declared noise-only calibration and appropriately fall back; its order mode still uses observed aggregate RPM. A bounded RPM improvement on one S7 case does not establish a better global operating policy. Agreement is not a usefulness certificate. Drift cannot in general be distinguished from real changes using timestamp/range checks alone.',
        '', '## Commands and artifact checks','',
        '- Local editable installation with analysis/test dependencies, `doctor`, manifest discovery plus hash audit.',
        '- Initial `select` and the documented order-selection extension, frozen before combined test evaluation.',
        '- Full `evaluate --config configs/frozen.yaml --output runs/combined_test --plots`.',
        '- `evaluate --event-probes --batch event_probe --methods complete --output runs/event_probes_test`.',
        '- `sensor-study --output runs/sensor_degradation`.',
        '- Single-file associated-source inference with telemetry and event clips; missing-sensor inference from a different working directory; Batch2 batch inference.',
        '- `pytest` plus delivery checks of all saved WAV lengths/finite values, completed ledgers, selected config equality and unchanged supplied source hashes.',
        '', 'Exact executed versions, test console log and machine-readable checks are under `runs/`. All 107 complete-evaluation cases contain pre/post audio and diagnostic traces. Pipeline, aligned signal/control, method comparison, paired parent comparison, fixed filter response, and sensor-degradation PNGs are included. Audio is float WAV with no hidden normalization.',
        '', '## Concrete limitations / unavailable work','',
        '- Exact archival reproduction remains separate: original ZIP bytes and the Linux DeepFilterNet comparator are absent. Inference on already extracted captures is implemented and executed.',
        '- No supplied preliminary report/PDF was available. This package does not write the later IEEE report.',
        '- The datasets are hybrid/synthetic and source-limited. No real-flight transfer, speaker identification, physical actuator-fault classifier, source localization, propagation delay or Doppler estimation is claimed.',
        '- Template calibration is conditional on an explicit noise-only prefix. Unknown-startup fallback and the new gain guard need further independent validation.',
        '- The optional low-band motion canceller is newly adapted and untuned, not a previously validated wind-suppression module.',
        '- No exact clean/noise component decomposition was scored; all reported SNRs are waveform reconstruction measures. `oracle.known_noise_wiener` is a separately named evaluation-only comparator, not used in normal inference or matched deployable results.',
        '- PESQ and optional neural comparisons were not run. STOI/ESTOI were installed and run where a non-silent reference made them applicable.',
        '- CPU timing was measured on this machine, sometimes with concurrent jobs, so it is descriptive rather than a controlled hardware benchmark. Algorithmic buffering is documented independently.',
        '- Reporting-only coverage labels were corrected after execution to separate structural validity from template/model eligibility. Python dataclass defaults were aligned to the already frozen explicit config, and full reset now clears prior-recording sensors. Representative delivery replay checks confirm unchanged new-recording outputs. The original freeze and delivery code record preserve this distinction; no frozen DSP setting changed.',
        '']
    (ROOT/'docs/VALIDATION_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':main()
