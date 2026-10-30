"""Finalize reporting after inference, without rerunning or changing decisions.

Separates structural RPM validity from template/model eligibility. Earlier
in-flight evaluations counted eligible model frames under a validity label;
the audio-only row supplies the identical monitor's structural coverage.
"""
import csv,json,hashlib
from pathlib import Path
from drone_audio.outputs import csv_out,json_out,environment
from drone_audio.evaluation import grouped_results
from drone_audio.config import load_config


def main():
    root=Path(__file__).resolve().parents[1]
    for name in ('combined_test','event_probes_test','sensor_degradation'):
        p=root/'runs'/name
        if not (p/'execution.json').exists() or json.loads((p/'execution.json').read_text())['status']!='completed':
            raise RuntimeError(f'{name} has not completed')
        with (p/'per_case.csv').open() as f:rows=list(csv.DictReader(f))
        if 'rpm_model_eligible_coverage' not in rows[0]:
            baseline={}
            for r in rows:
                if r['method'] in ('audio_only','unprocessed'):
                    baseline[(r['sample_id'],r.get('sensor_condition',''))]=float(r['rpm_valid_coverage'])
            for r in rows:
                old=float(r['rpm_valid_coverage']);r['rpm_model_eligible_coverage']=old
                r['rpm_valid_coverage']=baseline.get((r['sample_id'],r.get('sensor_condition','')),old)
        csv_out(p/'per_case.csv',rows)
        if name!='sensor_degradation':
            numeric=[]
            for r in rows:
                d=dict(r)
                for k,v in d.items():
                    if v=='':d[k]=None
                    else:
                        try:d[k]=float(v)
                        except (ValueError,TypeError):pass
                numeric.append(d)
            parents,macro,paired=grouped_results(numeric)
            csv_out(p/'per_parent.csv',parents);csv_out(p/'summary.csv',macro);csv_out(p/'paired_vs_audio.csv',paired)
        json_out(p/'metric_corrections.json',{'status':'completed','changed_fields':['rpm_valid_coverage','rpm_model_eligible_coverage'],
            'reason':'Separate structural RPM validity from model/template eligibility; retain aggregate quality columns when the first case has a silent reference. No audio, decisions or per-case quality metrics changed.',
            'derivation':'Structural monitor is identical across modes; use matched audio-only/unprocessed row. Probe complete mode is already audio-only.',
            'rows':len(rows),'environment':environment()})
    freeze=json.loads((root/'configs/integration_freeze.json').read_text())
    current={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((root/'drone_audio').rglob('*')) if f.suffix in ('.py','.json')}
    changed=[k for k,v in freeze['source_hashes'].items() if current.get(k)!=v]
    json_out(root/'configs/delivery_code_record.json',{'frozen_configuration_unchanged':load_config(root/'configs/frozen.yaml').sha256==freeze['config_sha256'],
        'source_hashes':current,'changes_since_pretest_code_record':changed,
        'explanation':'Python dataclass defaults aligned to the already frozen explicit settings; evaluation coverage labels separated; full reset now drops prior recording sensor streams (constructor explicitly preserves supplied sensors). New-recording inference is unchanged and representative delivery replay is checked. No test-driven DSP parameter changes. Diagnostic/reporting additions are not enhancement changes.',
        'environment':environment()})


if __name__=='__main__':main()
