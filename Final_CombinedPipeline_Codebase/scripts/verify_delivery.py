"""Verify actual delivery artifacts and write a machine-readable audit."""
from pathlib import Path
import hashlib,json,csv,re
import numpy as np
from scipy.io import wavfile
from drone_audio.config import PipelineConfig,load_config
from drone_audio.outputs import json_out,environment


def main():
    root=Path(__file__).resolve().parents[1];checks={};wav_count=0
    for name,count in [('combined_test',376),('event_probes_test',60),('sensor_degradation',84)]:
        p=root/'runs'/name
        ledger=json.loads((p/'execution.json').read_text());assert ledger['status']=='completed'
        assert not ledger.get('failed')
        with (p/'per_case.csv').open() as f:rows=list(csv.DictReader(f))
        assert len(rows)==count,(name,len(rows))
        checks[name]={'status':'completed','rows':len(rows)}
        for record in (p/'cases').glob('*/complete/execution.json'):
            meta=json.loads(record.read_text());n=meta['output_samples']
            for filename in ('final.wav','enhanced.wav'):
                sr,x=wavfile.read(record.parent/filename)
                assert sr==16000 and x.ndim==1 and len(x)==n and np.isfinite(x).all()
                if filename=='final.wav':assert max(abs(x),default=0)<=.950001
                wav_count+=1
    for item in json.loads((root/'configs/source_provenance.json').read_text()):
        assert hashlib.sha256((root.parent/item['source']).read_bytes()).hexdigest()==item['sha256']
    frozen=load_config(root/'configs/frozen.yaml')
    assert frozen.sha256==load_config(root/'configs/default.yaml').sha256==PipelineConfig().sha256
    assert frozen.sha256==json.loads((root/'configs/integration_freeze.json').read_text())['config_sha256']
    raw=(root/'runs/validation_tests.log').read_bytes()
    log=raw.decode('utf-16' if raw[:2] in (b'\xff\xfe',b'\xfe\xff') else 'utf-8-sig')
    match=re.search(r'(\d+) passed',log);assert match
    replay=json.loads((root/'runs/delivery_replay.json').read_text());assert replay['status']=='completed'
    assert all(r['passed'] for r in replay['rows'])
    figures=list((root/'runs/combined_test/figures').glob('*.png'));assert len(figures)>=6
    assert (root/'runs/sensor_degradation/sensor_degradation.png').exists()
    checks.update(tests_passed=int(match.group(1)),wav_files_verified=wav_count,figure_count=len(figures)+1,
        delivered_code_replay=replay['rows'],
        source_artifact_hashes_unchanged=True,default_and_frozen_identical=True,config_sha256=frozen.sha256,
        environment=environment(),status='completed')
    json_out(root/'runs/delivery_validation.json',checks);print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
