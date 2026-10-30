"""Recorded second development round: exercise calibration-free RPM order mode.

Round 1 exposed missing observed RPM in Batch2 development. This extension
adds one deployment-eligible order candidate using the round-1 audio pole.
Only previously declared development/validation cases are opened.
"""
from pathlib import Path
import csv,json
from copy import deepcopy
import numpy as np
from drone_audio.config import load_config,save_config
from drone_audio.datasets import discover,hash_file
from drone_audio.evaluation import bind_input,load_target,si_sdr
from drone_audio.audio import read_audio
from drone_audio.orchestration import process_audio
from drone_audio.outputs import csv_out,json_out,environment


def main():
    root=Path(__file__).resolve().parents[1]
    source=root/'runs/development_selection';out=root/'runs/development_selection_with_order'
    out.mkdir(exist_ok=False)
    protocol=json.loads((source/'protocol.json').read_text());prior=json.loads((source/'selection.json').read_text())
    with (source/'selection_rows.csv').open() as f:rows=list(csv.DictReader(f))
    for row in rows:row['si_sdr_db']=float(row['si_sdr_db'])
    ids={r['sample_id'] for r in rows};samples,audit=discover(root.parent/'Data',include_variants=False)
    samples=[s for s in samples if s.id in ids]
    assert all(s.split in ('development','validation') for s in samples)
    for s in samples:assert hash_file(s.inference.microphone)==prior['input_hashes'][s.id]
    base=load_config(root/'configs/frozen.yaml');candidate=deepcopy(base)
    candidate.enhancement.mode='order';candidate.amplitude.enabled=False;candidate.detector_enabled=False
    for i,s in enumerate(samples):
        cfg=bind_input(candidate,s.inference);x,_=read_audio(s.inference.microphone)
        p=process_audio(x,cfg,telemetry=s.inference.load_telemetry());target=load_target(s);mask=np.ones(len(x),bool)
        if s.evaluation.active_interval_s:
            t=np.arange(len(x))/16000;a,b=s.evaluation.active_interval_s;mask=(t>=a)&(t<b)
        rows.append({'sample_id':s.id,'parent_id':s.parent_id,'batch':s.batch,'split':s.split,'candidate':'order_a0.9_b1.0','si_sdr_db':si_sdr(target[mask],p.enhanced[mask])})
        print(f'Order selection {i+1}/{len(samples)}: {s.id}',flush=True)
    candidates=dict(protocol['candidates']);candidates['order_a0.9_b1.0']=candidate.to_dict()
    def objective(name,split):
        values=[]
        for batch in ('batch1','batch2'):
            v=[r['si_sdr_db'] for r in rows if r['candidate']==name and r['split']==split and r['batch']==batch]
            if v:values.append(np.mean(v))
        return float(np.mean(values))
    ranked=sorted(candidates,key=lambda n:objective(n,'development'),reverse=True)
    winner=max(ranked[:3],key=lambda n:objective(n,'validation'))
    selected=load_config(root/'configs/frozen.yaml');selected.enhancement=type(selected.enhancement)(**candidates[winner]['enhancement']);selected.validate()
    save_config(selected,out/'selected.yaml');csv_out(out/'selection_rows.csv',rows)
    record={'winner':winner,'finalists':ranked[:3],'config_sha256':selected.sha256,
        'round_1':str(source),'round_2_reason':'Only S7 has observed per-motor RPM; add template-free order mode using observed aggregate RPM on Batch1 development.',
        'candidates':[{'name':n,'development_objective':objective(n,'development'),'validation_objective':objective(n,'validation')} for n in candidates],
        'test_opened':False,'cases':[s.id for s in samples],'configurations':candidates,'environment':environment()}
    json_out(out/'selection.json',record)
    print(json.dumps({'winner':winner,'sha256':selected.sha256},indent=2))


if __name__=='__main__':main()
