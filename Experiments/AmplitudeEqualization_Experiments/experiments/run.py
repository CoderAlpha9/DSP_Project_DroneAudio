"""Target-only experiments. Never reads mixture.wav, noise.wav or noise PSD/SNR."""
from pathlib import Path
from dataclasses import asdict,replace
import argparse,csv,json,hashlib,itertools,time,sys,shutil
import numpy as np
from scipy.io import wavfile
from scipy import signal

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pipeline import Config,process_audio


def json_out(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def csv_out(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def rows_in(p):
    with Path(p).open(encoding='utf-8') as f:return list(csv.DictReader(f))


def wav_in(p):
    sr,x=wavfile.read(p)
    return sr,x.astype(float)/(32767 if x.dtype==np.int16 else 1.)


def pcm(x):return np.round(np.clip(x,-1,1)*32767).astype(np.int16)
def rmsf(x):return np.sqrt(np.mean(x.reshape(-1,320)**2,axis=1)+1e-30)


def load(data,row):
    d=Path(data)/row.get('relative_path','cases/'+row['scenario_id'])
    sr,x=wav_in(d/'target_at_drone.wav');_,ref=wav_in(d/'reference.wav')
    # No other waveform is opened, even for evaluation.
    gt=np.genfromtxt(d/'ground_truth_trajectory.csv',delimiter=',',names=True)
    obs=np.genfromtxt(d/'telemetry_observed.csv',delimiter=',',names=True)
    t=(np.arange(len(x)//320)+1)*.02
    ii=np.clip(np.searchsorted(obs['time_s'],t,side='right')-1,0,len(obs)-1)
    tele=[dict(pose=[obs['pose_enu_'+v+'_m'][j] for v in 'xyz'],pose_valid=bool(obs['pose_valid'][j]),
               imu_force=[obs['accel_body_'+v+'_m_s2'][j] for v in 'xyz'],imu_valid=bool(obs['imu_valid'][j])) for j in ii]
    ta=np.arange(len(x))/sr;r=np.interp(ta,gt['time_s'],gt['distance_m'])
    ideal=np.maximum(1,r/np.minimum.accumulate(r))
    goal=x*ideal
    active=rmsf(ref)>max(.1*np.percentile(rmsf(ref),95),1e-6)
    meta=json.loads((d/'metadata.json').read_text())
    return dict(row=row,path=d,sr=sr,x=x,reference=ref,goal=goal,ideal=ideal,
                distance=r,telemetry=tele,active=active,meta=meta)


def evaluate(sc,y,g,logs,policy='event',full=True):
    goal=sc['goal'] if policy=='event' else sc['reference']
    ideal=sc['ideal'] if policy=='event' else sc['distance']/2
    active=sc['active'];db=20*np.log10(np.maximum(g.reshape(-1,320).mean(axis=1),1e-12))
    targetdb=20*np.log10(ideal.reshape(-1,320).mean(axis=1))
    env=20*np.log10(rmsf(y)+1e-12);envgoal=20*np.log10(rmsf(goal)+1e-12)
    d=env-envgoal;need=active&(targetdb>1)
    m=dict(active_gain_mae_db=float(np.mean(abs(db[active]-targetdb[active]))),
           needed_gain_mae_db=float(np.mean(abs(db[need]-targetdb[need]))) if need.any() else float('nan'),
           active_level_mae_db=float(np.mean(abs(d[active]))),active_level_bias_db=float(np.mean(d[active])),
           envelope_shape_rmse_db=float(np.std(d[active])),gain_step_p95_db=float(np.percentile(abs(np.diff(db)),95)),
           gain_step_max_db=float(max(abs(np.diff(db)))),clipping_fraction=float(np.mean(abs(y)>=1)),
           peak=float(max(abs(y))),active_frames=int(sum(active)),needed_frames=int(sum(need)),
           mean_gain_db=float(np.mean(db)),max_gain_db=float(max(db)),
           limiter_fraction=float(np.mean([a.get('limiter_active',0) for a in logs])))
    if not full:return m
    m['waveform_rmse']=float(np.sqrt(np.mean((y-goal)**2)))
    m['correlation']=float(np.corrcoef(y,goal)[0,1])
    m['reconstruction_snr_uncapped_db']=float(10*np.log10((np.sum(goal**2)+1e-24)/(np.sum((y-goal)**2)+1e-24)))
    m['reconstruction_snr_db']=min(80.,m['reconstruction_snr_uncapped_db'])
    m['identity_reconstruction_snr_uncapped_db']=float(10*np.log10((np.sum(goal**2)+1e-24)/(np.sum((sc['x']-goal)**2)+1e-24)))
    m['identity_reconstruction_snr_db']=min(80.,m['identity_reconstruction_snr_uncapped_db'])
    m['reconstruction_snr_change_db']=m['reconstruction_snr_db']-m['identity_reconstruction_snr_db']
    _,_,A=signal.stft(y,nperseg=1024,noverlap=768);_,_,B=signal.stft(goal,nperseg=1024,noverlap=768)
    floor=max(np.max(abs(B))*1e-4,1e-12)
    err=20*np.log10(np.maximum(abs(A),floor)/np.maximum(abs(B),floor))
    m['log_spectral_distance_db']=float(np.mean(np.sqrt(np.mean(err**2,axis=0))))
    return m


def run(sc,cfg):return process_audio(sc['x'],cfg,telemetry=sc['telemetry'],event_active=True)


def objective(ms):
    # Internal development-only score; no development performance table is exported.
    return float(np.mean([m['active_level_mae_db']+.25*m['envelope_shape_rmse_db']+.1*m['gain_step_p95_db'] for m in ms]))


def tune(train):
    groups=sorted(set(s['row']['source_group'] for s in train));cal=[s for s in train if s['row']['source_group']!=groups[-1]]
    val=[s for s in train if s['row']['source_group']==groups[-1]]
    grid=list(itertools.product([12.,18.,24.],[.04,.12,.30],[.8,1.],['hold','unity','energy']))
    ranking=[]
    for i,(cap,tau,w,fallback) in enumerate(grid):
        cfg=Config(max_gain_db=cap,smooth_s=tau,geometry_weight=w,fallback=fallback)
        ms=[]
        for sc in cal:
            y,g,logs=run(sc,cfg);ms.append(evaluate(sc,y,g,logs,full=False))
        ranking.append((objective(ms),i,cfg))
        print(f'Development configuration {i+1}/{len(grid)} completed',flush=True)
    finalists=sorted(ranking,key=lambda v:v[0])[:3];vresults=[]
    for _,i,cfg in finalists:
        ms=[]
        for sc in val:
            y,g,logs=run(sc,cfg);ms.append(evaluate(sc,y,g,logs,full=False))
        vresults.append((objective(ms),i,cfg))
    _,best,cfg=min(vresults,key=lambda v:v[0])
    json_out(ROOT/'configs/selected.json',asdict(cfg))
    json_out(ROOT/'configs/selection_protocol.json',dict(calibration_scenarios=len(cal),validation_scenarios=len(val),
        calibration_groups=groups[:-1],validation_group=groups[-1],candidate_count=len(grid),
        candidates=[dict(candidate_id=i,max_gain_db=cap,smooth_s=tau,geometry_weight=w,fallback=fb) for i,(cap,tau,w,fb) in enumerate(grid)],
        shortlisted_ids=[i for _,i,_ in finalists],selected_id=best,
        criterion='Mean active-level MAE + 0.25 envelope-shape error + 0.1 gain-step p95, development only.',
        performance_reporting='Test-set performance metrics only; development ranking values are not exported.',
        caution='Existing held-out recording reused from the earlier noisy study; not a newly collected blind test.'))
    return cfg


def paper_baseline(x,kind,cap):
    """Algorithm cores from 2016 feedback AGC and 2020/2021 power-law DRC.

    One-band specialization for DRC; common practical gain/peak bounds are
    declared adaptations. No multi-band or original-paper benchmark claim.
    """
    sr=16000;gmax=10**(cap/20);g=np.empty_like(x);upper=.04
    if kind=='feedback_agc_2016':
        current=1.;d=1/(2*.2*sr)
        for i,sample in enumerate(x):
            g[i]=current
            y=current*sample
            current=float(np.clip(current*(1+d*(1-(y/upper)**2)),1.,gmax))
    elif kind=='powerlaw_drc_2021':
        power=max(x[0]**2,1e-12);attack=math_exp(-1/(sr*.005));release=math_exp(-1/(sr*.1))
        for i,sample in enumerate(x):
            p=sample*sample;a=attack if p>power else release;power=a*power+(1-a)*p
            # C(v)=v_ref^(1-1/CR)*v^(1/CR), CR=3, so unity at v_ref.
            g[i]=np.clip((upper**2/max(power,1e-16))**(1/3),1.,gmax)
    else:raise ValueError(kind)
    y=x*g;logs=[]
    for i in range(len(x)//320):
        sl=slice(i*320,(i+1)*320);peak=max(abs(y[sl]));guard=min(1.,.95/max(peak,1e-12))
        g[sl]*=guard;y[sl]*=guard;logs.append(dict(limiter_active=int(guard<.999999)))
    return y,g,logs


def math_exp(x):return float(np.exp(x))


def export_case(sc,cfg,names):
    sid=sc['row']['scenario_id'];row=sc['row'];case=ROOT/'data/cases'/sid;case.mkdir(parents=True,exist_ok=True)
    for fn in ['target_at_drone.wav','reference.wav','telemetry_observed.csv','ground_truth_trajectory.csv']:
        if (sc['path']/fn).resolve()!=(case/fn).resolve():shutil.copy2(sc['path']/fn,case/fn)
    json_out(case/'metadata.json',{k:sc['meta'][k] for k in ['scenario_id','source_offset_samples','source_position_enu_m','reference_distance_m'] if k in sc['meta']})
    y,g,logs=run(sc,cfg)
    fixed=replace(cfg,reference_distance_m=2.)
    ya,ga,logsa=run(sc,fixed)
    # Outputs all use the same physical level. Never normalize each WAV independently.
    index=[]
    for original in names:
        rec=Path(original).stem;out=ROOT/'listening'/rec/sid;out.mkdir(parents=True,exist_ok=True)
        wavfile.write(out/'output_event_relative.wav',sc['sr'],pcm(y))
        wavfile.write(out/'output_fixed_2m.wav',sc['sr'],pcm(ya))
        index.append(dict(rec_number=rec,original_filename=original,scenario_id=sid,split=row['split'],profile=row['profile'],
            sensor_condition=row['sensor_condition'],duplicate_alias=int(rec=='rec5'),
            input_path=str((case/'target_at_drone.wav').relative_to(ROOT)).replace('\\','/'),
            reference_path=str((case/'reference.wav').relative_to(ROOT)).replace('\\','/'),
            event_output_path=str((out/'output_event_relative.wav').relative_to(ROOT)).replace('\\','/'),
            calibrated_output_path=str((out/'output_fixed_2m.wav').relative_to(ROOT)).replace('\\','/')))
    audit=ROOT/'audio_logs'/sid;audit.mkdir(parents=True,exist_ok=True)
    csv_out(audit/'event_trace.csv',logs);csv_out(audit/'fixed_2m_trace.csv',logsa)
    np.save(audit/'event_gain.npy',g.astype(np.float32));np.save(audit/'fixed_2m_gain.npy',ga.astype(np.float32))
    return index


def test_metrics(test,cfg):
    methods={'identity':replace(cfg,method='identity'),'energy_only':replace(cfg,method='energy'),
             'geometry_unchecked':replace(cfg,method='geometry',confidence_checks=False),'selected':cfg,
             'abl_no_smoothing':replace(cfg,smooth_s=.0001,max_slew_db_s=10000),
             'abl_no_hold':replace(cfg,fallback='unity'),
             'abl_max_reference':replace(cfg,method='energy',percentile=100)}
    out=[]
    for sc in test:
        for policy in ['event','fixed_2m']:
            for name,base in methods.items():
                c=replace(base,reference_distance_m=2. if policy=='fixed_2m' else None)
                start=time.perf_counter();y,g,logs=run(sc,c);elapsed=time.perf_counter()-start
                y=pcm(y).astype(float)/32767 # Score the exact delivered PCM convention.
                r={k:sc['row'][k] for k in ['scenario_id','split','source_group','profile','sensor_condition']}
                r.update(method=name,reference_policy=policy,**evaluate(sc,y,g,logs,policy),runtime_s=elapsed,real_time_factor=elapsed/10)
                out.append(r)
            for name in ['feedback_agc_2016','powerlaw_drc_2021']:
                y,g,logs=paper_baseline(sc['x'],name,cfg.max_gain_db);y=pcm(y).astype(float)/32767
                r={k:sc['row'][k] for k in ['scenario_id','split','source_group','profile','sensor_condition']}
                r.update(method=name,reference_policy=policy,**evaluate(sc,y,g,logs,policy));out.append(r)
        print('Test comparisons:',sc['row']['scenario_id'],flush=True)
    assert all(r['split']=='test' for r in out)
    csv_out(ROOT/'metrics/test_per_case.csv',out)
    summaries=[]
    for policy in ['event','fixed_2m']:
        for name in list(methods)+['feedback_agc_2016','powerlaw_drc_2021']:
            rs=[r for r in out if r['method']==name and r['reference_policy']==policy]
            rr=dict(method=name,reference_policy=policy,split='test',n_cases=len(rs))
            for k in rs[0]:
                if isinstance(rs[0][k],(int,float)):
                    a=np.array([r.get(k,np.nan) for r in rs],float)
                    if np.isfinite(a).any():rr[k+'_mean']=float(np.nanmean(a));rr[k+'_std']=float(np.nanstd(a))
            summaries.append(rr)
    csv_out(ROOT/'metrics/test_summary.csv',summaries)
    # Frozen-parameter robustness probes on TEST inputs only; never used for selection.
    fallback_rows=[]
    for sc in test:
        for mode in ['nominal','no_pose','no_imu','frozen_pose','pose_bias_3m']:
            ss=dict(sc);tele=[dict(v) for v in sc['telemetry']]
            for t in tele:
                if mode=='no_pose':t['pose_valid']=False
                if mode=='no_imu':t['imu_valid']=False
                if mode=='frozen_pose':t['pose']=sc['telemetry'][0]['pose']
                if mode=='pose_bias_3m':t['pose']=(np.asarray(t['pose'])+[3,0,0]).tolist()
            ss['telemetry']=tele;y,g,logs=run(ss,cfg);y=pcm(y).astype(float)/32767
            fallback_rows.append(dict(scenario_id=sc['row']['scenario_id'],split='test',condition=mode,**evaluate(ss,y,g,logs)))
    csv_out(ROOT/'metrics/test_sensor_probes.csv',fallback_rows)
    return out


def listening_index(index):
    import html
    body=['<!doctype html><meta charset="utf-8"><title>Target-only amplitude outputs</title><style>body{font:15px system-ui;max-width:1400px;margin:28px auto;color:#193444}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;padding:9px;text-align:left}audio{width:215px}small{color:#52636b}</style><h1>Target-only amplitude outputs</h1><p>No synthetic drone noise or denoising was used. Files share a fixed level convention: do not normalize them separately when comparing. Rec5 is a duplicate alias of rec4, not a separate test source.</p><p><b>Event output:</b> boost relative to closest observed range so far. <b>2 m output:</b> optional calibration to the dataset\'s known 2 m reference level.</p><table><tr><th>Rec / case</th><th>Input at drone</th><th>Event output</th><th>2 m output</th><th>Reference at 2 m</th></tr>']
    for r in index:
        label=html.escape(r['rec_number']+' / '+r['scenario_id'])
        body.append('<tr><td>'+label+'<br><small>'+html.escape(r['split']+'; '+r['sensor_condition'])+'</small></td>'+''.join('<td><audio controls preload="none" src="../'+html.escape(r[k],quote=True)+'"></audio></td>' for k in ['input_path','event_output_path','calibrated_output_path','reference_path'])+'</tr>')
    body.append('</table>')
    (ROOT/'listening/index.html').write_text('\n'.join(body),encoding='utf-8')


def main(args):
    data=Path(args.data).resolve();manifest=rows_in(data/'manifest.csv')
    train=[load(data,r) for r in manifest if r['split']=='train']
    cfg=Config(**json.loads((ROOT/'configs/selected.json').read_text())) if args.resume else tune(train)
    freeze=hashlib.sha256((ROOT/'configs/selected.json').read_bytes()).hexdigest()
    json_out(ROOT/'configs/freeze_record.json',dict(configuration_sha256=freeze,selected_before_new_test_scoring=True,
        source_sha256=hashlib.sha256((ROOT/'pipeline/core.py').read_bytes()).hexdigest(),
        input_policy='Only target_at_drone.wav; reference and true trajectory are evaluation-only; no mixture/noise waveform opened.',
        report_policy='Performance tables only on test; train and validation audio exported without performance metrics.'))
    test=[load(data,r) for r in manifest if r['split']=='test']
    test_metrics(test,cfg)
    groups=json.loads((data/'source_groups.json').read_text());names={g['source_group']:g['original_filenames'] for g in groups}
    index=[]
    for i,sc in enumerate(train+test):
        index.extend(export_case(sc,cfg,names[sc['row']['source_group']]))
        print(f'Exported {i+1}/75 unique cases',flush=True)
    csv_out(ROOT/'listening/audio_index.csv',index);listening_index(index)
    clean_manifest=[{k:r[k] for k in ['scenario_id','split','source_group','profile','sensor_condition','source_offset_samples']} for r in manifest]
    csv_out(ROOT/'data/manifest.csv',clean_manifest);json_out(ROOT/'data/source_groups.json',groups)
    assert hashlib.sha256((ROOT/'configs/selected.json').read_bytes()).hexdigest()==freeze
    json_out(ROOT/'data/export_audit.json',dict(unique_cases=75,unique_train_cases=60,unique_test_cases=15,listening_rows=len(index),
        output_wavs=2*len(index),duplicate_alias_cases=15,source_rec_numbers=['rec1','rec2','rec3','rec4','rec5'],
        synthetic_noise_added=False,denoising_applied=False,performance_tables_test_only=True,
        input_wavs_copied_bit_exact=True,output_format='Mono 16 kHz PCM16, fixed scale, no per-file normalization.'))
    print('Target-only experiments and all-case listening exports complete.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',default=str(ROOT/'data'));p.add_argument('--resume',action='store_true')
    main(p.parse_args())
