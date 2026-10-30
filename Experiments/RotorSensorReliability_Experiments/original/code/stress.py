"""Secondary frozen-configuration ablations and diagnostics, no holdout tuning."""
from experiment import *
import warnings
warnings.filterwarnings('ignore',category=wavfile.WavFileWarning)

_,runs=inventory();runs=[r for r in runs if r['seed']==303]
cfgs={c['family']:c for c in json.loads((ROOT/'results/frozen_configs.json').read_text())}
rows=[];linear=[];audio=ROOT/'listening';audio.mkdir(exist_ok=True)
for run in runs:
    s,y,rpm=load(run,-5);T=len(rpm);tt=np.arange(T)*HOP/FS
    cases={'nominal':rpm,'gain_x1.2':rpm*1.2,'per_motor_gain':rpm*np.array([.7,1.2,1.5,.9]),
           'frozen':np.broadcast_to(np.median(rpm[4:51],axis=0),rpm.shape).copy(),
           'reversed':rpm[::-1].copy(),'gain_drift_20pct':rpm*(1+.2*tt/15)[:,None]}
    for lag in [.05,.2,.5]:
        cases[f'lag_{lag}s']=np.stack([np.interp(tt-lag,tt,rpm[:,j]) for j in range(4)],axis=1)
    rr=rpm.copy();rr[(tt>5)&(tt<9)]=rr[np.searchsorted(tt,5)-1];cases['dropout_4s']=rr
    # This family of controls is fixed before inspecting held-out metric values.
    for method in ['static','spp','rrst','rrst_guard']:
        ref=None
        for case,r in cases.items():
            if method in ['static','spp'] and case!='nominal':continue
            out,sec=enhance(y,r,cfgs[method]);ref=out if ref is None else ref
            rows.append(dict(scenario=run['scenario'],method=method,case=case,max_output_delta=float(np.max(abs(out-ref))),**metrics(s,out)))
    for method in ['rrst','rrst_guard']:
        for cal in [.2,.4,.8,1.5,3.]:
            cfg=dict(cfgs[method],cal=cal);out,sec=enhance(y,rpm,cfg)
            rows.append(dict(scenario=run['scenario'],method=method,case=f'calibration_{cal}s',max_output_delta=0,**metrics(s,out)))
    for method in ['static','spp','rrst','rrst_guard','rrst_broadband']:
        cfg=cfgs[method];out,sec,state=enhance(y,rpm,cfg,True);G=state['G']
        # Apply the identical mixture-derived gain to each component: exact conditional linear audit.
        speech=istft(stft(s)*G,len(s));noise=istft(stft(y-s)*G,len(s));sl=slice(32000,224000)
        speech_dist=10*np.log10((np.sum(s[sl]**2)+EPS)/(np.sum((speech[sl]-s[sl])**2)+EPS))
        attenuation=10*np.log10((np.sum((y[sl]-s[sl])**2)+EPS)/(np.sum(noise[sl]**2)+EPS))
        low=signal.sosfiltfilt(signal.butter(6,3900,fs=FS,output='sos'),out)
        ss=signal.sosfiltfilt(signal.butter(6,3900,fs=FS,output='sos'),s)
        linear.append(dict(scenario=run['scenario'],method=method,speech_distortion_snr=speech_dist,noise_attenuation_db=attenuation,reconstruction_max_error=float(np.max(abs(out-speech-noise))),lowpass_sisdr=metrics(ss,low)['sisdr']))
        if run['scenario'] in ['S3','S7']:wavfile.write(audio/f'{run["scenario"]}_{method}.wav',FS,out.astype(np.float32))
    if run['scenario'] in ['S3','S7']:
        wavfile.write(audio/f'{run["scenario"]}_clean.wav',FS,s.astype(np.float32));wavfile.write(audio/f'{run["scenario"]}_noisy.wav',FS,y.astype(np.float32))
    pd.DataFrame(rows).to_csv(ROOT/'results/stress.csv',index=False);pd.DataFrame(linear).to_csv(ROOT/'results/linear_audit.csv',index=False)
    print('stress',run['scenario'],flush=True)
