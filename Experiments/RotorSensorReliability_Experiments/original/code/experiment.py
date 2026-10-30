"""Auditable offline monaural DSP experiment. Run with BLAS threads=1.
No reference signals or latent noise stems enter non-oracle enhancement.
"""
import os, json, time, argparse, hashlib, platform
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal, ndimage, special
from scipy.io import wavfile

ROOT=Path(os.environ.get('RPM_OUTPUT_ROOT', Path(__file__).resolve().parents[1]))
DATA=Path(os.environ.get('RPM_DATA_ROOT', Path(__file__).resolve().parents[1]/'data'))
FS=16000; NFFT=1024; HOP=256; EPS=1e-16

def inventory():
    allruns=[]
    for p in sorted(DATA.glob('S*/*/manifest.json')):
        m=json.loads(p.read_text())
        allruns.append(dict(path=str(p.parent),run=p.parent.name,scenario=m['scenario'],seed=m['seed'],duration=m['duration_seconds'],valid=m.get('speech_valid_for_metrics',False),manifest=m))
    # Freeze latest valid capture for each scenario / seed, avoiding convenience replication.
    keep={}
    for x in allruns:
        if x['valid']:keep[x['scenario'],x['seed']]=x
    return allruns, sorted(keep.values(),key=lambda x:(x['scenario'],x['seed']))

def stft(x):return signal.stft(x,FS,nperseg=NFFT,noverlap=NFFT-HOP,boundary='zeros',padded=True)[2]
def istft(X,n):return signal.istft(X,FS,nperseg=NFFT,noverlap=NFFT-HOP,boundary=True)[1][:n]
def load(run,snr=-5):
    p=Path(run['path']);sub={-5:'audio',-10:'audio_snr_sweep/m10p0dB',0:'audio_snr_sweep/p0p0dB',5:'audio_snr_sweep/p5p0dB'}[snr]
    _,s=wavfile.read(p/sub/'clean_speech.wav');_,y=wavfile.read(p/sub/'mic_noisy.wav')
    obs=p/'rpm_sensor_observed.csv';table=pd.read_csv(obs if obs.exists() else p/'rpm_aligned.csv')
    cols=[c for c in table if c.startswith('rpm_')][:4]
    # Missing telemetry is filled from the last observation; leading absence uses first valid value.
    raw=table[cols].to_numpy(float);raw[raw<=0]=np.nan
    raw=pd.DataFrame(raw).ffill().bfill().to_numpy()
    t=table.sim_time_s.to_numpy()-run['manifest']['audio_t0_sim_ns']/1e9
    tf=np.arange(stft(y).shape[1])*HOP/FS
    rpm=np.stack([np.interp(tf,t,raw[:,j]) for j in range(4)],1)
    return s.astype(float),y.astype(float),rpm

def initialize(Y,cal=.8):
    t=np.arange(Y.shape[1])*HOP/FS
    # Exclude center-zero padding and stop before known 1s speech-free prefix ends.
    use=(t>=.064)&(t<=cal)
    return np.maximum(np.mean(abs(Y[:,use])**2,axis=1),EPS)

def transport(n0,rpm,power=3.,mode='relative',tonal=True):
    f=np.fft.rfftfreq(NFFT,1/FS);T=rpm.shape[0]
    ref=np.median(rpm[4:51],axis=0)
    ratios=np.clip(rpm/ref,0.35,2.5)
    smooth=ndimage.percentile_filter(n0,percentile=25,size=13,mode='nearest')
    smooth=ndimage.gaussian_filter1d(smooth,3)
    lines=np.maximum(n0-smooth,0)
    if not tonal:smooth=n0;lines=np.zeros_like(n0)
    out=np.zeros((len(f),T))
    for j in range(4):
        for t in range(T):
            r=ratios[t,j] if mode!='fixed' else 1.
            # Frequency-axis transport; amplitude law is a development-tuned nuisance parameter.
            out[:,t]+=(smooth + np.interp(f/r,f,lines,left=lines[0],right=0))*r**power/4
    return np.maximum(out,EPS)

def tracker(P,n0,alpha=.95):
    # Speech-presence-probability recursive noise tracking, fixed 15 dB speech hypothesis.
    n=n0.copy();prob=np.zeros(len(n));out=np.empty_like(P);xi=31.6227766
    for t in range(P.shape[1]):
        gamma=P[:,t]/np.maximum(n,EPS)
        pp=1/(1+(1+xi)*np.exp(np.maximum(-gamma*xi/(1+xi),-700)))
        prob=.9*prob+.1*pp
        pp=np.where(prob>.99,np.minimum(pp,.99),pp)
        n=alpha*n+(1-alpha)*((1-pp)*P[:,t]+pp*n)
        out[:,t]=n
    return np.maximum(out,EPS)

def gains(Y,N,beta=1.,floor=.04,kind='dd',alpha=.96):
    P=abs(Y)**2;gamma=np.minimum(P/np.maximum(N,EPS),1e5)
    if kind=='sub':return np.sqrt(np.maximum(1-beta/np.maximum(gamma,EPS),floor**2))
    prev=np.zeros(Y.shape[0]);G=np.empty_like(P)
    for t in range(Y.shape[1]):
        xi=alpha*prev+(1-alpha)*np.maximum(gamma[:,t]-beta,0)
        g=xi/(xi+beta)
        if kind=='lsa':
            v=np.maximum(g*gamma[:,t],1e-8)
            g=np.minimum(g*np.exp(.5*special.exp1(v)),1)
        g=np.maximum(g,floor)
        G[:,t]=g;prev=g*g*gamma[:,t]
    return G

def enhance(y,rpm,cfg,return_state=False):
    start=time.perf_counter();Y=stft(y);P=abs(Y)**2;n0=initialize(Y,cfg.get('cal',.8));name=cfg['family']
    if name=='raw':G=np.ones_like(P)
    elif name=='bandpass':
        f=np.fft.rfftfreq(NFFT,1/FS);G=np.broadcast_to(((f>=80)&(f<=3900))[:,None].astype(float),P.shape).copy()
    else:
        if name in ['static','sub','lsa']:N=np.broadcast_to(n0[:,None],P.shape)
        elif name in ['spp','rrst_guard']:
            N=tracker(P,n0,cfg.get('noise_alpha',.95))
        elif name=='quantile':
            sm=ndimage.uniform_filter1d(P,5,axis=1,mode='nearest')
            N=ndimage.percentile_filter(sm,percentile=20,size=(1,63),mode='nearest')*2.1
        if name in ['rrst','rrst_guard','rrst_broadband']:
            prior=transport(n0,rpm,cfg.get('power',3),tonal=name!='rrst_broadband')
            if name=='rrst_guard':
                # Conservative bounded prior: no unbounded PSD extrapolation versus audio estimate.
                c=cfg.get('cap',4.);prior=np.clip(prior,N/c,N*c)
                N=cfg.get('weight',.75)*prior+(1-cfg.get('weight',.75))*N
            else:N=prior
        G=gains(Y,N,cfg.get('beta',1),cfg.get('floor',.04),kind={'sub':'sub','lsa':'lsa'}.get(name,'dd'),alpha=cfg.get('alpha',.96))
        if cfg.get('band',False):
            f=np.fft.rfftfreq(NFFT,1/FS);G[(f<80)|(f>3900)]=0
    out=istft(Y*G,len(y));rt=time.perf_counter()-start
    return (out,rt,dict(Y=Y,G=G)) if return_state else (out,rt)

def metrics(s,x):
    # Fixed evaluation interval, same for every method; never used to enhance.
    a=s[32000:224000].copy();b=x[32000:224000].copy()
    snr=10*np.log10((np.dot(a,a)+EPS)/(np.dot(a-b,a-b)+EPS))
    a-=a.mean();b-=b.mean();proj=np.dot(a,b)/(np.dot(a,a)+EPS)*a
    si=10*np.log10((np.dot(proj,proj)+EPS)/(np.dot(b-proj,b-proj)+EPS))
    # Framewise SNR 20ms with a clean-energy gate only for evaluation.
    aa=a[:len(a)//320*320].reshape(-1,320);bb=b[:len(b)//320*320].reshape(-1,320)
    pa=(aa**2).sum(1);pe=((aa-bb)**2).sum(1);active=pa>pa.max()*.001
    seg=np.mean(np.clip(10*np.log10((pa[active]+EPS)/(pe[active]+EPS)),-10,35))
    r=dict(snr=float(snr),sisdr=float(si),segsnr=float(seg))
    try:
        from pystoi import stoi
        r['stoi']=float(stoi(a,b,FS,extended=False));r['estoi']=float(stoi(a,b,FS,extended=True))
    except ImportError:pass
    try:
        from pesq import pesq
        r['pesq_nb']=float(pesq(8000,signal.resample_poly(a,1,2),signal.resample_poly(b,1,2),'nb'))
    except ImportError:pass
    return r

def configurations():
    return [dict(id='raw',family='raw'),dict(id='bandpass',family='bandpass')]+[
        dict(id=f'{fam}_b{b}_f{f}',family=fam,beta=b,floor=f)
        for fam in ['static','sub','lsa','spp','quantile'] for b in [.7,1.,1.5] for f in [.04,.12]
    ]+[
        dict(id=f'{fam}_p{p}_b{b}',family=fam,power=p,beta=b,floor=.04)
        for fam in ['rrst','rrst_guard','rrst_broadband'] for p in [0.,2.,4.,6.] for b in [.7,1.,1.5]
    ]

def development():
    _,runs=inventory();runs=[x for x in runs if x['seed']==101 and x['scenario']!='S7']
    cfgs=configurations();rows=[]
    for run in runs:
        s,y,rpm=load(run)
        for cfg in cfgs:
            out,sec=enhance(y,rpm,cfg)
            # Tuning uses objective waveform SI-SDR only, on development partition.
            a=s[32000:224000];b=out[32000:224000];a=a-a.mean();b=b-b.mean();p=np.dot(a,b)/(np.dot(a,a)+EPS)*a
            si=10*np.log10((np.dot(p,p)+EPS)/(np.dot(b-p,b-p)+EPS))
            rows.append(dict(scenario=run['scenario'],seed=run['seed'],config=cfg['id'],family=cfg['family'],sisdr=si,seconds=sec))
        print('development',run['scenario'],flush=True)
    df=pd.DataFrame(rows);df.to_csv(ROOT/'results/development.csv',index=False)
    scores=df.groupby(['family','config']).sisdr.mean().reset_index()
    winners=scores.loc[scores.groupby('family').sisdr.idxmax()]
    chosen=[next(c for c in cfgs if c['id']==i) for i in winners.config]
    (ROOT/'results/frozen_configs.json').write_text(json.dumps(chosen,indent=2))
    print(winners.to_string(index=False),flush=True)

def evaluate():
    _,runs=inventory();
    if len(runs)!=19: raise ValueError(f'Expected 19 selected captures, found {len(runs)}; run doctor first')
    cfgs=json.loads((ROOT/'results/frozen_configs.json').read_text());rows=[]
    for run in runs:
        for snr in [-10,-5,0,5]:
            s,y,rpm=load(run,snr)
            for cfg in cfgs:
                out,sec=enhance(y,rpm,cfg)
                rows.append(dict(scenario=run['scenario'],seed=run['seed'],run=run['run'],input_snr=snr,method=cfg['family'],split='development' if run['seed']==101 and run['scenario']!='S7' else ('validation' if run['seed']==202 and run['scenario']!='S7' else 'test'),seconds=sec,rtf=sec/(len(y)/FS),**metrics(s,out)))
            # Oracle ideal ratio mask: reference-dependent diagnostic, not deployable upper bound.
            Y=stft(y);S=stft(s);V=stft(y-s);G=abs(S)**2/(abs(S)**2+abs(V)**2+EPS)
            out=istft(Y*G,len(y))
            rows.append(dict(scenario=run['scenario'],seed=run['seed'],run=run['run'],input_snr=snr,method='oracle_irm',split=rows[-1]['split'],seconds=0,rtf=0,**metrics(s,out)))
        pd.DataFrame(rows).to_csv(ROOT/'results/main.csv',index=False)
        print('evaluated',run['scenario'],run['seed'],flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['development','evaluate']);a=ap.parse_args()
    globals()[a.action]()
