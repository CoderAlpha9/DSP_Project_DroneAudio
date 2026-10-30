import json, time, os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal, ndimage
from scipy.io import wavfile

ROOT=Path(os.environ.get('RPM_FRESH_OUTPUT', Path(__file__).resolve().parents[1]/'runs/fresh'))
ROOT.mkdir(parents=True,exist_ok=True)
SPEECH=Path(os.environ.get('RPM_SPEECH', Path(__file__).resolve().parents[1]/'original/reference/data/clean_speech/OSR_us_000_0010_8k.wav'))
FS=16000; NFFT=512; HOP=256; EPS=1e-12; DUR=6.0; CAL=.8
SNR_LEVELS=[-10,-5,0,5]
CONDITIONS=['nominal','scale_1p2','drift_m20','lag_200ms','hold_2s','time_reverse']
LAMBDAS=np.round(np.linspace(0,1,11),1)
ALL_METHODS=[f'fixed_{l:.1f}' for l in LAMBDAS] + ['bounded_0.75','adaptive_soft','adaptive_capped','adaptive_hard','oracle_fixed','oracle_psd']

# ---------- signal utilities ----------
def stft(x): return signal.stft(x,FS,nperseg=NFFT,noverlap=NFFT-HOP,boundary='zeros',padded=True)[2]
def istft(X,n): return signal.istft(X,FS,nperseg=NFFT,noverlap=NFFT-HOP,boundary=True)[1][:n]
def sisdr(a,b):
    a=a.astype(float).copy(); b=b.astype(float).copy(); a-=a.mean(); b-=b.mean(); p=np.dot(a,b)/(np.dot(a,a)+EPS)*a
    return float(10*np.log10((np.dot(p,p)+EPS)/(np.dot(b-p,b-p)+EPS)))
def snr(a,b): return float(10*np.log10((np.dot(a,a)+EPS)/(np.dot(a-b,a-b)+EPS)))
def spec_sdr(S,Sh): return float(10*np.log10((np.sum(np.abs(S)**2)+EPS)/(np.sum(np.abs(S-Sh)**2)+EPS)))

fs,x=wavfile.read(SPEECH); x=x.astype(float)/32768.0
SPEECH_ALL=signal.resample_poly(x,2,1) if fs==8000 else x
SPEECH_ALL=SPEECH_ALL/(np.max(np.abs(SPEECH_ALL))+EPS)*0.20

# ---------- generator ----------
def rpm_trajectory(rng,n):
    t=np.arange(n)/FS; base=rng.uniform(3200,5000,4); rpm=np.empty((n,4))
    for j in range(4):
        f1=rng.uniform(.04,.14); f2=rng.uniform(.15,.35)
        r=base[j]+rng.uniform(250,750)*np.sin(2*np.pi*f1*t+rng.uniform(0,6.28))+rng.uniform(70,300)*np.sin(2*np.pi*f2*t+rng.uniform(0,6.28))
        r+=rng.uniform(-350,350)*(t/DUR)
        c=rng.uniform(2,4.8); r+=rng.uniform(-450,450)*np.tanh((t-c)/rng.uniform(.25,.7))
        rpm[:,j]=r
    return np.clip(rpm,2200,6500)

def rotor_noise(rng,rpm):
    n=len(rpm); t=np.arange(n)/FS; y=np.zeros(n); orders=[2,4,6,8]
    for j in range(4):
        ph=np.cumsum(2*np.pi*(rpm[:,j]/60)/FS)+rng.uniform(0,6.28)
        q=rpm[:,j]/np.median(rpm[:int(CAL*FS),j]); env=rng.lognormal(-.1,.3)*(0.72+.28*q**2)
        env*=1+.12*np.sin(2*np.pi*rng.uniform(.04,.12)*t+rng.uniform(0,6.28))
        for oi,k in enumerate(orders): y += [1,.55,.30,.17][oi]*rng.uniform(.8,1.2)*env*np.sin(k*ph+rng.uniform(0,6.28))
    y/=np.std(y)+EPS
    b,a=signal.butter(3,[80/(FS/2),3900/(FS/2)],btype='band'); broad=signal.lfilter(b,a,rng.normal(size=n)); broad/=np.std(broad)+EPS
    speed=(rpm.mean(1)/np.median(rpm[:int(CAL*FS)].mean(1)))**1.4; broad*=speed
    b2,a2=signal.butter(3,220/(FS/2),btype='low'); wind=signal.lfilter(b2,a2,rng.normal(size=n)); wind/=np.std(wind)+EPS
    env=.15+sum(rng.uniform(.15,.6)*np.exp(-.5*((t-rng.uniform(1.5,5.5))/rng.uniform(.15,.6))**2) for _ in range(2)); wind*=env
    extra=np.zeros(n)
    if rng.random()<.55: extra=.18*np.sin(2*np.pi*rng.uniform(260,1700)*t+rng.uniform(0,6.28))
    z=.64*y+.30*broad+.17*wind+extra; return z/(np.std(z)+EPS)

def make_case(cid):
    rng=np.random.default_rng(51000+cid); n=int(DUR*FS); rpm=rpm_trajectory(rng,n); noise=rotor_noise(rng,rpm)
    nsp=n-FS; start=int(rng.integers(0,max(1,len(SPEECH_ALL)-nsp-1))); sp=SPEECH_ALL[start:start+nsp].copy(); sp=np.pad(sp,(0,max(0,nsp-len(sp))))[:nsp]
    taps=signal.firwin(13,rng.uniform(2600,3900),fs=FS); sp=signal.lfilter(taps,[1],sp); sp/=np.sqrt(np.mean(sp**2)+EPS)
    s=np.zeros(n); s[FS:]=sp; db=SNR_LEVELS[cid%4]
    scale=np.sqrt((np.mean(s[FS:]**2)+EPS)/(np.mean(noise[FS:]**2)+EPS)/(10**(db/10))); v=noise*scale; y=s+v
    q=.90/max(np.max(np.abs(y)),.90); return s*q,v*q,y*q,rpm,db

def corrupt(r,cond):
    x=r.copy(); n=len(x)
    if cond=='scale_1p2': x*=1.2
    elif cond=='drift_m20': x*=np.linspace(1,.8,n)[:,None]
    elif cond=='lag_200ms':
        d=int(.2*FS); x=np.vstack([np.repeat(x[:1],d,axis=0),x[:-d]])
    elif cond=='hold_2s':
        a=int(2.5*FS); b=min(n,a+int(2*FS)); x[a:b]=x[a]
    elif cond=='time_reverse': x=x[::-1].copy()
    return x

# ---------- estimators ----------
def rpm_frames(r,nframes):
    tf=np.arange(nframes)*HOP/FS; t=np.arange(len(r))/FS
    return np.stack([np.interp(tf,t,r[:,j]) for j in range(4)],1)
def init_psd(Y):
    tf=np.arange(Y.shape[1])*HOP/FS; use=(tf>=.032)&(tf<=CAL); return np.maximum(np.mean(np.abs(Y[:,use])**2,1),EPS)
def audio_tracker(P,n0,alpha=.95):
    n=n0.copy(); prob=np.zeros(len(n)); out=np.empty_like(P); xi=31.6228
    for t in range(P.shape[1]):
        gam=P[:,t]/np.maximum(n,EPS); pp=1/(1+(1+xi)*np.exp(np.maximum(-gam*xi/(1+xi),-700)))
        prob=.9*prob+.1*pp; pp=np.where(prob>.99,np.minimum(pp,.99),pp); n=alpha*n+(1-alpha)*((1-pp)*P[:,t]+pp*n); out[:,t]=n
    return np.maximum(out,EPS)
def transport(n0,rf,power=2.0):
    K=len(n0); T=len(rf); ref=np.median(rf[2:min(26,T)],0); q=np.clip(rf/np.maximum(ref,1),.35,2.5)
    sm=ndimage.percentile_filter(n0,25,size=9,mode='nearest'); sm=ndimage.gaussian_filter1d(sm,2); lines=np.maximum(n0-sm,0)
    bins=np.arange(K,dtype=float)[:,None]; out=np.zeros((K,T))
    for j in range(4):
        src=bins/ q[:,j][None,:]; i0=np.floor(src).astype(int); frac=src-i0; i1=i0+1
        valid=(i0>=0)&(i1<K); warped=np.zeros((K,T));
        ii0=np.clip(i0,0,K-1); ii1=np.clip(i1,0,K-1); warped[valid]=(lines[ii0][valid]*(1-frac[valid])+lines[ii1][valid]*frac[valid])
        out+=(sm[:,None]+warped)*(q[:,j][None,:]**power)/4
    return np.maximum(out,EPS)
def corr_score(P,NA,NR):
    f=np.fft.rfftfreq(NFFT,1/FS); b=(f>=100)&(f<=3200)
    yp=(np.log(P+EPS)-np.log(NA+EPS))[b]; rp=(np.log(NR+EPS)-np.log(NA+EPS))[b]
    yp-=ndimage.gaussian_filter1d(yp,4,axis=0,mode='nearest'); rp-=ndimage.gaussian_filter1d(rp,4,axis=0,mode='nearest')
    yp-=yp.mean(0,keepdims=True); rp-=rp.mean(0,keepdims=True); c=np.sum(yp*rp,0)/np.sqrt(np.sum(yp*yp,0)*np.sum(rp*rp,0)+EPS)
    return ndimage.uniform_filter1d(np.clip(c,-1,1),7,mode='nearest')
def gain(P,N,beta=1.0,floor=.04):
    # Same monotone Wiener-like gain for every fusion strategy in this auxiliary stress test.
    return np.maximum(P/(P+beta*N+EPS),floor)
def lam_from_corr(c,lo,hi): return np.clip((c-lo)/max(hi-lo,1e-6),0,1)

def state(s,v,y,r):
    Y=stft(y); S=stft(s); V=stft(v); P=np.abs(Y)**2; n0=init_psd(Y); NA=audio_tracker(P,n0); NR=transport(n0,rpm_frames(r,Y.shape[1])); NRc=np.clip(NR,NA/4,NA*4); c=corr_score(P,NA,NR)
    return Y,S,V,P,NA,NR,NRc,c

def N_for(method,NA,NR,NRc,c,gate,oracle_lam=None,V=None):
    if method.startswith('fixed_'): l=float(method.split('_')[1]); return (1-l)*NA+l*NR,l
    if method=='bounded_0.75': return .25*NA+.75*NRc,.75
    if method.startswith('adaptive_'):
        l=lam_from_corr(c,*gate)
        if method=='adaptive_hard': l=(l>=.5).astype(float)
        prior=NRc if method=='adaptive_capped' else NR; return NA*(1-l[None,:])+prior*l[None,:],float(np.mean(l))
    if method=='oracle_fixed': return (1-oracle_lam)*NA+oracle_lam*NR,oracle_lam
    if method=='oracle_psd': return np.maximum(np.abs(V)**2,EPS),np.nan
    raise ValueError(method)

# ---------- development tuning using reference only in development objective ----------
def tune():
    cases=[]
    for cid in range(16):
        s,v,y,r,db=make_case(cid)
        for cond in CONDITIONS:
            Y,S,V,P,NA,NR,NRc,c=state(s,v,y,corrupt(r,cond)); cases.append((cid,cond,db,Y,S,V,P,NA,NR,NRc,c))
    fixed=[]
    for l in LAMBDAS:
        vals=[]
        for *_,Y,S,V,P,NA,NR,NRc,c in cases:
            N,_=N_for(f'fixed_{l:.1f}',NA,NR,NRc,c,(0,1),V=V); vals.append(spec_sdr(S,gain(P,N)*Y))
        fixed.append((float(l),np.mean(vals),np.percentile(vals,10)))
    cand=[]
    for lo in [-.15,-.05,.05,.15,.25,.35]:
        for width in [.15,.25,.35,.50]:
            hi=lo+width; vals=[]
            for *_,Y,S,V,P,NA,NR,NRc,c in cases:
                N,_=N_for('adaptive_capped',NA,NR,NRc,c,(lo,hi),V=V); vals.append(spec_sdr(S,gain(P,N)*Y))
            cand.append((lo,hi,np.mean(vals),np.percentile(vals,10)))
    fdf=pd.DataFrame(fixed,columns=['lambda','mean_stft_sdr','p10_stft_sdr']); gdf=pd.DataFrame(cand,columns=['lo','hi','mean_stft_sdr','p10_stft_sdr'])
    fdf.to_csv(ROOT/'development_fixed_grid_v2.csv',index=False); gdf.to_csv(ROOT/'development_gate_grid_v2.csv',index=False)
    nonzero=max([x for x in fixed if x[0]>0],key=lambda z:z[1]+.25*z[2]); bestinc=max(fixed,key=lambda z:z[1]+.25*z[2]); bg=max(cand,key=lambda z:z[2]+.25*z[3])
    cfg={'best_nonzero_fixed_lambda':nonzero[0],'best_fixed_including_audio_lambda':bestinc[0],'gate_lo':bg[0],'gate_hi':bg[1],
         'development_objective':'mean complex-STFT SDR + 0.25 * 10th-percentile across 16 development acoustic realizations x 6 telemetry states; test SI-SDR never used for tuning'}
    (ROOT/'frozen_reliability_config_v2.json').write_text(json.dumps(cfg,indent=2)); return cfg

def run():
    t=time.time(); cfg=tune() if os.environ.get('RPM_RETUNE')=='1' else json.loads((ROOT/'frozen_reliability_config_v2.json').read_text()); gate=(cfg['gate_lo'],cfg['gate_hi']); bf=cfg['best_nonzero_fixed_lambda']; selected=['fixed_0.0',f'fixed_{bf:.1f}','fixed_1.0','bounded_0.75','adaptive_capped','adaptive_hard','oracle_psd']
    # 4896 spectral processing evaluations across all 48 fresh realizations and all 17 variants.
    spectral=[]; waveform=[]
    for cid in range(48):
        split='development' if cid<16 else 'test'; s,v,y,r,db=make_case(cid)
        for cond in CONDITIONS:
            Y,S,V,P,NA,NR,NRc,c=state(s,v,y,corrupt(r,cond))
            # oracle fixed lambda for diagnostic variant
            gvals=[]
            for l in LAMBDAS:
                N,_=N_for(f'fixed_{l:.1f}',NA,NR,NRc,c,gate,V=V); gvals.append((float(l),spec_sdr(S,gain(P,N)*Y)))
            ol=max(gvals,key=lambda x:x[1])[0]
            for m in ALL_METHODS:
                N,ml=N_for(m,NA,NR,NRc,c,gate,oracle_lam=ol,V=V); G=gain(P,N); spectral.append(dict(case_id=cid,split=split,input_snr=db,condition=cond,method=m,stft_sdr=spec_sdr(S,G*Y),mean_lambda=ml,corr_mean=float(np.mean(c))))
                if split=='test' and m in selected:
                    out=istft(G*Y,len(s)); so=istft(G*S,len(s)); no=istft(G*V,len(s)); a=s[FS:]; b=out[FS:]; sa=so[FS:]; nv=no[FS:]
                    waveform.append(dict(case_id=cid,input_snr=db,condition=cond,method=m,sisdr=sisdr(a,b),snr=snr(a,b),speech_sisdr=sisdr(a,sa),noise_atten_db=float(10*np.log10((np.mean(nv**2)+EPS)/(np.mean(v[FS:]**2)+EPS))),mean_lambda=ml,corr_mean=float(np.mean(c))))
        if (cid+1)%8==0: print('case',cid+1,flush=True)
    sdf=pd.DataFrame(spectral); wdf=pd.DataFrame(waveform); sdf.to_csv(ROOT/'fresh_reliability_4896_v2.csv',index=False); wdf.to_csv(ROOT/'fresh_test_waveform_v2.csv',index=False)
    sm=sdf[sdf.split=='test'].groupby(['condition','method']).agg(mean_stft_sdr=('stft_sdr','mean'),p10_stft_sdr=('stft_sdr',lambda x:np.percentile(x,10)),mean_lambda=('mean_lambda','mean')).reset_index(); sm.to_csv(ROOT/'fresh_test_spectral_summary_v2.csv',index=False)
    wm=wdf.groupby(['condition','method']).agg(mean_sisdr=('sisdr','mean'),p10_sisdr=('sisdr',lambda x:np.percentile(x,10)),mean_snr=('snr','mean'),speech_sisdr=('speech_sisdr','mean'),noise_atten_db=('noise_atten_db','mean'),mean_lambda=('mean_lambda','mean')).reset_index(); wm.to_csv(ROOT/'fresh_test_waveform_summary_v2.csv',index=False)
    ov=wdf.groupby('method').agg(mean_sisdr=('sisdr','mean'),median_sisdr=('sisdr','median'),p10_sisdr=('sisdr',lambda x:np.percentile(x,10)),worst_sisdr=('sisdr','min'),speech_sisdr=('speech_sisdr','mean'),noise_atten_db=('noise_atten_db','mean')).reset_index().sort_values('mean_sisdr',ascending=False); ov.to_csv(ROOT/'fresh_test_waveform_overall_v2.csv',index=False)
    piv=wdf.pivot_table(index=['case_id','condition','input_snr'],columns='method',values='sisdr').reset_index(); piv.to_csv(ROOT/'fresh_test_paired_v2.csv',index=False)
    rng=np.random.default_rng(9137); ids=piv.case_id.unique(); stats={}
    for target in [f'fixed_{bf:.1f}','fixed_1.0','fixed_0.0','bounded_0.75']:
        d=(piv['adaptive_capped']-piv[target]).to_numpy(); boots=[]
        for _ in range(3000):
            ss=rng.choice(ids,len(ids),replace=True); boots.append(np.mean(np.concatenate([d[piv.case_id.to_numpy()==i] for i in ss])))
        stats[target]={'mean_delta':float(np.mean(d)),'median_delta':float(np.median(d)),'ci95_case_cluster':[float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))],'win_rate':float(np.mean(d>0)),'harm_gt_0p5_rate':float(np.mean(d<-.5))}
    aud=piv['fixed_0.0'].to_numpy()
    for m in [f'fixed_{bf:.1f}','fixed_1.0','bounded_0.75','adaptive_capped']:
        d=piv[m].to_numpy()-aud; stats[m+'_vs_audio']={'mean_delta':float(np.mean(d)),'harm_gt_0p5_rate':float(np.mean(d<-.5)),'benefit_gt_0p5_rate':float(np.mean(d>.5))}
    res={'config':cfg,'spectral_evaluations':len(sdf),'test_waveform_reconstructions':len(wdf),'stats':stats,'elapsed_s':time.time()-t}
    (ROOT/'fresh_reliability_statistics_v2.json').write_text(json.dumps(res,indent=2)); print(json.dumps(res,indent=2)); print('\n',ov.to_string(index=False)); print('\nCONDITION\n',wm.to_string(index=False))
if __name__=='__main__': run()
