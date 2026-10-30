import json,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy import ndimage
sys.path.insert(0,str(Path(__file__).resolve().parent))
import fresh_reliability_study_v2 as m
R=m.ROOT; ORDERS=[2,4,6,8]

def ridge_score(P,rf):
    # Deployable score: are telemetry-predicted rotor ridges locally supported by microphone power?
    K,T=P.shape; df=m.FS/m.NFFT; scores=[]
    for j in range(4):
        for order in ORDERS:
            bins=np.rint((order*rf[:,j]/60)/df).astype(int)
            sc=np.zeros(T)
            for t,b in enumerate(bins):
                if b<8 or b>=K-8: continue
                loc=P[b-1:b+2,t]; peak_i=int(np.argmax(loc)); peak=float(loc[peak_i]); offset=peak_i-1
                bg=np.concatenate([P[b-7:b-3,t],P[b+4:b+8,t]])
                base=float(np.median(bg))+m.EPS
                cdb=10*np.log10((peak+m.EPS)/base)
                contrast=1/(1+np.exp(-(cdb-2.0)/1.5))
                align=np.exp(-0.5*(offset/0.65)**2)
                sc[t]=contrast*align
            scores.append(sc)
    A=np.stack(scores,axis=0)
    # Trim outliers from speech overlap by 35th-75th percentile averaging across predicted ridges.
    lo=np.percentile(A,35,axis=0); hi=np.percentile(A,75,axis=0)
    mask=(A>=lo[None,:])&(A<=hi[None,:]); s=np.sum(A*mask,axis=0)/np.maximum(mask.sum(axis=0),1)
    return ndimage.uniform_filter1d(s,size=9,mode='nearest')

def state(s,v,y,r):
    Y=m.stft(y); S=m.stft(s); V=m.stft(v); P=np.abs(Y)**2; n0=m.init_psd(Y); NA=m.audio_tracker(P,n0); rf=m.rpm_frames(r,Y.shape[1]); NR=m.transport(n0,rf); NRc=np.clip(NR,NA/4,NA*4); c=ridge_score(P,rf)
    return Y,S,V,P,NA,NR,NRc,c

def tune():
    cases=[]
    for cid in range(16):
        s,v,y,r,db=m.make_case(cid)
        for cond in m.CONDITIONS:
            Y,S,V,P,NA,NR,NRc,c=state(s,v,y,m.corrupt(r,cond)); cases.append((Y,S,V,P,NA,NR,NRc,c))
    cand=[]
    # Reliability score is in [0,1].
    for lo in [.15,.20,.25,.30,.35,.40,.45,.50]:
        for width in [.10,.15,.20,.30,.40]:
            hi=min(.95,lo+width); vals=[]; lams=[]
            for Y,S,V,P,NA,NR,NRc,c in cases:
                l=m.lam_from_corr(c,lo,hi); N=NA*(1-l[None,:])+NRc*l[None,:]; vals.append(m.spec_sdr(S,m.gain(P,N)*Y)); lams.append(np.mean(l))
            cand.append((lo,hi,np.mean(vals),np.percentile(vals,10),np.mean(lams)))
    df=pd.DataFrame(cand,columns=['lo','hi','mean_stft_sdr','p10_stft_sdr','mean_lambda']); df.to_csv(R/'ridge_gate_development_grid_v3.csv',index=False)
    # reward useful telemetry use mildly, but only after quality criterion. This avoids the trivial always-audio selector.
    # Primary selection remains quality. Among candidates within 0.01 dB of best quality objective, choose largest mean lambda.
    df['objective']=df.mean_stft_sdr+.25*df.p10_stft_sdr; best=df.objective.max(); pool=df[df.objective>=best-.01]; row=pool.sort_values(['mean_lambda','objective'],ascending=False).iloc[0]
    cfg={'gate_lo':float(row.lo),'gate_hi':float(row.hi),'dev_mean_lambda':float(row.mean_lambda),'dev_mean_stft_sdr':float(row.mean_stft_sdr),'dev_p10_stft_sdr':float(row.p10_stft_sdr),'selection':'maximize mean STFT-SDR + .25*p10; within 0.01 dB choose the candidate with greater telemetry use to reject the trivial always-audio solution'}
    (R/'ridge_gate_config_v3.json').write_text(json.dumps(cfg,indent=2)); print(json.dumps(cfg,indent=2)); print(df.sort_values('objective',ascending=False).head(12).to_string(index=False))
if __name__=='__main__': tune()
