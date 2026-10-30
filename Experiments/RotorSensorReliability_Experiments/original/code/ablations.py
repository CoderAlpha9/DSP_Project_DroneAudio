"""Matched-setting ablations, fault windows, reference and source verification."""
from experiment import *
import warnings,itertools
warnings.filterwarnings('ignore',category=wavfile.WavFileWarning)
_,runs=inventory();runs=[r for r in runs if r['seed']==303]
cfg=next(c for c in json.loads((ROOT/'results/frozen_configs.json').read_text()) if c['family']=='rrst')
rows=[];fault=[];mask_audit=[]
def short_si(a,b):
    a=a-a.mean();b=b-b.mean();p=np.dot(a,b)/(np.dot(a,a)+EPS)*a
    return float(10*np.log10((np.dot(p,p)+EPS)/(np.dot(b-p,b-p)+EPS)))
for r in runs:
    s,y,rpm=load(r,-5)
    tests=[('full',cfg,rpm),('no_amplitude',dict(cfg,power=0),rpm),('no_frequency_transport',dict(cfg,family='rrst_broadband'),rpm),('one_mean_speed',cfg,np.repeat(rpm.mean(1)[:,None],4,axis=1)),('no_rpm',dict(cfg,family='static'),rpm),('spp_same_gain',dict(cfg,family='spp'),rpm),('capped_fusion_same_gain',dict(cfg,family='rrst_guard'),rpm)]
    for name,c,rr in tests:
        out,sec=enhance(y,rr,c);rows.append(dict(scenario=r['scenario'],ablation=name,**metrics(s,out)))
        if r['scenario']=='S6':
            for label,a,b in [('pre_fault',2,7.5),('post_fault',7.5,14)]:
                sl=slice(int(a*FS),int(b*FS));fault.append(dict(method=name,window=label,sisdr=short_si(s[sl],out[sl])))
    pd.DataFrame(rows).to_csv(ROOT/'results/ablations.csv',index=False)
    print('ablation',r['scenario'],flush=True)
pd.DataFrame(fault).to_csv(ROOT/'results/fault_windows.csv',index=False)
# Deterministic theoretical identity check, not a speech experiment.
rng=np.random.default_rng(184);S=10**rng.uniform(-3,3,10000);N=10**rng.uniform(-3,3,10000);Nh=N*10**rng.uniform(-2,2,10000)
g=S/(S+Nh);g0=S/(S+N);R=(1-g)**2*S+g*g*N;R0=S*N/(S+N)
err=np.max(abs((R-R0)-(S+N)*(g-g0)**2))
(ROOT/'results/theory_check.json').write_text(json.dumps({'wiener_identity_max_absolute_error':float(err),'tested_positive_power_pairs':10000},indent=2))
