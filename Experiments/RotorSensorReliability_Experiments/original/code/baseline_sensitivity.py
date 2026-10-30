"""Expanded baseline-only development tuning. Not used to tune RPM method."""
from experiment import *
import warnings
warnings.filterwarnings('ignore',category=wavfile.WavFileWarning)
_,runs=inventory();dev=[r for r in runs if r['seed']==101 and r['scenario']!='S7']
cfgs=[]
for family in ['static','spp','lsa']:
    for a in [.9,.96,.98]:
        for b in [.7,1,1.5,2.]:
            for na in ([.85,.95,.98] if family=='spp' else [.95]):
                cfgs.append(dict(id=f'{family}_a{a}_b{b}_na{na}',family=family,alpha=a,beta=b,noise_alpha=na,floor=.04))
if os.environ.get('RPM_RETUNE')=='1':
    rows=[]
    for r in dev:
        s,y,rpm=load(r)
        for c in cfgs:
            out,sec=enhance(y,rpm,c);a=s[32000:224000];b=out[32000:224000];a=a-a.mean();b=b-b.mean();p=np.dot(a,b)/(np.dot(a,a)+EPS)*a
            rows.append(dict(scenario=r['scenario'],config=c['id'],family=c['family'],sisdr=10*np.log10((np.dot(p,p)+EPS)/(np.dot(b-p,b-p)+EPS))))
    df=pd.DataFrame(rows);df.to_csv(ROOT/'results/baseline_development.csv',index=False)
    scores=df.groupby(['family','config']).sisdr.mean().reset_index();w=scores.loc[scores.groupby('family').sisdr.idxmax()]
    chosen=[next(c for c in cfgs if c['id']==i) for i in w.config]
    (ROOT/'results/baseline_frozen.json').write_text(json.dumps(chosen,indent=2));print(w.to_string(index=False),flush=True)
else:
    chosen=json.loads((ROOT/'results/baseline_frozen.json').read_text())
rows=[]
for r in runs:
    for snr in [-10,-5,0,5]:
        s,y,rpm=load(r,snr)
        for c in chosen:
            out,sec=enhance(y,rpm,c)
            rows.append(dict(scenario=r['scenario'],seed=r['seed'],run=r['run'],input_snr=snr,method=c['family']+'_expanded',split='development' if r['seed']==101 and r['scenario']!='S7' else ('validation' if r['seed']==202 and r['scenario']!='S7' else 'test'),seconds=sec,rtf=sec/15,**metrics(s,out)))
    pd.DataFrame(rows).to_csv(ROOT/'results/baseline_sensitivity.csv',index=False)
    print('baseline',r['scenario'],r['seed'],flush=True)
