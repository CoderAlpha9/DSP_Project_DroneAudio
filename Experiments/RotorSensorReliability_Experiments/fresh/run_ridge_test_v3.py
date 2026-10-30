import json,sys,time
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
import fresh_reliability_study_v2 as m
import ridge_audit_v3 as r3
R=m.ROOT
cfg=json.loads((R/'ridge_gate_config_v3.json').read_text()); gate=(cfg['gate_lo'],cfg['gate_hi'])
rows=[]; t0=time.time()
for cid in range(16,48):
    s,v,y,r,db=m.make_case(cid)
    for cond in m.CONDITIONS:
        Y,S,V,P,NA,NR,NRc,c=r3.state(s,v,y,m.corrupt(r,cond))
        methods={
          'audio_only': (NA,0.0),
          'fixed_0.1': (.9*NA+.1*NR,0.1),
          'pure_rpm': (NR,1.0),
          'bounded_0.75': (.25*NA+.75*NRc,0.75),
        }
        lam=m.lam_from_corr(c,*gate); methods['ridge_adaptive_capped']=(NA*(1-lam[None,:])+NRc*lam[None,:],float(np.mean(lam)))
        for name,(N,ml) in methods.items():
            G=m.gain(P,N); out=m.istft(G*Y,len(s)); a=s[m.FS:]; b=out[m.FS:]
            rows.append(dict(case_id=cid,input_snr=db,condition=cond,method=name,sisdr=m.sisdr(a,b),mean_lambda=ml,ridge_score_mean=float(np.mean(c))))
    if (cid-15)%4==0: print('case',cid,'elapsed',round(time.time()-t0,1),flush=True)
df=pd.DataFrame(rows); df.to_csv(R/'ridge_gate_test_v3.csv',index=False)
ov=df.groupby('method').agg(mean_sisdr=('sisdr','mean'),p10_sisdr=('sisdr',lambda x:np.percentile(x,10)),mean_lambda=('mean_lambda','mean')).reset_index().sort_values('mean_sisdr',ascending=False)
ov.to_csv(R/'ridge_gate_test_overall_v3.csv',index=False)
piv=df.pivot_table(index=['case_id','condition'],columns='method',values='sisdr').reset_index()
rng=np.random.default_rng(31881); ids=piv.case_id.unique(); stats={}
for target in ['audio_only','fixed_0.1','pure_rpm','bounded_0.75']:
    d=(piv['ridge_adaptive_capped']-piv[target]).to_numpy(); boots=[]
    for _ in range(5000):
        ss=rng.choice(ids,len(ids),replace=True); boots.append(np.mean(np.concatenate([d[piv.case_id.to_numpy()==i] for i in ss])))
    stats[target]={'mean_delta':float(d.mean()),'median_delta':float(np.median(d)),'ci95_case_cluster':[float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))], 'win_rate':float(np.mean(d>0)), 'harm_gt_0p5_rate':float(np.mean(d<-.5)), 'benefit_gt_0p5_rate':float(np.mean(d>.5))}
res={'config':cfg,'n_case_conditions':int(len(piv)),'stats':stats,'overall':ov.to_dict(orient='records')}
(R/'ridge_gate_test_stats_v3.json').write_text(json.dumps(res,indent=2)); print('\n',ov.to_string(index=False)); print(json.dumps(res,indent=2))
