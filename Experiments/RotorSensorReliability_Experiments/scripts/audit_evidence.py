"""Check archived row-level evidence against paper counts and headline values."""
import json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];o=ROOT/'evidence/original';f=ROOT/'evidence/fresh';checks={}
d=pd.read_csv(o/'all_results.csv');checks['original_1216_rows']=len(d)==1216;checks['original_16_methods']=d.method.nunique()==16;checks['original_76_mixtures']=len(d[['run','input_snr']].drop_duplicates())==76;checks['original_19_captures']=d.run.nunique()==19;checks['no_original_duplicates']=not d.duplicated(['run','input_snr','method']).any()
checks['development_640_total']=len(pd.read_csv(o/'development.csv'))+len(pd.read_csv(o/'baseline_development.csv'))==640
hold=d[d.seed==303];checks['primary_6_captures']=hold.run.nunique()==6
means=hold.groupby('method').sisdr.mean();checks['paper_original_rrst']=bool(abs(means.rrst-8.309)<.001);checks['paper_original_static']=bool(abs(means.static_expanded-7.760)<.001)
s=pd.read_csv(f/'fresh_reliability_4896_v2.csv');w=pd.read_csv(f/'fresh_test_waveform_v2.csv');r=pd.read_csv(f/'ridge_gate_test_v3.csv')
checks['fresh_4896_rows']=len(s)==4896;checks['fresh_1344_waveforms']=len(w)==1344;checks['fresh_48_cases']=s.case_id.nunique()==48;checks['fresh_17_variants']=s.method.nunique()==17;checks['fresh_6_conditions']=s.condition.nunique()==6;checks['fresh_test_ids_16_47']=set(w.case_id)==set(range(16,48));checks['ridge_960_waveforms']=len(r)==960
p=w.pivot(index=['case_id','condition'],columns='method',values='sisdr');harm=float(((p['fixed_1.0']-p['fixed_0.0'])<-.5).mean());checks['paper_49pct_harm']=abs(harm-94/192)<1e-12
b=s[s.condition=='nominal'].set_index(['case_id','method']).stft_sdr;c=s[s.condition=='scale_1p2'].set_index(['case_id','method']).stft_sdr
checks['fresh_constant_scale_negative_control']=bool(np.max(np.abs(b-c))<1e-10)
for name,table,cols in [('original',d,['sisdr','snr','stoi','estoi','pesq_nb']),('fresh',w,['sisdr','snr'])]:checks[name+'_metrics_finite']=bool(np.isfinite(table[cols]).all().all())
result={'status':'PASS' if all(checks.values()) else 'FAIL','scope':'Archived evidence audit, not an original raw-data rerun','checks':checks,'headline_values':{'rrst_sisdr':float(means.rrst),'expanded_static_sisdr':float(means.static_expanded),'fresh_pure_rpm_harm_rate':harm}}
(ROOT/'validation/evidence_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if not all(checks.values()):raise SystemExit(1)
