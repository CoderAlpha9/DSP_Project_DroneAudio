"""Numerically compare reruns with archived rows; timing columns are deliberately excluded."""
import json,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];suite=sys.argv[1];run=Path(sys.argv[2])
files={'fresh': [('fresh_reliability_4896_v2.csv',['case_id','condition','method'],['stft_sdr']),('fresh_test_waveform_v2.csv',['case_id','condition','method'],['sisdr','snr']),('ridge_gate_test_v3.csv',['case_id','condition','method'],['sisdr','mean_lambda'])], 'original':[('all_results.csv',['scenario','seed','input_snr','method'],['sisdr','stoi','estoi','pesq_nb']),('stress.csv',['scenario','method','case'],['sisdr'])]}
checks=[]
for file,keys,metrics in files[suite]:
    a=pd.read_csv(ROOT/'evidence'/suite/file).set_index(keys).sort_index();b=pd.read_csv(run/file).set_index(keys).sort_index()
    if not a.index.equals(b.index):raise ValueError('Row mismatch: '+file)
    for metric in metrics:
        delta=float(np.max(np.abs(a[metric].to_numpy()-b[metric].to_numpy())))
        checks.append({'file':file,'metric':metric,'max_absolute_delta':delta,'within_tolerance':bool(delta<1e-7)})
result={'tolerance':1e-7,'all_within_tolerance':all(x['within_tolerance'] for x in checks),'checks':checks}
(run/'archive_comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if not result['all_within_tolerance']:raise SystemExit('Numerical mismatch with archived evidence; inspect archive_comparison.json')
