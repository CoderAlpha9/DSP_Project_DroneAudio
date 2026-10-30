"""Isolated enhancement timing rerun, excluding I/O and objective metric evaluation."""
import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'original/code'))
from experiment import *
_,runs=inventory();run=next(r for r in runs if r['scenario']=='S3' and r['seed']==303)
s,y,rpm=load(run);rows=[]
for cfg in json.loads((ROOT/'results/frozen_configs.json').read_text()):
    enhance(y,rpm,cfg)
    for repetition in range(5):
        _,seconds=enhance(y,rpm,cfg)
        rows.append(dict(method=cfg['family'],repetition=repetition,seconds=seconds,rtf=seconds/(len(y)/FS)))
pd.DataFrame(rows).to_csv(ROOT/'results/isolated_timing_rerun.csv',index=False)
