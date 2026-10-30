from pathlib import Path
import argparse,csv
import numpy as np
from scipy.io import wavfile
from .detector import detect
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args()
sr,x=wavfile.read(a.input)
if sr!=16000 or x.ndim!=1 or len(x)==0:raise ValueError('Use nonempty mono 16 kHz audio')
if x.dtype!=np.int16 and not np.issubdtype(x.dtype,np.floating):raise ValueError('Use PCM16 or float WAV')
x=x.astype(float)/(32767 if x.dtype==np.int16 else 1)
if not np.isfinite(x).all() or np.max(abs(x))>1.0001:raise ValueError('Use finite full-scale-normalized audio')
logs=detect(x);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
with (out/'trace.csv').open('w',newline='') as f:w=csv.DictWriter(f,logs[0]);w.writeheader();w.writerows(logs)
mask=np.array([r['event_active'] for r in logs],bool);ed=np.diff(np.r_[False,mask,False].astype(int))
for i,(s,e) in enumerate(zip(np.flatnonzero(ed==1),np.flatnonzero(ed==-1)),1):wavfile.write(out/f'event_{i:03d}.wav',sr,np.round(x[max(0,s*160-1600):min(len(x),e*160)]*32767).astype(np.int16))
print('Wrote event clips and general/speech trace')
