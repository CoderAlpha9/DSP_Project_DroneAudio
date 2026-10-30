"""Verify original ZIP bytes, safely extract captures and audit the selected dataset."""
import hashlib,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
src=Path(sys.argv[1]);expected=json.loads((BASE/'evidence/original/execution_environment.json').read_text())['input_archives']
# Match bytes rather than filenames, so renamed uploads remain usable.
found={}
for p in src.glob('*.zip'):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    for name,digest in expected.items():
        if h.hexdigest()==digest:found[name]=p
missing=sorted(set(expected)-set(found))
if missing:raise SystemExit('Missing exact source archive bytes: '+', '.join(missing)+'. No extraction performed.')
for name,p in found.items():
    with zipfile.ZipFile(p) as z:
        bad=z.testzip()
        if bad:raise ValueError('CRC mismatch: '+bad)
        for info in z.infolist():
            n=info.filename;rel=Path(n.replace('\\','/'))
            if rel.is_absolute() or '..' in rel.parts or (rel.parts and ':' in rel.parts[0]):raise ValueError('Unsafe archive path: '+n)
            if info.is_dir() or n.endswith('.log') or '/plots/' in n:continue
            out=ROOT/('reference' if name=='data.zip' else 'data')/rel
            out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(z.read(info))
    print('Verified and extracted',p.name,'as',name)
from experiment import inventory,load
import numpy as np
_,runs=inventory()
if len(runs)!=19:raise ValueError(f'Expected 19 latest valid captures, found {len(runs)}')
for run in runs:
    for snr in [-10,-5,0,5]:
        s,y,r=load(run,snr)
        if len(s)!=240000 or len(y)!=240000:raise ValueError('Expected 15 seconds at 16 kHz')
        if not (np.isfinite(s).all() and np.isfinite(y).all() and np.isfinite(r).all()):raise ValueError('Nonfinite data')
        if np.max(np.abs(s[1024:12800]))!=0:raise ValueError('Speech overlaps calibration interval')
print('Validated 19 captures and 76 paired mixtures')
