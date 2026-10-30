"""Untuned pretrained DeepFilterNet release binary, with compensated lookahead."""
from experiment import *
import subprocess,warnings
warnings.filterwarnings('ignore',category=wavfile.WavFileWarning)
binary=Path(os.environ.get('RPM_DFN_BINARY',Path(__file__).resolve().parents[2]/'tools/deep-filter'))
expected='70775e251eee44c0f2451a1e833326cf8bcbbe304d3e7cd12851e6fce72ef7da'
if not binary.is_file() or hashlib.sha256(binary.read_bytes()).hexdigest()!=expected: raise ValueError('Missing or mismatched DeepFilterNet v0.5.6 binary; see README')
_,runs=inventory();cfg=next(c for c in json.loads((ROOT/'results/frozen_configs.json').read_text()) if c['family']=='rrst')
rows=[];tmp=ROOT/'neural_work';tmp.mkdir(exist_ok=True);outdir=tmp/'out';outdir.mkdir(exist_ok=True)
for run in runs:
    for snr in [-10,-5,0,5]:
        s,y,rpm=load(run,snr)
        for method in ['deepfilter','rrst_deepfilter']:
            if method=='rrst_deepfilter':audio,pre=enhance(y,rpm,cfg)
            else:audio=y;pre=0
            name=f'{run["scenario"]}_{run["seed"]}_{snr}_{method}.wav';inp=tmp/name
            wavfile.write(inp,FS,audio.astype(np.float32))
            start=time.perf_counter()
            result=subprocess.run([str(binary),'-D','-o',str(outdir),str(inp)],capture_output=True,text=True,check=True)
            sec=time.perf_counter()-start+pre
            fs,x=wavfile.read(outdir/name)
            if np.issubdtype(x.dtype,np.integer):x=x.astype(float)/32768
            if fs!=FS:raise ValueError(fs)
            if abs(len(x)-len(y))>FS*.1:raise ValueError('Unexpected output length')
            x=np.pad(x,(0,max(0,len(y)-len(x))))[:len(y)]
            split='development' if run['seed']==101 and run['scenario']!='S7' else ('validation' if run['seed']==202 and run['scenario']!='S7' else 'test')
            rows.append(dict(scenario=run['scenario'],seed=run['seed'],run=run['run'],input_snr=snr,method=method,split=split,seconds=sec,rtf=sec/15,**metrics(s,x)))
            if snr==-5 and run['seed']==303 and run['scenario'] in ['S3','S7']:
                wavfile.write(ROOT/'listening'/f'{run["scenario"]}_{method}.wav',FS,x.astype(np.float32))
            inp.unlink();(outdir/name).unlink()
    pd.DataFrame(rows).to_csv(ROOT/'results/neural.csv',index=False)
    print('neural',run['scenario'],run['seed'],flush=True)
