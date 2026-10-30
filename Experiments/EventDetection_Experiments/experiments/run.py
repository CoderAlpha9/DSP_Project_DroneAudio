from pathlib import Path
import sys,json,csv,hashlib,itertools,shutil,time,html
import numpy as np
from scipy.io import wavfile
from scipy import signal
from sklearn.linear_model import LogisticRegression
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pipeline.detector import extract,from_features,SETS,NAMES
from pipeline.amplitude import Config,AmplitudePipeline
from baselines import rvad_fast_2020,statistical_sad_2020

def jout(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def cout(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if not rows:return
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w',newline='') as f:w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def cin(p):
 with Path(p).open() as f:return list(csv.DictReader(f))
def win(p):
 sr,x=wavfile.read(p);assert sr==16000 and x.ndim==1;return x.astype(float)/32767
def wout(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);wavfile.write(p,16000,np.round(np.clip(x,-1,1)*32767).astype(np.int16))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def labels(row):
 ann=json.loads((ROOT/'configs/source_intervals.json').read_text());ts=(np.arange(1000)+.5)*.01+int(row['source_offset_samples'])/16000
 y=np.zeros(1000,bool);v=np.ones(1000,bool)
 for a,b in ann['sources'][row['source_group']]:y|=(ts>=a)&(ts<b);v&=(abs(ts-a)>.08)&(abs(ts-b)>.08)
 return y,v

def generic(noise,truth,kind,seed):
 rng=np.random.default_rng(seed);n=len(noise);fg=np.zeros(n);lab=np.zeros(n//160,bool);valid=np.ones_like(lab)
 times=[(rng.uniform(1,2),rng.uniform(.3,1.4)),(rng.uniform(5,7),rng.uniform(.5,2))]
 for start,duration in times:
  aa=int(start*16000);bb=min(n,aa+int(duration*16000));t=np.arange(bb-aa)/16000
  if kind=='chirp':s=signal.chirp(t,rng.uniform(300,900),t[-1],rng.uniform(1600,4300))
  elif kind=='tone':s=np.sin(2*np.pi*rng.uniform(450,2300)*t)+.3*np.sin(2*np.pi*rng.uniform(2500,4500)*t)
  else:
   s=signal.sosfilt(signal.butter(2,[rng.uniform(300,700),rng.uniform(2200,6000)],btype='bandpass',fs=16000,output='sos'),rng.normal(size=len(t)))
   s*=np.exp(-rng.uniform(1,5)*t)
  s*=signal.windows.tukey(len(s),.15);s/=max(np.sqrt(np.mean(s*s)),1e-9)
  # Foreground and original background have overlapping level ranges, not class-coded levels.
  level=np.sqrt(np.mean(noise*noise))*10**(rng.uniform(-3,15)/20)
  fg[aa:bb]+=level*s*np.interp(np.arange(aa,bb)/16000,truth['time_s'],truth['distance_m'][0]/truth['distance_m'])
  lab[(np.arange(len(lab))+.5)*.01>=start] |= False
  tt=(np.arange(len(lab))+.5)*.01;lab|=(tt>=start)&(tt<bb/16000);valid&=(abs(tt-start)>.08)&(abs(tt-bb/16000)>.08)
 return noise+fg,fg,lab,valid

def prepare(original):
 rows=cin(Path(original)/'manifest.csv');manifest=[]
 for j,r in enumerate(rows):
  sid=r['scenario_id'];src=Path(original)/r['relative_path'];dst=ROOT/'data/original'/sid;dst.mkdir(parents=True,exist_ok=True)
  for fn in ['mixture.wav','noise.wav','target_at_drone.wav','reference.wav','telemetry_observed.csv','ground_truth_trajectory.csv','metadata.json']:shutil.copy2(src/fn,dst/fn)
  noise=win(dst/'noise.wav');mix=win(dst/'mixture.wav');y,v=labels(r)
  variants=[('speech',mix,win(dst/'target_at_drone.wav'),y,v),('background',noise,np.zeros_like(noise),np.zeros_like(y),np.ones_like(v))]
  truth=np.genfromtxt(dst/'ground_truth_trajectory.csv',delimiter=',',names=True)
  for k,kind in enumerate(['chirp','tone','impact_texture']):
   xx,fg,yy,vv=generic(noise,truth,kind,98211+j*47+k*7919);variants.append((kind,xx,fg,yy,vv))
  for kind,x,fg,yy,vv in variants:
   cid=sid+'__'+kind;path=ROOT/'data/cases'/cid;path.mkdir(parents=True,exist_ok=True)
   # Original speech input copied bit for bit, not remixed/re-normalized.
   if kind=='speech':shutil.copy2(dst/'mixture.wav',path/'input.wav')
   else:wout(path/'input.wav',x)
   np.savez_compressed(path/'truth.npz',activity=yy,speech=yy if kind=='speech' else np.zeros_like(yy),valid=vv,foreground=fg)
   manifest.append(dict(case_id=cid,base_id=sid,kind=kind,split=r['split'],source_group=r['source_group'],profile=r['profile'],snr_initial=r['initial_nominal_active_snr_db'],input_sha256=sha(path/'input.wav')))
  if j%10==0:print('Prepared',j+1,'/75 base cases',flush=True)
 cout(ROOT/'data/manifest.csv',manifest);shutil.copy2(Path(original)/'source_groups.json',ROOT/'data/source_groups.json')

def load(row):
 p=ROOT/'data/cases'/row['case_id'];x=win(p/'input.wav');truth=np.load(p/'truth.npz');fp=p/'features.npy'
 f=np.load(fp) if fp.exists() else extract(x)
 if not fp.exists():np.save(fp,f)
 return dict(row=row,x=x,f=f,y=truth['activity'].astype(bool),speech=truth['speech'].astype(bool),valid=truth['valid'].astype(bool))

def metric(y,p,v):
 y=np.asarray(y,bool)[v];p=np.asarray(p,bool)[v];tp=int(sum(y&p));fp=int(sum(~y&p));fn=int(sum(y&~p));tn=int(sum(~y&~p))
 pr=tp/(tp+fp) if tp+fp else float(not tp+fn);re=tp/(tp+fn) if tp+fn else 1.;miss=fn/max(1,tp+fn);fa=fp/max(1,tn+fp)
 return dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=pr,recall=re,f1=2*pr*re/(pr+re) if pr+re else 0.,dcf=.75*miss+.25*fa,miss_rate=miss,false_alarm_rate=fa)

def fit(cases,ids,C):
 f=np.concatenate([s['f'][s['valid']][::5] for s in cases]);y=np.concatenate([s['y'][s['valid']][::5] for s in cases]);sp=np.concatenate([s['speech'][s['valid']][::5] for s in cases])
 center=np.median(f,axis=0);scale=np.maximum(np.percentile(f,75,axis=0)-np.percentile(f,25,axis=0),.01)
 z=np.clip((f-center)/scale,-12,12);out=dict(center=center.tolist(),scale=scale.tolist())
 for name,target in [('activity',y),('speech',sp)]:
  model=LogisticRegression(C=C,max_iter=200,class_weight='balanced',solver='lbfgs').fit(z[:,ids],target)
  co=np.zeros(len(NAMES));co[ids]=model.coef_[0];out[name+'_coef']=co.tolist();out[name+'_intercept']=float(model.intercept_[0])
 return out

def tune(dev):
 cal=[s for s in dev if s['row']['source_group']!='c414202e8f9c'];val=[s for s in dev if s['row']['source_group']=='c414202e8f9c'];models={};protocol=[]
 for name,ids in SETS.items():
  best=None
  for C in [.1,1.]:
   model=fit(cal,ids,C)
   for on,release in itertools.product([.5,.7],[10,25]):
    m=dict(**model,state=dict(on=on,off=on*.6,onset=2,release=release,gap=40),feature_set=name,C=C)
    scores=[]
    for s in val:
     logs=from_features(s['f'],m);a=[v['activity_active'] for v in logs];sp=[v['speech_active'] for v in logs]
     scores.append(metric(s['y'],a,s['valid'])['dcf']+.5*metric(s['speech'],sp,s['valid'])['dcf'])
    score=float(np.mean(scores));protocol.append(dict(feature_set=name,C=C,on=on,release=release))
    if best is None or score<best[0]:best=(score,m)
  models[name]=best[1];print('Selected development model',name,flush=True)
 best_name=min(models,key=lambda n:np.mean([metric(s['y'],[v['activity_active'] for v in from_features(s['f'],models[n])],s['valid'])['dcf']+.5*metric(s['speech'],[v['speech_active'] for v in from_features(s['f'],models[n])],s['valid'])['dcf'] for s in val]))
 for k,m in models.items():jout(ROOT/'configs'/('model_'+k+'.json'),m)
 jout(ROOT/'configs/selected.json',models[best_name]);jout(ROOT/'configs/protocol.json',dict(calibration_cases=len(cal),validation_cases=len(val),candidates=protocol,selected_feature_set=best_name,objective='Validation mean activity DCF + .5 speech DCF; fitted on calibration only; development scores not exported',split='60 development / 15 test base recordings; all five descendants keep parent split',random_seed=98211))
 return models,best_name

def operate(s,model):
 logs=from_features(s['f'],model);core=AmplitudePipeline(Config(**json.loads((ROOT/'configs/amplitude.json').read_text())))
 base=ROOT/'data/original'/s['row']['base_id'];obs=np.genfromtxt(base/'telemetry_observed.csv',delimiter=',',names=True)
 ys=[];gs=[]
 for i in range(len(s['x'])//320):
  end=(i+1)*.02;j=max(0,np.searchsorted(obs['time_s'],end,side='right')-1)
  ev=any(l['event_active'] for l in logs[i*2:i*2+2]);pose=[obs['pose_enu_'+v+'_m'][j] for v in 'xyz']
  y,g,_=core.process_block(s['x'][i*320:(i+1)*320],pose=pose,pose_valid=bool(obs['pose_valid'][j]),event_active=True,reference_update=ev)
  ys.append(y);gs.append(g)
 return np.concatenate(ys),np.concatenate(gs),logs

def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--original');p.add_argument('--resume',action='store_true');a=p.parse_args()
 if a.original:prepare(a.original)
 rows=cin(ROOT/'data/manifest.csv');dev=[]
 for i,r in enumerate(rows):
  if r['split']=='train':dev.append(load(r))
  if i%40==0:print('Feature loading',i,flush=True)
 if a.resume:models={n:json.loads((ROOT/'configs'/('model_'+n+'.json')).read_text()) for n in SETS};best=json.loads((ROOT/'configs/selected.json').read_text())['feature_set']
 else:models,best=tune(dev)
 jout(ROOT/'configs/freeze.json',dict(selected_sha256=sha(ROOT/'configs/selected.json'),core_sha256=sha(ROOT/'pipeline/detector.py'),frozen_before_test_load=True,test_status='Reused recording benchmark, not fresh blind validation. Synthetic non-speech waveforms have disjoint seeds.'))
 test=[load(r) for r in rows if r['split']=='test'];metrics=[];frame=[];lit=[];amp=[]
 for s in test:
  r=s['row']
  for name,m in models.items():
   logs=from_features(s['f'],m)
   for task,key,truth in [('activity','activity_active',s['y']),('speech','speech_active',s['speech'])]:
    metrics.append(dict(case_id=r['case_id'],kind=r['kind'],split='test',method=name,task=task,**metric(truth,[l[key] for l in logs],s['valid'])))
  logs=from_features(s['f'],models[best])
  frame.extend(dict(case_id=r['case_id'],split='test',reference_activity=int(s['y'][i]),reference_speech=int(s['speech'][i]),scored=int(s['valid'][i]),**v) for i,v in enumerate(logs))
  if r['kind']=='speech':
   for name,pred in [('rvad_fast_2020_adapted',rvad_fast_2020(s['x'])),('statistical_sad_2020_adapted',statistical_sad_2020(s['x']))]:lit.append(dict(case_id=r['case_id'],split='test',method=name,**metric(s['speech'],pred,s['valid'])))
   yy,gg,_=operate(s,models[best]);path=ROOT/'data/original'/r['base_id'];target=win(path/'target_at_drone.wav');noise=win(path/'noise.wav');ref=win(path/'reference.wav');act=np.repeat(s['y']&s['valid'],160)
   sn=lambda t,n:float(10*np.log10((np.sum(t[act]**2)+1e-20)/(np.sum(n[act]**2)+1e-20)))
   amp.append(dict(case_id=r['case_id'],split='test',input_component_snr_db=sn(target,noise),output_component_snr_db=sn(target*gg,noise*gg),output_vs_reference_rmse=float(np.sqrt(np.mean((yy[act]-ref[act])**2))),note='Gain applies equally to target and noise; not denoising'))
 cout(ROOT/'metrics/test_per_case.csv',metrics);cout(ROOT/'metrics/test_frames.csv',frame);cout(ROOT/'metrics/test_literature_speech.csv',lit);cout(ROOT/'metrics/test_mixture_amplitude.csv',amp)
 # All-case processed mixed audio and generic-event clips, including development cases.
 groups=json.loads((ROOT/'data/source_groups.json').read_text());mapping={g['source_group']:g['original_filenames'] for g in groups};index=[]
 for i,s in enumerate(dev+test):
  r=s['row'];logs=from_features(s['f'],models[best]);d=ROOT/'outputs'/r['case_id'];d.mkdir(parents=True,exist_ok=True);cout(d/'trace.csv',logs)
  mask=np.array([v['event_active'] for v in logs],bool);ed=np.diff(np.r_[False,mask,False].astype(int));events=[]
  for k,(aa,bb) in enumerate(zip(np.flatnonzero(ed==1),np.flatnonzero(ed==-1)),1):
   start=max(0,aa*160-1600);end=min(len(s['x']),bb*160);wout(d/f'events/{k:03d}.wav',s['x'][start:end]);events.append(dict(event_id=k,start_s=start/16000,end_s=end/16000,decision_available_s=(aa+1)*.01))
  cout(d/'events.csv',events)
  if r['kind']=='speech':
   y,_,_=operate(s,models[best]);wout(d/'amplitude_on_mixture.wav',y)
  index.append(dict(**r,rec_labels=' / '.join(mapping[r['source_group']]),event_count=len(events),output_dir=str(d.relative_to(ROOT))))
  if i%50==0:print('Exported',i,flush=True)
 cout(ROOT/'listening/index.csv',index)
 parts=['<!doctype html><meta charset="utf-8"><title>Mixed audio event outputs</title><style>body{font:14px system-ui;margin:24px}td,th{border:1px solid #ccc;padding:8px}audio{width:240px}</style><h1>Mixed-audio acoustic event detector</h1><p>Original mixtures, and non-speech probes over the original moving background. General events and speech flags are separate. Amplitude output is not denoised.</p><table><tr><th>Case / recording</th><th>Mixed input</th><th>Operated mixture / event list</th></tr>']
 for r in index:
  cid=r['case_id'];extra=f'<a href="../outputs/{cid}/events.csv">event timestamps</a>' if r['event_count'] else 'No events'
  if r['kind']=='speech':extra=f'<audio controls preload="none" src="../outputs/{cid}/amplitude_on_mixture.wav"></audio><br>'+extra
  extra+=''.join(f'<a href="../outputs/{cid}/events/{k:03d}.wav">clip {k}</a> ' for k in range(1,r['event_count']+1))
  parts.append(f'<tr><td>{html.escape(r["rec_labels"])}<br>{cid}<br>{r["split"]}</td><td><audio controls preload="none" src="../data/cases/{cid}/input.wav"></audio></td><td>{extra}</td></tr>')
 (ROOT/'listening/index.html').write_text(''.join(parts)+'</table>')
 print('Done',best,flush=True)
if __name__=='__main__':main()
