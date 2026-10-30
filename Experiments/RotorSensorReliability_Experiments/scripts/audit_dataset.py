"""Raw capture audit: source inclusion, audio validity, time and additive stems."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'original/code'))
from experiment import *
allruns,selected=inventory();selected_paths={r['path'] for r in selected};rows=[]
if len(selected)!=19:raise ValueError(f'Expected 19 selected captures, found {len(selected)}')
for run in allruns:
 p=Path(run['path']);m=run['manifest'];row={'scenario':run['scenario'],'run':run['run'],'seed':run['seed'],'seconds':run['duration'],'speech_valid':run['valid'],'selected':run['path'] in selected_paths,'source_start':m.get('speech_selection',{}).get('selected_start_s'),'speaker':m.get('speech_speaker_id'),'source_hash':m.get('speech_sha256')}
 audio=p/'audio';clean=audio/'clean_speech.wav'
 if clean.exists():
  fs,s=wavfile.read(clean);row['calibration_speech_peak']=float(np.max(abs(s[1024:min(len(s),12800)])))
 if row['selected']:
  if fs!=FS or s.ndim!=1 or len(s)!=240000:raise ValueError('Invalid clean audio: '+str(p))
  for db,sub in [(-5,'audio'),(-10,'audio_snr_sweep/m10p0dB'),(0,'audio_snr_sweep/p0p0dB'),(5,'audio_snr_sweep/p5p0dB')]:
   folder=p/sub;fy,y=wavfile.read(folder/'mic_noisy.wav');fc,c=wavfile.read(folder/'clean_speech.wav')
   if fy!=FS or fc!=FS or len(y)!=240000 or y.ndim!=1:raise ValueError('Invalid mixture: '+str(folder))
   npath=folder/'drone_noise.wav'
   if npath.exists():
    fn,n=wavfile.read(npath);error=float(np.max(np.abs(y.astype(float)-c.astype(float)-n.astype(float))))
    row[f'additive_max_error_{db}']=error
    if error>1e-6:raise ValueError('Nonadditive mixture: '+str(folder))
   _,_,rpm=load(run,db)
   if rpm.shape[1]!=4 or not np.isfinite(rpm).all():raise ValueError('Invalid aligned telemetry')
  if row['calibration_speech_peak']!=0:raise ValueError('Calibration contains clean speech')
 rows.append(row)
pd.DataFrame(rows).to_csv(ROOT/'results/dataset_audit_rerun.csv',index=False)
print('PASS: 19 selected captures, 76 paired mixtures; data and timing contract checked')
