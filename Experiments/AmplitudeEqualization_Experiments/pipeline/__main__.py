import argparse,csv,json
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from .core import Config,process_audio


def main():
    p=argparse.ArgumentParser(description='Amplitude control for already isolated target audio; no denoising or synthetic noise.')
    p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--config',default=str(Path(__file__).resolve().parents[1]/'configs/selected.json'))
    p.add_argument('--telemetry');p.add_argument('--source-position',type=float,nargs=3,default=[0,0,0])
    p.add_argument('--fixed-reference-m',type=float);p.add_argument('--trace')
    a=p.parse_args();cfg=Config(**json.loads(Path(a.config).read_text()))
    if a.fixed_reference_m is not None:cfg.reference_distance_m=a.fixed_reference_m
    sr,x=wavfile.read(a.input)
    if sr!=cfg.sample_rate or x.ndim!=1:raise ValueError('Explicitly convert to mono at the configured sample rate')
    if x.dtype==np.int16:x=x.astype(float)/32767
    elif x.dtype.kind=='f':x=x.astype(float)
    else:raise ValueError('Use PCM16 or float WAV')
    n=round(sr*cfg.block_ms/1000);tele=None
    if a.telemetry:
        v=np.genfromtxt(a.telemetry,delimiter=',',names=True)
        if not np.all(np.diff(v['time_s'])>0):raise ValueError('Telemetry timestamps must strictly increase')
        tele=[]
        ends=(np.arange((len(x)+n-1)//n)+1)*(n/sr)
        for i in range(len(ends)):
            end=ends[i];j=int(np.searchsorted(v['time_s'],end,side='right')-1)
            fresh=j>=0 and end-v['time_s'][max(j,0)]<=.2;j=max(j,0)
            tele.append(dict(pose=[v['pose_enu_'+q+'_m'][j] for q in 'xyz'],pose_valid=fresh and bool(v['pose_valid'][j]),
                             imu_force=[v['accel_body_'+q+'_m_s2'][j] for q in 'xyz'],imu_valid=fresh and bool(v['imu_valid'][j])))
    y,g,logs=process_audio(x,cfg,telemetry=tele,source_position=a.source_position,event_active=True)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    wavfile.write(a.output,sr,np.round(np.clip(y,-1,1)*32767).astype(np.int16))
    if a.trace:
        Path(a.trace).parent.mkdir(parents=True,exist_ok=True)
        with open(a.trace,'w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(logs[0]));w.writeheader();w.writerows(logs)
    print(f'Processed {len(x)/sr:.2f} seconds; peak {max(abs(y)):.4f}')


if __name__=='__main__':main()
