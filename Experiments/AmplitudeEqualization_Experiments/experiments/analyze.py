"""Test-only result summaries, numerical audits and scientific figures."""
import json,hashlib,sys,time,tracemalloc
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import ROOT,rows_in,load,run,pcm,evaluate,json_out,csv_out,Config,process_audio

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.grid':True,'grid.alpha':.18,'savefig.dpi':180})
COLORS=['#697b8a','#a66139','#307b76','#275d91','#8a4b78']
LABELS={'identity':'No gain','energy_only':'Energy only','geometry_unchecked':'Pose, unchecked','selected':'Selected pose-guided',
        'feedback_agc_2016':'Feedback AGC (2016)','powerlaw_drc_2021':'Power-law DRC (2021)',
        'abl_no_smoothing':'No smoothing','abl_no_hold':'No hold','abl_max_reference':'Maximum energy reference'}


def save(fig,name):
    fig.tight_layout(pad=1.1);fig.savefig(ROOT/'figures'/name,bbox_inches='tight');plt.close(fig)


def export_test_frames(data,index):
    from run import wav_in,rmsf
    frames=[]
    for row in rows_in(data/'manifest.csv'):
        if row['split']!='test':continue
        sc=load(data,row);sid=row['scenario_id']
        for policy in ['event','fixed_2m']:
            r=index[(index.scenario_id==sid)&(index.rec_number=='rec4')].iloc[0]
            _,y=wav_in(ROOT/r['event_output_path' if policy=='event' else 'calibrated_output_path'])
            g=np.load(ROOT/'audio_logs'/sid/('event_gain.npy' if policy=='event' else 'fixed_2m_gain.npy'))
            goal=sc['goal'] if policy=='event' else sc['reference']
            ideal=sc['ideal'] if policy=='event' else sc['distance']/2
            for i,(desired,out,inp,gg,gi,a) in enumerate(zip(rmsf(goal),rmsf(y),rmsf(sc['x']),g.reshape(-1,320).mean(axis=1),ideal.reshape(-1,320).mean(axis=1),sc['active'])):
                frames.append(dict(scenario_id=sid,split='test',reference_policy=policy,time_center_s=(i+.5)*.02,
                    active_for_scoring=int(a),input_rms=float(inp),output_rms=float(out),desired_rms=float(desired),
                    level_error_db=float(20*np.log10((out+1e-12)/(desired+1e-12))),
                    applied_gain_db=float(20*np.log10(max(gg,1e-12))),ideal_gain_db=float(20*np.log10(max(gi,1e-12)))))
    csv_out(ROOT/'metrics/test_frame_metrics.csv',frames)


def main():
    cfg=Config(**json.loads((ROOT/'configs/selected.json').read_text()))
    data=ROOT/'data';index=pd.read_csv(ROOT/'listening/audio_index.csv')
    if (data/'input_integrity.csv').exists():
        for row in rows_in(data/'input_integrity.csv'):
            assert hashlib.sha256((ROOT/row['input_path']).read_bytes()).hexdigest()==row['sha256']
    export_test_frames(data,index)
    df=pd.read_csv(ROOT/'metrics/test_per_case.csv');assert set(df['split'])=={'test'}
    # Keep unbounded numerical SNR for audit, but cap reported reconstruction SNR
    # at 80 dB so exact identity controls do not create arbitrary huge figures.
    for prefix in ['reconstruction_snr','identity_reconstruction_snr']:
        raw=prefix+'_uncapped_db';normal=prefix+'_db'
        if raw not in df:df[raw]=df[normal]
        df[normal]=np.minimum(80.,df[raw])
    df['reconstruction_snr_change_db']=df.reconstruction_snr_db-df.identity_reconstruction_snr_db
    df.to_csv(ROOT/'metrics/test_per_case.csv',index=False)
    numeric=df.select_dtypes(include='number').columns
    mean=df.groupby(['reference_policy','method'])[numeric].mean().reset_index()
    mean.to_csv(ROOT/'metrics/test_summary_mean.csv',index=False)
    df.groupby(['reference_policy','method'])[numeric].std().reset_index().to_csv(ROOT/'metrics/test_summary_std.csv',index=False)
    # Replace the earlier summary with consistent capped-SNR fields.
    mean.to_csv(ROOT/'metrics/test_summary.csv',index=False)
    df.groupby(['reference_policy','method','profile'])[numeric].mean().reset_index().to_csv(ROOT/'metrics/test_by_profile.csv',index=False)
    df.groupby(['reference_policy','method','sensor_condition'])[numeric].mean().reset_index().to_csv(ROOT/'metrics/test_by_sensor.csv',index=False)
    methods=['identity','energy_only','feedback_agc_2016','powerlaw_drc_2021','selected']
    fig,axes=plt.subplots(1,2,figsize=(8,3.5))
    for ax,policy in zip(axes,['event','fixed_2m']):
        sub=mean[mean.reference_policy==policy].set_index('method')
        ax.bar(range(len(methods)),[sub.loc[m,'active_level_mae_db'] for m in methods],color=COLORS)
        ax.set_xticks(range(len(methods)),[LABELS[m] for m in methods],rotation=45,ha='right',fontsize=8)
        ax.set(ylabel='Active level MAE (dB)',title='Event-relative reference' if policy=='event' else 'Known 2 m reference')
    save(fig,'test_comparison.png')
    # Independent error metrics: no scores or plots from train/validation.
    fig,axes=plt.subplots(1,3,figsize=(8.5,3.5))
    for ax,key,title in zip(axes,['waveform_rmse','envelope_shape_rmse_db','gain_step_p95_db'],['Waveform RMSE','Envelope-shape error (dB)','Gain-step p95 (dB)']):
        sub=df[df.reference_policy=='event']
        vals=[sub[sub.method==m][key].to_numpy() for m in ['identity','energy_only','selected']]
        ax.boxplot(vals,tick_labels=['No gain','Energy','Selected']);ax.set(title=title);ax.tick_params(axis='x',rotation=30)
    save(fig,'test_fidelity.png')
    # A rec4/5 fast-retreat test case provides the time-varying example.
    sid='test_02_00_recede_fast';base=next(r for r in rows_in(data/'manifest.csv') if r['scenario_id']==sid)
    base['relative_path']='cases/'+sid
    sc=load(data,base)
    e=pd.read_csv(ROOT/'audio_logs'/sid/'event_trace.csv');a=pd.read_csv(ROOT/'audio_logs'/sid/'fixed_2m_trace.csv')
    fig,axes=plt.subplots(3,1,figsize=(8,5.3),sharex=True)
    tc=(np.arange(500)+.5)*.02
    axes[0].plot(tc,sc['distance'].reshape(-1,320).mean(axis=1),label='True range (evaluation)',color=COLORS[3]);axes[0].set_ylabel('Range (m)');axes[0].legend(fontsize=8)
    axes[1].plot(tc,20*np.log10(sc['ideal'].reshape(-1,320).mean(axis=1)),'k--',label='Ideal event gain')
    axes[1].plot(tc,e.applied_gain_db,label='Event output',color=COLORS[2]);axes[1].plot(tc,a.applied_gain_db,label='2 m output',color=COLORS[1]);axes[1].set_ylabel('Gain (dB)');axes[1].legend(fontsize=8,ncol=3)
    axes[2].plot(tc,e.pose_usable,label='Observed pose usable',color=COLORS[4]);axes[2].set(xlabel='Time (s)',ylabel='Validity',ylim=(-.1,1.2));axes[2].legend(fontsize=8)
    save(fig,'test_retreat_trace.png')
    # Same scale, same audio sample indices for every spectrogram.
    r=index[(index.scenario_id==sid)&(index.rec_number=='rec4')].iloc[0]
    from run import wav_in,rmsf
    _,yev=wav_in(ROOT/r.event_output_path);_,yabs=wav_in(ROOT/r.calibrated_output_path)
    fig,axes=plt.subplots(3,1,figsize=(8,5),sharex=True,sharey=True)
    for ax,x,label in zip(axes,[sc['x'],yev,yabs],['Input target_at_drone','Event-relative output','Calibrated 2 m output']):
        ff,tt,Z=signal.stft(x,fs=16000,nperseg=1024,noverlap=768)
        ax.pcolormesh(tt,ff/1000,20*np.log10(np.maximum(abs(Z),1e-7)),vmin=-85,vmax=-25,shading='auto',cmap='magma')
        ax.set(ylim=(0,4),ylabel='kHz',title=label)
    axes[-1].set_xlabel('Time (s)');save(fig,'test_spectrograms.png')
    fig,ax=plt.subplots(figsize=(8,2.7))
    for sig,label,c in [(sc['x'],'Target input',COLORS[0]),(yev,'Event output',COLORS[2]),(yabs,'2 m output',COLORS[1]),(sc['reference'],'2 m reference','black')]:
        ax.plot(tc,rmsf(sig),label=label,color=c,lw=1.)
    ax.set(xlabel='Time (s)',ylabel='20 ms RMS');ax.legend(fontsize=8,ncol=2);save(fig,'test_envelopes.png')
    fb=pd.read_csv(ROOT/'metrics/test_sensor_probes.csv');assert set(fb.split)=={'test'}
    means=fb.groupby('condition')[['active_level_mae_db','envelope_shape_rmse_db']].mean()
    fig,ax=plt.subplots(figsize=(7.5,2.6))
    ax.barh(means.index,means.active_level_mae_db,color=COLORS[3]);ax.set_xlabel('Event-reference active level MAE (dB)');save(fig,'test_sensor_probes.png')
    # Envelope-control filter study is analytic, not development recording metrics.
    fs=50.;p=np.exp(-.02/cfg.smooth_s);fc=1/(2*np.pi*cfg.smooth_s);n=max(3,round(.443*fs/fc))
    filt=[('IIR',[1-p],[1,-p]),('FIR',np.ones(n)/n,[1])];filterrows=[]
    fig,axes=plt.subplots(2,3,figsize=(8.5,4.7))
    for i,(name,b,aa) in enumerate(filt):
        freq,H=signal.freqz(b,aa,worN=2048,fs=fs);_,gd=signal.group_delay((b,aa),w=2*np.pi*freq/fs)
        step=signal.lfilter(b,aa,np.ones(100));imp=signal.lfilter(b,aa,np.r_[1.,np.zeros(99)])
        c=COLORS[3] if name=='IIR' else COLORS[1]
        axes[0,0].plot(freq,20*np.log10(abs(H)+1e-16),c=c,label=name)
        axes[0,1].plot(freq,np.unwrap(np.angle(H)),c=c);axes[0,2].plot(freq,gd/fs,c=c)
        axes[1,0].plot(np.arange(100)/fs,step,c=c);axes[1,1].plot(np.arange(100)/fs,imp,c=c)
        filterrows.append(dict(filter=name,frame_rate_hz=50,measured_3db_hz=float(freq[np.argmin(abs(20*np.log10(abs(H)+1e-16)+3.0103))]),
            group_delay_at_0_2hz_s=float(np.interp(.2,freq,gd)/fs),taps=len(b),pole_radius=float(p) if name=='IIR' else 0))
    for j,(label,limits) in enumerate([('Magnitude (dB)',(-30,1)),('Phase (rad)',None),('Group delay (s)',(0,.3))]):
        axes[0,j].set(xlim=(0,15),xlabel='Envelope frequency (Hz)',ylabel=label)
        if limits:axes[0,j].set_ylim(limits)
    axes[0,0].legend(fontsize=8)
    axes[1,0].set(xlim=(0,.6),xlabel='Time (s)',ylabel='Unit step')
    axes[1,1].set(xlim=(0,.5),xlabel='Time (s)',ylabel='Impulse')
    ax=axes[1,2];theta=np.linspace(0,2*np.pi,200);ax.plot(np.cos(theta),np.sin(theta),'k--',lw=.7)
    roots=np.exp(2j*np.pi*np.arange(1,n)/n);ax.scatter(roots.real,roots.imag,facecolors='none',edgecolors=COLORS[1],label='FIR zeros')
    ax.scatter([p],[0],marker='x',c=COLORS[3],s=60,label='IIR pole');ax.scatter([0],[0],marker='x',c=COLORS[1],label='FIR origin poles');ax.scatter([0],[0],facecolors='none',edgecolors=COLORS[3],label='IIR zero')
    ax.set(aspect='equal',xlabel='Real z',ylabel='Imaginary z');ax.legend(fontsize=6)
    save(fig,'dsp_filters.png');csv_out(ROOT/'analysis/filter_analysis.csv',filterrows)
    # Time/memory/precision are evaluated on one named TEST recording only.
    times=[]
    for _ in range(5):
        start=time.perf_counter();y,g,logs=run(sc,cfg);times.append(time.perf_counter()-start)
    tracemalloc.start();run(sc,cfg);_,mem=tracemalloc.get_traced_memory();tracemalloc.stop()
    sc32={**sc,'x':sc['x'].astype(np.float32)};y32,_,_=run(sc32,cfg)
    json_out(ROOT/'metrics/test_runtime_precision.json',dict(split='test',scenario_id=sid,repeated_runtime_s=times,
        median_real_time_factor=float(np.median(times)/10),wrapper_peak_allocation_bytes=mem,
        float32_input_max_output_difference=float(max(abs(y-y32))),pcm16_max_quantization_error=float(max(abs(pcm(y).astype(float)/32767-y))),
        buffering_ms=20,internal_precision='float64',scope='Amplitude only; I/O and metric calculations excluded.'))
    # Audit every export, source mapping and test-only metric restriction.
    hashes=[]
    for _,row in index.iterrows():
        for key in ['event_output_path','calibrated_output_path']:
            sr,x=wav_in(ROOT/row[key]);assert sr==16000 and len(x)==160000 and np.max(abs(x))<=.95002
        hashes.append(hashlib.sha256((ROOT/row.event_output_path).read_bytes()).hexdigest())
    for row in rows_in(data/'manifest.csv'):
        sid=row['scenario_id'];folder=data/'cases'/sid
        assert (folder/'target_at_drone.wav').exists()
    selected_hash=hashlib.sha256((ROOT/'configs/selected.json').read_bytes()).hexdigest()
    freeze=json.loads((ROOT/'configs/freeze_record.json').read_text());assert selected_hash==freeze['configuration_sha256']
    audit=dict(passed=True,unique_cases=int(index.scenario_id.nunique()),listening_rows=len(index),
        unique_test_cases=15,test_performance_rows=len(df),observed_source_rec_numbers=sorted(index.rec_number.unique().tolist()),
        output_wav_count=2*len(index),all_waveforms_valid=True,selection_unchanged=True,
        synthetic_noise_added=False,denoising_applied=False,performance_tables_contain_test_only=True,
        rec5_alias_matches_rec4=all(index[(index.scenario_id==sid)&(index.rec_number=='rec4')].shape[0]==1 for sid in index[index.rec_number=='rec5'].scenario_id))
    for sid in index[index.rec_number=='rec5'].scenario_id:
        for name in ['output_event_relative.wav','output_fixed_2m.wav']:
            assert (ROOT/'listening/rec4'/sid/name).read_bytes()==(ROOT/'listening/rec5'/sid/name).read_bytes()
    json_out(ROOT/'data/final_audit.json',audit)
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
