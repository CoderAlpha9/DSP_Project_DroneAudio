"""Executed scientific figures and explicit observed-sensor perturbations."""
from copy import deepcopy
from pathlib import Path
import csv
import json
import numpy as np
from scipy import signal
from .audio import read_audio
from .datasets import discover
from .evaluation import bind_input, score
from .orchestration import process_audio
from .outputs import csv_out,json_out,environment
from .telemetry import ObservedTelemetry,SensorSeries


def sensor_study(data_root,config,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    samples,_=discover(data_root,include_variants=False)
    selected=[next(s for s in samples if s.batch=='batch1' and s.split=='test' and s.metadata['profile']=='recede_slow'),
              next(s for s in samples if s.batch=='batch2' and s.split=='test' and s.metadata['scenario']=='S7')]
    conditions=['nominal','missing','delay_200ms','hold_2s','drift_m20','scale_per_motor','stopped']
    modes=['audio_only','fixed_rpm','bounded_rpm','correlation','ridge','order'];rows=[]
    for sample in selected:
        x,_=read_audio(sample.inference.microphone);base=sample.inference.load_telemetry()
        for condition in conditions:
            tele=deepcopy(base)
            if condition=='missing':tele.rpm=None
            elif tele.rpm is not None:
                s=tele.rpm;t=s.times.copy();v=s.values.copy()
                if condition=='delay_200ms':t+=.2
                elif condition=='hold_2s':
                    start=np.searchsorted(t,2.5);end=np.searchsorted(t,4.5)
                    if start<len(v):v[start:end]=v[start]
                elif condition=='drift_m20':v*=np.linspace(1.,.8,len(v))[:,None]
                elif condition=='scale_per_motor':v*=np.linspace(.7,1.3,v.shape[1])[None,:]
                elif condition=='stopped':v[:]=0
                tele.rpm=SensorSeries(t,v,s.valid.copy(),units='RPM')
            for mode in modes:
                cfg=bind_input(config,sample.inference);cfg.amplitude.enabled=False;cfg.enhancement.mode=mode
                cfg.enhancement.rpm_weight=.75 if mode=='bounded_rpm' else .1
                p=process_audio(x,cfg,telemetry=tele);row=score(sample,p,cfg,intelligibility=False)
                row.update(sensor_condition=condition,method=mode);rows.append(row)
            csv_out(out/'per_case.csv',rows)
            print(f"Sensor study: {sample.id} / {condition}",flush=True)
    json_out(out/'execution.json',{'status':'completed','rows':len(rows),'cases':[s.id for s in selected],
        'new_integration_study':True,'reused_test':True,'environment':environment(),'not_a_fault_diagnosis':True})
    _sensor_plot(out,rows)
    return rows


def _matplotlib():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'savefig.dpi':160,'axes.spines.top':False,'axes.spines.right':False})
    return plt


def _sensor_plot(out,rows):
    plt=_matplotlib();fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,batch in zip(axes,['batch1','batch2']):
        conditions=list(dict.fromkeys(r['sensor_condition'] for r in rows if r['batch']==batch))
        for mode in ['audio_only','fixed_rpm','bounded_rpm','correlation','ridge','order']:
            vals=[next(r['pre_si_sdr_db'] for r in rows if r['batch']==batch and r['sensor_condition']==c and r['method']==mode) for c in conditions]
            ax.plot(range(len(vals)),vals,'o-',label=mode,markersize=3)
        ax.set_xticks(range(len(conditions)),conditions,rotation=45,ha='right');ax.set_title(batch+' representative reused test case');ax.set_ylabel('Pre-amplitude SI-SDR (dB)');ax.grid(alpha=.2)
    axes[-1].legend(fontsize=7);fig.savefig(out/'sensor_degradation.png');plt.close(fig)


def make_plots(run):
    run=Path(run);out=run/'figures';out.mkdir(parents=True,exist_ok=True);plt=_matplotlib()
    # Diagram is generated from the implemented graph, no claimed system-wide LTI model.
    fig,ax=plt.subplots(figsize=(11,3));ax.set_axis_off()
    boxes=[(.02,.35,'16 kHz mono\nobserved sensors'),(.25,.35,'SPP tracker / RPM\nDD Wiener + WOLA'),(.52,.35,'Bounded pose gain\nsource + speech control'),(.80,.35,'Continuous audio\ntraces / event clips'),(.25,.78,'Raw microphone detector\n25 ms / 10 ms hop')]
    for x,y,label in boxes:ax.text(x,y,label,va='center',ha='left',bbox=dict(boxstyle='round',fc='#edf3fa',ec='#3b5574'))
    for x1,y1,x2,y2 in [(.18,.35,.24,.35),(.47,.35,.51,.35),(.75,.35,.79,.35),(.12,.43,.25,.75),(.46,.75,.63,.44)]:
        ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',color='#3b5574'))
    fig.savefig(out/'pipeline.png',bbox_inches='tight');plt.close(fig)
    # Actual recursive PSD smoothing pole: update clock is hop/fs, not fs.
    config=json.loads(json.dumps({}))
    from .config import load_config
    c=load_config(run/'frozen.yaml') if (run/'frozen.yaml').exists() else load_config()
    alpha=c.enhancement.noise_alpha;update_rate=16000/c.enhancement.hop
    w,H=signal.freqz([1-alpha],[1,-alpha],worN=1024,fs=update_rate)
    omega,gd=signal.group_delay(([1-alpha],[1,-alpha]),w=1024,fs=update_rate)
    fig,axes=plt.subplots(2,2,figsize=(9,6),layout='constrained')
    axes[0,0].plot(w,20*np.log10(abs(H)));axes[0,0].set_ylabel('Magnitude (dB)')
    axes[0,1].plot(w,np.unwrap(np.angle(H)));axes[0,1].set_ylabel('Phase (rad)')
    axes[1,0].plot(omega,gd/update_rate);axes[1,0].set_ylabel('Group delay (s)')
    for ax in [axes[0,0],axes[0,1],axes[1,0]]:ax.set_xlabel('PSD modulation frequency (Hz)');ax.grid(alpha=.2)
    theta=np.linspace(0,2*np.pi,200);axes[1,1].plot(np.cos(theta),np.sin(theta),'--',color='gray');axes[1,1].plot(alpha,0,'x',label=f'pole {alpha}');axes[1,1].plot(0,0,'o',mfc='none',label='zero at 0');axes[1,1].set_aspect('equal');axes[1,1].legend();axes[1,1].set_xlabel('Real');axes[1,1].set_ylabel('Imaginary')
    fig.suptitle('Fixed PSD one-pole smoother, not the adaptive pipeline transfer function');fig.savefig(out/'psd_smoother_response.png');plt.close(fig)
    sos=signal.butter(2,[100,6500],btype='bandpass',fs=16000,output='sos');f,h=signal.sosfreqz(sos,worN=2048,fs=16000)
    fig,axes=plt.subplots(1,2,figsize=(9,3),layout='constrained');axes[0].plot(f,20*np.log10(np.maximum(abs(h),1e-8)));axes[0].set_ylabel('Magnitude (dB)');axes[1].plot(f,np.unwrap(np.angle(h)));axes[1].set_ylabel('Phase (rad)')
    for ax in axes:ax.set_xlabel('Audio frequency (Hz)');ax.grid(alpha=.2)
    fig.suptitle('Fixed causal detector input bandpass');fig.savefig(out/'detector_bandpass_response.png');plt.close(fig)
    rows=[]
    if (run/'summary.csv').exists():
        with (run/'summary.csv').open() as fp:rows=list(csv.DictReader(fp))
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        for ax,batch in zip(axes,['batch1','batch2']):
            group=[r for r in rows if r['batch']==batch];names=[r['method'] for r in group]
            ax.barh(names,[float(r['pre_si_sdr_db']) for r in group]);ax.set_title(batch+' macro parent means');ax.set_xlabel('Pre-amplitude SI-SDR (dB)')
        fig.savefig(out/'method_comparison.png');plt.close(fig)
    if (run/'paired_vs_audio.csv').exists():
        with (run/'paired_vs_audio.csv').open() as fp:paired=list(csv.DictReader(fp))
        fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
        methods=[m for m in dict.fromkeys(r['method'] for r in paired) if m!='audio_only']
        vals=[[float(r['pre_si_sdr_delta_vs_audio_db']) for r in paired if r['method']==m and r['pre_si_sdr_delta_vs_audio_db']] for m in methods]
        ax.boxplot(vals,tick_labels=methods,vert=True);ax.axhline(0,color='gray',ls='--');ax.tick_params(axis='x',labelrotation=30);ax.set_ylabel('Paired parent SI-SDR delta vs audio-only (dB)');fig.savefig(out/'paired_comparison.png');plt.close(fig)
    cases=sorted((run/'cases').glob('*/complete'))
    if cases:
        p=next((p for p in cases if 'recede_slow' in str(p)),cases[0]);final,_=read_audio(p/'final.wav');enhanced,_=read_audio(p/'enhanced.wav')
        with (p/'execution.json').open() as f:record=json.load(f)
        mic=record['input_record']['sample']['inference']['microphone'];raw,_=read_audio(mic)
        with (p/'enhancement.csv').open() as f:trace=list(csv.DictReader(f))
        with (p/'amplitude.csv').open() as f:amp=list(csv.DictReader(f))
        fig,axes=plt.subplots(4,1,figsize=(11,9),layout='constrained',sharex=True)
        t=np.arange(len(raw))/16000;axes[0].plot(t,raw,label='microphone',alpha=.5,lw=.4);axes[0].plot(t,final,label='final',alpha=.7,lw=.4);axes[0].legend();axes[0].set_ylabel('Amplitude')
        for ax,x,label in [(axes[1],raw,'Microphone'),(axes[2],enhanced,'Pre-amplitude enhanced')]:
            f,tt,Z=signal.stft(x,16000,nperseg=512,noverlap=384);ax.pcolormesh(tt,f,20*np.log10(np.maximum(abs(Z),1e-7)),vmin=-80,vmax=-20,shading='auto',rasterized=True);ax.set_ylim(0,4500);ax.set_ylabel(label+'\nHz')
        axes[3].plot([float(r['time_s']) for r in amp],[float(r['applied_gain_db']) for r in amp],label='applied gain dB')
        axes[3].plot([float(r['available_time_s']) for r in trace],[float(r['fusion_weight']) for r in trace],label='RPM weight');axes[3].legend();axes[3].set_xlabel('Audio time (s)')
        fig.suptitle(p.parent.name+' — aligned signals and controls');fig.savefig(out/'representative_panels.png');plt.close(fig)
    json_out(out/'figure_manifest.json',{'generated':[p.name for p in sorted(out.glob('*.png'))],'source_run':str(run),'no_adaptive_LTI_claim':True})
