from experiment import *
import itertools,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

names={'raw':'Unprocessed','bandpass':'80-3900 Hz','sub':'Spectral subtraction','static':'Static Wiener','lsa':'Log-spectral gain','spp':'SPP Wiener','quantile':'Local quantile','rrst':'RPM transport','rrst_guard':'Capped fusion','rrst_broadband':'RPM amplitude only','oracle_irm':'Oracle IRM','deepfilter':'DeepFilterNet3','rrst_deepfilter':'RPM + DFN3','static_expanded':'Static Wiener (expanded)','lsa_expanded':'LSA (expanded)','spp_expanded':'SPP (expanded)'}
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight','pdf.fonttype':42})
frames=[pd.read_csv(ROOT/'results'/f) for f in ['main.csv','baseline_sensitivity.csv','neural.csv']]
d=pd.concat(frames,ignore_index=True);d.to_csv(ROOT/'results/all_results.csv',index=False)
assert len(d)==1216 and d.groupby(['run','input_snr','method']).size().max()==1
cols=['sisdr','snr','segsnr','stoi','estoi','pesq_nb','rtf']
hold=d[d.seed==303].copy();summary=hold.groupby('method')[cols].mean().sort_values('sisdr',ascending=False)
summary.to_csv(ROOT/'results/holdout_summary.csv')
hold.groupby(['scenario','method'])[cols].mean().to_csv(ROOT/'results/holdout_by_condition.csv')
d.groupby(['split','method'])[cols].mean().to_csv(ROOT/'results/all_split_summary.csv')
hold.groupby(['input_snr','method'])[cols].mean().to_csv(ROOT/'results/holdout_by_snr.csv')
rng=np.random.default_rng(742);stat=[]
for base in ['static','static_expanded','spp_expanded','deepfilter']:
    for met in ['sisdr','estoi','pesq_nb']:
        p=hold.pivot_table(index='run',columns='method',values=met);delta=(p.rrst-p[base]).to_numpy()
        bt=delta[rng.integers(0,len(delta),(50000,len(delta)))].mean(1)
        signs=np.array(list(itertools.product([-1,1],repeat=len(delta))))
        prob=float(np.mean(abs((signs*delta).mean(1))>=abs(delta.mean())-1e-12))
        stat.append(dict(comparison='rrst - '+base,metric=met,n_flights=len(delta),mean=float(delta.mean()),median=float(np.median(delta)),ci_low=float(np.quantile(bt,.025)),ci_high=float(np.quantile(bt,.975)),exact_two_sided_signflip_p=prob,wins=int(np.sum(delta>0)),individual_deltas=delta.tolist()))
pd.DataFrame(stat).to_csv(ROOT/'results/paired_statistics.csv',index=False)
(ROOT/'results/paired_statistics.json').write_text(json.dumps(stat,indent=2))

selected=['raw','static_expanded','spp_expanded','rrst','rrst_guard','deepfilter','rrst_deepfilter','oracle_irm']
fig,axes=plt.subplots(1,2,figsize=(7.1,2.7))
for m in selected[:-1]:
    tab=hold[hold.method==m].groupby('input_snr')[['sisdr','estoi']].mean()
    axes[0].plot(tab.index,tab.sisdr,'o-',label=names[m],ms=3)
    axes[1].plot(tab.index,tab.estoi,'o-',ms=3)
axes[0].set(xlabel='Nominal active-speech SNR (dB)',ylabel='SI-SDR (dB)');axes[1].set(xlabel='Nominal active-speech SNR (dB)',ylabel='ESTOI')
for ax in axes:ax.grid(alpha=.2)
axes[0].legend(fontsize=6,ncol=2,loc='upper left');fig.tight_layout();fig.savefig(ROOT/'figures/snr_curves.png',dpi=220);plt.close(fig)

fig,ax=plt.subplots(figsize=(7.1,2.5));p=hold.pivot_table(index='scenario',columns='method',values='sisdr');xx=np.arange(len(p));width=.18
for k,m in enumerate(['static_expanded','spp_expanded','rrst','deepfilter']):ax.bar(xx+(k-1.5)*width,p[m],width,label=names[m])
ax.set(xticks=xx,xticklabels=p.index,ylabel='Mean SI-SDR over four SNRs (dB)');ax.legend(fontsize=7,ncol=2);ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(ROOT/'figures/condition_comparison.png',dpi=220);plt.close(fig)

stress=pd.read_csv(ROOT/'results/stress.csv');cases=['nominal','gain_x1.2','per_motor_gain','lag_0.05s','lag_0.2s','lag_0.5s','dropout_4s','frozen','reversed','gain_drift_20pct']
fig,ax=plt.subplots(figsize=(7.1,2.7))
for method in ['rrst','rrst_guard']:
    v=stress[(stress.method==method)&stress.case.isin(cases)].groupby('case').sisdr.mean().reindex(cases)
    ax.plot(range(len(v)),v,'o-',label=names[method])
ax.axhline(stress[(stress.method=='static')].sisdr.mean(),ls='--',c='grey',label='Static Wiener')
ax.set(xticks=range(len(cases)),xticklabels=['Nominal','Gain 1.2','Per-motor\ngains','Lag 50ms','Lag 200ms','Lag 500ms','4s hold','Frozen','Reversed','20% drift'],ylabel='SI-SDR at -5 dB (dB)');ax.tick_params(axis='x',labelsize=7);ax.legend(fontsize=7);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(ROOT/'figures/telemetry_stress.png',dpi=220);plt.close(fig)

a=pd.read_csv(ROOT/'results/ablations.csv');v=a.groupby('ablation').sisdr.mean().sort_values()
fig,ax=plt.subplots(figsize=(7.1,2.7));ax.barh(v.index.str.replace('_',' '),v,color='#227c9d');ax.set(xlabel='SI-SDR at -5 dB, matched settings (dB)');ax.grid(axis='x',alpha=.2);fig.tight_layout();fig.savefig(ROOT/'figures/ablations.png',dpi=220);plt.close(fig)

fig,axes=plt.subplots(2,3,figsize=(7.1,4.4),sharex=True,sharey=True)
for ax,sc in zip(axes.flat,['S1','S3','S4','S5','S6','S7']):
    run=next(r for r in inventory()[1] if r['scenario']==sc and r['seed']==303)
    _,_,rpm=load(run);t=np.arange(len(rpm))*HOP/FS
    for j in range(4):ax.plot(t,rpm[:,j]/rpm[4:51,j].mean(),lw=.8)
    ax.set_title(sc,fontsize=9);ax.grid(alpha=.2)
for ax in axes[-1]:ax.set_xlabel('Audio time (s)')
for ax in axes[:,0]:ax.set_ylabel('Relative observed RPM')
fig.tight_layout();fig.savefig(ROOT/'figures/rpm_traces.png',dpi=220);plt.close(fig)

# Show one prespecified dynamic case, not the best individual clip.
run=next(r for r in inventory()[1] if r['scenario']=='S3' and r['seed']==303);s,y,rpm=load(run)
cfg=next(c for c in json.loads((ROOT/'results/frozen_configs.json').read_text()) if c['family']=='rrst');x,_=enhance(y,rpm,cfg)
fig,axes=plt.subplots(3,1,figsize=(7.1,4.7),sharex=True)
for ax,(label,audio) in zip(axes,[('Clean reference',s),('Noisy microphone',y),('RPM transport output',x)]):
    X=stft(audio);im=ax.imshow(20*np.log10(abs(X)+1e-7),origin='lower',aspect='auto',extent=[0,len(audio)/FS,0,FS/2],vmin=-90,vmax=-35,cmap='magma');ax.set(ylim=(0,4000),ylabel='Hz',title=label)
axes[-1].set_xlabel('Time (s)');fig.tight_layout();fig.savefig(ROOT/'figures/spectrogram.png',dpi=220);plt.close(fig)
print(summary.round(4).to_string());print(pd.DataFrame(stat).drop(columns='individual_deltas').round(5).to_string(index=False))
