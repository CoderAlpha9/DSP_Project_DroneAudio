"""Recreate fresh-study plots from row-level CSV evidence."""
import sys
from pathlib import Path
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
src=Path(sys.argv[1]);out=Path(sys.argv[2]) if len(sys.argv)>2 else src/'figures';out.mkdir(parents=True,exist_ok=True)
d=pd.read_csv(src/'fresh_test_waveform_v2.csv');p=d.pivot(index=['case_id','condition'],columns='method',values='sisdr');delta=p.subtract(p['fixed_0.0'],axis=0)
methods=['fixed_0.0','fixed_0.1','bounded_0.75','fixed_1.0','adaptive_capped','oracle_psd'];labels=['Audio only','Fixed 10%','Bounded 75%','RPM only','Correlation','Oracle PSD']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
fig,ax=plt.subplots(figsize=(8,3.8));ax.bar(labels,[p[m].mean() for m in methods],color=['#506a86','#4c9e95','#e7a653','#c76a58','#8477b4','#a4b86b']);ax.set_ylabel('Mean SI-SDR (dB)');ax.tick_params(axis='x',labelrotation=20);ax.set_title('Fresh sealed test: 32 cases x 6 telemetry states');ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(out/'fresh_summary.png',dpi=200);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,3.8));delta.groupby(level='condition').mean()[['fixed_0.1','bounded_0.75','fixed_1.0','adaptive_capped']].plot.bar(ax=ax);ax.axhline(0,color='black',lw=.7);ax.set_ylabel('SI-SDR difference vs audio only (dB)');ax.set_xlabel('Telemetry condition');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'fresh_telemetry_conditions.png',dpi=200);plt.close(fig)
fig,ax=plt.subplots(figsize=(7,3.5));ax.bar(labels[:-1],[100*(delta[m]<-.5).mean() for m in methods[:-1]],color='#c76a58');ax.set_ylabel('Cases >0.5 dB worse than audio only (%)');ax.tick_params(axis='x',labelrotation=20);ax.set_ylim(0,100);fig.tight_layout();fig.savefig(out/'fresh_harm_rate.png',dpi=200);plt.close(fig)
f=src/'development_fixed_grid_v2.csv'
if f.exists():
 g=pd.read_csv(f);fig,ax=plt.subplots(figsize=(6,3.5));ax.plot(g['lambda'],g.mean_stft_sdr,'o-',label='Mean');ax.plot(g['lambda'],g.p10_stft_sdr,'s--',label='10th percentile');ax.set(xlabel='RPM blend weight',ylabel='Development complex STFT-SDR (dB)');ax.legend();fig.tight_layout();fig.savefig(out/'development_fixed_grid.png',dpi=200);plt.close(fig)
r=pd.read_csv(src/'ridge_gate_test_v3.csv');r.groupby('method').agg(mean_sisdr=('sisdr','mean'),mean_lambda=('mean_lambda','mean')).to_csv(out/'ridge_summary.csv')
print(out)
