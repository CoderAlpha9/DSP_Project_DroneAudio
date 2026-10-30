"""Checks that protect substantive scientific conclusions, not implementation mirrors."""
from experiment import *
import warnings
warnings.filterwarnings('ignore',category=wavfile.WavFileWarning)
allruns,runs=inventory();d=pd.read_csv(ROOT/'results/all_results.csv')
assert len(runs)==19 and len(d)==1216
assert d[['run','input_snr','method']].duplicated().sum()==0
assert d.method.nunique()==16
for met in ['sisdr','snr','segsnr','stoi','estoi','pesq_nb']:assert np.isfinite(d[met]).all()
assert d.groupby(['run','method']).input_snr.nunique().eq(4).all()
assert d[d.seed==303].run.nunique()==6
cfg=next(c for c in json.loads((ROOT/'results/frozen_configs.json').read_text()) if c['family']=='rrst')
r=next(r for r in runs if r['seed']==303 and r['scenario']=='S3');s,y,rpm=load(r)
# Exact reconstruction baseline: catches STFT timing or scale bugs that invalidate all metrics.
reconstruction=istft(stft(y),len(y));err=float(np.max(abs(y-reconstruction)));assert err<1e-10
# Verify scale invariance numerically without using reference speech in enhancement.
x,_=enhance(y,rpm,cfg);z,_=enhance(y,rpm*np.array([.7,1.2,1.5,.9]),cfg)
invariance=float(np.max(abs(x-z)));assert invariance<1e-10
# Verify frozen config winner really follows development scores, not a heldout metric.
dev=pd.read_csv(ROOT/'results/development.csv')
best=dev[dev.family=='rrst'].groupby('config').sisdr.mean().idxmax();assert cfg['id']==best
for r in runs:
    p=Path(r['path']);_,s=wavfile.read(p/'audio/clean_speech.wav');assert np.max(abs(s[1024:12800]))==0
report=dict(passed=True,processing_evaluations=len(d),selected_flights=len(runs),methods=16,metric_nonfinite_values=0,duplicate_processing_rows=0,stft_identity_max_error=err,per_motor_scale_invariance_max_error=invariance,development_winner_verified=True,calibration_speech_absent_in_all_selected_captures=True)
(ROOT/'results/verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
