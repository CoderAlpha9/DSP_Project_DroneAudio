from pathlib import Path
import sys,json,time
import numpy as np,pandas as pd
from scipy import signal
from scipy.optimize import linear_sum_assignment
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from run import cin,cout,jout,load,metric,win,sha
from pipeline.detector import detect,extract,from_features,NAMES

def regions(m):
 e=np.diff(np.r_[False,m,False].astype(int));return list(zip(np.flatnonzero(e==1),np.flatnonzero(e==-1)))
def emetric(y,p):
 rr=regions(y);truth=[]
 for a,b in rr:
  if truth and a-truth[-1][1]<=50:truth[-1]=(truth[-1][0],b)
  else:truth.append((a,b))
 pred=regions(p);I=np.zeros((len(truth),len(pred)))
 for i,(a,b) in enumerate(truth):
  for j,(c,d) in enumerate(pred):I[i,j]=max(0,min(b,d)-max(a,c))/max(1,max(b,d)-min(a,c))
 pairs=[]
 if I.size:
  ii,jj=linear_sum_assignment(-I);pairs=[(i,j) for i,j in zip(ii,jj) if I[i,j]>=.3]
 onset=[(pred[j][0]+1-truth[i][0])*10 for i,j in pairs];end=[(pred[j][1]-truth[i][1])*10 for i,j in pairs]
 return dict(true_events=len(truth),predicted_events=len(pred),matched_events=len(pairs),event_f1=2*len(pairs)/max(1,len(truth)+len(pred)) if truth or pred else 1.,onset_abs_ms=float(np.mean(abs(np.asarray(onset)))) if onset else None,offset_abs_ms=float(np.mean(abs(np.asarray(end)))) if end else None)
def save(name):plt.tight_layout();plt.savefig(ROOT/'figures'/name,dpi=150,bbox_inches='tight');plt.close()
def analyze():
 (ROOT/'figures').mkdir(exist_ok=True)
 d=pd.read_csv(ROOT/'metrics/test_per_case.csv')
 for keys,fn in [(['task','method'],'summary'),(['task','kind','method'],'by_kind')]:
  d.groupby(keys).mean(numeric_only=True).to_csv(ROOT/'metrics'/('test_'+fn+'.csv'));d.groupby(keys).std(numeric_only=True).to_csv(ROOT/'metrics'/('test_'+fn+'_std.csv'))
 selected=json.loads((ROOT/'configs/selected.json').read_text());best=selected['feature_set'];ev=[];sensitivity=[]
 test=[r for r in cin(ROOT/'data/manifest.csv') if r['split']=='test']
 for r in test:
  s=load(r);logs=from_features(s['f'],selected);e=emetric(s['y'],np.array([v['event_active'] for v in logs],bool));ev.append(dict(case_id=r['case_id'],kind=r['kind'],split='test',**e))
  if r['kind']=='speech':
   for factor in [.5,1.,2.]:
    noise=win(ROOT/'data/original'/r['base_id']/'noise.wav');target=win(ROOT/'data/original'/r['base_id']/'target_at_drone.wav');xx=target+noise*factor
    l=detect(xx,selected);sensitivity.append(dict(case_id=r['case_id'],split='test',noise_scale=factor,**metric(s['y'],[v['activity_active'] for v in l],s['valid'])))
 cout(ROOT/'metrics/test_events.csv',ev);cout(ROOT/'metrics/test_noise_sweep.csv',sensitivity)
 # Open recorded speech mixed with the same synthetic background: not real-flight data.
 from scipy.io import wavfile
 op=ROOT/'data/open/source.wav'
 if op.exists():
  xx=win(op)[:160000];noise=win(ROOT/'data/original/test_00_00_hover/noise.wav');xx=xx+noise
  ann=json.loads((ROOT/'configs/source_intervals.json').read_text())['sources']['open_librispeech'];t=(np.arange(1000)+.5)*.01;y=np.zeros(1000,bool);v=np.ones(1000,bool)
  for a,b in ann:y|=(t>=a)&(t<b);v&=(abs(t-a)>.08)&(abs(t-b)>.08)
  from run import wout
  wout(ROOT/'data/open_mixed.wav',xx);xx=win(ROOT/'data/open_mixed.wav')
  logs=detect(xx,selected);cout(ROOT/'metrics/external_test_open_mixed.csv',[dict(split='external_test',task=task,**metric(y,[l[key] for l in logs],v)) for task,key in [('activity','activity_active'),('speech','speech_active')]])
  cout(ROOT/'outputs/open_mixed_trace.csv',logs)
 # All selected test cached/stream paths agree; precision/runtime only named test case.
 s=load(next(r for r in test if r['kind']=='speech'));start=time.perf_counter();l=detect(s['x'],selected);elapsed=time.perf_counter()-start
 cached=from_features(s['f'],selected);assert l==cached
 l32=detect(s['x'].astype(np.float32),selected)
 jout(ROOT/'metrics/test_runtime_precision.json',dict(case_id=s['row']['case_id'],duration_s=10,detector_seconds=elapsed,real_time_factor=elapsed/10,window_ms=25,hop_ms=10,
 input_rounding_decision_mismatches=sum(a['event_active']!=b['event_active'] for a,b in zip(l,l32)),float64_internal=True,stream_cache_exact=True))
 fig,ax=plt.subplots(1,2,figsize=(10,3.5))
 for aa,task in zip(ax,['activity','speech']):
  pivot=d[d.task==task].groupby('method')[['f1','dcf']].mean();pivot.plot.bar(ax=aa);aa.set_title(task+' (75 test descendants)');aa.set_ylim(0,1);aa.tick_params(axis='x',rotation=25)
 save('feature_comparison.png')
 fig,ax=plt.subplots(figsize=(10,3.4));pivot=d[(d.method==best)&(d.task=='activity')].groupby('kind')[['f1','false_alarm_rate','miss_rate']].mean();pivot.plot.bar(ax=ax);ax.tick_params(axis='x',rotation=15);ax.set_ylim(0,1);save('event_types.png')
 ll=pd.DataFrame(cached);fig,ax=plt.subplots(3,1,figsize=(10,5.3),sharex=True);t=np.arange(1000)*.01
 ax[0].plot(np.arange(len(s['x']))/16000,s['x'],lw=.4);ax[0].set_ylabel('Mixture')
 ax[1].plot(t,ll.activity_score,label='activity');ax[1].plot(t,ll.speech_score,label='speech');ax[1].legend();ax[1].set_ylabel('Model score')
 ax[2].step(t,s['y'],label='visual reference');ax[2].step(t,ll.event_active*.7,label='event');ax[2].step(t,ll.speech_active*.4,label='speech');ax[2].legend(ncol=3);ax[2].set_xlabel('Time (s)');save('timeline.png')
 fig,ax=plt.subplots(2,1,figsize=(10,4.8),sharex=True)
 for aa,cols in zip(ax,[['complex_error','phase_error'],['js_change','gd_change']]):
  for c in cols:aa.plot(t,ll[c],label=c)
  aa.legend();aa.grid(alpha=.2)
 ax[-1].set_xlabel('Time (s)');save('phase_distribution.png')
 # Filter analysis is analytic, not a data-tuned alternative algorithm.
 sos=signal.butter(2,[100,6500],btype='bandpass',fs=16000,output='sos');b,a=signal.sos2tf(sos);freq,h=signal.freqz(b,a,fs=16000,worN=1024);fg=np.linspace(150,6000,200);_,gd=signal.group_delay((b,a),w=fg,fs=16000);z,p,k=signal.sos2zpk(sos)
 fig,ax=plt.subplots(1,3,figsize=(10,3));ax[0].plot(freq,20*np.log10(abs(h)+1e-12));ax[0].set_ylim(-60,5);ax[0].set_title('Analysis filter (dB)');ax[1].plot(freq,np.unwrap(np.angle(h)));ax[1].set_title('Filter phase (rad)');ax[2].plot(fg,gd/16);ax[2].set_title('Filter group delay (ms)');save('filter.png')
 jout(ROOT/'metrics/filter_analysis.json',dict(maximum_pole_radius=float(max(abs(p))),group_delay_ms_min=float(min(gd)/16),group_delay_ms_max=float(max(gd)/16),sos=sos.tolist(),note='Filter delay is distinct from signal group-delay change feature'))
 report(d,ev,best)

def report(d,events,best):
 from reportlab.pdfbase import pdfmetrics
 from reportlab.pdfbase.ttfonts import TTFont
 from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
 from reportlab.lib.units import mm
 from reportlab.lib import colors
 from reportlab.platypus import SimpleDocTemplate,Paragraph,PageBreak,Table,TableStyle,Image,Spacer
 fp=Path(matplotlib.get_data_path())/'fonts/ttf'
 pdfmetrics.registerFont(TTFont('Body',str(fp/'STIXGeneral.ttf')));pdfmetrics.registerFont(TTFont('Head',str(fp/'DejaVuSans-Bold.ttf')))
 S=getSampleStyleSheet();S.add(ParagraphStyle(name='B',fontName='Body',fontSize=10.6,leading=14.5,spaceAfter=10));S.add(ParagraphStyle(name='H',fontName='Head',fontSize=16,leading=21,spaceAfter=15));S.add(ParagraphStyle(name='Small',fontName='Body',fontSize=8.8,leading=11))
 story=[]
 def p(t):story.append(Paragraph(t,S['B']))
 def page(t):
  if story:story.append(PageBreak())
  story.append(Paragraph(t,S['H']))
 def table(head,rows):
  data=[[Paragraph(str(v),S['Small']) for v in r] for r in [head]+rows];n=len(head);tb=Table(data,colWidths=[154*mm/n]*n,repeatRows=1);tb.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7eff5')),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,-1),.3,colors.lightgrey)]));story.append(tb);story.append(Spacer(1,10))
 def fig(fn):
  from PIL import Image as PImage
  path=ROOT/'figures'/fn
  with PImage.open(path) as im:w,h=im.size
  story.append(Image(str(path),width=154*mm,height=154*mm*h/w));story.append(Spacer(1,10))
 summary=d.groupby(['task','method']).mean(numeric_only=True);by=d.groupby(['task','kind','method']).mean(numeric_only=True)
 page('Mixed-audio acoustic event detection with a separate speech flag')
 p('Sensor-Assisted Adaptive Drone Noise suppression with Motion Artefact mitigation and Actuator Fault awareness<br/>Jaiwanth Karthi M K | Group 4 | ED5017 | IIT Madras')
 p('This revision trains and evaluates on the original synthetic mixtures, not isolated target audio. The module detects general acoustic activity and retains a distinct speech-presence output. It tests energy, spectral-distribution change, phase-prediction error and a regularized signal group-delay change measure. A small non-neural statistical classifier calibrates these DSP features; it is not a semantic sound recognizer or a deep neural replacement for the DSP question.')
 p(f'Development selection chose the {best} feature set. The report retains all five feature-family comparisons, including negative results. Original test speech boundaries remain approximate visual annotations, while synthetic non-speech event timing is exactly constructed. The original test recording has already been used in earlier work; this is a reused regression benchmark, not a new blind validation.')
 p('The delivered amplitude-on-mixture files are explicitly not denoised. Gain multiplies both target and noise. The package includes the experiment code, frozen inference coefficients, source inputs, test-only metric data, traces and all-case listening outputs.')
 page('1. Objective, scope and revised plan')
 p('A speech-only gate is insufficient for non-speech foreground events. The revised objective is acoustic activity above an evolving drone background, with separate speech sensitivity. activity_active describes foreground evidence; event_active bridges short gaps for extraction; speech_active reports calibrated speech evidence while a general event is present. No arbitrary event labels such as alarm or person identity are inferred.')
 p('The original synthetic drone background is not constant. Rotor RPM, broadband noise and wind vary with motion. Background-only cases therefore test whether expected variations are mistaken for events. Existing noise is preserved instead of adding a second unrelated drone-noise generator.')
 p('Signal model: x[n] = s_target[n] + b_drone[n] for the original mixture. Synthetic non-speech probes replace s_target with a controlled foreground over that same b_drone. Every algorithm sees x only. Separate components and true motion are available exclusively for generation and scoring. Observed telemetry is used by the amplitude demonstration, not detector inference.')
 p('The detector aims to recognize acoustic deviation within the tested foreground families. An unseen sound with little energy or little spectral deviation may be missed; changing background conditions may trigger false events. Sustained events must be detected after their onset, so phase novelty alone cannot substitute for an activity measure.')
 page('2. Data, labels and 80/20 partition')
 table(['Partition','Base cases','Descendants','Role'],[['Development','60','300','40 base calibration, 20 validation'],['Test','15','75','Original held-out recording and controls']])
 p('Each base scenario produces five cases: original speech mixture, original background only, chirp, tone and impact/texture. There are 375 ten-second inputs. Original mixture WAVs are copied byte for byte. All descendants keep their base split. rec1 and rec3 calibrate models; rec2 selects parameters. rec4 and rec5 are duplicate source recordings and are never counted twice. The 15 test trajectories are correlated conditions of one source recording, not 15 independent speakers.')
 p('Speech intervals are inherited from the earlier visually annotated source regions and offset-mapped into each case. Primary scores exclude an 80 ms collar near each boundary. These labels are not independently listener-verified or forced-aligned, so reported speech metrics are agreement with this approximate protocol. Generic-event labels have exact constructed support; their fade boundaries still receive the same collar.')
 p('Generic events use disjoint deterministic seeds, randomized frequencies, durations, positions and levels, with 1/r-related level variation over the original time-varying backgrounds. Chirps sweep across frequencies; tones remain narrowband; impact/texture events use decaying filtered stochastic excitation. These are controlled proxies, not a library of real alarms, claps or impacts. Dataset seed and exact generated component arrays are packaged.')
 p('PCM conversion saturates extreme samples: four impact/texture inputs contain 19 full-scale samples in total (three samples in one test input). This is retained and disclosed rather than silently changing the dataset after evaluation. Generic foreground arrays are pre-quantization construction signals; exact linear component decomposition does not hold at those saturated samples. Original speech mixtures are unchanged and have no such clipping.')
 page('3. Spectral and energy DSP features')
 p('Analysis uses a causal fourth-order 100-6500 Hz bandpass in second-order sections, a trailing 25 ms Hann window, a 10 ms hop and a 512-point FFT at 16 kHz. Audio samples are full-scale-normalized PCM. Bins k map to k x 16000/512 Hz; radians/sample equal 2 pi f/16000. The original waveform is not filtered in delivered event clips.')
 p('A per-bin asymmetric minimum follower tracks background power: B[t,k] = min(P[t,k], 0.997 B[t-1,k] + 0.003 P[t,k]). This is a heuristic minimum follower, not a calibrated noise PSD or known stationary background. It adapts down immediately and rises slowly. Initialization uses the first available frame; no oracle noise-only initialization or mandatory silent lead-in is supplied.')
 p('Features include broadband excess in dB, mean excess in the strongest 40 bins, positive normalized spectral flux, Jensen-Shannon divergence between consecutive normalized power spectra, flatness, normalized autocorrelation periodicity and centroid. Flux and distribution divergence expose abrupt changes even when total energy changes little. Strongest-bin excess supports narrowband events that average broadband energy can hide.')
 p('Feature vectors are robustly centered and scaled using calibration frames only. Values are clipped at 12 scaled units. The binary logistic readouts use L2 regularization and balanced class weights. They provide calibrated decision scores for this study; probability calibration on a new deployment population is not established.')
 page('4. Phase and group-delay change experiments')
 p('Complex novelty predicts the next spectrum as |X[t-1,k]| exp(j(2 phase[t-1,k] - phase[t-2,k])). The normalized complex residual measures failure of the constant phase-advance model. A second feature averages wrapped phase-prediction error only in bins exceeding a background ratio and a relative-power floor. This rejects weak bins whose phases are intrinsically unstable. The feature is inspired by complex-domain onset detection [1], adapted to noisy event activity rather than musical note scoring.')
 p('The group-delay proxy uses Re(FFT(n x[n] w[n]) conj(X)) / (|X| squared + 0.01 max(|X| squared)), clips the sample-valued result to the window extent and normalizes by 400 samples. A weighted difference of this proxy across frames measures temporal-structure change. Regularization prevents division near spectral zeros. This is a bounded heuristic, not an exact implementation of modified-group-delay cepstra [2].')
 p('Crucially, these are signal phase/structure features, not changes in the causal filter group delay or estimated source propagation time. The original generator changes amplitude instantaneously with distance and contains no propagation delay or Doppler. Thus a phase feature cannot infer missing physical delay from this dataset. Phase also varies in stationary stochastic noise; it may add variance rather than discrimination. Development selection and held-out ablations determine whether it helps.')
 fig('phase_distribution.png')
 page('5. Selection and event/speech state machines')
 p('Five feature sets are compared: energy; energy plus spectral/distribution/shape features; spectral plus phase/complex residual; spectral plus group-delay change; and all features. For each family, two regularization values (C=0.1, 1) and four temporal operating points are evaluated, for 40 candidates. Logistic coefficients are fitted on calibration cases only; validation selects thresholds and the final feature family. No test score selects a threshold or a feature set.')
 p('The state machine confirms onset for two 10 ms hops. General onset thresholds are 0.5 or 0.7, with release thresholds equal to 0.6 times the onset threshold. Activity releases after 100 or 250 ms; event state retains an additional 400 ms gap. The speech flag uses its independent model with 0.6/0.35 hysteresis, 30 ms onset and 120 ms release, and is reported while a general event is present.')
 p('The objective is validation mean activity DCF + 0.5 speech DCF, with DCF = 0.75 missed-positive rate + 0.25 false-alarm rate. Per-case means prevent long or numerous frames from one source silently dominating selection. Frame fitting uses every fifth valid calibration frame for tractability. Numeric development scores are internal; all exported performance tables use test or explicitly external-test items.')
 p(f'The frozen selected feature family is {best}. General events do not automatically imply speech. Normal event closure does not reset an amplitude source-session anchor. VAD and general acoustic change do not prove a new source identity; session resets require an application signal. Cold-start behavior is included in scoring and is not retrospectively corrected.')
 page('6. Test results: feature-family comparison')
 table(['Task / features','F1','Recall','False alarm','DCF'],[[task+' / '+name,*[f'{summary.loc[(task,name),k]:.3f}' for k in ['f1','recall','false_alarm_rate','dcf']]] for task in ['activity','speech'] for name in ['energy','spectral','phase','group_delay','full']])
 fig('feature_comparison.png')
 p('These are macro means over 75 test descendants. Background-only cases have no positive frames; an empty-positive convention gives recall 1 when appropriate, so confusion counts, false-alarm rate and DCF are essential alongside F1. Feature-family comparisons are independently calibrated models selected on validation, not arbitrary coefficient removal from the final model. The models and coefficient names are supplied for inspection.')
 page('7. Non-speech events and event timing')
 table(['Kind','Activity F1','Speech false alarm','Activity miss'],[[kind,f'{by.loc[("activity",kind,best),"f1"]:.3f}',f'{by.loc[("speech",kind,best),"false_alarm_rate"]:.3f}',f'{by.loc[("activity",kind,best),"miss_rate"]:.3f}'] for kind in ['speech','background','chirp','tone','impact_texture']])
 fig('event_types.png')
 ee=pd.DataFrame(events).groupby('kind').mean(numeric_only=True)
 table(['Kind','Event overlap F1','Onset |error| ms','Offset |error| ms'],[[k,*[(f'{ee.loc[k,c]:.2f}' if np.isfinite(ee.loc[k,c]) else 'N/A') for c in ['event_f1','onset_abs_ms','offset_abs_ms']]] for k in ee.index])
 p('Reference regions separated by at most 500 ms are merged. Predicted event intervals are paired by maximum IoU with threshold 0.3. Boundary errors are reported only for matched pairs; missed or spurious events remain visible in raw counts. The export includes 100 ms pre-roll, but scored onset uses decision availability, not retrospectively shifted clip start. Release tails can produce imprecise endpoints. Class-specific thresholds were not tuned on test.')
 page('8. Mixed speech, literature baselines and robustness')
 lit=pd.read_csv(ROOT/'metrics/test_literature_speech.csv').groupby('method').mean(numeric_only=True)
 speech=d[(d.method==best)&(d.task=='speech')&(d.kind=='speech')].mean(numeric_only=True)
 table(['Speech-mixture method','F1','Recall','DCF'],[['Selected speech flag',f'{speech.f1:.3f}',f'{speech.recall:.3f}',f'{speech.dcf:.3f}']]+[[k,*[f'{lit.loc[k,c]:.3f}' for c in ['f1','recall','dcf']]] for k in lit.index])
 p('Two post-2015 literature methods from the preceding study are re-evaluated on mixed inputs: the rVAD-fast adaptation [3] and the statistical SAD branch adaptation [4]. They are speech comparators, not general-event classifiers. rVAD-fast uses spectral anchors and weighted energy difference; the statistical SAD uses iterative Wiener analysis, LPC, band energy, GMMs and HMM/Viterbi decoding. Their declared approximations remain: fixed-bias minimum tracking and project-specific framing/realization. Exact author-code parity and their published corpus results are not claimed. Both can use offline context; the selected detector is causal.')
 p('The selected speech flag is not the best speech comparator here: the offline statistical SAD adaptation has higher F1. Selection jointly targets general activity and speech DCF, rather than maximizing speech F1 alone. On the original mixtures, residual false speech flags and missed boundaries remain substantial. Do not infer reliable speech recognition from successful detection of the synthetic non-speech events.')
 sweep=pd.read_csv(ROOT/'metrics/test_noise_sweep.csv').groupby('noise_scale').mean(numeric_only=True)
 table(['Background multiplier','Activity F1','Miss rate','False alarm'],[[k,*[f'{sweep.loc[k,c]:.3f}' for c in ['f1','miss_rate','false_alarm_rate']]] for k in sweep.index])
 p('The frozen detector is also evaluated on target + 0.5, 1 or 2 times the original background. These diagnostic test sweeps do not alter selected parameters. Existing test sources and backgrounds are reused, so they quantify controlled sensitivity, not new independent trials. Phase and group-delay tests should be judged by all families and operating conditions, not one attractive example.')
 page('9. Amplitude integration and mixed-audio outputs')
 amp=pd.read_csv(ROOT/'metrics/test_mixture_amplitude.csv');table(['Test mean','Value'],[['Input target/noise component SNR (dB)',f'{amp.input_component_snr_db.mean():.3f}'],['After common gain (dB)',f'{amp.output_component_snr_db.mean():.3f}'],['Mixture output vs source reference RMSE',f'{amp.output_vs_reference_rmse.mean():.5f}']])
 p('All 75 original mixed speech cases have amplitude demonstration WAVs using the inherited 18 dB gain cap, 40 ms smoothing, 36 dB/s nominal slew, 3.5 s pose hold and 0.95 sample-peak guard. Event activity permits range-anchor updates; event closure does not clear the source-session anchor. The gain law is inherited, not retuned on mixtures. Generic-event exports preserve their mixed input and provide detected-event clips.')
 p('Applying a gain to a mixture amplifies target and noise together. Per-sample component SNR is unchanged by a common nonzero gain; aggregate speech-region SNR can change because a time-varying gain reweights time. The component decomposition above is an evaluation diagnostic, not an inference input or evidence of denoising. Original component quantization can make the stored mixture differ slightly from the sum of separately stored components.')
 p('These outputs should not be presented as clean recovered speech. A separate noise-suppression module and a validated low-SNR gain guard are still needed for the complete project. A generic event could be a different source; its presence alone does not authorize source-distance compensation. The demonstration assumes the original known source origin and available observed pose.')
 fig('timeline.png')
 page('10. Filter analysis, complexity and external check')
 fig('filter.png')
 runtime=json.loads((ROOT/'metrics/test_runtime_precision.json').read_text());fa=json.loads((ROOT/'metrics/filter_analysis.json').read_text())
 p(f'The analysis IIR maximum pole radius is {fa["maximum_pole_radius"]:.6f}, below one. Its phase and group delay characterize only the analysis filter; they are distinct from the signal group-delay feature. The nonlinear background follower, classifier and event state machine have no single LTI group-delay function. The feature window is 25 ms and availability updates every 10 ms; confirmation and release add separate application delays.')
 p(f'On a named 10-second test case, detector processing took {runtime["detector_seconds"]:.3f} s (real-time factor {runtime["real_time_factor"]:.3f}). Float32-rounded input gives {runtime["input_rounding_decision_mismatches"]} event-decision differences relative to float64 input, with float64 accumulation in both paths. Streaming and cached-feature predictions match exactly. Dominant cost is 100 hops/s of 512-point FFTs, autocorrelation, fixed-size vector operations and two linear classifiers. No GPU or downloaded neural model is used.')
 p('A retained LibriSpeech recorded-speech excerpt is additionally mixed with the original hover background for an external-speech check. It is a semi-synthetic open-recording probe, not a fully real noisy event dataset. Its visual labels remain approximate and the synthetic background is reused. CC BY 4.0 attribution is supplied. Real recorded non-speech events and real flight audio remain unvalidated.')
 external=pd.read_csv(ROOT/'metrics/external_test_open_mixed.csv')
 table(['External mixed task','F1','Recall','False alarm'],[[r['task'],*[f'{r[k]:.3f}' for k in ['f1','recall','false_alarm_rate']]] for _,r in external.iterrows()])
 page('11. Reproducibility, conclusions and limitations')
 p(f'The selected {best} model was frozen before test loading. Test activity F1 is {summary.loc[("activity",best),"f1"]:.3f}, compared with {summary.loc[("activity","phase"),"f1"]:.3f} for the phase family and {summary.loc[("activity","group_delay"),"f1"]:.3f} for the group-delay family. Thus these data do not support forcing either phase addition into deployment. The module detects the tested non-speech foregrounds as general events and reports speech separately; it does not recognize arbitrary sound identities, speaker changes or actuator faults. The lower impact/texture recall and imprecise event endings remain deployment limitations.')
 p('Moodle coverage: signal/reconstruction evidence is supplied by mixture-component diagnostics and ungated clip preservation; frequency behavior by spectral features and filter response; phase/time behavior by phase/group-delay analysis and event boundaries; robustness by five event/background conditions and noise sweep; task outcome by confusion counts, DCF/F1 and event matching; practicality by measured runtime and reproducible code. This does not establish complete author-code parity or broad real-world validity. The controlled non-speech event waveforms and single test speech recording constrain any claim of generality.')
 p('Run from the extracted project root: pip install -r requirements.txt; python -m unittest discover -s tests -v; python experiments/run.py --resume; python experiments/analyze_report.py. Omit --resume to retrain/select on packaged development inputs. No original archive is required for reruns. Frozen coefficients are plain JSON; inference needs only NumPy and SciPy. The listening index maps all original rec labels, with rec5 as an alias rather than another test contribution.')
 p('References: [1] Duxbury, Bello, Davies and Sandler, Complex domain onset detection for musical signals, DAFx 2003, dafx.de/paper-archive/2003/pdfs/dafx81.pdf. [2] Murthy, Hegde and Gadde, The modified group delay feature: a new spectral representation of speech, Interspeech 2004, DOI 10.21437/Interspeech.2004-335. [3] Tan, Sarkar and Dehak, rVAD, Computer Speech and Language 59 (2020), DOI 10.1016/j.csl.2019.06.005. [4] Heitkaemper et al., Statistical and Neural Network Based Speech Activity Detection in Non-Stationary Acoustic Environments, Interspeech 2020, DOI 10.21437/Interspeech.2020-1252. [5] LibriSpeech SLR12, openslr.org/12/, CC BY 4.0. [6] Supplied ED5017 Moodle course instructions, October 2026.')
 def footer(c,doc):c.setFont('Body',8);c.drawString(28*mm,14*mm,'Group 4 | Mixed-audio acoustic events and speech');c.drawRightString(182*mm,14*mm,str(doc.page))
 SimpleDocTemplate(str(ROOT/'report/mixed_event_report.pdf'),title='Mixed-audio acoustic events and speech - Group 4',author='Jaiwanth Karthi M K, Group 4',pagesize=(210*mm,297*mm),leftMargin=28*mm,rightMargin=28*mm,topMargin=23*mm,bottomMargin=24*mm).build(story,onFirstPage=footer,onLaterPages=footer)
if __name__=='__main__':analyze()
