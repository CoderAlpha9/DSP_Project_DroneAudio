"""Build the standalone target-only report. All empirical scores are test-only."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image as PIL
from run import ROOT

for name,file in [('Body','DejaVuSerif.ttf'),('Bold','DejaVuSerif-Bold.ttf'),('Sans','DejaVuSans.ttf')]:
    p=Path(matplotlib.get_data_path())/'fonts/ttf'/file
    pdfmetrics.registerFont(TTFont(name,str(p)))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='Bold',italic='Body',boldItalic='Bold')
NAVY=colors.HexColor('#213f59');TEAL=colors.HexColor('#26776e');W=A4[0]-56*mm
styles={k:ParagraphStyle(k,fontName=font,fontSize=size,leading=lead,spaceAfter=space,textColor=color) for k,font,size,lead,space,color in [
    ('body','Body',10.3,14.3,9,colors.HexColor('#202e37')),('small','Body',9.1,12.3,7,colors.black),
    ('h','Bold',15,19,12,NAVY),('sub','Bold',11.2,15,7,TEAL),('title','Bold',19,24,15,NAVY),
    ('caption','Sans',8.5,11,9,colors.HexColor('#52646e')),('cell','Sans',8.4,11.2,0,colors.black),
    ('eq','Sans',10.2,15,10,colors.black)]}
story=[]
def p(s,kind='body'):story.append(Paragraph(s,styles[kind]))
def h(s):p(s,'sub')
def page(n,s):
    if story:story.append(PageBreak())
    p(f'{n}. {s}','h')
def table(head,rows,widths):
    a=[[Paragraph(str(x),styles['cell']) for x in row] for row in [head]+rows]
    t=Table(a,colWidths=widths,hAlign='LEFT',repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eaf0f4')),('LINEBELOW',(0,0),(-1,0),.7,NAVY),
                          ('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#d3dfe5')),('VALIGN',(0,0),(-1,-1),'TOP'),
                          ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
                          ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
    story.extend([t,Spacer(1,8)])
def fig(name,caption,limit=250):
    path=ROOT/'figures'/name
    with PIL.open(path) as im:ratio=im.height/im.width
    width=min(W,limit/ratio);story.append(Image(str(path),width=width,height=width*ratio,hAlign='CENTER'));p(caption,'caption')
def fmt(x,n=2):return 'NA' if not np.isfinite(float(x)) else f'{float(x):.{n}f}'


def build():
    df=pd.read_csv(ROOT/'metrics/test_summary_mean.csv')
    def get(policy,method,key):return float(df[(df.reference_policy==policy)&(df.method==method)][key].iloc[0])
    cfg=json.loads((ROOT/'configs/selected.json').read_text());protocol=json.loads((ROOT/'configs/selection_protocol.json').read_text())
    runtime=json.loads((ROOT/'metrics/test_runtime_precision.json').read_text())
    # PAGE 1
    p('Sensor-Assisted Adaptive Drone Noise suppression with Motion Artefact mitigation and Actuator Fault awareness','title')
    p('Target-only amplitude restoration<br/>Experimental report and final implementation','sub')
    p('ED5017 | IIT Madras | Group 4<br/>ED24B011, ED24B027, ED24B034, ED24B051<br/>29 October 2026','small')
    h('Abstract')
    p('This study isolates amplitude restoration using the existing target_at_drone recordings. No synthetic drone noise is added, no mixture/noise waveform is opened, and no denoising front end is applied. Fifty-four DSP controller settings are examined internally on source-disjoint development groups, then frozen before scoring the existing fifteen-case test partition. Reported performance tables contain test cases only. Processed audio is exported for all seventy-five unique scenarios, including every development case and every source rec number.')
    p(f'The selected setting is effectively pose-guided: geometry weight {cfg["geometry_weight"]:.1f}, an {cfg["max_gain_db"]:.0f} dB gain ceiling, {cfg["smooth_s"]:.2f} s smoothing and hold fallback for short sensor gaps. Mean active level error relative to the closest range so far falls from {get("event","identity","active_level_mae_db"):.2f} to {get("event","selected","active_level_mae_db"):.2f} dB. With the benchmark\'s known 2 m reference, error falls from {get("fixed_2m","identity","active_level_mae_db"):.2f} to {get("fixed_2m","selected","active_level_mae_db"):.2f} dB. Missing geometry, drift and gain limits still prevent exact recovery in some cases.')
    table(['Scope','Delivered evidence'],[
        ['Input','Only target_at_drone.wav, with natural source-recording defects retained.'],
        ['Evaluation','15 unique test conditions; no development performance tables.'],
        ['Listening','75 unique cases; 90 rec-labelled rows because rec5 duplicates rec4; two operated outputs per row.'],
        ['Claim boundary','Controlled amplitude restoration, not noise suppression, SNR-gate validation or real-flight recovery.']],[100,W-100])
    p('Keywords: inverse-distance pressure law, causal gain, pose validity, envelope preservation, sample-peak protection.','small')
    # PAGE 2
    page(1,'Scope change and course requirements')
    p('The project title remains unchanged [1]. This is a revised amplitude-section experiment. The earlier study coupled amplitude control to synthetic residual interference. The present request deliberately removes that interaction by operating directly on each existing target_at_drone waveform. Consequently there is no noise PSD estimator, spectral subtraction, Wiener filter or estimated-SNR gate in the tested amplitude module.')
    h('Research question')
    p('Can a physically informed, causal gain restore range-dependent amplitude while preserving the speaker\'s natural dynamics? The primary comparison is against identity gain. Two recent literature-based gain controls provide non-neural comparison methods. Sensor rejection, temporary hold and temporal smoothing are tested as components rather than assumed to be helpful.')
    table(['Course category [1]','Evidence in this report'],[
        ['Reconstruction fidelity','Test-only level error, waveform RMSE, correlation and reference-relative reconstruction SNR.'],
        ['Frequency-domain behavior','Test log-spectral distance and equal-scale spectrograms; envelope-smoother magnitude response.'],
        ['Phase / time behavior','Smoother phase/group delay, gain-step variation, response shape, sample alignment and latency.'],
        ['Robustness','All fifteen motion profiles, existing corrupted sensors, and frozen-setting test sensor probes.'],
        ['Task-level outcome','Range-compensation error and preservation of speech envelope, with two declared reference policies.'],
        ['Complexity','Test-case runtime and memory, numerical precision, bounded controller state and reproducible code.']],[122,W-122])
    p('The mathematical contribution is the source-relative controller and its failure analysis. It is not merely a comparison of filter names. The SNR values reported here quantify reconstruction error against a specified waveform; they are not target-to-drone-noise ratios, because no separate synthetic drone noise is present.')
    p('The broader course project still requires its own open real-world drone validation, rotor filtering and fault experiments. This restricted study does not claim those requirements have been completed. The user-provided recordings are real speech sources, but the propagation and telemetry remain synthetic. No deep network is used.','small')
    # PAGE 3
    page(2,'Data, rec mapping and performance partitions')
    p('The dataset supplies 75 cases of ten seconds each at 16 kHz mono. Observed telemetry is 100 Hz, while gain decisions occur every 20 ms. Each input already contains the generator\'s inverse-distance attenuation. That waveform is copied bit-for-bit into this package; the study does not regenerate it, add another noise track or normalize it again.')
    table(['Rec label','Unique cases','Partition and role'],[
        ['rec1','20','Development: calibration'],['rec2','20','Development: validation'],['rec3','20','Development: calibration'],
        ['rec4','15','Final performance test'],['rec5','0 additional','Exact duplicate of rec4; 15 mirrored listening entries only']],[78,83,W-161])
    p('Thus the scenario split remains exactly 60 development / 15 test, or 80/20. There are three independent development recordings and only one independent test recording. The duplicated rec4/rec5 audio is never counted twice for metrics. Source-disjoint recording groups do not establish a speaker-independent population study.')
    h('What may enter the controller')
    p('Allowed runtime inputs are the target waveform, available pose/IMU measurements and a declared source location in the same coordinate frame. True position and the reference waveform define scoring targets only. Source activity for metrics is estimated from the reference envelope; it is never supplied as a frame-level oracle event detector.')
    p('For this amplitude-isolation protocol, each complete ten-second case is one event. The controller receives event_active=True during the clip and is reset between clips. Ordinary word pauses are therefore not interpreted as a new acoustic event. The reusable API also accepts externally supplied event closure or a simple minimum-RMS/hangover detector.')
    h('Remaining recording imperfections')
    p('The target is not certified anechoic, studio-clean speech. Original room noise, microphone coloration and compression remain. In particular, upsampling the 8 kHz source recording to the 16 kHz benchmark does not restore bandwidth. This experiment removes added synthetic drone noise only; it does not erase the original recording conditions.','small')
    # PAGE 4
    page(3,'Physical model and two reference policies')
    p('Let s[n] be the source recording under the generator\'s 2 m convention and r[n] the true source-to-microphone range in meters. The saved target is x[n] = a[n] s[n], where a[n] = 2/r[n]. This is a free-field pressure model: amplitude falls as 1/r and squared pressure falls as 1/r². Doubling distance halves pressure and requires +6.02 dB of gain to restore its level.')
    h('Primary: closest-so-far event reference')
    p('The operational event reference does not presume knowledge of the original speaking level. Define rmin[n] as the smallest range seen so far in the current clip. The evaluation waveform is')
    p('g*[n] = max(1, r[n]/rmin[n]);  d_event[n] = g*[n] x[n].','eq')
    p('A retreat should be compensated; an approach should not be attenuated just to match an older quieter point. After the drone has passed its closest point and recedes, the reference remains that earlier close range. This causal boost-only policy preserves natural variations in the underlying speech waveform rather than forcing every word to a constant RMS.')
    h('Secondary: known fixed 2 m calibration')
    p('When the reference distance is explicitly known, the geometry target is g2m[n] = r[n]/2 and the desired waveform is the supplied reference.wav. This policy restores the generator\'s level convention, including the initial attenuation. Its absolute level cannot be inferred reliably from target audio alone. The known 2 m value is a declared calibration assumption, not an inferred real microphone distance.')
    table(['Listening file','Reference to use when judging it'],[
        ['output_event_relative.wav','Closest observed range so far. It need not match the absolute level of reference.wav.'],
        ['output_fixed_2m.wav','Known 2 m benchmark reference. Compare directly with reference.wav.'],
        ['target_at_drone.wav','Original attenuated input, unchanged.'],
        ['reference.wav','Generator level reference, used for scoring/listening, never fed to the gain controller.']],[166,W-166])
    p('Neither policy reverses propagation delay, Doppler, reverberation or occlusion. Those effects were not simulated. The original 1/r synthesis and PCM quantization set the available information; gain cannot recreate discarded spectral content.','small')
    # PAGE 5
    page(4,'Final target-only controller')
    p('For each buffered 320-sample block, a valid observed pose and the declared source position give a range estimate. The smallest accepted range in the current event becomes the online reference, unless a fixed reference distance is configured. Requested geometry gain is 20 log10(current range / reference range), bounded below by zero and above by the selected ceiling.')
    h('Observed-sensor checks and fallback')
    p('The controller rejects missing/nonfinite positions, range below 1.5 m, apparent pose speed above 14 m/s, and available IMU specific-force norms outside 3-22 m/s². A valid bit is not treated as proof that a sensor is physically correct. The reported confidence trace is diagnostic; the selected geometry weight is fixed at 1.0 when pose is accepted.')
    p('When pose is rejected temporarily, the selected hold policy retains the last geometry request for up to 3.5 s. It then releases toward unity if observations remain unavailable. If geometry has never been available, the request stays at unity. Holding is not dead reckoning and does not claim that the drone stopped moving. Raw IMU is never double-integrated to manufacture an absolute source distance.')
    h('Smoothing, gain limits and peak control')
    p(f'Greq is limited to {cfg["max_gain_db"]:.0f} dB. The command follows a one-pole recursion with {cfg["smooth_s"]:.2f} s time constant and a 36 dB/s nominal slew limit. Linear sample interpolation bridges successive block gains. A common block headroom factor keeps sample peaks at or below 0.95. Warm start permits the first buffered block to use its available geometry, especially in fixed-reference mode.')
    p('One block of input must be buffered, giving 20 ms of implementation latency. Output sample indices remain aligned with input. A large overload can invoke abrupt emergency block attenuation, so this is a sample-peak guard rather than a true-peak or transparent lookahead limiter. The nominal slew bound does not constrain that emergency action.')
    h('Why the final configuration is effectively geometry only')
    p('The search allowed a robust energy reference to contribute to a hybrid gain. Development selected a geometry weight of 1.0. The reusable energy method remains available as a comparison or explicit fallback, but it is not silently substituted when all geometry is missing in the selected configuration. Even target-only audio cannot distinguish a naturally quieter syllable from increased source range.')
    # PAGE 6
    page(5,'Controls and post-2015 published gain baselines')
    p('All methods receive the same target_at_drone waveform. No baseline is a denoiser. Literature methods are implemented as gain-control cores with explicit task adaptations; reproductions of the original papers\' benchmark scores, multiband studies or hardware are not claimed.')
    h('Feedback AGC: Tisserand and Berviller, 2016 [2]')
    p('The first-order multiplicative feedback form updates gain using output squared level: g_next = g(1 + d(1 - (y/Yref)²)). Here d = 1/(2 tau fs), tau = 0.2 s and Yref = 0.04 in normalized amplitude units. This implements the software recursion form described in the paper. We add common boost-only, upper-gain and sample-peak bounds; these are declared practical adaptations, not changes hidden inside a claim of exact benchmark reproduction.','small')
    h('Power-law DRC: Corey and Singer, 2021 [3]')
    p('A one-band specialization uses the paper\'s attack/release power envelope and power-law mapping C(v) = g0² v^(1/CR), with CR = 3. We set 5 ms attack, 100 ms release and unity gain at RMS 0.04, then apply the same practical gain and peak bounds. Although the article discusses mixtures, this comparison uses only the single-target gain law. Its six-band experiments and mixture analysis are not reproduced.','small')
    table(['Additional comparison','Purpose'],[
        ['Identity','Determine whether processing improves the specified restoration target.'],
        ['Robust energy tracker','85th-percentile causal reference, slow envelope and bounded gain; requires no pose.'],
        ['Unchecked geometry','Same observed pose and limits, without plausibility rejection.'],
        ['Selected controller','Accepted geometry, temporary hold, selected cap and smoothing.'],
        ['No smoothing / no hold','Isolate gain interpolation/control response and dropout bridging.'],
        ['Maximum energy reference','Assess the effect of replacing the robust percentile with an absolute history maximum.']],[126,W-126])
    p('AGC and DRC ordinarily target an output loudness distribution rather than undoing distance attenuation. Their lower scores on a physical restoration task do not mean they are poor compressors for their intended uses. The fixed 2 m comparison additionally gives geometry methods a known calibration that audio-only methods cannot identify.','small')
    # PAGE 7
    page(6,'Development selection and test-only scoring')
    p('Internal development uses 40 calibration cases from rec1/rec3 and 20 validation cases from rec2. The fixed grid has 54 combinations: caps {12, 18, 24} dB; smoothing constants {0.04, 0.12, 0.30} s; geometry weights {0.8, 1.0}; and fallback policies {hold, unity, energy}. Calibration shortlists three candidates and source-disjoint validation selects among them.')
    p('The development-only objective is mean active-frame level MAE + 0.25 x envelope-shape error + 0.1 x gain-step p95. This is an engineering tradeoff, not an SNR or probability. Ranking values and development performance tables are deliberately not exported, in accordance with the request that reported metrics be test-only.')
    table(['Selected parameter','Value'],[
        ['Candidate ID',protocol['selected_id']],['Gain ceiling',str(cfg['max_gain_db'])+' dB'],['Geometry weight',str(cfg['geometry_weight'])],
        ['Smoothing / nominal slew',str(cfg['smooth_s'])+' s / 36 dB/s'],['Fallback / maximum hold',cfg['fallback']+' / 3.5 s'],
        ['Reference policy','Event-relative by default; optional known 2 m'],['Headroom ceiling','0.95 sample amplitude']],[168,W-168])
    h('How performance is scored')
    p('Test activity uses reference-frame RMS above 10% of its 95th percentile. Active level MAE compares output and desired frame RMS in dB. Waveform RMSE, correlation, envelope-shape error, STFT log-spectral distance, gain-step statistics and clipping are also measured. Every compared output is quantized to the delivered PCM16 convention before waveform metrics are calculated.')
    p('Metrics are first calculated per case, then macro-averaged across fifteen test conditions. Blank needed-gain values mean no eligible frames. Reconstruction SNR is scale-sensitive waveform fidelity, not a physical signal-to-noise ratio. Individual numerical scores are capped at 80 dB for reporting, with uncapped values retained for audit; exact identity controls otherwise create arbitrary large values.')
    p('The selected JSON is hashed before scoring. No test outcome changes the parameters. However, the same held-out source was already inspected in the prior noisy study. This is a reused benchmark under a changed protocol, not a newly collected or fully blind independent test.','small')
    # PAGE 8
    page(7,'Held-out restoration performance')
    labels={'identity':'No gain','energy_only':'Energy only','feedback_agc_2016':'Feedback AGC 2016','powerlaw_drc_2021':'Power-law DRC 2021','selected':'Selected pose-guided'}
    table(['Method','Event level MAE dB','Event waveform RMSE','2 m level MAE dB'],[
        [lab,fmt(get('event',m,'active_level_mae_db')),fmt(get('event',m,'waveform_rmse'),4),fmt(get('fixed_2m',m,'active_level_mae_db'))] for m,lab in labels.items()
    ],[149,94,105,W-348])
    fig('test_comparison.png','Figure 1. Mean active-level error on the fifteen test cases. Two reference policies answer different questions; they should not be conflated.',235)
    p('For the primary event-relative target, the selected controller reduces level MAE from 4.24 to 1.23 dB, about 71%. Mean waveform RMSE falls from 0.0050 to 0.0018, while correlation rises from 0.9808 to 0.9931. This supports physical amplitude recovery under the declared clean-target and available-pose conditions.')
    p('With known 2 m calibration, level MAE falls from 15.42 to 3.92 dB and RMSE from 0.0627 to 0.0189. Energy-only has similar level MAE (3.94 dB) but RMSE 0.0569: a level metric alone can hide waveform distortion. Calibrated mode restores initial attenuation but requires a known reference; it is not an audio-only inference.')
    p('The energy-only tracker produces 11.73 dB primary level error. Its reference over-amplifies natural speech variation, illustrating why simply making every block similarly loud is not the desired reconstruction. No result from train or validation is mixed into these performance figures.','small')
    # PAGE 9
    page(8,'Envelope preservation and sensor failures')
    fig('test_fidelity.png','Figure 2. Test-case distributions of waveform fidelity, natural-envelope preservation and applied-gain variation. All cases share one held-out source recording.',235)
    fb=pd.read_csv(ROOT/'metrics/test_sensor_probes.csv').groupby('condition')[['active_level_mae_db','envelope_shape_rmse_db']].mean()
    table(['Test sensor condition','Level MAE dB','Envelope error dB'],[[name,fmt(r.active_level_mae_db),fmt(r.envelope_shape_rmse_db)] for name,r in fb.iterrows()],[210,109,W-319])
    p('The nominal selected test envelope-shape error is about 0.66 dB, compared with 2.17 dB without correction and 3.75 dB for energy-only control. Geometry can change slow distance gain while preserving naturally quiet and loud syllables. This distinction remains important even after added noise has been removed.')
    p('With no pose available from the start, the selected hold fallback has nothing to hold and returns unity gain. It cannot recover unknown source distance from IMU alone. A frozen position can look like a real hover; a constant bias also distorts relative range. The table reports those failures instead of treating valid flags as ground truth.')
    p('Removing smoothing reduces lag but can increase abrupt gain variation. Removing hold loses motion correction during telemetry gaps. These comparisons reuse test inputs with frozen settings for diagnosis only; they are not used to select a new threshold. No independent-speaker statistical significance is claimed.','small')
    # PAGE 10
    page(9,'Time-frequency and DSP interpretation')
    fig('test_retreat_trace.png','Figure 3. A held-out fast retreat. True range is an evaluation trace; the controller uses only observed telemetry. The short missing-pose interval invokes hold, not oracle motion knowledge.',270)
    fig('test_envelopes.png','Figure 4. The same test input and both operated outputs. No waveform is separately peak-normalized. The event output and fixed-reference output have intentionally different targets.',190)
    p('Gain is shared across all samples/frequencies of a block rather than estimated independently per FFT bin. A slowly varying positive multiplier preserves sample timing and introduces no explicit phase-rotation filter. Nevertheless time variation multiplies the waveform and therefore convolves its spectrum; fast changes can create sidebands. Preserving sample indices is not a claim that the complete adaptive system has one fixed phase response.')
    p('Hann STFT plots are diagnostic only: 1024 samples, 256-sample hop at 16 kHz, with one fixed color scale. No STFT masking or spectral noise filtering is applied by this target-only pipeline. Original source coloration and limited bandwidth remain.','small')
    # PAGE 11
    page(10,'Envelope FIR/IIR response and stability')
    p('The controller\'s nominal smoothing operates at 50 Hz in the gain-control domain. For a constant-mode one-pole smoother, p = exp(-0.02/tau) and H(z) = (1-p)/(1-p z^(-1)). The selected tau = 0.04 s gives p about 0.6065, inside the unit circle. This is not an audio-rate rotor notch.')
    fig('dsp_filters.png','Figure 5. A fixed-mode IIR smoother and a nearest-integer boxcar FIR with approximately matched bandwidth. Magnitude, phase, group delay, step, impulse and pole-zero structure explain the smoothing tradeoff.',325)
    filters=pd.read_csv(ROOT/'analysis/filter_analysis.csv')
    table(['Filter','Measured -3 dB Hz','Delay at 0.2 Hz','Taps / coefficients'],[
        [r['filter'],fmt(r.measured_3db_hz,3),fmt(r.group_delay_at_0_2hz_s,4)+' s',int(r.taps)] for _,r in filters.iterrows()
    ],[73,134,123,W-330])
    p('The FIR has linear phase and a finite response but requires a small delay line; the IIR uses a compact state and an adjustable time constant. Integer FIR order limits exact bandwidth matching, so measured corners are reported instead of calling specifications identical. The chosen controller is not globally LTI: pose rejection, reference minima, event reset, slew saturation and emergency peak protection make it nonlinear and time varying.')
    p('No single pole-zero plot or group-delay curve is assigned to the complete amplitude module. Its directly measurable implementation latency is one 20 ms block; its response lag depends on smoothing and the rate of geometric change.','small')
    # PAGE 12
    page(11,'All-case listening outputs and reproducibility')
    p('The single ZIP is self-contained. It includes the final pipeline, experiment scripts, selected configuration, test metrics, diagnostic figures, this PDF, target-only input/reference data and every operated output. No older noisy experiment, synthetic noise track or mixture waveform is required.')
    table(['Location','Use'],[
        ['listening/index.html','Open locally in a browser. Every row has input, event output, calibrated output and reference audio controls.'],
        ['listening/audio_index.csv','Rec-number mapping, scenario, split, sensor condition and exact relative file paths.'],
        ['listening/rec1 ... rec5/','Operated PCM16 WAVs. Rec5 mirrors the rec4 outputs bit-for-bit.'],
        ['data/cases/scenario/','Unchanged target_at_drone.wav and reference.wav, observed telemetry and evaluation trajectory.'],
        ['audio_logs/scenario/','Controller traces and sample gain arrays for all unique cases; no train performance-error tables.'],
        ['metrics/test_*.csv','All dataset performance scores are from the test partition.'],
        ['pipeline/ and configs/','Reusable final module, command line and frozen selected parameters.']],[153,W-153])
    p('There are 75 unique input cases and 150 unique operated output files, one per reference policy. For convenience, 30 rec5 alias files mirror the fifteen rec4 cases in both policies, giving 180 listening output WAVs and 90 labelled rows. The aliases do not add independent samples and are not re-scored.')
    h('Listening and rerunning')
    p('Start with listening/index.html. For the primary mode, judge whether the speech remains at the closest-so-far level while preserving its natural rhythm and emphasis. For fixed 2 m mode, compare with reference.wav. Do not separately peak-normalize files: doing so would erase the amplitude effect being evaluated.')
    p('From the extracted package root, install requirements and run python experiments/run.py --resume, then experiments/analyze.py and experiments/report.py. A full development search is available by omitting --resume. It computes ranking internally but exports no development performance table. The included data subset is sufficient to reproduce the study.','small')
    # PAGE 13
    page(12,'Complexity, numerical checks and limitations')
    table(['Test runtime / precision item','Measured scope'],[
        ['Named runtime case','test_02_00_recede_fast, ten seconds, five repeats.'],
        ['Median real-time factor',fmt(runtime['median_real_time_factor'],4)+'; amplitude computation only.'],
        ['Peak wrapper allocation',fmt(runtime['wrapper_peak_allocation_bytes']/1024**2,2)+' MiB by tracemalloc, including output/gain/log arrays.'],
        ['Float32 input sensitivity',f'{runtime["float32_input_max_output_difference"]:.2e}'+' maximum output difference; internal arithmetic remains float64.'],
        ['PCM16 delivery error',f'{runtime["pcm16_max_quantization_error"]:.2e}'+' maximum waveform quantization error.'],
        ['Latency / state','20 ms buffered block; bounded histories and a small set of gain/pose states.']],[164,W-164])
    p('Seven automated checks verify identity, missing-pose unity behavior, distance doubling, constant-range prosody preservation, causality, reset/peak protection and invalid-input rejection. File audits verify all outputs, source labels, rec4/rec5 duplicate equality, frozen selection and test-only performance tables. Input target copies retain their original bytes.')
    h('What the results do not establish')
    p('No low-SNR refusal threshold can be validated here, because synthetic interference is absent. The source recording may retain noise of its own, but it is not separated or used to claim noise reduction. The SNR gate should be reintroduced and independently validated when this stage is integrated after a real noise-suppression pipeline.')
    p('Only one independent source recording supplies the fifteen held-out conditions. Aggregate improvements are conditional on those words, voices, trajectories and sensor assumptions. The test source was reused from earlier work, so the result is not a fresh blind validation. No listening panel, intelligibility metric or universal speech-quality claim is supplied.')
    p('A hold is reliable only as a short continuity policy; it cannot know current range during a true navigation failure. The known source location, free-field law and fixed microphone sensitivity may fail in deployment. Large required gains exceed the 18 dB cap. The peak guard protects sample amplitudes but is not a true-peak limiter and can create an abrupt change on overload.','small')
    # PAGE 14
    page(13,'Inference, final selection and references')
    p('Under the requested target-only protocol, source-relative geometry provides useful information that energy normalization lacks. Development selects zero energy contribution whenever pose is accepted. Test level errors improve for both the online event reference and the known fixed-distance reference, while the selected controller better preserves the source envelope than energy-only control.')
    p('The final selection is an 18 dB cap, 0.04 s smoothing, 36 dB/s nominal slew and a 3.5 s temporary hold. No additional drone noise or denoising is involved. Keep the primary event policy for unknown source loudness; use the fixed 2 m output only when that calibration is genuinely available. If geometry is missing from the start, the selected fallback is unity rather than an invented IMU range estimate.')
    p('These findings refine the amplitude section without replacing the original project\'s sensor-assisted filtering question. In the full report, the defensible claim is partial range-induced level restoration on isolated targets. Do not substitute this result for evidence of rotor-noise suppression, fault awareness, low-SNR gating or real-flight performance.')
    h('References')
    for ref in [
        '[1] IIT Madras, ED5017 Course Research Project. Moodle evaluation instructions and Group 4 problem statement, supplied course documents, October 2026. Original project title retained.',
        '[2] E. Tisserand and Y. Berviller. Design and implementation of a new digital automatic gain control. Electronics Letters 52(22), 1847-1849, 2016. DOI: 10.1049/el.2016.1398. Software feedback recursion used with declared practical bounds.',
        '[3] R. M. Corey and A. C. Singer. Modeling the effects of dynamic range compression on signals in noise. Journal of the Acoustical Society of America 150(1), 159-170, 2021. DOI: 10.1121/10.0005314. Author preprint: arXiv:2012.03860. One-band envelope/power-law specialization only.',
    ]:p(ref,'small')
    h('Data and implementation record')
    p('Source speech is the user-provided rec1-rec5 collection. Rec4 and rec5 decode identically. Synthetic source-to-drone geometry and telemetry come from the earlier supplied dataset. This package includes only its target/reference subset and explicitly excludes added-noise and mixture audio. Original recording imperfections are retained. No neural network, external account data or fabricated measurement is used.','small')
    p('The audio index, selected parameter JSON, per-case test metrics, source/selection hashes and export audit provide the exact operational record. Reported reconstruction SNR uses a defined waveform reference and an 80 dB reporting ceiling; it does not indicate a measured acoustic noise-floor improvement.','small')
    out=ROOT/'report/target_only_amplitude_report.pdf';out.parent.mkdir(parents=True,exist_ok=True)
    doc=SimpleDocTemplate(str(out),pagesize=A4,leftMargin=28*mm,rightMargin=28*mm,topMargin=21*mm,bottomMargin=20*mm,
                         title='Target-only amplitude restoration: experiments and implementation',author='Group 4, IIT Madras ED5017')
    def decorate(c,d):
        c.saveState();c.setFont('Sans',7.5);c.setFillColor(colors.HexColor('#566b76'))
        c.drawString(28*mm,A4[1]-13*mm,'ED5017 | TARGET-ONLY AMPLITUDE STUDY')
        c.setStrokeColor(colors.HexColor('#ccdbe3'));c.line(28*mm,A4[1]-16*mm,A4[0]-28*mm,A4[1]-16*mm)
        c.drawString(28*mm,12*mm,'Group 4 | 29 October 2026');c.drawRightString(A4[0]-28*mm,12*mm,f'{d.page} / 14');c.restoreState()
    doc.build(story,onFirstPage=decorate,onLaterPages=decorate)
    from pypdf import PdfReader
    count=len(PdfReader(out).pages);print('Report:',out,'pages:',count)
    assert count==14,'Report overflow'


if __name__=='__main__':build()
