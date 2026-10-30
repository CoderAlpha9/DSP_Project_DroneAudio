"""Streaming SPP noise tracker, DD Wiener gain, and optional RPM models.

The original rotor study used NFFT=1024, hop=256. Its offline centered
agreement smoothers and calibration are made causal here; these are new
integration results, not numerical replays of archived waveforms.
"""
import time
import numpy as np
from scipy import ndimage, signal
from .telemetry import ReliabilityMonitor

EPS = 1e-18


def pole_time_constant(alpha, hop=256, sample_rate=16000):
    if not np.isfinite(alpha) or not 0 <= alpha < 1: raise ValueError("Stable pole requires 0 <= alpha < 1")
    return 0. if alpha == 0 else -(hop/sample_rate)/np.log(alpha)


def relative_transport(template, rpm, reference_rpm, power=2.):
    """Positive per-motor constant scale cancels in rpm/reference_rpm."""
    rpm = np.asarray(rpm, float); ref = np.asarray(reference_rpm, float)
    if rpm.shape != ref.shape or not np.isfinite(rpm).all() or not np.isfinite(ref).all() or np.any(ref <= 0) or np.any(rpm <= 0):
        raise ValueError("Transport requires matching, positive, finite motor calibration")
    template = np.asarray(template, float)
    if not np.isfinite(template).all() or np.any(template < 0): raise ValueError("Invalid PSD template")
    sm = ndimage.percentile_filter(template, percentile=25, size=13, mode="nearest")
    sm = ndimage.gaussian_filter1d(sm, 3)
    return _transport_parts(sm, np.maximum(template-sm,0), rpm, ref, power)


def _transport_parts(smooth, lines, rpm, ref, power):
    bins = np.arange(len(lines)); out = np.zeros_like(lines)
    ratios = np.clip(rpm/ref,.35,2.5)
    for ratio in ratios:
        out += (smooth+np.interp(bins/ratio,bins,lines,left=lines[0],right=0))*ratio**power/len(ratios)
    return np.maximum(out, EPS)


def harmonic_gain(freqs, rpm, *, blades=2, harmonics=8, width_hz=18., attenuation=.15):
    """Blade-passing harmonic k is blades*k*rpm/60 Hz, not rpm Hz."""
    gain = np.ones_like(freqs, dtype=float)
    for motor in np.asarray(rpm):
        if not np.isfinite(motor) or motor <= 0: continue
        for k in range(1, harmonics+1):
            frequency = blades*k*motor/60.
            if frequency <= 0 or frequency > freqs[-1]: continue
            gain[np.abs(freqs-frequency) <= width_hz] = attenuation
    return gain


def ridge_agreement(power, rpm, df):
    scores = []
    for motor in rpm:
        for order in (2,4,6,8):
            b = int(round(order*motor/60/df))
            if b < 8 or b >= len(power)-8: continue
            loc = power[b-1:b+2]; offset = int(np.argmax(loc))-1
            base = np.median(np.r_[power[b-7:b-3],power[b+4:b+8]])+EPS
            cdb = 10*np.log10((np.max(loc)+EPS)/base)
            contrast = 1/(1+np.exp(np.clip(-(cdb-2)/1.5,-700,700)))
            scores.append(contrast*np.exp(-.5*(offset/.65)**2))
    if not scores: return 0.
    a = np.asarray(scores); lo,hi = np.percentile(a,[35,75]); keep=a[(a>=lo)&(a<=hi)]
    return float(np.mean(keep)) if len(keep) else 0.


class MotionCanceller:
    """New conservative causal NLMS adaptation of the filter suite.

    Only the <=20 Hz microphone component is an adaptation target. Held IMU
    is low-pass filtered; no high-frequency vibration is synthesized. Weight
    updates stop during speech. This optional component has no field claim.
    """
    def __init__(self, config):
        self.c = config
        self.sos = signal.butter(2, config.motion_cutoff_hz, fs=16000, output="sos")
        self.zi_audio = np.zeros((len(self.sos),2)); self.zi_ref = self.zi_audio.copy()
        self.weights = np.zeros(16); self.buffer=np.zeros(16)

    def process(self, x, imu, speech_active):
        low,self.zi_audio=signal.sosfilt(self.sos,x,zi=self.zi_audio)
        if imu is None:
            _,self.zi_ref=signal.sosfilt(self.sos,np.zeros(len(x)),zi=self.zi_ref)
            return x.copy(), "missing_imu"
        reference = np.full(len(x), np.clip(np.linalg.norm(imu)-9.80665,-5,5))
        ref,self.zi_ref=signal.sosfilt(self.sos,reference,zi=self.zi_ref)
        estimate=np.zeros(len(x)); bound=.2*np.sqrt(np.mean(x*x)+EPS)
        for i,(target,r) in enumerate(zip(low,ref)):
            self.buffer[1:]=self.buffer[:-1];self.buffer[0]=r
            prediction=float(self.weights@self.buffer)
            if not speech_active:
                self.weights += self.c.motion_mu*(target-prediction)*self.buffer/(self.buffer@self.buffer+1e-5)
                norm=np.linalg.norm(self.weights)
                if norm>1:self.weights/=norm
            estimate[i]=np.clip(prediction,-bound,bound)
        return x-estimate,"speech_update_hold" if speech_active else "bounded_low_band_nlms"


class SpectralEnhancer:
    def __init__(self, config, telemetry):
        self.config=config;self.c=config.enhancement;self.telemetry=telemetry
        c=self.c; self.window=signal.windows.hann(c.n_fft,sym=False)
        self.history=np.zeros(c.n_fft);self.ola=np.zeros(c.n_fft);self.norm=np.zeros(c.n_fft)
        self.seen=0;self.noise=None;self.prob=None;self.previous=None
        self.cal_psd=[];self.cal_rpm=[];self.template=None;self.reference_rpm=None
        self.parts=None;self.cal_complete=False;self.corr_ema=0.;self.ridge_ema=0.
        self.monitor=ReliabilityMonitor(config);self.motion=MotionCanceller(c) if c.motion_enabled else None
        self.freqs=np.fft.rfftfreq(c.n_fft,1/16000);self.traces=[];self.gains=[]
        self.seconds=0.;self.telemetry_seconds=0.;self.motion_seconds=0.

    def process_hop(self, x, *, speech_active=False, valid_samples=None):
        start=time.perf_counter();c=self.c
        x=np.asarray(x,float)
        if x.shape!=(c.hop,): raise ValueError("Enhancer requires one configured hop")
        self.seen+=c.hop;time_s=self.seen/16000
        sensor_start=time.perf_counter()
        sensor_time=min(time_s,valid_samples/16000) if valid_samples is not None else time_s
        snapshot=self.telemetry.at(sensor_time,self.config.telemetry_max_age_s)
        rpm,reason=self.monitor.inspect(snapshot,sensor_time)
        self.telemetry_seconds+=time.perf_counter()-sensor_start
        motion_reason="disabled"
        if self.motion is not None and c.enabled:
            motion_start=time.perf_counter();x,motion_reason=self.motion.process(x,snapshot["imu"],speech_active)
            self.motion_seconds+=time.perf_counter()-motion_start
        self.history[:-c.hop]=self.history[c.hop:];self.history[-c.hop:]=x
        X=np.fft.rfft(self.history*self.window);P=abs(X)**2
        prefix=self.config.speech_free_prefix_s
        full_frame=self.seen>=c.n_fft
        in_cal=prefix is not None and time_s<=prefix+1e-12
        if in_cal and full_frame:
            self.cal_psd.append(P.copy())
            if rpm is not None:self.cal_rpm.append(rpm.copy())
        if not self.cal_complete and prefix is not None and time_s>prefix:
            if self.cal_psd:
                self.template=np.maximum(np.mean(self.cal_psd,axis=0),EPS)
                self.noise=self.template.copy()
                if len(self.cal_rpm)>=3:
                    self.reference_rpm=np.median(self.cal_rpm,axis=0)
                    sm=ndimage.gaussian_filter1d(ndimage.percentile_filter(self.template,25,size=13,mode="nearest"),3)
                    self.parts=(sm,np.maximum(self.template-sm,0))
            self.cal_psd.clear();self.cal_rpm.clear();self.cal_complete=True
        if self.noise is None:
            self.noise=np.maximum(P*(1. if prefix else .1),EPS)
            self.prob=np.zeros(len(P));self.previous=np.zeros(len(P))
        gamma=P/np.maximum(self.noise,EPS)
        pp=1/(1+32.6227766*np.exp(np.maximum(-gamma*31.6227766/32.6227766,-700)))
        self.prob=.9*self.prob+.1*pp
        pp=np.where(self.prob>.99,np.minimum(pp,.99),pp)
        self.noise=np.maximum(c.noise_alpha*self.noise+(1-c.noise_alpha)*((1-pp)*P+pp*self.noise),EPS)
        NA=self.noise;N=NA;weight=0.;corr=0.;ridge=0.
        if rpm is not None and c.mode!="audio_only":
            ridge=ridge_agreement(P,rpm,16000/c.n_fft)
            self.ridge_ema+=(ridge-self.ridge_ema)/9
            if c.mode!="order":
                if self.parts is None or self.reference_rpm is None or self.reference_rpm.shape!=rpm.shape:
                    reason="no_declared_rpm_template_calibration"
                else:
                    NR=_transport_parts(*self.parts,rpm,self.reference_rpm,c.rpm_power)
                    band=(self.freqs>=100)&(self.freqs<=3200)
                    a=np.log(P+EPS)-np.log(NA+EPS);b=np.log(NR+EPS)-np.log(NA+EPS)
                    a=a-ndimage.gaussian_filter1d(a,4);b=b-ndimage.gaussian_filter1d(b,4)
                    a=a[band]-a[band].mean();b=b[band]-b[band].mean()
                    corr=float(a@b/np.sqrt((a@a)*(b@b)+EPS));self.corr_ema+=(corr-self.corr_ema)/7
                    if c.mode=="correlation":weight=float(np.clip((self.corr_ema-.35)/.5,0,1))
                    elif c.mode=="ridge":weight=float(np.clip((self.ridge_ema-.2)/.4,0,1))
                    else:weight=c.rpm_weight
                    if c.mode!="fixed_rpm":NR=np.clip(NR,NA/c.psd_cap,NA*c.psd_cap)
                    N=(1-weight)*NA+weight*NR
        posterior=np.minimum(P/np.maximum(N,EPS),1e5)
        xi=c.dd_alpha*self.previous+(1-c.dd_alpha)*np.maximum(posterior-c.beta,0)
        gain=np.maximum(xi/(xi+c.beta),c.floor)
        if c.mode=="order" and rpm is not None:
            gain*=harmonic_gain(self.freqs,rpm,blades=c.blades,harmonics=c.harmonics,width_hz=c.order_width_hz,attenuation=c.order_attenuation)
            weight=1.
        if prefix is None and time_s<c.startup_s:gain=np.maximum(gain,.5)
        if in_cal or not c.enabled:gain=np.ones_like(P);weight=0.
        self.previous=gain*gain*posterior
        self.ola+=np.fft.irfft(X*gain,n=c.n_fft)*self.window;self.norm+=self.window**2
        output=np.divide(self.ola[:c.hop],self.norm[:c.hop],out=np.zeros(c.hop),where=self.norm[:c.hop]>1e-15)
        self.ola[:-c.hop]=self.ola[c.hop:];self.ola[-c.hop:]=0
        self.norm[:-c.hop]=self.norm[c.hop:];self.norm[-c.hop:]=0
        # SNR estimate used only as a conservative integrated gain guard, not measured truth.
        snr_est=float(10*np.log10((np.sum(np.maximum(P-N,0))+EPS)/(np.sum(N)+EPS)))
        log={"frame_start_s":(self.seen-c.n_fft)/16000,"available_time_s":time_s,
             "rpm_valid":int(rpm is not None),"rpm_status":reason,"rpm_age_s":snapshot["rpm_age_s"],
             "correlation_raw":corr,"correlation_causal":self.corr_ema,"ridge_raw":ridge,"ridge_causal":self.ridge_ema,
             "fusion_weight":weight,"snr_estimate_db":snr_est,"mean_spectral_gain":float(gain.mean()),
             "declared_calibration":int(in_cal),"motion_status":motion_reason}
        self.traces.append(log);self.gains.append(gain)
        self.seconds+=time.perf_counter()-start
        return self.seen-c.n_fft,output,log
