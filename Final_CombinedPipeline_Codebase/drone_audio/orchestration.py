"""Common sample timeline; independent 160/256/320-sample module clocks."""
from copy import deepcopy
import time
import numpy as np
from .config import PipelineConfig
from .detection import Detector
from .enhancement import SpectralEnhancer
from .amplitude import AmplitudeRestorer
from .telemetry import ObservedTelemetry


class DronePipeline:
    def __init__(self, config=None, *, telemetry=None):
        self.config=deepcopy(config or PipelineConfig()).validate()
        self.telemetry=telemetry or ObservedTelemetry()
        self.reset(telemetry=self.telemetry)

    def reset(self, *, telemetry=None):
        """Full reset between unrelated recordings; unlike new_source_session."""
        self.telemetry=telemetry if telemetry is not None else ObservedTelemetry()
        self.detector=Detector() if self.config.detector_enabled else None
        self.enhancer=SpectralEnhancer(self.config,self.telemetry)
        self.amplitude=AmplitudeRestorer(self.config)
        self.pending_detection=np.array([],float);self.pending_enhancement=np.array([],float)
        self.pending_amplitude=np.array([],float);self.logs_detection=[];self.logs_amplitude=[]
        self.pre_blocks=[];self.output_blocks=[];self.gain_blocks=[]
        self.received=0;self.emitted=0;self.closed=False;self.last_snr=0.
        self.detector_seconds=0.;self.amplitude_seconds=0.;self.total_seconds=0.

    def new_source_session(self, **kwargs):
        """Application-authorized source change, without clearing noise/rotor state."""
        if len(self.pending_amplitude):
            raise ValueError("Change source sessions at an emitted 20 ms boundary")
        self.amplitude.new_source_session(**kwargs)

    def _decision(self,end_sample):
        # Never read a detector decision from later on the microphone timeline.
        index=min(end_sample//160,len(self.logs_detection))-1
        if index<0:return {"activity_active":False,"event_active":False,"speech_active":False}
        return self.logs_detection[index]

    def _detect(self, x, final=False):
        self.pending_detection=np.r_[self.pending_detection,x]
        while len(self.pending_detection)>=160:
            hop=self.pending_detection[:160];self.pending_detection=self.pending_detection[160:]
            start=time.perf_counter()
            if self.detector is not None:r=self.detector.process_hop(hop)
            else:
                i=len(self.logs_detection);r={"time_start_s":i*.01,"available_time_s":(i+1)*.01,
                    "activity_active":0,"event_active":0,"speech_active":0,"event_opened":0,"event_closed":0,"event_id":0,
                    "activity_score":0.,"speech_score":0.,"detector_bypassed":1}
            self.detector_seconds+=time.perf_counter()-start;self.logs_detection.append(r)
        if final and len(self.pending_detection):
            n=len(self.pending_detection);self._detect(np.zeros(160-n))

    def _amplify(self, final=False):
        while len(self.pending_amplitude)>=320 or (final and len(self.pending_amplitude)):
            n=min(320,len(self.pending_amplitude));x=self.pending_amplitude[:n];self.pending_amplitude=self.pending_amplitude[n:]
            end=self.emitted+n;t=end/16000
            decision=self._decision(end)
            start=time.perf_counter();snapshot=self.telemetry.at(t,self.config.telemetry_max_age_s)
            y,g,log=self.amplitude.process_block(x,time_s=t,snapshot=snapshot,
                activity_active=decision["activity_active"],event_active=decision["event_active"],speech_active=decision["speech_active"],
                snr_estimate_db=self.last_snr)
            self.amplitude_seconds+=time.perf_counter()-start
            self.pre_blocks.append(x.copy());self.output_blocks.append(y);self.gain_blocks.append(g);self.logs_amplitude.append(log)
            self.emitted=end

    def _enhance_hop(self, hop, *, final=False):
        decision=self._decision(min(self.enhancer.seen+len(hop),self.received))
        position,y,log=self.enhancer.process_hop(hop,speech_active=decision["speech_active"],valid_samples=self.received)
        self.last_snr=log["snr_estimate_db"]
        if position>=0 and position<self.received:
            y=y[:min(len(y),self.received-position)]
            self.pending_amplitude=np.r_[self.pending_amplitude,y]
            self._amplify()

    def process_block(self, samples):
        """Accept arbitrary nonempty/empty mono chunks; return ready output.

        Output is delayed, not shifted: concatenating returns plus flush gives
        exactly the input sample count at the original sample indices.
        """
        if self.closed:raise RuntimeError("Pipeline is flushed; call reset before processing another recording")
        x=np.asarray(samples,float)
        if x.ndim!=1 or not np.isfinite(x).all():raise ValueError("Supply finite mono audio")
        start=time.perf_counter();before=len(self.output_blocks)
        self.received+=len(x);self._detect(x)
        self.pending_enhancement=np.r_[self.pending_enhancement,x]
        hop=self.config.enhancement.hop
        while len(self.pending_enhancement)>=hop:
            block=self.pending_enhancement[:hop];self.pending_enhancement=self.pending_enhancement[hop:]
            self._enhance_hop(block)
        self.total_seconds+=time.perf_counter()-start
        new=self.output_blocks[before:]
        return np.concatenate(new) if new else np.array([],float)

    def flush(self):
        if self.closed:return np.array([],float)
        start=time.perf_counter();before=len(self.output_blocks)
        self._detect(np.array([],float),final=True)
        hop=self.config.enhancement.hop
        if len(self.pending_enhancement):
            self._enhance_hop(np.pad(self.pending_enhancement,(0,hop-len(self.pending_enhancement))),final=True)
            self.pending_enhancement=np.array([],float)
        if self.received:
            for _ in range((self.config.enhancement.n_fft-hop)//hop):self._enhance_hop(np.zeros(hop),final=True)
        self._amplify(final=True);self.closed=True
        self.total_seconds+=time.perf_counter()-start
        if self.emitted!=self.received:raise RuntimeError(f"Output length mismatch {self.emitted} != {self.received}")
        new=self.output_blocks[before:]
        return np.concatenate(new) if new else np.array([],float)

    @property
    def audio(self):return np.concatenate(self.output_blocks) if self.output_blocks else np.array([],float)

    @property
    def enhanced(self):return np.concatenate(self.pre_blocks) if self.pre_blocks else np.array([],float)

    @property
    def sample_gain(self):return np.concatenate(self.gain_blocks) if self.gain_blocks else np.array([],float)

    @property
    def timing(self):
        return {"compute_total_s":self.total_seconds,"detection_s":self.detector_seconds,
            "enhancement_including_telemetry_motion_s":self.enhancer.seconds,"telemetry_within_enhancement_s":self.enhancer.telemetry_seconds,
            "motion_within_enhancement_s":self.enhancer.motion_seconds,"amplitude_including_pose_lookup_s":self.amplitude_seconds,
            "real_time_factor":self.total_seconds/max(self.received/16000,1e-12),
            "stft_buffer_s":self.config.enhancement.n_fft/16000,
            "steady_reconstruction_lookahead_s":(self.config.enhancement.n_fft-self.config.enhancement.hop)/16000,
            "amplitude_block_s":.02,"detector_hop_s":.01,
            "maximum_output_buffer_s":(self.config.enhancement.n_fft+320)/16000,
            "streaming_note":"Buffering plus detector confirmation/release; telemetry always held from past observations. CPU time is not latency."}


def process_audio(samples, config=None, *, telemetry=None, chunk_size=2048):
    if not isinstance(chunk_size,int) or chunk_size<=0:raise ValueError("chunk_size must be a positive integer")
    p=DronePipeline(config,telemetry=telemetry);x=np.asarray(samples,float)
    if x.ndim!=1 or not np.isfinite(x).all():raise ValueError("Supply finite mono audio")
    for start in range(0,len(x),chunk_size):p.process_block(x[start:start+chunk_size])
    p.flush();return p
