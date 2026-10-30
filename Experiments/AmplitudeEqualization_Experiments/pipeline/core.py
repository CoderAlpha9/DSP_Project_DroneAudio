"""Amplitude-only DSP for already isolated target audio. No noise or denoiser."""
from dataclasses import dataclass
from collections import deque
import math
import numpy as np


@dataclass
class Config:
    sample_rate: int = 16000
    block_ms: float = 20.
    method: str = 'hybrid'
    max_gain_db: float = 24.
    geometry_weight: float = 1.
    smooth_s: float = .12
    max_slew_db_s: float = 36.
    reference_distance_m: float | None = None
    envelope_s: float = .25
    percentile: float = 85.
    history_s: float = 2.
    fallback: str = 'hold'
    hold_s: float = 3.5
    confidence_checks: bool = True
    max_pose_speed_m_s: float = 14.
    peak_ceiling: float = .95
    warm_start: bool = True
    min_event_rms: float = 1e-5
    event_close_s: float = 1.
    reset_on_close: bool = True


class AmplitudePipeline:
    """One buffered block of latency. Source and observed pose must share ENU.

    For this controlled experiment every complete dataset clip is one event.
    Supply an upstream event state in a larger pipeline. No SNR is inferred or
    used: the caller explicitly assumes already isolated target audio.
    """
    def __init__(self,config=None):
        self.c=config or Config()
        c=self.c
        if c.sample_rate<=0 or c.block_ms<=0 or c.smooth_s<=0 or c.envelope_s<=0 or c.history_s<=0:
            raise ValueError('Invalid sample rate, block duration or time constant')
        if c.max_gain_db<0 or not 0<=c.geometry_weight<=1 or not 0<c.peak_ceiling<1:
            raise ValueError('Invalid gain, weight or peak ceiling')
        if c.reference_distance_m is not None and c.reference_distance_m<=0:raise ValueError('Reference distance must be positive')
        if c.fallback not in ['hold','unity','energy']:raise ValueError('Unknown fallback')
        if c.method not in ['identity','energy','geometry','hybrid']:raise ValueError('Unknown method')
        self.n=round(c.sample_rate*c.block_ms/1000);self.dt=self.n/c.sample_rate
        self.t=0.;self.gain_db=0.;self.prev_gain=1.;self.prev_pose=None;self.prev_pose_t=None
        self.confidence=0.;self.last_geometry_gain_db=0.;self.last_valid_time=None;self.event=False;self.quiet=0
        self.history=deque(maxlen=max(3,round(c.history_s/self.dt)))
        self.reference_db=None;self.level_db=None;self.closest_range=None;self.started=False

    def reset_event(self):
        self.history.clear();self.reference_db=None;self.level_db=None;self.closest_range=None
        self.last_geometry_gain_db=0.;self.last_valid_time=None

    def process_block(self,x,*,pose=None,pose_valid=False,imu_force=None,imu_valid=False,
                      source_position=(0.,0.,0.),event_active=None):
        c=self.c;x=np.asarray(x,dtype=np.float64)
        if x.ndim!=1 or len(x)!=self.n or not np.isfinite(x).all():raise ValueError('Supply one finite mono block')
        src=np.asarray(source_position,float)
        if src.shape!=(3,) or not np.isfinite(src).all():raise ValueError('Supply a finite source position')
        rms=float(np.sqrt(np.mean(x*x)+1e-30))
        if event_active is None:
            self.quiet=0 if rms>c.min_event_rms else self.quiet+1
            event=rms>c.min_event_rms or (self.event and self.quiet*self.dt<c.event_close_s)
        else:event=bool(event_active)
        closed=self.event and not event
        if closed and c.reset_on_close:self.reset_event()
        self.event=event
        usable=bool(pose_valid and pose is not None and np.asarray(pose).shape==(3,) and np.isfinite(pose).all())
        distance=None;reason='missing'
        if usable:
            p=np.asarray(pose,float);distance=float(np.linalg.norm(p-src));reason='valid'
            if c.confidence_checks:
                if distance<1.5:usable=False;reason='range_floor'
                if self.prev_pose is not None:
                    elapsed=self.t-self.prev_pose_t
                    if 0<elapsed<.5 and np.linalg.norm(p-self.prev_pose)/elapsed>c.max_pose_speed_m_s:
                        usable=False;reason='pose_jump'
                if imu_valid and imu_force is not None and np.isfinite(imu_force).all():
                    if not 3<np.linalg.norm(imu_force)<22:usable=False;reason='imu_outlier'
            self.prev_pose=p.copy();self.prev_pose_t=self.t
        a=math.exp(-self.dt/.08)
        self.confidence=a*self.confidence+(1-a) if usable else 0.
        instant=20*math.log10(max(rms,1e-12));before=self.level_db
        a=math.exp(-self.dt/c.envelope_s)
        self.level_db=instant if before is None else a*before+(1-a)*instant
        transient=before is not None and instant-before>12
        if event and not transient and rms>c.min_event_rms:
            self.history.append(self.level_db)
            if len(self.history)>=3:
                ref=float(np.percentile(self.history,c.percentile))
                self.reference_db=ref if self.reference_db is None else max(self.reference_db,ref)
        energy=0. if self.reference_db is None else max(0.,self.reference_db-self.level_db)
        if usable and event:
            self.closest_range=distance if self.closest_range is None else min(self.closest_range,distance)
            rr=c.reference_distance_m or self.closest_range
            geo=max(0.,20*math.log10(distance/rr))
            self.last_geometry_gain_db=geo;self.last_valid_time=self.t
        else:
            held=self.last_valid_time is not None and self.t-self.last_valid_time<=c.hold_s
            geo=self.last_geometry_gain_db if c.fallback=='hold' and held else (energy if c.fallback=='energy' else 0.)
        if c.method=='identity':request=0.
        elif c.method=='energy':request=energy
        elif c.method=='geometry':request=geo
        elif usable:
            weight=c.geometry_weight # confidence rejects a sample; it does not invent source level
            request=weight*geo+(1-weight)*energy
        else:request=geo
        if not event:request=0.
        request=float(np.clip(request,0,c.max_gain_db))
        if not self.started and c.warm_start:
            self.gain_db=request;self.prev_gain=10**(request/20)
        self.started=True
        delta=(1-math.exp(-self.dt/c.smooth_s))*(request-self.gain_db)
        self.gain_db+=float(np.clip(delta,-c.max_slew_db_s*self.dt,c.max_slew_db_s*self.dt))
        end=10**(self.gain_db/20);gain=np.linspace(self.prev_gain,end,self.n)
        unguarded=x*gain;peak=float(max(abs(unguarded)))
        guard=min(1.,c.peak_ceiling/max(peak,1e-30))
        gain*=guard;y=x*gain;self.prev_gain=end
        log=dict(time_s=self.t,event_active=int(event),event_closed=int(closed),pose_usable=int(usable),
                 pose_status=reason,pose_confidence=self.confidence,observed_distance_m=distance,
                 energy_gain_db=energy,geometry_gain_db=geo,requested_gain_db=request,
                 applied_gain_db=float(20*np.log10(max(np.mean(gain),1e-12))),peak_guard=guard,
                 limiter_active=int(guard<.999999))
        self.t+=self.dt
        return y,gain,log


def process_audio(audio,config=None,*,telemetry=None,source_position=(0.,0.,0.),event_active=None):
    e=AmplitudePipeline(config);x=np.asarray(audio,float)
    if x.ndim!=1 or not np.isfinite(x).all():raise ValueError('Supply finite mono audio')
    count=(len(x)+e.n-1)//e.n
    if telemetry is not None and len(telemetry)!=count:raise ValueError('One observed telemetry item per block is required')
    ys=[];gs=[];logs=[]
    for i in range(count):
        block=x[i*e.n:(i+1)*e.n];length=len(block)
        if length<e.n:block=np.pad(block,(0,e.n-length))
        kwargs=telemetry[i] if telemetry is not None else {}
        ev=event_active if event_active is None or np.asarray(event_active).ndim==0 else event_active[i]
        y,g,log=e.process_block(block,source_position=source_position,event_active=ev,**kwargs)
        ys.append(y[:length]);gs.append(g[:length]);logs.append(log)
    return (np.concatenate(ys),np.concatenate(gs),logs) if ys else (np.array([]),np.array([]),[])
