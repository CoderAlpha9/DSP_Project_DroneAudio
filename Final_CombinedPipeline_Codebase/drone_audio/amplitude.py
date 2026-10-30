"""Selected pose-guided gain law, adapted to explicit source sessions.

Inherited: 18 dB ceiling, 40 ms pole, 36 dB/s slew, 3.5 s hold, .95 peak.
New: speech/source association and optional inference-estimated SNR guard.
"""
import math
import numpy as np


class AmplitudeRestorer:
    def __init__(self, config):
        self.config=config;self.c=config.amplitude;self.reset()

    def reset(self):
        self.closest_range=None;self.last_request=0.;self.last_valid_s=None
        self.last_pose=None;self.last_pose_s=None;self.gain_db=0.;self.previous_gain=1.

    def new_source_session(self, source_position_enu_m=None, source_associated=None):
        self.reset()
        if source_position_enu_m is not None:self.c.source_position_enu_m=tuple(source_position_enu_m)
        if source_associated is not None:self.c.source_associated=bool(source_associated)
        self.c.validate()

    def process_block(self,x,*,time_s,snapshot,activity_active,event_active,speech_active,snr_estimate_db):
        x=np.asarray(x,float);c=self.c;dt=len(x)/16000
        if x.ndim!=1 or not len(x) or not np.isfinite(x).all(): raise ValueError("Finite nonempty mono block required")
        reason="valid";pose=snapshot["pose"];distance=None
        associated=c.source_associated and c.source_position_enu_m is not None
        allowed=associated and ((speech_active if c.speech_only else event_active) if c.event_control else True)
        usable=pose is not None and associated
        if not associated:reason="source_not_associated_or_geometry_unknown"
        elif pose is None:reason="pose_"+snapshot["pose_status"]
        if usable:
            distance=float(np.linalg.norm(pose-np.asarray(c.source_position_enu_m)))
            if distance<1.5:usable=False;reason="range_floor"
            if self.last_pose is not None:
                elapsed=time_s-self.last_pose_s
                if 0<elapsed<.5 and np.linalg.norm(pose-self.last_pose)/elapsed>c.max_pose_speed_m_s:
                    usable=False;reason="pose_jump"
            imu=snapshot["imu"]
            if imu is not None and not 3<np.linalg.norm(imu)<22:usable=False;reason="imu_outlier"
            # Only accepted poses become the plausibility anchor.
            if usable:self.last_pose=pose.copy();self.last_pose_s=time_s
        geometry=0.
        if usable and allowed:
            self.closest_range=distance if self.closest_range is None else min(self.closest_range,distance)
            reference=c.reference_distance_m if c.reference_mode=="fixed" else self.closest_range
            geometry=max(0.,20*math.log10(distance/reference))
            self.last_request=geometry;self.last_valid_s=time_s
        elif allowed and self.last_valid_s is not None and time_s-self.last_valid_s<=c.hold_s:
            geometry=self.last_request;reason="bounded_hold:"+reason
        elif not allowed:reason="source_control_inactive" if associated else reason
        else:reason="hold_expired:"+reason
        # Closing an acoustic event NEVER erases the source anchor.
        guard=1.
        if c.snr_guard_enabled:
            guard=float(np.clip((snr_estimate_db-c.snr_guard_low_db)/(c.snr_guard_high_db-c.snr_guard_low_db),0,1))
        request=float(np.clip(geometry,0,c.max_gain_db))*guard if allowed and c.enabled else 0.
        if not c.enabled:
            self.gain_db=0.;self.previous_gain=1.
            return x.copy(),np.ones(len(x)),{"time_s":time_s,"geometry_gain_db":geometry,"requested_gain_db":0.,"applied_gain_db":0.,"peak_guard":1.,"limiter_active":0,"pose_status":"amplitude_bypassed","snr_guard":guard,"closest_range_m":self.closest_range}
        delta=(1-math.exp(-dt/c.smooth_s))*(request-self.gain_db)
        self.gain_db+=float(np.clip(delta,-c.max_slew_db_s*dt,c.max_slew_db_s*dt))
        end=10**(self.gain_db/20);gain=np.linspace(self.previous_gain,end,len(x));self.previous_gain=end
        peak=float(np.max(np.abs(x*gain)));peak_guard=min(1.,c.peak_ceiling/max(peak,1e-30))
        gain*=peak_guard;y=x*gain
        log={"time_s":time_s,"geometry_gain_db":geometry,"requested_gain_db":request,
             "applied_gain_db":float(20*np.log10(max(gain.mean(),1e-12))),"peak_guard":peak_guard,
             "limiter_active":int(peak_guard<.999999),"pose_status":reason,"pose_usable":int(usable),
             "observed_distance_m":distance,"closest_range_m":self.closest_range,"snr_guard":guard,
             "source_control_active":int(allowed),"activity_active":int(activity_active),"event_active":int(event_active),"speech_active":int(speech_active)}
        return y,gain,log
