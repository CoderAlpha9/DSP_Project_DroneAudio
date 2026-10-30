from copy import deepcopy
import builtins
import io
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
from scipy.io import wavfile
from drone_audio import DronePipeline, PipelineConfig, process_audio
from drone_audio.amplitude import AmplitudeRestorer
from drone_audio.audio import read_audio
from drone_audio.datasets import discover, InferenceInput, split_audit, Sample, EvaluationInput
from drone_audio.detection import Detector, extract
from drone_audio.enhancement import harmonic_gain, relative_transport, pole_time_constant
from drone_audio.evaluation import bind_input
from drone_audio.telemetry import ObservedTelemetry, SensorSeries, ReliabilityMonitor, read_observed_csv

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT.parent/"Data"


@pytest.mark.parametrize("n",[0,1,17,255,256,319,320,1023,1024,17329])
def test_identity_length_flush(n):
    c=PipelineConfig();c.enhancement.enabled=False;c.amplitude.enabled=False
    x=np.random.default_rng(2).normal(0,.03,n)
    p=process_audio(x,c,chunk_size=117)
    np.testing.assert_allclose(p.audio,x,atol=2e-14);np.testing.assert_array_equal(p.audio,p.enhanced)
    assert len(p.flush())==0
    with pytest.raises(RuntimeError):p.process_block([])


@pytest.mark.parametrize("chunk",[1,113,160,320,4093,20000])
def test_arbitrary_chunks_match(chunk):
    x=np.random.default_rng(4).normal(0,.02,6537)
    c=PipelineConfig();c.amplitude.enabled=False
    whole=process_audio(x,c,chunk_size=len(x));other=process_audio(x,c,chunk_size=chunk)
    np.testing.assert_array_equal(whole.audio,other.audio)
    assert whole.logs_detection==other.logs_detection


def test_full_reset_and_reference_separation(monkeypatch):
    samples,_=discover(DATA,include_variants=False)
    s=next(s for s in samples if s.batch=="batch1" and s.split=="test")
    original=builtins.open
    def guarded(path,*args,**kwargs):
        name=str(path).lower()
        assert not any(v in name for v in ("reference.wav","target_at_drone","noise.wav","ground_truth","truth.npz","rpm_aligned","vehicle_state","gazebo_rotors")),name
        return original(path,*args,**kwargs)
    monkeypatch.setattr(builtins,"open",guarded)
    monkeypatch.setattr(io,"open",guarded)
    x,_=read_audio(s.inference.microphone);tele=s.inference.load_telemetry();c=bind_input(PipelineConfig(),s.inference)
    p=process_audio(x[:1600],c,telemetry=tele);first=p.audio.copy()
    p.reset(telemetry=tele);p.process_block(x[:1600]);p.flush();np.testing.assert_array_equal(first,p.audio)
    assert not any("reference" in k or "truth" in k for k in vars(s.inference))


def test_detector_matches_supplied_source():
    path=ROOT.parent/"Experiments/EventDetection_Experiments/pipeline/detector.py"
    spec=importlib.util.spec_from_file_location("source_detector",path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    x=np.random.default_rng(9).normal(0,.03,3217)
    np.testing.assert_array_equal(extract(x),m.extract(x))
    model=json.loads((ROOT/"drone_audio/assets/detector_selected.json").read_text())
    a=Detector(model);b=m.Detector(model)
    for f in extract(x):assert a.step_features(f)==b.step_features(f)


def test_detector_distinct_flags_and_no_reference_reset():
    d=Detector();rows=[d.process_hop(np.zeros(160)) for _ in range(20)]
    assert all(set(("activity_active","event_active","speech_active","activity_score","speech_score")).issubset(r) for r in rows)
    assert all(r["reset_reference"]==0 for r in rows)


def test_no_future_sensor_interpolation_and_stale():
    s=SensorSeries([.5,1.],[[1000],[2000]],units="RPM")
    assert s.at(.49,.2)[1]=="leading_gap"
    assert s.at(.55,.2)[0][0]==1000
    assert s.at(.9,.2)[1]=="stale"
    assert s.at(1.,.2)[0][0]==2000


@pytest.mark.parametrize("times",[[0.,0.],[1.,0.],[0.,float('nan')]])
def test_malformed_timestamps_rejected(times):
    with pytest.raises(ValueError):SensorSeries(times,[[1.],[1.]],units="RPM")


def test_units_coordinate_and_csv_validation():
    with pytest.raises(ValueError):ObservedTelemetry(rpm=SensorSeries([0],[[10]],units="rad/s"))
    with pytest.raises(ValueError):ObservedTelemetry(pose=SensorSeries([0],[[0,0,2]],units="m",frame="NED"))


def test_telemetry_freeze_stop_and_invalid():
    c=PipelineConfig();m=ReliabilityMonitor(c)
    snapshot={"rpm":np.array([3000.,3100.]),"rpm_status":"valid"}
    assert m.inspect(snapshot,0.)[1]=="valid"
    assert "freeze" in m.inspect(snapshot,2.1)[1]
    snapshot["rpm"]=np.array([0.,3000.]);assert "stopped" in m.inspect(snapshot,2.2)[1]
    snapshot["rpm"]=None;snapshot["rpm_status"]="stale";assert m.inspect(snapshot,3.)[0] is None
    tele=ObservedTelemetry(rpm=SensorSeries([0.],[np.array([np.nan,3000.])],units="RPM"))
    assert tele.at(0.)["rpm_status"]=="invalid_observation"


def test_rpm_constant_scale_invariance_and_invalid_calibration():
    n=np.exp(np.random.default_rng(5).normal(size=513));r=np.array([3000.,4000.,3500.,4100.]);ref=np.array([2900.,4100.,3400.,4000.]);scale=np.array([.7,1.2,1.5,.9])
    np.testing.assert_allclose(relative_transport(n,r,ref),relative_transport(n,r*scale,ref*scale),rtol=1e-12)
    with pytest.raises(ValueError):relative_transport(n,r,ref*0)
    drift=relative_transport(n,r*.8,ref)
    assert np.max(abs(drift-relative_transport(n,r,ref)))>1e-3


def test_blade_passing_and_nyquist():
    f=np.arange(0.,8001.)
    g=harmonic_gain(f,[3000.,0.,np.nan,1e8],blades=2,harmonics=1,width_hz=1,attenuation=.15)
    assert g[100]==.15 and g[50]==1 and np.isfinite(g).all()


def test_missing_rpm_is_exact_audio_fallback():
    x=np.random.default_rng(1).normal(0,.02,5000);a=PipelineConfig();a.amplitude.enabled=False
    b=deepcopy(a);b.enhancement.mode="ridge"
    np.testing.assert_array_equal(process_audio(x,a).audio,process_audio(x,b).audio)


def test_declared_calibration_and_causal_rpm_chunk_equivalence():
    x=np.random.default_rng(6).normal(0,.03,24013)
    ts=np.arange(0,2,.02);values=np.stack([3500+100*np.sin(ts+j) for j in range(4)],axis=1)
    tele=ObservedTelemetry(rpm=SensorSeries(ts,values,units="RPM"))
    c=PipelineConfig();c.speech_free_prefix_s=.8;c.amplitude.enabled=False;c.enhancement.mode="bounded_rpm";c.enhancement.rpm_weight=.75
    a=process_audio(x,c,telemetry=tele,chunk_size=len(x));b=process_audio(x,c,telemetry=tele,chunk_size=71)
    np.testing.assert_array_equal(a.audio,b.audio)
    assert max(r["fusion_weight"] for r in a.enhancer.traces)==.75


def snapshot(pose):
    return {"pose":None if pose is None else np.asarray(pose,float),"pose_status":"missing" if pose is None else "valid","imu":None}


def amplitude_step(a,pose,t,active=True,x=None):
    return a.process_block(np.full(320,.01) if x is None else x,time_s=t,snapshot=snapshot(pose),activity_active=active,event_active=active,speech_active=active,snr_estimate_db=30.)


def test_amplitude_range_law_anchor_retention_and_session_reset():
    c=PipelineConfig();c.amplitude.source_position_enu_m=(0,0,0);c.amplitude.source_associated=True;c.amplitude.snr_guard_enabled=False
    a=AmplitudeRestorer(c);amplitude_step(a,[0,0,2],.02)
    for t in np.arange(.04,1.02,.02):amplitude_step(a,[0,0,2+min(t,.5)*4],t)
    assert a.closest_range==2
    _,g,log=amplitude_step(a,[0,0,4],1.04)
    assert log['geometry_gain_db']==pytest.approx(20*np.log10(2))
    assert np.mean(g)==pytest.approx(2,abs=.03)
    amplitude_step(a,None,1.06,active=False);assert a.closest_range==2
    _,_,log=amplitude_step(a,None,1.08);assert "bounded_hold" in log["pose_status"]
    _,_,log=amplitude_step(a,None,5.);assert "hold_expired" in log["pose_status"]
    a.new_source_session();assert a.closest_range is None


def test_unknown_source_bypass_gain_bounds_and_peak_guard():
    c=PipelineConfig();a=AmplitudeRestorer(c)
    _,g,log=amplitude_step(a,[0,0,30],.02);np.testing.assert_array_equal(g,np.ones(320))
    c.amplitude.source_associated=True;c.amplitude.source_position_enu_m=(0,0,0);c.amplitude.reference_mode="fixed";c.amplitude.reference_distance_m=2.;c.amplitude.snr_guard_enabled=False
    a=AmplitudeRestorer(c);previous=0.
    for t in np.arange(.02,2,.02):
        y,g,log=amplitude_step(a,[0,0,100],t,x=np.ones(320)*.8)
        assert a.gain_db-previous<=.7200001;previous=a.gain_db
        assert max(abs(y))<=.95000000001 and max(g)<=10**(18/20)
    assert log['limiter_active']==1


def test_snr_guard_and_unassociated_generic_event():
    c=PipelineConfig();c.amplitude.source_associated=True;c.amplitude.source_position_enu_m=(0,0,0);c.amplitude.reference_mode="fixed";c.amplitude.reference_distance_m=2
    a=AmplitudeRestorer(c)
    _,g,log=a.process_block(np.ones(320)*.01,time_s=.02,snapshot=snapshot([0,0,20]),activity_active=True,event_active=True,speech_active=False,snr_estimate_db=30)
    assert log['requested_gain_db']==0 and np.max(g)==1
    _,g,log=a.process_block(np.ones(320)*.01,time_s=.04,snapshot=snapshot([0,0,20]),activity_active=True,event_active=True,speech_active=True,snr_estimate_db=-5)
    assert log['requested_gain_db']==0 and log['snr_guard']==0


def test_explicit_rate_channels_and_invalid_audio(tmp_path):
    # All generated test artifacts remain in the combined codebase via --basetemp.
    p=tmp_path/'stereo.wav';wavfile.write(p,8000,np.zeros((40,2),np.int16))
    with pytest.raises(ValueError):read_audio(p)
    x,info=read_audio(p,allow_resample=True,downmix=True);assert len(x)==80 and info['resampled'] and info['downmixed']
    for x in [[float('nan')],[[1.,2.]]]:
        with pytest.raises(ValueError):process_audio(x)
    c=PipelineConfig();c.amplitude.reference_mode='fixed'
    with pytest.raises(ValueError):c.validate()
    for alpha in [-.1,1.,float('nan')]:
        with pytest.raises(ValueError):pole_time_constant(alpha)
    assert pole_time_constant(0)==0


def test_dataset_splits_sources_and_manifest_selection():
    samples,audit=discover(DATA)
    assert sum(s.batch=='batch1' and s.split=='test' for s in samples)==15
    assert len({s.source_id for s in samples if s.batch=='batch1' and s.split=='test'})==1
    assert sum(s.batch=='batch2' for s in samples)==76
    assert not audit['conflicting_parent_splits'] and not audit['conflicting_capture_splits']
    assert audit['overlapping_source_splits']  # Explicitly disclosed shared Batch2 source.
    assert all(s.inference.observed_csv is None or 'rpm_sensor_observed' in s.inference.observed_csv.name for s in samples if s.batch=='batch2')
    for s in samples:
        assert s.inference.microphone.exists() and s.evaluation.target_at_microphone.exists()
        if s.batch=='batch2':assert s.inference.source_position_enu_m is None
    duplicate=deepcopy(samples[0]);duplicate.split='test'
    assert split_audit([samples[0],duplicate])['conflicting_parent_splits']


def test_event_probe_adapter():
    samples,_=discover(DATA,include_event_probes=True)
    probes=[s for s in samples if s.batch=='event_probe' and s.split=='test']
    assert len(probes)==60
    assert all(s.inference.microphone.exists() and s.evaluation.event_labels.exists() and not s.inference.source_associated for s in probes)


def test_pose_control_chunk_equivalence_and_no_sensor_queries_after_audio_end():
    samples,_=discover(DATA,include_variants=False)
    s=next(s for s in samples if s.batch=='batch1' and s.split=='test' and s.metadata['profile']=='recede_slow')
    c=bind_input(PipelineConfig(),s.inference);x,_=read_audio(s.inference.microphone);x=x[:3237]
    t=s.inference.load_telemetry();calls=[];original=t.at
    def tracked(time_s,max_age_s=.2):calls.append(time_s);return original(time_s,max_age_s)
    t.at=tracked
    a=process_audio(x,c,telemetry=t,chunk_size=len(x));b=process_audio(x,c,telemetry=t,chunk_size=173)
    np.testing.assert_array_equal(a.audio,b.audio);assert a.logs_amplitude==b.logs_amplitude
    assert max(calls)<=len(x)/16000+1e-12


def test_new_source_reset_keeps_noise_model():
    p=DronePipeline();p.process_block(np.ones(2048)*.01)
    assert p.enhancer.noise is not None and not len(p.pending_amplitude)
    model=p.enhancer
    p.new_source_session(source_position_enu_m=(0,0,0),source_associated=True)
    assert p.enhancer is model


def test_python_and_cli_defaults_match_frozen_selection():
    from drone_audio.config import load_config
    assert PipelineConfig().sha256==load_config(ROOT/'configs/frozen.yaml').sha256


def test_full_reset_drops_previous_recording_sensors():
    t=ObservedTelemetry(rpm=SensorSeries([0.],[[3000.]],units='RPM'))
    p=DronePipeline(telemetry=t);assert p.telemetry.at(0.)['rpm'] is not None
    p.process_block(np.zeros(320));p.reset()
    assert p.telemetry.at(0.)['rpm'] is None and p.enhancer.noise is None and p.received==0


def test_parent_summary_preserves_metrics_after_silent_first_case():
    from drone_audio.evaluation import grouped_results
    rows=[{'batch':'event_probe','method':'complete','parent_id':'a','pre_si_sdr_db':None},
          {'batch':'event_probe','method':'complete','parent_id':'a','pre_si_sdr_db':5.,'pre_stoi':.6},
          {'batch':'event_probe','method':'complete','parent_id':'b','pre_si_sdr_db':3.,'pre_stoi':.4}]
    parents,summary,_=grouped_results(rows)
    assert len(parents)==2 and summary[0]['parents']==2
    assert summary[0]['pre_si_sdr_db']==4. and summary[0]['pre_stoi']==.5


def test_oracle_explicitly_separate_and_finite():
    from drone_audio.oracle import known_noise_wiener
    rng=np.random.default_rng(4);noise=rng.normal(0,.01,700);mixture=noise+np.sin(np.arange(700)*.1)*.03
    out=known_noise_wiener(mixture,noise);assert len(out)==700 and np.isfinite(out).all()
