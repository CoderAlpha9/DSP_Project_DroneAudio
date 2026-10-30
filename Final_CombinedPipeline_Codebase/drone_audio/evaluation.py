"""Evaluation references are opened here, strictly after inference finishes."""
from copy import deepcopy
from pathlib import Path
import csv
import hashlib
import json
import time
import numpy as np
from scipy import signal
from .audio import read_audio
from .config import PipelineConfig, save_config
from .datasets import discover, hash_file
from .orchestration import process_audio
from .outputs import csv_out, json_out, export_run, environment


def bind_input(config, inference):
    c=deepcopy(config)
    if c.amplitude.source_position_enu_m is None:
        c.amplitude.source_position_enu_m=inference.source_position_enu_m
        c.amplitude.source_associated=inference.source_associated
    if c.speech_free_prefix_s is None:c.speech_free_prefix_s=inference.speech_free_prefix_s
    return c.validate()


def si_sdr(reference, estimate):
    a=np.asarray(reference,float).copy();b=np.asarray(estimate,float).copy()
    if len(a)!=len(b):raise ValueError("Metric inputs must align")
    if not len(a) or np.sum(a*a)<1e-18:return float("nan")
    a-=a.mean();b-=b.mean();p=(a@b)/(a@a+1e-18)*a
    return float(10*np.log10((p@p+1e-18)/((b-p)@(b-p)+1e-18)))


def reconstruction_snr(a,b):
    if not len(a) or np.sum(a*a)<1e-18:return float("nan")
    return float(10*np.log10((np.sum(a*a)+1e-18)/(np.sum((a-b)**2)+1e-18)))


def quality(reference, estimate, mask=None, *, intelligibility=True):
    a=np.asarray(reference,float);b=np.asarray(estimate,float)
    if len(a)!=len(b):raise ValueError("Metric inputs must have identical length")
    active=np.ones(len(a),bool) if mask is None else np.asarray(mask,bool)
    d={"si_sdr_db":si_sdr(a[active],b[active]),"reconstruction_snr_db":reconstruction_snr(a[active],b[active]),
       "rmse":float(np.sqrt(np.mean((a-b)**2))) if len(a) else 0.}
    if len(a)>=512:
        A=abs(signal.stft(a,16000,nperseg=512,noverlap=384)[2]);B=abs(signal.stft(b,16000,nperseg=512,noverlap=384)[2])
        use=A>max(A.max()*1e-3,1e-10)
        d["log_spectral_distance_db"]=float(np.sqrt(np.mean((20*np.log10(np.maximum(A[use],1e-10))-20*np.log10(np.maximum(B[use],1e-10)))**2))) if np.any(use) else float("nan")
    n=len(a)//320
    if n:
        ar=np.sqrt(np.mean(a[:n*320].reshape(n,320)**2,axis=1));br=np.sqrt(np.mean(b[:n*320].reshape(n,320)**2,axis=1))
        use=ar>max(.1*np.percentile(ar,95),1e-6)
        d["active_level_mae_db"]=float(np.mean(abs(20*np.log10((br[use]+1e-12)/(ar[use]+1e-12))))) if use.any() else float("nan")
    d["stoi"]=d["estoi"]=None;d["intelligibility_status"]="not_requested"
    if intelligibility and len(a)>=1600 and np.sum(a*a)>1e-15:
        try:
            from pystoi import stoi
            d["stoi"]=float(stoi(a,b,16000,extended=False));d["estoi"]=float(stoi(a,b,16000,extended=True));d["intelligibility_status"]="executed"
        except ImportError:d["intelligibility_status"]="unavailable:pystoi"
        except (ValueError,RuntimeError) as e:d["intelligibility_status"]="unavailable:"+str(e)
    elif intelligibility:d["intelligibility_status"]="not_applicable:short_or_silent_reference"
    return d


def _intervals(flags):
    d=np.diff(np.r_[False,np.asarray(flags,bool),False].astype(int))
    return list(zip(np.flatnonzero(d==1)*.01,np.flatnonzero(d==-1)*.01))


def classification(truth,predicted,valid):
    y=np.asarray(truth,bool)[valid];p=np.asarray(predicted,bool)[valid]
    tp=int(np.sum(y&p));fp=int(np.sum(~y&p));fn=int(np.sum(y&~p));tn=int(np.sum(~y&~p))
    precision=tp/(tp+fp) if tp+fp else float(tp+fn==0);recall=tp/(tp+fn) if tp+fn else 1.
    return {"tp":tp,"fp":fp,"fn":fn,"tn":tn,"precision":precision,"recall":recall,
            "f1":2*precision*recall/(precision+recall) if precision+recall else 0.,
            "false_alarm_rate":fp/max(1,fp+tn),"miss_rate":fn/max(1,tp+fn)}


def event_metrics(sample,logs):
    count=len(logs)
    if sample.batch=="event_probe":
        with np.load(sample.evaluation.event_labels) as truth:
            activity=truth["activity"].astype(bool);speech=truth["speech"].astype(bool);valid=truth["valid"].astype(bool)
        definition="packaged synthetic support, 80 ms collars"
    elif sample.batch=="batch1":
        annotations=json.loads((Path(__file__).parent/"assets/source_intervals.json").read_text())
        t=(np.arange(count)+.5)*.01+sample.evaluation.source_offset_samples/16000
        activity=np.zeros(count,bool);valid=np.ones(count,bool)
        for a,b in annotations["sources"][sample.source_id]:
            activity|=(t>=a)&(t<b);valid&=(abs(t-a)>.08)&(abs(t-b)>.08)
        speech=activity.copy();definition="approximate visual source intervals, 80 ms collars"
    else:return {"event_scoring_status":"unavailable:no compatible event annotation"}
    n=min(count,len(activity));valid=valid[:n];result={"event_scoring_status":definition}
    for name,truth,key in [("activity",activity,"activity_active"),("speech",speech,"speech_active")]:
        result.update({name+"_"+k:v for k,v in classification(truth[:n],[r[key] for r in logs[:n]],valid).items()})
    predicted=_intervals([r["event_active"] for r in logs[:n]]);truth_events=_intervals(activity[:n])
    pairs=[]
    for i,(a,b) in enumerate(truth_events):
        for j,(c,d) in enumerate(predicted):
            overlap=max(0.,min(b,d)-max(a,c))
            if overlap>0:pairs.append((overlap/max(b-a,d-c,1e-12),i,j))
    used_truth=set();used_pred=set();errors=[]
    for _,i,j in sorted(pairs,reverse=True):
        if i in used_truth or j in used_pred:continue
        used_truth.add(i);used_pred.add(j)
        errors.append((abs(predicted[j][0]-truth_events[i][0]),abs(predicted[j][1]-truth_events[i][1])))
    result.update({"matched_events":len(errors),"unmatched_reference_events":len(truth_events)-len(errors),
                   "unmatched_predicted_events":len(predicted)-len(errors),
                   "matched_onset_mae_s":float(np.mean([e[0] for e in errors])) if errors else None,
                   "matched_offset_mae_s":float(np.mean([e[1] for e in errors])) if errors else None})
    return result


def load_target(sample):
    if sample.batch=="event_probe":
        with np.load(sample.evaluation.target_at_microphone) as truth:return truth["foreground"].astype(float)
    return read_audio(sample.evaluation.target_at_microphone)[0]


def amplitude_goal(sample,target,config):
    e=sample.evaluation
    if sample.batch!="batch1":return target,"microphone_target_no_range_calibration"
    if config.amplitude.reference_mode=="fixed":
        # Reference WAV is valid at declared generator distance, not arbitrary requested distance.
        reference=read_audio(e.source_reference)[0]
        return reference*e.fixed_reference_m/config.amplitude.reference_distance_m,"fixed_declared_calibration"
    trajectory=np.genfromtxt(e.true_trajectory,delimiter=",",names=True)
    d=np.interp(np.arange(len(target))/16000,trajectory["time_s"],trajectory["distance_m"])
    return target*np.maximum(1.,d/np.minimum.accumulate(d)),"true_closest_so_far_session_evaluation_only"


def score(sample,pipeline,config,*,intelligibility=True):
    target=load_target(sample);n=len(pipeline.audio)
    if len(target)!=n:raise ValueError(f"Reference length mismatch in {sample.id}")
    mask=np.ones(n,bool)
    if sample.evaluation.active_interval_s:
        a,b=sample.evaluation.active_interval_s;t=np.arange(n)/16000;mask=(t>=a)&(t<b)
    pre=quality(target,pipeline.enhanced,mask,intelligibility=intelligibility)
    goal,convention=amplitude_goal(sample,target,config)
    post=quality(goal,pipeline.audio,mask,intelligibility=False)
    weights=[r["fusion_weight"] for r in pipeline.enhancer.traces if r["frame_start_s"]>=0 and r["available_time_s"]<=n/16000]
    reasons=[r["rpm_status"] for r in pipeline.enhancer.traces if r["available_time_s"]<=n/16000]
    row={"sample_id":sample.id,"parent_id":sample.parent_id,"source_id":sample.source_id,"batch":sample.batch,"split":sample.split,"condition":sample.condition,
         "config_sha256":config.sha256,"samples":n,"amplitude_reference":convention,"peak_abs":float(np.max(abs(pipeline.audio))) if n else 0.,
         "clipping_fraction":float(np.mean(abs(pipeline.audio)>=1.)) if n else 0.,"mean_rpm_weight":float(np.mean(weights)) if weights else 0.,
         "rpm_valid_coverage":reasons.count("valid")/max(1,len(reasons)),"mean_applied_gain_db":float(np.mean([r["applied_gain_db"] for r in pipeline.logs_amplitude])) if pipeline.logs_amplitude else 0.,
         "limiter_fraction":float(np.mean([r["limiter_active"] for r in pipeline.logs_amplitude])) if pipeline.logs_amplitude else 0.,
         "component_snr_status":"not_claimed: no replay of adaptive operator on stems",
         **{k:v for k,v in pipeline.timing.items() if isinstance(v,(int,float))}}
    row.update({"pre_"+k:v for k,v in pre.items()});row.update({"post_"+k:v for k,v in post.items()})
    row.update(event_metrics(sample,pipeline.logs_detection))
    return row


def matched_configurations(config):
    def clone():return deepcopy(config)
    methods={}
    c=clone();c.enhancement.enabled=False;c.amplitude.enabled=False;methods["unprocessed"]=c
    c=clone();c.enhancement.mode="audio_only";c.amplitude.enabled=False;methods["audio_only"]=c
    c=clone();c.enhancement.mode="fixed_rpm";c.enhancement.rpm_weight=.1;c.amplitude.enabled=False;methods["fixed_rpm_0p1"]=c
    c=clone();c.enhancement.mode="ridge";c.amplitude.enabled=False;methods["reliability_ridge"]=c
    c=clone();c.amplitude.enabled=False;methods["complete_no_amplitude"]=c
    methods["complete"]=clone()
    c=clone();c.amplitude.event_control=False;methods["no_event_control"]=c
    c=clone();c.enhancement.motion_enabled=True;methods["motion_ablation"]=c
    return methods


def grouped_results(rows):
    """Conditions -> parent means -> macro parent means, without frame CIs."""
    keys=[k for k in rows[0] if isinstance(rows[0][k],(float,int)) and k not in {"samples"}] if rows else []
    groups={}
    for r in rows:groups.setdefault((r["batch"],r["method"],r["parent_id"]),[]).append(r)
    parents=[]
    for (batch,method,parent),items in groups.items():
        row={"batch":batch,"method":method,"parent_id":parent,"conditions":len(items)}
        for k in keys:
            vals=[r[k] for r in items if isinstance(r.get(k),(float,int)) and np.isfinite(r[k])]
            row[k]=float(np.mean(vals)) if vals else None
        parents.append(row)
    macro=[]
    for batch,method in sorted({(r["batch"],r["method"]) for r in parents}):
        items=[r for r in parents if r["batch"]==batch and r["method"]==method]
        row={"batch":batch,"method":method,"parents":len(items)}
        for k in keys:
            vals=[r[k] for r in items if r.get(k) is not None]
            row[k]=float(np.mean(vals)) if vals else None
        macro.append(row)
    # Matched deltas are computed within parents, never as independent frames.
    paired=[];baseline={(r["batch"],r["parent_id"]):r for r in parents if r["method"]=="audio_only"}
    for r in parents:
        base=baseline.get((r["batch"],r["parent_id"]))
        if base:
            paired.append({"batch":r["batch"],"parent_id":r["parent_id"],"method":r["method"],
                "pre_si_sdr_delta_vs_audio_db":r["pre_si_sdr_db"]-base["pre_si_sdr_db"] if r.get("pre_si_sdr_db") is not None and base.get("pre_si_sdr_db") is not None else None})
    return parents,macro,paired


def evaluate_dataset(data_root,config,output,*,split="test",limit=None,event_probes=False,save_all_audio=False,methods=None,batch=None):
    out=Path(output).resolve()
    if out.exists():raise ValueError("Choose a new run directory; existing evidence is preserved")
    out.mkdir(parents=True)
    samples,audit=discover(data_root,include_event_probes=event_probes)
    if audit["conflicting_capture_splits"]:raise ValueError("Conflicting capture split assignments")
    selected=[s for s in samples if s.split==split]
    if batch:selected=[s for s in selected if s.batch==batch]
    if limit:selected=selected[:limit]
    variants=matched_configurations(config)
    if methods:variants={k:v for k,v in variants.items() if k in methods}
    if not selected or not variants:raise ValueError("No selected cases/methods")
    json_out(out/"environment.json",environment());json_out(out/"data_audit.json",audit);save_config(config,out/"frozen.yaml")
    ledger={"status":"running","stage":"combined_evaluation","split":split,"reused_test":True,"cases":len(selected),"methods":list(variants),"completed":[],"failed":[],"skipped":[]}
    json_out(out/"execution.json",ledger);rows=[]
    for i,sample in enumerate(selected):
        io=time.perf_counter();x,conversion=read_audio(sample.inference.microphone);telemetry=sample.inference.load_telemetry();io=time.perf_counter()-io
        cache={}
        for method,base in variants.items():
            cfg=bind_input(base,sample.inference)
            try:
                reused=cfg.sha256 in cache
                if reused:p=cache[cfg.sha256]
                else:p=process_audio(x,cfg,telemetry=telemetry);cache[cfg.sha256]=p
                metric_start=time.perf_counter()
                row=score(sample,p,cfg);row.update(method=method,input_io_s=io,evaluation_s=time.perf_counter()-metric_start,inference_cache_reused=int(reused))
                rows.append(row)
                write_start=time.perf_counter()
                if save_all_audio or method=="complete":export_run(out/"cases"/sample.id/method,p,record={"sample":sample.record(),"conversion":conversion})
                row['output_io_s']=time.perf_counter()-write_start
            except Exception as exc:
                ledger["failed"].append({"sample":sample.id,"method":method,"error":str(exc)})
                json_out(out/"execution.json",ledger);raise
        ledger["completed"].append(sample.id);csv_out(out/"per_case.csv",rows);json_out(out/"execution.json",ledger)
        print(f"Evaluated {i+1}/{len(selected)}: {sample.id}",flush=True)
    parents,macro,paired=grouped_results(rows)
    csv_out(out/"per_parent.csv",parents);csv_out(out/"summary.csv",macro);csv_out(out/"paired_vs_audio.csv",paired)
    ledger["status"]="completed";ledger["metric_rows"]=len(rows);json_out(out/"execution.json",ledger)
    return rows


def select_development(data_root,config,output,freeze_path):
    """Small declared comparison; no test sample or test reference is opened."""
    out=Path(output).resolve()
    if out.exists():raise ValueError("Choose a new selection output directory")
    out.mkdir(parents=True)
    samples,audit=discover(data_root,include_variants=False)
    if audit["conflicting_parent_splits"] or audit["conflicting_capture_splits"]:raise ValueError("Split conflict blocks tuning")
    pool=[]
    profiles={"hover","recede_slow","out_and_back","mixed_3d"}
    for s in samples:
        if s.split=="test":continue
        if s.batch=="batch1" and s.metadata["profile"] not in profiles:continue
        pool.append(s)
    candidates=[]
    for alpha,beta in [(.90,1.),(.95,1.),(.98,1.),(.95,1.5),(.98,1.5)]:
        c=deepcopy(config);c.amplitude.enabled=False;c.detector_enabled=False;c.enhancement.mode="audio_only";c.enhancement.noise_alpha=alpha;c.enhancement.beta=beta
        candidates.append((f"audio_a{alpha}_b{beta}",c))
    for mode,weight in [("fixed_rpm",.1),("bounded_rpm",.75),("ridge",.1)]:
        c=deepcopy(config);c.amplitude.enabled=False;c.detector_enabled=False;c.enhancement.mode=mode;c.enhancement.rpm_weight=weight
        candidates.append((mode,c))
    json_out(out/"protocol.json",{"cases":[s.record() for s in pool],"test_opened":False,"candidates":{name:c.to_dict() for name,c in candidates},
        "objective":"Batch-balanced macro parent SI-SDR; development ranks eight, validation chooses among top three. No nonzero-RPM tie-break.","source_overlap":audit["overlapping_source_splits"],
        "limitation":"Batch2 capture splits reuse one speech source; all test sets were previously inspected."})
    rows=[]
    for i,s in enumerate(pool):
        x,_=read_audio(s.inference.microphone);telemetry=s.inference.load_telemetry()
        for name,c in candidates:
            cfg=bind_input(c,s.inference);p=process_audio(x,cfg,telemetry=telemetry)
            # Read reference only after inference; selection set is development/validation.
            target=load_target(s);mask=np.ones(len(target),bool)
            if s.evaluation.active_interval_s:
                t=np.arange(len(target))/16000;a,b=s.evaluation.active_interval_s;mask=(t>=a)&(t<b)
            rows.append({"sample_id":s.id,"parent_id":s.parent_id,"batch":s.batch,"split":s.split,"candidate":name,"si_sdr_db":si_sdr(target[mask],p.enhanced[mask])})
        csv_out(out/"selection_rows.csv",rows);print(f"Selection {i+1}/{len(pool)}: {s.id}",flush=True)
    def objective(name,split):
        batches=[]
        for batch in ("batch1","batch2"):
            vals=[r["si_sdr_db"] for r in rows if r["candidate"]==name and r["split"]==split and r["batch"]==batch]
            if vals:batches.append(np.mean(vals))
        return float(np.mean(batches)) if batches else float("-inf")
    ranked=sorted(candidates,key=lambda nc:objective(nc[0],"development"),reverse=True)
    finalists=ranked[:3];winner=max(finalists,key=lambda nc:objective(nc[0],"validation"))
    frozen=deepcopy(config);frozen.enhancement=winner[1].enhancement;frozen.validate()
    save_config(frozen,freeze_path)
    record={"winner":winner[0],"finalists":[n for n,_ in finalists],"config_sha256":frozen.sha256,"freeze_path":str(Path(freeze_path).resolve()),
        "candidates":[{"name":n,"development_objective":objective(n,"development"),"validation_objective":objective(n,"validation")} for n,_ in candidates],
        "input_hashes":{s.id:hash_file(s.inference.microphone) for s in pool},"reference_access":"development/validation only", "environment":environment()}
    json_out(out/"selection.json",record);json_out(Path(freeze_path).with_suffix(".provenance.json"),record)
    return frozen,record
