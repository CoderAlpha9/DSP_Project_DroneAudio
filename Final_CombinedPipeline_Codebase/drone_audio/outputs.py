from pathlib import Path
import csv
import json
import platform
import sys
import importlib.metadata
from datetime import datetime, timezone
import numpy as np
from .audio import write_audio


def clean_json(value):
    if isinstance(value,dict):return {str(k):clean_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean_json(v) for v in value]
    if isinstance(value,np.ndarray):return clean_json(value.tolist())
    if isinstance(value,(np.floating,float)):return float(value) if np.isfinite(value) else None
    if isinstance(value,np.integer):return int(value)
    if isinstance(value,np.bool_):return bool(value)
    if isinstance(value,Path):return str(value)
    return value


def json_out(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(clean_json(value),indent=2,allow_nan=False)+"\n",encoding="utf-8")


def csv_out(path,rows):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);rows=list(rows)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(clean_json(rows))


def environment():
    versions={}
    for name in ("numpy","scipy","PyYAML","matplotlib","pytest","pystoi"):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]="unavailable"
    return {"utc":datetime.now(timezone.utc).isoformat(),"python":sys.version,"platform":platform.platform(),"versions":versions}


def event_intervals(logs,length,*,pre_roll_s=.1,post_roll_s=.1):
    if pre_roll_s<0 or post_roll_s<0:raise ValueError("Event padding must be nonnegative")
    events=[];current=None
    for row in logs:
        if row["event_active"] and current is None:
            current={"event_id":row["event_id"],"decision_available_s":row["available_time_s"],"start_s":row["time_start_s"]}
        if current is not None and not row["event_active"]:
            current["end_s"]=min(length/16000,row["time_start_s"]);events.append(current);current=None
    if current is not None:current["end_s"]=length/16000;events.append(current)
    for e in events:
        e["clip_start_sample"]=max(0,int((e["start_s"]-pre_roll_s)*16000))
        e["clip_end_sample"]=min(length,int((e["end_s"]+post_roll_s)*16000))
    return events


def export_run(path,pipeline,*,microphone=None,clips=False,pre_roll_s=.1,post_roll_s=.1,record=None):
    p=Path(path);p.mkdir(parents=True,exist_ok=True)
    write_audio(p/"final.wav",pipeline.audio);write_audio(p/"enhanced.wav",pipeline.enhanced)
    csv_out(p/"detector.csv",pipeline.logs_detection);csv_out(p/"enhancement.csv",pipeline.enhancer.traces);csv_out(p/"amplitude.csv",pipeline.logs_amplitude)
    events=event_intervals(pipeline.logs_detection,len(pipeline.audio),pre_roll_s=pre_roll_s,post_roll_s=post_roll_s)
    json_out(p/"events.json",events)
    if clips:
        # Explicit file buffer supplies historical samples. Does not change online availability.
        if microphone is None:raise ValueError("Event export requires an explicit microphone buffer")
        for e in events:write_audio(p/"clips"/f"event_{e['event_id']:04d}.wav",microphone[e["clip_start_sample"]:e["clip_end_sample"]])
    json_out(p/"execution.json",{"status":"completed","config":pipeline.config.to_dict(),"config_sha256":pipeline.config.sha256,
        "timing":pipeline.timing,"environment":environment(),"input_record":record or {},
        "output_samples":len(pipeline.audio),"audio_format":"float32 WAV, no normalization","events":len(events)})
    return events
