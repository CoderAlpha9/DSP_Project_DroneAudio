"""CPU command-line entry points; run directories never overwrite evidence."""
import argparse
from copy import deepcopy
from pathlib import Path
import importlib.util
import json
import time
from .config import load_config, save_config
from .audio import read_audio
from .datasets import discover, InferenceInput, input_overlap_audit
from .telemetry import ObservedTelemetry, read_observed_csv
from .orchestration import process_audio
from .outputs import json_out, csv_out, environment, export_run


def doctor():
    result={"environment":environment(),"dependencies":{n:importlib.util.find_spec(n) is not None for n in ("numpy","scipy","yaml","matplotlib","pytest","pystoi")},
        "inference_requires":["numpy","scipy","PyYAML"],"neural_comparator":"unavailable: optional historical runtime not installed by this package",
        "source_references_required_for_inference":False,"internal_rate_hz":16000}
    print(json.dumps(result,indent=2));return result


def _new_directory(path):
    p=Path(path).expanduser().resolve()
    package=Path(__file__).resolve().parents[1]
    if not p.is_relative_to(package):raise ValueError("Generated outputs must stay inside Final_CombinedPipeline_Codebase")
    if p.exists():raise ValueError(f"Output exists: {p}; choose a new run name")
    return p


def main(argv=None):
    parser=argparse.ArgumentParser(description="Combined drone-audio DSP; no reference signals enter inference")
    sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("doctor")
    d=sub.add_parser("discover");d.add_argument("--data-root",default="../Data");d.add_argument("--output",default="runs/discovery");d.add_argument("--hash-audit",action="store_true");d.add_argument("--event-probes",action="store_true")
    p=sub.add_parser("process");p.add_argument("--input",required=True);p.add_argument("--output",default="runs/single");p.add_argument("--config",default="configs/default.yaml")
    p.add_argument("--telemetry");p.add_argument("--telemetry-schema",choices=["batch1","batch2_rpm"],default="batch1");p.add_argument("--telemetry-time-origin-s",type=float,default=0.)
    p.add_argument("--source-position",type=float,nargs=3);p.add_argument("--source-associated",action="store_true");p.add_argument("--fixed-reference-m",type=float);p.add_argument("--speech-free-prefix-s",type=float)
    p.add_argument("--resample",action="store_true");p.add_argument("--downmix",action="store_true");p.add_argument("--clips",action="store_true");p.add_argument("--pre-roll-s",type=float,default=.1);p.add_argument("--post-roll-s",type=float,default=.1)
    for command in ("batch","evaluate","select"):
        p=sub.add_parser(command);p.add_argument("--data-root",default="../Data");p.add_argument("--config",default="configs/default.yaml");p.add_argument("--output",default="runs/"+command)
        if command=="select":p.add_argument("--freeze",default="configs/frozen.yaml")
        else:
            p.add_argument("--split",choices=["development","validation","test","all"],default="test");p.add_argument("--limit",type=int);p.add_argument("--event-probes",action="store_true")
            p.add_argument("--batch",choices=["batch1","batch2","event_probe"])
        if command=="evaluate":p.add_argument("--methods",nargs="+");p.add_argument("--save-all-audio",action="store_true");p.add_argument("--plots",action="store_true")
    p=sub.add_parser("plots");p.add_argument("--run",required=True)
    p=sub.add_parser("sensor-study");p.add_argument("--data-root",default="../Data");p.add_argument("--config",default="configs/frozen.yaml");p.add_argument("--output",default="runs/sensor_study")
    args=parser.parse_args(argv)
    if args.command=="doctor":doctor();return
    if args.command=="plots":
        from .diagnostics import make_plots
        make_plots(Path(args.run).resolve());return
    if args.command=="discover":
        out=_new_directory(args.output);samples,audit=discover(args.data_root,include_event_probes=args.event_probes)
        if args.hash_audit:audit.update(input_overlap_audit(samples))
        json_out(out/"samples.json",[s.record() for s in samples]);json_out(out/"audit.json",audit)
        counts={}
        for s in samples:counts[s.batch+":"+s.split]=counts.get(s.batch+":"+s.split,0)+1
        print(json.dumps({"samples":len(samples),"counts":counts,"audit":audit},indent=2));return
    config=load_config(args.config);out=_new_directory(args.output)
    if args.command=="process":
        if args.source_position is not None:config.amplitude.source_position_enu_m=tuple(args.source_position)
        if args.source_associated:config.amplitude.source_associated=True
        if args.fixed_reference_m is not None:config.amplitude.reference_mode="fixed";config.amplitude.reference_distance_m=args.fixed_reference_m
        if args.speech_free_prefix_s is not None:config.speech_free_prefix_s=args.speech_free_prefix_s
        config.validate();io=time.perf_counter();x,conversion=read_audio(args.input,allow_resample=args.resample,downmix=args.downmix)
        telemetry=read_observed_csv(args.telemetry,args.telemetry_schema,args.telemetry_time_origin_s) if args.telemetry else ObservedTelemetry()
        io=time.perf_counter()-io;p=process_audio(x,config,telemetry=telemetry)
        export_run(out,p,microphone=x,clips=args.clips,pre_roll_s=args.pre_roll_s,post_roll_s=args.post_roll_s,record={"microphone":str(Path(args.input).resolve()),"conversion":conversion,"file_input_s":io})
        print(json.dumps({"output":str(out),"samples":len(p.audio),"timing":p.timing},indent=2));return
    if args.command=="select":
        freeze=Path(args.freeze).resolve()
        if not freeze.is_relative_to(Path(__file__).resolve().parents[1]):raise ValueError("Frozen config must remain inside package")
        if freeze.exists():raise ValueError("Frozen config already exists; choose a new freeze filename")
        from .evaluation import select_development
        _,record=select_development(args.data_root,config,out,freeze);print(json.dumps(record,indent=2));return
    if args.command=="evaluate":
        if args.split=="all":raise ValueError("Evaluate one explicitly declared split at a time")
        from .evaluation import evaluate_dataset
        rows=evaluate_dataset(args.data_root,config,out,split=args.split,limit=args.limit,event_probes=args.event_probes,save_all_audio=args.save_all_audio,methods=args.methods,batch=args.batch)
        if args.plots:
            from .diagnostics import make_plots
            make_plots(out)
        print(f"Completed {len(rows)} new metric rows in {out}");return
    if args.command=="sensor-study":
        from .diagnostics import sensor_study
        sensor_study(args.data_root,config,out);return
    if args.command=="batch":
        from .evaluation import bind_input
        samples,audit=discover(args.data_root,include_event_probes=args.event_probes)
        samples=[s for s in samples if args.split=="all" or s.split==args.split]
        if args.batch:samples=[s for s in samples if s.batch==args.batch]
        if args.limit:samples=samples[:args.limit]
        json_out(out/"audit.json",audit);completed=[]
        for i,s in enumerate(samples):
            x,conversion=read_audio(s.inference.microphone);c=bind_input(config,s.inference)
            p=process_audio(x,c,telemetry=s.inference.load_telemetry())
            export_run(out/s.id,p,record={"sample_id":s.id,"microphone":str(s.inference.microphone),"conversion":conversion})
            completed.append(s.id);json_out(out/"execution.json",{"status":"running","completed":completed,"config_sha256":config.sha256})
            print(f"Processed {i+1}/{len(samples)}: {s.id}",flush=True)
        json_out(out/"execution.json",{"status":"completed","completed":completed,"config_sha256":config.sha256,"evaluation_references_opened":False})


if __name__=="__main__":main()
