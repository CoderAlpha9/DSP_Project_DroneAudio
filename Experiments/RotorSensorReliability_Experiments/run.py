"""Portable entry point for the experiments in the supplied reliability paper."""
import argparse, datetime, hashlib, importlib.util, json, os, platform, shutil, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('MPLBACKEND','Agg')

def execute(script,*args,env=None):
    subprocess.run([sys.executable,str(ROOT/script),*map(str,args)],cwd=ROOT,env=env or os.environ.copy(),check=True)

def doctor():
    checks={n:importlib.util.find_spec(n) is not None for n in ['numpy','scipy','pandas','matplotlib','pystoi','pesq']}
    p=ROOT/'original/data'; captures=list(p.glob('S*/*/manifest.json'))
    result={'python':sys.version,'platform':platform.platform(),'dependencies':checks,'raw_manifest_count':len(captures),'fresh_study_ready':all(checks[n] for n in ['numpy','scipy','pandas']),'original_benchmark_ready':False,'missing_raw_archives':['S3(1).zip','S6(1).zip','data.zip'] if not captures else []}
    if captures:
        sys.path.insert(0,str(ROOT/'original/code'));import experiment as m
        _,selected=m.inventory();result['selected_captures']=len(selected)
        result['original_benchmark_ready']=len(selected)==19 and all(checks.values())
    binary=ROOT/'tools/deep-filter';result['neural_binary_present']=binary.is_file()
    print(json.dumps(result,indent=2));return result

def run_suite(suite,output,retune):
    output=Path(output).resolve()
    if output.exists():raise ValueError('Output already exists; choose a new --output directory to preserve previous evidence')
    output.mkdir(parents=True)
    env=os.environ.copy(); env['RPM_RETUNE']='1' if retune else '0'
    if suite=='fresh':
        env['RPM_FRESH_OUTPUT']=str(output)
        for n in ['frozen_reliability_config_v2.json','ridge_gate_config_v3.json']:shutil.copy2(ROOT/'configs'/n,output/n)
        tasks=[('fresh/fresh_reliability_study_v2.py',[])]
        if retune:tasks.append(('fresh/ridge_audit_v3.py',[]))
        tasks.extend([('fresh/run_ridge_test_v3.py',[]),('scripts/plot_fresh.py',[output]),('scripts/compare.py',['fresh',output])])
    else:
        state=doctor()
        if not state['original_benchmark_ready']:raise RuntimeError('Original raw capture tree or full metrics dependencies missing. See README; no substitute data are used.')
        binary=ROOT/'tools/deep-filter'
        if not binary.is_file():raise RuntimeError('Full original study requires tools/deep-filter; see README')
        for d in ['results','figures','listening']:(output/d).mkdir()
        for n in ['frozen_configs.json','baseline_frozen.json']:shutil.copy2(ROOT/'configs'/n,output/'results'/n)
        shutil.copy2(ROOT/'evidence/original/development.csv',output/'results/development.csv')
        env['RPM_OUTPUT_ROOT']=str(output);env['RPM_DATA_ROOT']=str(ROOT/'original/data')
        tasks=[]
        if retune:tasks.append(('original/code/experiment.py',['development']))
        tasks.extend([('scripts/audit_dataset.py',[]),('original/code/experiment.py',['evaluate']),('original/code/baseline_sensitivity.py',[]),('original/code/neural.py',[]),('original/code/stress.py',[]),('original/code/ablations.py',[]),('scripts/timing.py',[]),('original/code/analyze.py',[]),('original/code/verify_results.py',[]),('scripts/compare.py',['original',output/'results'])])
    ledger={'suite':suite,'retuned':retune,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'python':sys.version,'platform':platform.platform(),'steps':[]}
    ledger['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.rglob('*.py')) if 'runs' not in p.parts}
    t=time.perf_counter()
    try:
        for script,args in tasks:
            step={'script':script,'status':'running'};ledger['steps'].append(step)
            (output/'execution_ledger.json').write_text(json.dumps(ledger,indent=2))
            start=time.perf_counter();print('Running',script,flush=True)
            with (output/(Path(script).stem+'.log')).open('w') as log:
                subprocess.run([sys.executable,str(ROOT/script),*map(str,args)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            step.update(status='executed',elapsed_s=time.perf_counter()-start)
        ledger['status']='EXECUTED'
    except Exception as exc:
        ledger['status']='FAILED';ledger['error']=str(exc)
        if ledger['steps']:ledger['steps'][-1]['status']='failed'
        raise
    finally:
        ledger['elapsed_s']=time.perf_counter()-t
        (output/'execution_ledger.json').write_text(json.dumps(ledger,indent=2))
    print('Completed:',output)

def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    for x in ['doctor','audit','test']:sub.add_parser(x)
    p=sub.add_parser('prepare');p.add_argument('archives',type=Path)
    p=sub.add_parser('fresh');p.add_argument('--output',default='runs/fresh');p.add_argument('--retune',action='store_true')
    p=sub.add_parser('original');p.add_argument('--output',default='runs/original');p.add_argument('--retune',action='store_true')
    p=sub.add_parser('figures');p.add_argument('--input',type=Path,default=ROOT/'evidence/fresh');p.add_argument('--output',type=Path,default=ROOT/'runs/archived_figures')
    a=ap.parse_args()
    if a.command=='doctor':doctor()
    elif a.command=='prepare':execute('original/code/prepare_data.py',a.archives.resolve())
    elif a.command=='audit':execute('scripts/audit_evidence.py')
    elif a.command=='test':subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,check=True)
    elif a.command=='figures':execute('scripts/plot_fresh.py',a.input.resolve(),a.output.resolve())
    else:run_suite(a.command,a.output,a.retune)
if __name__=='__main__':main()
