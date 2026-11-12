"""Generate DeliveryBench cases or admit one recipe against the real UE runtime."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import traceback
import uuid

from cases import ROOT,UPSTREAM,FAMILIES,City,generate,digest,content_hash,public_observation

HERE=Path(__file__).resolve().parent


def now():return datetime.now(timezone.utc).isoformat()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_suffix('.pending.json');pending.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');pending.replace(path)


def sources():
    local=list(HERE.glob('*.py'))
    native=list((UPSTREAM/'vlm_delivery').rglob('*.py'))
    return {str(p.relative_to(ROOT)):digest(p) for p in local+native}


def metadata(output,kind):
    return {'run_id':output.name,'project_id':'delivery-cases-'+uuid.uuid4().hex,
        'project_name':'DeliveryBench interaction cases','task_name':kind,'task_id':'delivery-cases-v1',
        'purpose':'Construct reproducible ability-demanding courier cases, reject shortcuts and inspect native interaction evidence.',
        'dataset':{'name':'Native DeliveryBench map and API recipes','version':'delivery-cases/1','split':'development','tasks':[],'materials':[]},
        'experiment_type':kind,'model':{'name':'No VLM; generation or privileged mechanical witness','precision':'Not applicable'},
        'environment':{'name':'SimWorld dedicated gym_citynav DeliveryBench build',
                       'package':'02_Simulator/DeliveryBench/Linux/gym_citynav','asset_revision':'64a4889d72a2c7b792d242083e53a926b43ceaa9'},
        'source':{'entrypoint':str(Path(__file__).relative_to(ROOT)),'commit':'Local sources identified by hash','sha256':sources()},
        'parameters':{},'slurm':{'job_id':os.environ.get('SLURM_JOB_ID','Not applicable'),'state':'RUNNING',
            'created_at':now(),'started_at':now(),'ended_at':'Not completed','exit_code':'Not completed',
            'failure_reason':'Not recorded','node':os.uname().nodename},
        'artifacts':{'result_dir':str(output.relative_to(ROOT)),'report':'summary.json','primary_metrics':'validation.json'},
        'notes':'Generated recipes are not UE-verified cases. Native admission uses actual order state and actor pose, not constant-zero Gym reward.'}


def generate_batch(args):
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    meta=metadata(output,'Deterministic recipe generation and graph screening')
    meta['dataset']['tasks']=list(args.families);meta['dataset']['materials']=args.maps
    meta['parameters']={'seed':args.seed,'maps':args.maps,'levels':args.levels,'per_level':args.per_level}
    write(output/'run_metadata.json',meta)
    for relative in meta['source']['sha256']:
        target=output/'source_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/relative,target)
    results=[];failures=[]
    try:
        for map_index,map_name in enumerate(args.maps):
            city=City(map_name)
            for family_index,family in enumerate(args.families):
                for level in args.levels:
                    for index in range(args.per_level):
                        seed=args.seed+map_index*10000+family_index*1000+level*100+index
                        try:case=generate(city,family,seed,level)
                        except ValueError as error:
                            failures.append({'family':family,'map':map_name,'seed':seed,'level':level,'error':str(error)});continue
                        relative='cases/'+case['case_id']+'.json';write(output/relative,case)
                        write(output/'public'/f"{case['case_id']}.json",public_observation(case,{}))
                        results.append({'case_id':case['case_id'],'family':family,'map':map_name,'level':level,
                                        'seed':seed,'status':case['status'],'path':relative})
        summary={'schema':'delivery-case-batch/1','cases':results,'rejected':failures,
                 'generated':len(results),'native_admitted':0,'model_evaluated':0}
        write(output/'summary.json',summary)
        # The index is convenient for data loading; private certificates stay in case files.
        (output/'cases.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in results))
        validation={'completed':not failures,'recipes':len(results),'native_execution':False,
                    'structural_levels':args.levels,'difficulty_calibrated':False,'rejections':len(failures)}
        write(output/'validation.json',validation)
        lines=['# DeliveryBench case batch','','Recipes and native graph screening only; no UE/model execution.','',
               '| Family | L1 | L2 | L3 |','| --- | ---: | ---: | ---: |']
        for family in args.families:
            counts=[sum(c['family']==family and c['level']==level for c in results) for level in (1,2,3)]
            lines.append('| '+family+' | '+' | '.join(map(str,counts))+' |')
        lines+=['','Native admission must verify collision/arrival, order state, bag placement, charging, deadlines and UI/UE position agreement.',
                'Visual handoff remains a candidate family until customer visibility and primitive-motion state synchronization pass.',
                'Use separate `public/` files for policy instructions; full `cases/` files include private certificates.']
        (output/'README.md').write_text('\n'.join(lines)+'\n')
        meta['slurm'].update(state='COMPLETED',exit_code=0,ended_at=now())
        print(json.dumps(validation))
    except BaseException as error:
        meta['slurm'].update(state='FAILED',exit_code=1,failure_reason=str(error),ended_at=now());raise
    finally:write(output/'run_metadata.json',meta)


def admit(args):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('UE admission requires a Slurm allocation')
    case=json.loads(args.case.resolve().read_text())
    check=dict(case);case_id=check.pop('case_id')
    if case_id!=case['family']+'-'+content_hash(check)[:12]:raise ValueError('Recipe content hash does not match case ID')
    for relative,expected in case['source_sha256'].items():
        if digest(ROOT/relative)!=expected:raise ValueError('Native assets or mechanics changed: '+relative)
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    meta=metadata(output,'Qwen fixed-case rollout' if args.model else 'Native case admission; no policy inference')
    if args.model:
        meta['model']={'name':args.model,'checkpoint':'01_Models/'+args.model,'precision':'bfloat16'}
        meta['parameters'].update(max_model_calls=args.max_model_calls,model_url=args.model_url)
    meta['dataset'].update(tasks=[case_id],materials=[str(args.case.resolve().relative_to(ROOT)),case['map']])
    meta['parameters'].update(case_id=case_id,seed=case['seed'],level=case['level'],map=case['map'],budget=case['budget'],time_scale=1.,native_control_action_reserve=4)
    meta['artifacts'].update(trajectory='native-trace.jsonl',snapshots='snapshots/')
    write(output/'run_metadata.json',meta);write(output/'case.json',case)
    for relative in meta['source']['sha256']:
        target=output/'source_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/relative,target)
    native=None;result={}
    try:
        from native import NativeCase
        native=NativeCase(case,output)
        recording=None
        if args.record_process:
            from process import ProcessRecording
            recording=ProcessRecording(native)
            meta['parameters']['process_capture_interval_ms']=500
        stopped=threading.Event()
        def worker():
            try:
                score=native.policy(args.model,args.model_url,args.max_model_calls) if args.model else native.reference()
                result.update(native_executed=True,native_admitted=score['success'],evaluation=score,
                              policy=args.model or 'privileged mechanical admission witness',model_evaluated=bool(args.model))
                if args.model:result.update(model_success=score['success'],native_admitted=None)
            except BaseException as error:
                result.update(native_executed=True,native_admitted=False,error=str(error))
                (output/'traceback.log').write_text(traceback.format_exc())
            finally:stopped.set()
        # The native API dispatches work onto the Qt main thread.
        from PyQt5.QtCore import QTimer
        timer=QTimer();timer.timeout.connect(lambda:native.env._app.quit() if stopped.is_set() else None);timer.start(100)
        thread=threading.Thread(target=worker,daemon=True);thread.start();native.env.run_qt_loop();thread.join(timeout=5)
        if thread.is_alive():raise RuntimeError('Admission worker did not stop cleanly')
        if recording:
            result['process_recording']=recording.finish()
        write(output/'validation.json',result)
        completed=bool(result.get('model_evaluated') or result.get('native_admitted'))
        meta['slurm'].update(state='COMPLETED' if completed else 'FAILED',
            exit_code=0 if completed else 1,ended_at=now(),failure_reason=result.get('error',result.get('evaluation',{}).get('reasons')))
        if not completed:raise RuntimeError('Native execution failed; inspect validation.json')
        print(json.dumps(result))
    except BaseException as error:
        meta['slurm'].update(state='FAILED',exit_code=1,failure_reason=str(error),ended_at=now())
        if not (output/'validation.json').exists():write(output/'validation.json',{'native_admitted':False,'error':str(error)})
        if not (output/'traceback.log').exists():(output/'traceback.log').write_text(traceback.format_exc())
        raise
    finally:
        if native:native.close()
        write(output/'run_metadata.json',meta)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='mode',required=True)
    gen=sub.add_parser('generate');gen.add_argument('--output',type=Path,required=True)
    gen.add_argument('--maps',nargs='+',default=['small-city-11']);gen.add_argument('--families',nargs='+',choices=FAMILIES,default=list(FAMILIES))
    gen.add_argument('--levels',nargs='+',type=int,choices=(1,2,3),default=[1,2,3]);gen.add_argument('--per-level',type=int,default=3)
    gen.add_argument('--seed',type=int,default=20261026)
    admission=sub.add_parser('admit');admission.add_argument('--case',type=Path,required=True);admission.add_argument('--output',type=Path,required=True)
    admission.add_argument('--model');admission.add_argument('--model-url',default='http://127.0.0.1:8000/v1')
    admission.add_argument('--max-model-calls',type=int,default=16)
    admission.add_argument('--record-process',action='store_true')
    args=parser.parse_args()
    if args.mode=='generate':
        if args.per_level<1 or args.seed<0:parser.error('per-level must be positive and seed nonnegative')
        generate_batch(args)
    else:admit(args)


if __name__=='__main__':main()
