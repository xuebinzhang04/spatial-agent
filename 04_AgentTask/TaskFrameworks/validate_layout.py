#!/usr/bin/env python3
"""Validate individual task layout, the shared spatial backend, scoped CLIs and replay.

Use the existing THOR Python plus --thor to include native integration tests.
No GPU submission, environment installation or model inference is performed.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

FRAMEWORKS=Path(__file__).resolve().parent
WORKSPACE=FRAMEWORKS.parents[1]
CUSTOM=FRAMEWORKS.parent/'CustomAgentTasks'


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--thor',action='store_true')
    args=parser.parse_args()
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    if (output/'validation.json').exists():
        raise FileExistsError('Use a new validation output directory')
    directories={p.name:p for p in CUSTOM.iterdir() if p.is_dir() and (p/'task.json').is_file()}
    names=sorted(directories)
    definitions={name:json.loads((directories[name]/'task.json').read_text()) for name in names}
    metadata_path=output/'run_metadata.json'
    if metadata_path.exists():
        metadata=json.loads(metadata_path.read_text())
    else:
        metadata={'run_id':output.name,'project_id':str(uuid.uuid4()),'project_name':'Independent custom task layout',
          'task_name':'Task entrypoint regression','task_id':'custom-task-layout',
          'purpose':'Verify the eight independent native/geometry task entrypoints.',
          'dataset':{'name':'eight custom tasks','version':'post-directory-reorganization',
                     'split':'train/development smoke','tasks':names,'materials':['task.json in each task directory']},
          'experiment_type':'functional regression','model':{'name':'scripted/oracle','checkpoint':'not applicable','precision':'not applicable'},
          'environment':{'name':'NumPy + cached THOR' if args.thor else 'NumPy','version':sys.version},
          'source':{'commit':'Not recorded (no workspace Git metadata)','entrypoint':str(Path(__file__).relative_to(WORKSPACE)),'config':'task.json in each task directory'},
          'slurm':{'job_id':'not applicable','partition':'not applicable','resources':'local CPU; no GPU allocation',
                   'state':'RUNNING','created_at':now(),'started_at':now(),'ended_at':'Not recorded','exit_code':'Not recorded','failure_reason':'Not recorded'},
          'artifacts':{'result_dir':str(output),'primary_metrics':'validation.json','report':'validation.json','trajectory':'task-smokes/','logs':[]},
          'notes':'No model inference/training. Historical experiment snapshots and manifests remain unchanged.'}
    metadata['parameters']={**metadata.get('parameters',{}),'include_native_thor':args.thor,
                            'task_definitions':definitions,'python':sys.executable,'spatial_seed':42}
    metadata['slurm'].update(state='RUNNING',started_at=now())
    write(metadata_path,metadata)
    source_files=[]
    for directory in (CUSTOM,FRAMEWORKS,FRAMEWORKS.parent/'Demos'):
        source_files.extend(p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts and
                            (p.suffix in ('.py','.toml') or p.name=='task.json' or 'scene_states' in p.parts))
    write(output/'source_hashes.json',{str(p.relative_to(WORKSPACE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_files)})
    report={'completed':False,'tasks':names,'suites':[],'task_smokes':[],'generators':[],'replays':[]}
    environment={**os.environ,'SPATIAL_TEST_THOR':'1' if args.thor else '0'}

    def execute(command,log_name,cwd=WORKSPACE,timeout=300):
        log=output/log_name
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('w') as stream:
            result=subprocess.run(command,cwd=cwd,env=environment,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
        metadata['artifacts']['logs'].append(log_name)
        if result.returncode:
            raise RuntimeError(f'Command failed ({result.returncode}); inspect {log}')
        return log

    try:
        suites=[('layout',FRAMEWORKS,FRAMEWORKS/'tests'),
                ('individual-tasks',CUSTOM,CUSTOM),
                ('spatial',FRAMEWORKS/'spatial_interaction',FRAMEWORKS/'spatial_interaction/tests')]
        for name,cwd,tests in suites:
            log=execute([sys.executable,'-m','unittest','discover','-s',str(tests),'-v'],f'{name}-suite.log',cwd)
            text=log.read_text(); match=re.search(r'Ran (\d+) tests?',text)
            count=int(match.group(1)) if match else 0
            if not count:
                raise RuntimeError(f'Test discovery ran no tests: {name}')
            report['suites'].append({'name':name,'passed':True,'tests':count,'log':log.name,
                                    'native_enabled':args.thor if name=='spatial' else False})
            print(name,count,'tests passed',flush=True)
            write(output/'validation.json',report)
        for name in names:
            definition=definitions[name]
            native=definition.get('environment') in ('dynamic_memory','thor_tasks')
            if native and not args.thor:
                report['task_smokes'].append({'task':name,'status':'skipped','reason':'--thor not selected'})
                continue
            entry=directories[name]/'run.py'
            target=output/'task-smokes'/name
            execute([sys.executable,str(entry),'run','--episodes',str(CUSTOM/name/'benchmarks/development.jsonl'),
                     '--policy','oracle','--debug','--output',str(target)],f'{name}-run.log')
            result=json.loads((target/'report.json').read_text())
            assert result['task_type']==name and result['metrics']['success']
            replay_log=execute([sys.executable,str(entry),'replay',str(target),'--verify'],f'{name}-replay.log')
            replay=json.loads(replay_log.read_text().splitlines()[0]); assert replay['verified']
            report['replays'].append({'task':name,**replay})
            generated=output/'generated'/name/'development.jsonl'
            execute([sys.executable,str(entry),'generate','--count','1','--output',str(generated)],f'{name}-generate.log')
            specs=[json.loads(line) for line in generated.read_text().splitlines() if line.strip()]
            assert len(specs)==1 and {s['task_type'] for s in specs}=={name}
            certificate=json.loads(Path(str(generated)+'.validation.json').read_text())
            assert certificate[0]['metrics']['success']
            report['task_smokes'].append({'task':name,'status':'passed','artifact_dir':str(target.relative_to(output))})
            report['generators'].append({'task':name,'status':'passed','own_task_only':True})
            print(name,'scoped CLI / oracle / generation passed',flush=True)
            write(output/'validation.json',report)
        report['completed']=True
        metadata['slurm'].update(state='COMPLETED',exit_code=0,failure_reason='not applicable')
    except Exception as error:
        report['error']=f'{type(error).__name__}: {error}'
        metadata['slurm'].update(state='FAILED',exit_code=1,failure_reason=report['error'])
        raise
    finally:
        metadata['slurm']['ended_at']=now()
        write(output/'validation.json',report)
        write(metadata_path,metadata)
    print(json.dumps({'completed':True,'tests':sum(s['tests'] for s in report['suites']),
                      'task_smokes':sum(s['status']=='passed' for s in report['task_smokes']),
                      'generators':len(report['generators']),'replays':len(report['replays']),
                      'output':str(output)},indent=2))


if __name__=='__main__':
    main()
