"""Submit one sequential native-task validation; manifests precede execution."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import uuid

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--task',required=True,choices=('memory_maze','dependency_doors','relational_rearrangement','dynamic_replanning'));p.add_argument('--submit',action='store_true');p.add_argument('--cases',type=int,choices=(1,2,3),default=3);p.add_argument('--level',type=int,choices=(1,2,3))
    args=p.parse_args();levels=[args.level] if args.level else list(range(1,args.cases+1));script=ROOT/'scripts/experiments/native_task_migration.slurm'
    # A preceding completed migration must already be reflected in the report.
    order=('memory_maze','dependency_doors','relational_rearrangement','dynamic_replanning')
    if order.index(args.task):
        previous=order[order.index(args.task)-1]
        marker=ROOT/'04_AgentTask/CustomAgentTasks'/previous/'migration.json'
        assert marker.is_file(), 'Previous task must complete native admission first: '+previous
        meeting=ROOT/'survey/10-19 Meeting'
        summary=json.loads((meeting/'demo-summary.json').read_text())
        check=json.loads((meeting/'report-validation.json').read_text())
        assert check['completed'] and any(r['task']==previous for r in summary['native_migration']), 'Update and validate the meeting report before starting '+args.task
        relative=str(marker.relative_to(ROOT))
        assert summary['source_sha256'].get(relative)==hashlib.sha256(marker.read_bytes()).hexdigest(), 'Meeting report refers to an outdated preceding migration'
    usage=subprocess.run(['hyakusage','-a','video'],capture_output=True,text=True,check=True)
    if 'Permission denied' in usage.stdout+usage.stderr:raise RuntimeError('Account unavailable')
    tested=subprocess.run(['sbatch','--test-only',str(script)],capture_output=True,text=True,check=True)
    print(tested.stdout+tested.stderr,end='')
    if not args.submit:return
    session=ROOT/'05_Experiment/results/native-migration-session-20261019.json'
    if session.exists():group=json.loads(session.read_text())
    else:
        group={'project_id':str(uuid.uuid4()),'project_name':'Interaction Lab native simulator migration','jobs':[]}
    submitted=subprocess.run(['sbatch','--hold','--parsable','--export=ALL,MIGRATION_TASK='+args.task+',MIGRATION_CASES='+str(args.cases)+',MIGRATION_LEVEL='+str(args.level or 0),str(script)],capture_output=True,text=True,check=True)
    job=submitted.stdout.strip().split(';')[0];assert job.isdigit()
    out=ROOT/'05_Experiment/results'/f'native-migration-{args.task}-{job}';out.mkdir()
    custom=ROOT/'04_AgentTask/CustomAgentTasks'
    task=custom/args.task
    sources=list(task.rglob('*.py'))+list((ROOT/'04_AgentTask/TaskFrameworks/spatial_interaction/spatial_interaction').rglob('*.py'))
    sources += [Path(__file__),script,ROOT/'scripts/experiments/validate_native_task_migration.py']
    meta={'run_id':out.name,'project_id':group['project_id'],'project_name':group['project_name'],
      'task_name':args.task,'task_id':args.task,'purpose':'Native task migration admission; verify oracle solvability, premature finish, invalid actions, random rollouts and replay',
      'dataset':{'name':'native task procedural admission','version':'1','split':'development/validation/test scene separation',
                 'tasks':[args.task+f':L{i}:seed{41+i}' for i in levels],'materials':[f'FloorPlan{i}' for i in levels]},
      'experiment_type':'native simulator functional validation','model':{'name':'privileged oracle and observation-only random','checkpoint':'Not applicable','precision':'Not applicable'},
      'environment':{'name':'AI2-THOR 5.0.0 CloudRendering via CPU lavapipe','version':'cached f0825767cd50d69f666c7f282e54abfe58f1e917'},
      'source':{'commit':'Not recorded: workspace has no Git metadata','entrypoint':'scripts/experiments/validate_native_task_migration.py','config':'per-episode episode.json',
                'sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}},
      'parameters':{'seeds':[41+i for i in levels],'levels':levels,'max_steps':360,'image_size':128,'exploration_budget':12},
      'slurm':{'job_id':job,'partition':'gpu-h200','resources':'1 H200 allocation / 8 CPU / 50 GB / 20 min; software rendering only',
               'state':'HELD','created_at':datetime.now(timezone.utc).isoformat(),'started_at':'Not recorded','ended_at':'Not recorded','exit_code':'Not recorded','failure_reason':'Not recorded'},
      'artifacts':{'result_dir':str(out.relative_to(ROOT)),'primary_metrics':'validation.json','report':'validation.json','trajectory':'episodes/','logs':[f'05_Experiment/logs/native-migration-{job}.log']},
      'notes':'No model inference or RL training. Feasible oracle uses privileged state; not observation-only policy performance. Failed runs retained.'}
    (out/'run_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    (out/'account-usage.txt').write_text(usage.stdout+usage.stderr)
    (out/'scheduler-test.txt').write_text(tested.stdout+tested.stderr)
    group['jobs'].append({'task':args.task,'job_id':job,'result_dir':str(out.relative_to(ROOT))});session.write_text(json.dumps(group,indent=2)+'\n')
    subprocess.run(['scontrol','release',job],check=True)
    print(json.dumps({'job_id':job,'result_dir':str(out)},indent=2))


if __name__=='__main__':main()
