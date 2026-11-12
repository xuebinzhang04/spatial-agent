"""Combine real VLM smoke attempts without hiding infrastructure failures."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    selected={};attempts=[];provenance={};jobs=[]
    for raw in args.runs:
        run=raw.resolve();summary=json.loads((run/'summary.json').read_text())
        meta=json.loads((run/'run_metadata.json').read_text());jobs.append(str(meta['job_id']))
        for path in [run/'summary.json',run/'run_metadata.json']:
            provenance[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        for case in summary['cases']:
            attempts.append({'run_dir':str(run.relative_to(ROOT)),**case})
            task=case['task_type']
            if case['completed'] or task not in selected:selected[task]=(case,run/'episodes'/task)
    assert set(selected)=={'missing_object','moved_object','shape_identification','pose_alignment'}
    cases=[];call_count=0
    for task in ('shape_identification','pose_alignment','missing_object','moved_object'):
        case,source=selected[task];cases.append(case)
        shutil.copytree(source,out/'episodes'/task)
        calls=[json.loads(s) for s in (source/'model_calls.jsonl').read_text().splitlines()]
        rows=[json.loads(s) for s in (source/'trajectory.jsonl').read_text().splitlines()]
        assert len(calls)==case['model_calls']==len(rows)-1
        assert all(not c.get('transport_error') for c in calls)
        assert not case['oracle_uses_privileged_state']
        for row in rows:
            assert set(row['observation'])=={'instruction','phase','action_history','memory'}
        call_count+=len(calls)
    summary={'completed':all(c['completed'] for c in cases),'model':'Qwen3.8-27B','policy':'vlm',
             'cases':cases,'successful_episodes':sum(bool(c.get('metrics',{}).get('success')) for c in cases),
             'total_model_calls':call_count,'source_jobs':jobs,'attempts':attempts,
             'scope':'Four live smoke episodes, not a calibrated benchmark; original infrastructure failures retained',
             'source_sha256':provenance}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    module_path=Path(__file__).with_name('stage1_vlm_smoke.py')
    spec=importlib.util.spec_from_file_location('vlm_smoke_report',module_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.save_report(out,summary)
    accounting=subprocess.run(['sacct','-j',','.join(jobs),'--format=JobID,State,Elapsed,ExitCode,NodeList,AllocTRES','-nP'],text=True,capture_output=True,check=True)
    (out/'slurm-accounting.txt').write_text(accounting.stdout)
    print(json.dumps({'completed':summary['completed'],'successes':summary['successful_episodes'],
                      'model_calls':call_count,'report':str(out/'REPORT.md')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
