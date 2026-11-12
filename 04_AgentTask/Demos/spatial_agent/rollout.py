"""Run a model in an environment independently of the presentation server."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

from session import ROOT, Session

sys.path.insert(0, str(ROOT/'scripts/experiments'))
from analyze_interaction import write_analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', choices=('shape3d','native'), default='shape3d')
    parser.add_argument('--task', default='shape_identification')
    parser.add_argument('--model', default='Qwen3.8-27B')
    parser.add_argument('--model-url', default='http://127.0.0.1:8000/v1')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--difficulty', type=int, default=1)
    parser.add_argument('--budget', type=int, default=50)
    parser.add_argument('--scene', default='FloorPlan1')
    parser.add_argument('--run-dir', type=Path, default=ROOT/'05_Experiment/results'/('model-rollout-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')))
    args = parser.parse_args()
    if (args.run_dir/'run_metadata.json').exists():
        parser.error('Choose a fresh run directory; existing evidence is preserved')
    session = Session(args.run_dir, args.model_url, args.model)
    session.metadata['source']['entrypoint'] = '04_AgentTask/Demos/spatial_agent/rollout.py'
    session.metadata['purpose'] = 'Model inference and environment interaction independent of the presentation server'
    session.metadata['experiment_type'] = 'Headless inference rollout; no benchmark claim'
    session.save_metadata()
    try:
        state = session.reset({k:getattr(args,k) for k in ('environment','task','seed','difficulty','budget','scene')})
        write_analysis(session.run, session.run/'interaction-analysis.json')
        print('RUN '+str(session.run), flush=True)
        while not state['terminated'] and not state['truncated']:
            state = session.infer(True)['state']
            write_analysis(session.run, session.run/'interaction-analysis.json')
            print(f"step={state['step']} policy={state['policy']} action={state.get('action')} reward={state['reward']}", flush=True)
    except BaseException as error:
        session.metadata['slurm'].update(state='FAILED', exit_code=1, failure_reason=str(error))
        raise
    finally:
        session.close()


if __name__ == '__main__':
    main()
