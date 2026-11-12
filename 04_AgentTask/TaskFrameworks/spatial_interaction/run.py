#!/usr/bin/env python3
"""No installation needed: python run.py --help from this task directory."""
import argparse
import json
from pathlib import Path
from spatial_interaction import EpisodeConfig, make_env
from spatial_interaction.benchmarks import generate_episodes
from spatial_interaction.core.episode import save_episodes, load_episodes
from spatial_interaction.core.trajectory import replay, export_gif
from spatial_interaction.runner import run_episode


def main(argv=None, task_type=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    generate = sub.add_parser('generate')
    generate.add_argument('--count', type=int, default=8)
    generate.add_argument('--split', choices=['development','validation','test'], default='development')
    generate.add_argument('--environment', choices=['dynamic_memory','shape3d','thor_tasks'])
    generate.add_argument('--seed', type=int, default=42)
    generate.add_argument('--output', required=True)
    generate.add_argument('--validate', action='store_true', default=True, help='Validate with real simulators (default)')
    generate.add_argument('--recipes-only', action='store_true', help='Export unvalidated recipes; not a validated benchmark')
    run = sub.add_parser('run')
    run.add_argument('--episodes')
    run.add_argument('--index', type=int, default=0)
    from spatial_interaction.core.episode import TASKS
    run.add_argument('--task', choices=[task_type] if task_type else sum((list(v) for v in TASKS.values()),[]), default=task_type or 'shape_identification')
    run.add_argument('--seed', type=int, default=42)
    run.add_argument('--max-steps', type=int)
    run.add_argument('--difficulty', type=int)
    run.add_argument('--scene', default='FloorPlan1')
    run.add_argument('--policy', choices=['oracle','random','vlm'], default='oracle')
    run.add_argument('--output', required=True)
    run.add_argument('--debug', action='store_true')
    run.add_argument('--history', choices=['rollout','memory'], default='rollout',
                     help='VLM input: multi-turn rollout history (default) or single-message notebook')
    run.add_argument('--max-actions', type=int, default=3, help='VLM actions per turn')
    run.add_argument('--image-window', type=int, help='Keep images only in the last K observation messages')
    replay_parser = sub.add_parser('replay')
    replay_parser.add_argument('directory')
    replay_parser.add_argument('--verify', action='store_true', help='Re-execute and compare every frame/reward')
    evaluate = sub.add_parser('evaluate')
    evaluate.add_argument('directory')
    args = parser.parse_args(argv)
    if args.command == 'generate':
        episodes = generate_episodes(args.count, args.split, args.seed, args.environment, task_type)
        if not args.recipes_only:
            reports = []
            for config in episodes:
                report = run_episode(config, Path(args.output).parent / (Path(args.output).stem+'-validation') / config.episode_id)
                reports.append(report)
                if not report['metrics']['success']:
                    raise RuntimeError(f'Unsolvable recipe {config.episode_id}')
            Path(str(args.output)+'.validation.json').write_text(json.dumps(reports, indent=2))
        save_episodes(episodes, args.output)
        print(json.dumps({'episodes':len(episodes), 'path':args.output, 'simulator_validated':not args.recipes_only}))
    elif args.command == 'run':
        if args.episodes:
            configurations = load_episodes(args.episodes)
            if task_type and any(c.task_type != task_type for c in configurations):
                parser.error(f'This entrypoint only accepts {task_type} episodes')
            config = configurations[args.index]
        else:
            dynamic = args.task in ('missing_object','moved_object')
            native = args.task in TASKS['thor_tasks']
            config = EpisodeConfig(environment='thor_tasks' if native else ('dynamic_memory' if dynamic else 'shape3d'), task_type=args.task,
                scene_id=args.scene if dynamic or native else 'procedural', seed=args.seed,
                max_steps=args.max_steps if args.max_steps is not None else (240 if native else 50),
                difficulty=args.difficulty or (3 if args.task=='moved_object' else 1))
        options = ({'history':args.history, 'max_actions':args.max_actions, 'image_window':args.image_window}
                   if args.policy == 'vlm' else None)
        print(json.dumps(run_episode(config, args.output, args.policy, args.debug, agent_options=options), indent=2))
    elif args.command == 'replay':
        if args.verify:
            config = load_episodes(Path(args.directory)/'episode.json')[0]
            if task_type and config.task_type != task_type:
                parser.error(f'This entrypoint only replays {task_type} episodes')
            env = make_env(config)
            try:
                print(json.dumps(replay(env,args.directory)))
            finally:
                env.close()
        if task_type:
            config = load_episodes(Path(args.directory)/'episode.json')[0]
            if config.task_type != task_type:
                parser.error(f'This entrypoint only replays {task_type} episodes')
        print(export_gif(args.directory))
    else:
        reports = [json.loads(p.read_text()) for p in Path(args.directory).rglob('report.json')]
        if task_type:
            reports = [r for r in reports if r['task_type'] == task_type]
        groups = {}
        for report in reports:
            key = report['task_type'] + '/' + report['policy']
            groups.setdefault(key, []).append(report)
        print(json.dumps({key:{'episodes':len(rows), 'success_rate':sum(r['metrics']['success'] for r in rows)/len(rows),
                              'mean_reward':sum(r['reward'] for r in rows)/len(rows)} for key,rows in groups.items()}, indent=2))


if __name__ == '__main__':
    main()
