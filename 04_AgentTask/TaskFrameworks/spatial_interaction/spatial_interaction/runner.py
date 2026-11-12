"""One execution path for all policies, with public/privileged artifact separation."""
from pathlib import Path
import json
from PIL import Image
from . import make_env
from .agents.policies import RandomAgent, OracleAgent
from .agents.vlm_adapter import VLMAdapter
from .core.trajectory import TrajectoryRecorder, export_gif, export_debug_page


def run_turn(env, actions, recorder=None, debug=False):
    """Execute one agent turn: actions in order, stopping at the first failure, phase change or episode end.

    Only the frame after the last executed action is presented; earlier frames of the turn
    are withheld from the policy (still saved for researchers). Returns the final transition
    and per-action public results for the next prompt.
    """
    results, total = [], 0.
    for index, action in enumerate(actions):
        observation, reward, terminated, truncated, info = env.step(action)
        total += reward
        results.append({'action':action, 'valid':info['action_valid'], 'error':info.get('error'),
                        'phase_transition':info.get('phase_transition')})
        # A phase change (e.g. MEMORIZE) alters the scene and allowed actions: show it before continuing.
        last = (index == len(actions)-1 or env.done or not info['action_valid'] or bool(info.get('phase_transition')))
        if not last:
            env.withhold_view(env.steps)
        if recorder:
            recorder.record(observation, action, reward, terminated, truncated, info,
                            env.privileged_state() if debug else None, withheld=not last)
        if last:
            if index < len(actions)-1:
                results.append({'skipped':len(actions)-index-1})
            return observation, total, terminated, truncated, info, results


def run_episode(config, output, policy='oracle', debug=False, env=None, agent_options=None):
    """agent_options configure the VLM harness: history ('rollout'|'memory'), max_actions, image_window."""
    output = Path(output)
    owns_env = env is None
    env = env or make_env(config, **({'runtime_dir':str(output.parent / 'thor-runtime')}
                                    if config.environment == 'dynamic_memory' else {}))
    recorder = None
    agent = None
    try:
        observation, info = env.reset()
        recorder = TrajectoryRecorder(output, config, debug)
        recorder.record(observation, info=info, state=env.privileged_state() if debug else None)
        if debug and hasattr(env, 'initial_rgb'):
            Image.fromarray(env.initial_rgb).save(output / 'debug_initial.png')
        agent = {'oracle':lambda:OracleAgent(env), 'random':lambda:RandomAgent(config.seed),
                 'vlm':lambda:VLMAdapter(**(agent_options or {}))}[policy]()
        total, results, turns = 0., None, 0
        while not env.done:
            # Scripted agents act one step per turn; the VLM harness may return several actions.
            actions = (agent.act_turn(observation, env.available_actions, results) if hasattr(agent, 'act_turn')
                       else [agent.act(observation, env.available_actions)])
            observation, reward, terminated, truncated, info, results = run_turn(env, actions, recorder, debug)
            total += reward
            turns += 1
        report = {'episode_id':config.episode_id, 'task_type':config.task_type, 'policy':policy,
                  'seed':config.seed, 'terminated':terminated, 'truncated':truncated,
                  'reward':total, 'metrics':env.metrics, 'oracle_uses_privileged_state':policy=='oracle',
                  'turns':turns, 'environment_steps':env.steps,
                  'harness':{'history':agent.history, 'max_actions':agent.max_actions,
                             'image_window':agent.image_window} if policy == 'vlm' else None}
        (output / 'report.json').write_text(json.dumps(report, indent=2))
        if debug:
            (output / 'ground_truth.json').write_text(json.dumps(env.privileged_state(), indent=2))
            if hasattr(env, 'changed_rgb'):
                Image.fromarray(env.changed_rgb).save(output / 'debug_changed.png')
        recorder.close()
        recorder = None
        export_gif(output)
        if debug:
            export_debug_page(output)
        return report
    except Exception as error:
        if recorder:
            (output / 'failure.json').write_text(json.dumps({'error':str(error), 'type':type(error).__name__,
                                                            'completed':False}, indent=2))
        raise
    finally:
        if policy == 'vlm' and agent is not None and output.is_dir():
            (output / 'model_calls.jsonl').write_text('\n'.join(json.dumps(c) for c in agent.calls)+'\n')
        if recorder:
            recorder.close()
        if owns_env:
            env.close()
