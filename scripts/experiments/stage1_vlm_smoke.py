"""Four live VLM episodes. Success/failure are model outcomes, not assertions.

Public RGB/history/memory enter VLMAdapter; evaluator state is logged separately.
"""
import argparse
from html import escape
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import traceback
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'04_AgentTask/TaskFrameworks/spatial_interaction'))
from spatial_interaction import EpisodeConfig, make_env
from spatial_interaction.agents.vlm_adapter import VLMAdapter
from spatial_interaction.core.trajectory import TrajectoryRecorder, export_gif, export_debug_page

TASKS=('shape_identification','pose_alignment','missing_object','moved_object')


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def save_report(out,summary):
    """Automatic result inspection after each completed/failed real model episode."""
    names={'shape_identification':'形状识别','pose_alignment':'姿态对齐','missing_object':'物体缺失','moved_object':'物体移动'}
    lines=['# Stage 1 真实 VLM 联调','',
           '模型：Qwen3.8-27B。每类一个样例、最多 20 步；仅作为接口与交互流程检查，不是正式性能结论。',
           'VLM 仅接收当前 RGB、任务说明、动作历史与自身记忆；真值只供评分和单独的研究者日志。','',
           '|任务|执行完成|任务成功|模型调用|无效动作|终止方式|', '|---|---|---|---:|---:|---|']
    blocks=[]
    for case in summary['cases']:
        task=case['task_type'];folder=out/'episodes'/task
        success=case.get('metrics',{}).get('success',False)
        ending='运行错误' if not case['completed'] else ('步数耗尽' if case.get('truncated') else '提交答案')
        lines.append(f'|{names[task]}|{case["completed"]}|{success}|{case["model_calls"]}|{case.get("invalid_actions","—")}|{ending}|')
        frames=[]
        if (folder/'trajectory.jsonl').exists():
            for raw in (folder/'trajectory.jsonl').read_text().splitlines():
                row=json.loads(raw)
                frames.append(f'<figure><img loading="lazy" src="episodes/{task}/{row["frame"]}"><figcaption>Step {row["step"]} · {escape(row["observation"]["phase"])}<pre>{escape(json.dumps(row["action"],ensure_ascii=False))}</pre></figcaption></figure>')
        blocks.append(f'<section><h2>{names[task]}</h2><p>完成：{case["completed"]} · 成功：{success} · {ending}</p>'
                      f'<p>{escape(case.get("error",""))}</p><a href="episodes/{task}/model_calls.jsonl">原始模型输出</a>'
                      f'<div class="frames">{"".join(frames)}</div></section>')
    lines+=['','执行完成不等于回答正确。答案错误和无效动作是实际模型结果，不由 oracle 修正。',
            '查看 results.html 可逐帧检查；原始模型输出位于各任务的 model_calls.jsonl。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (out/'results.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Stage 1 VLM 实测</title>'
        '<style>body{font:17px/1.6 system-ui;margin:32px;background:#f4f5f1;color:#18323d}section{margin:30px 0}'
        '.frames{display:flex;gap:16px;overflow-x:auto}figure{margin:0;flex:0 0 420px;background:white;padding:12px;border-radius:12px}'
        'img{width:100%;height:320px;object-fit:contain;background:#192332}pre{white-space:pre-wrap;font-size:13px}</style>'
        '<h1>Stage 1 · 真实 VLM 联调</h1><p>Qwen3.8-27B；每类一个样例，最多 20 步。当前 RGB + 自身文本记忆，不向模型提供评分真值。</p>'
        '<p>画面为实际模型动作后的观察；横向滚动逐帧查看。结果只代表小规模联调，未训练。</p>'+''.join(blocks)+'</html>')


def episode(config,output):
    env=make_env(config,**({'runtime_dir':output.parent/'thor-runtime'} if config.environment=='dynamic_memory' else {}))
    agent=VLMAdapter(timeout=60)
    recorder=None
    started=time.monotonic()
    try:
        obs,info=env.reset()
        recorder=TrajectoryRecorder(output,config,debug=True)
        recorder.record(obs,info=info,state=env.privileged_state())
        if hasattr(env,'initial_rgb'):Image.fromarray(env.initial_rgb).save(output/'debug_initial.png')
        total=0.;invalid=0
        while not env.done:
            if time.monotonic()-started>300:raise TimeoutError('Five-minute episode wall-clock budget exceeded')
            call_start=time.monotonic()
            action=agent.act(obs,env.available_actions)
            agent.calls[-1]['latency_seconds']=round(time.monotonic()-call_start,3)
            obs,reward,terminated,truncated,info=env.step(action)
            invalid+=int(not info['action_valid']);total+=reward
            recorder.record(obs,action,reward,terminated,truncated,info,env.privileged_state())
            print(config.task_type,env.steps,action['action'],'valid=',info['action_valid'],flush=True)
        result={'episode_id':config.episode_id,'task_type':config.task_type,'seed':config.seed,
                'policy':'vlm','completed':True,'terminated':terminated,'truncated':truncated,
                'reward':total,'metrics':env.metrics,'invalid_actions':invalid,
                'model_calls':len(agent.calls),'oracle_uses_privileged_state':False,
                'elapsed_seconds':round(time.monotonic()-started,3)}
        write(output/'report.json',result)
        write(output/'ground_truth.json',env.privileged_state())
        if hasattr(env,'changed_rgb'):Image.fromarray(env.changed_rgb).save(output/'debug_changed.png')
        recorder.close();recorder=None
        export_gif(output,duration=650);export_debug_page(output)
        return result
    except Exception as error:
        output.mkdir(exist_ok=True,parents=True)
        result={'episode_id':config.episode_id,'task_type':config.task_type,'policy':'vlm','completed':False,
                'error':f'{type(error).__name__}: {error}','model_calls':len(agent.calls),
                'oracle_uses_privileged_state':False,'elapsed_seconds':round(time.monotonic()-started,3)}
        write(output/'failure.json',result)
        (output/'traceback.log').write_text(traceback.format_exc())
        return result
    finally:
        if recorder:recorder.close()
        if output.exists():
            (output/'model_calls.jsonl').write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in agent.calls))
        env.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-steps',type=int,default=20)
    p.add_argument('--tasks',nargs='+',choices=TASKS,default=list(TASKS))
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise FileExistsError('Preserve previous results; choose a new run directory')
    summary={'completed':False,'model':'Qwen3.8-27B','policy':'vlm','cases':[],
             'scope':'Live four-episode integration smoke; not a calibrated benchmark',
             'started_at_utc':datetime.now(timezone.utc).isoformat()}
    for task in args.tasks:
        dynamic=task in ('missing_object','moved_object')
        seed={'missing_object':42,'moved_object':43,'shape_identification':44,'pose_alignment':45}[task]
        config=EpisodeConfig(episode_id='vlm_smoke_'+task,environment='dynamic_memory' if dynamic else 'shape3d',
            task_type=task,scene_id='FloorPlan1' if dynamic else 'procedural',seed=seed,
            difficulty=3 if task=='moved_object' else 1,max_steps=args.max_steps,
            task_config={'image_size':320,'exploration_budget':6,'min_target_pixels':500} if dynamic else
                        {'image_size':224,'split':'development'})
        result=episode(config,out/'episodes'/task)
        summary['cases'].append(result);write(out/'summary.json',summary);save_report(out,summary)
    summary['completed']=all(c['completed'] for c in summary['cases'])
    summary['successful_episodes']=sum(bool(c.get('metrics',{}).get('success')) for c in summary['cases'])
    summary['ended_at_utc']=datetime.now(timezone.utc).isoformat();write(out/'summary.json',summary);save_report(out,summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    if not summary['completed']:sys.exit(1)


if __name__=='__main__':main()
