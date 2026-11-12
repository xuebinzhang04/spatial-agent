"""Import older public demonstrations into the shared episode schema."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import re


def embedded(path, variable):
    text = path.read_text()
    match = re.search(r"const " + variable + r"\s*=\s*", text)
    if not match:
        raise ValueError(f"Missing embedded data: {path}")
    return json.JSONDecoder().raw_decode(text[match.end():])[0]


def import_legacy(root, output, track):
    archive = root / "archive/agent-interaction/legacy-demos"
    assets, tasks, episodes = {}, {}, []

    def save(raw, suffix):
        key = hashlib.sha256(raw).hexdigest()
        name = "assets/" + key[:24] + suffix
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(raw)
        assets[name] = key
        return name

    source = archive / "deliverybench/demo.html"
    track(source)
    data = embedded(source, "DATA")
    frames = []
    fields = ("position_cm", "energy_pct", "battery_pct", "earnings", "transport", "sim_time_s", "active_orders", "order_pool", "prompt", "action_errors")
    for step, row in enumerate(data.get("replay", [])):
        state = row["state"]
        views = []
        for name, value in zip(("Global map", "Local map", "First-person RGB"), state["images"]):
            if value:
                views.append({"name": name, "image": save(base64.b64decode(value.split(",", 1)[1]), ".png")})
        observation = {k: state.get(k) for k in fields}
        observation.update(instruction="Operate the native delivery task; this recording is a demonstration cutoff, not a completed benchmark.",
                           phase="delivery", feedback=state.get("action_errors") or "")
        transition = state.get("transition", {})
        frames.append({"step": step, "image": views[-1]["image"], "views": views, "action": row["command"],
            "observation": observation, "reward": transition.get("reward", 0), "terminated": transition.get("terminated", False),
            "truncated": transition.get("truncated", False), "error": state.get("action_errors")})
    episode_id = "deliverybench-ue-345795"
    tasks["deliverybench"] = {"id": "deliverybench", "title": "DeliveryBench", "category": "Native UE delivery",
        "summary": "Recorded manual/script interaction with three camera streams, orders, and native resource state.",
        "objectives": ["Inspect and accept real orders", "Navigate, collect, and deliver under native constraints"],
        "environment": "simworld_delivery", "episode_ids": [episode_id], "status": "recorded"}
    episodes.append({"id": episode_id, "task": "deliverybench", "agent": "manual-script", "scene": data["replay"][0]["state"]["map"],
        "seed": data["replay"][0]["state"]["seed"], "difficulty": 0, "budget": 100, "instruction": frames[0]["observation"]["instruction"],
        "frames": frames, "result": {"success": None, "metrics": {"completion": "Demonstration cutoff; no benchmark result"},
            "reward": sum(f["reward"] for f in frames), "terminated": frames[-1]["terminated"], "truncated": frames[-1]["truncated"]},
        "provenance": {"directory": data["provenance"]["recording"], "policy": "manual/script", "privileged_policy": False,
                       "recording_complete": False, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}})
    model = {"id": "manual-script", "name": "Manual / scripted API", "kind": "manual", "checkpoint": None}
    return list(tasks.values()), episodes, [model], assets, data


def city_data(root):
    task = root / "04_AgentTask/SpatialAgentBench/DeliveryBench"
    maps = {}
    for folder in sorted((task / "maps").iterdir()):
        if not (folder / "roads.json").is_file():
            continue
        world = json.loads((folder / "progen_world_enriched.json").read_text())
        maps[folder.name] = {"roads": json.loads((folder / "roads.json").read_text())["roads"],
            "buildings": json.loads((folder / "buildings.json").read_text())["buildings"],
            "pois": [{"id": n["id"], "type": n["properties"].get("poi_type", n["instance_name"]),
                      "x": n["properties"]["location"]["x"] / 100, "y": n["properties"]["location"]["y"] / 100}
                     for n in world["nodes"] if n["properties"].get("poi_type") or n["instance_name"].startswith("POI_")]}
    tree = ast.parse((task / "vlm_delivery/gameplay/action_space.py").read_text())
    api = next(n.value.args[0].value for n in tree.body if isinstance(n, ast.AnnAssign)
               and isinstance(n.target, ast.Name) and n.target.id == "ACTION_API_SPEC")
    return {"maps": maps, "api": api, "mechanics": json.loads((task / "vlm_delivery/input/game_mechanics_config.json").read_text())}


def import_delivery_cases(root,output,track):
    """Import actual fixed-case snapshots and exact model request text, never witnesses as inputs."""
    tasks={};episodes=[];assets={}
    sources=[]
    for index in sorted((root/'05_Experiment/results').glob('delivery-cases-*/admissions.json')):
        track(index)
        sources += [root/entry['validation'] for entry in json.loads(index.read_text())['entries'] if entry.get('native_admitted')]
    sources+=sorted((root/'05_Experiment/results').glob('delivery-qwen-*/episode/validation.json'))
    for source in sources:
        if not source.exists():continue
        folder=source.parent;validation=json.loads(source.read_text())
        if not validation.get('native_executed'):continue
        snapshots=sorted((folder/'snapshots').glob('*.json'))
        calls_path=folder/'model-calls.jsonl'
        calls=[json.loads(line) for line in calls_path.read_text().splitlines()] if calls_path.exists() else []
        if not snapshots:continue
        case=json.loads((folder/'case.json').read_text());task_id='delivery_'+case['family']
        for p in (source,folder/'case.json'):
            track(p)
        if calls_path.exists():track(calls_path)
        frames=[]
        for index,path in enumerate(snapshots):
            track(path);snapshot=json.loads(path.read_text());views=[]
            for number,name in enumerate(('Global map','Local map','First-person RGB')):
                original=path.with_name(path.stem+'-'+str(number)+'.png')
                if not original.exists():continue
                track(original);raw=original.read_bytes();key=hashlib.sha256(raw).hexdigest();target='assets/'+key[:24]+'.png'
                (output/target).write_bytes(raw);assets[target]=key;views.append({'name':name,'image':target})
            public=json.loads(snapshot['prompt']);observation=dict(public['state']);observation['instruction']=case['public']['instruction']
            frame={'step':index,'image':views[-1]['image'],'views':views,'observation':observation,'action':None,
                   'reward':0,'terminated':False,'truncated':False}
            if index and index-1<len(calls):
                call=calls[index-1];frame['action']=call.get('command');frame['model_output']=call.get('model_output','')
                frame['model_input']={'input_step':call['input_step'],'text':call['request']['messages'][0]['content'][0]['text'],
                    'image_streams':['Global map','Local map','First-person RGB'],'image_observation_step':index-1,'views':[]}
                images=[part['image_url']['url'] for part in call['request']['messages'][0]['content'] if part['type']=='image_url']
                for name,value in zip(frame['model_input']['image_streams'],images):
                    raw=base64.b64decode(value.split(',',1)[1]);key=hashlib.sha256(raw).hexdigest();target='assets/'+key[:24]+'.png'
                    (output/target).write_bytes(raw);assets[target]=key
                    frame['model_input']['views'].append({'name':name,'image':target})
            elif index:
                trace=[json.loads(line) for line in (folder/'native-trace.jsonl').read_text().splitlines()]
                frame['action']=trace[index]['command']
            frames.append(frame)
        model=bool(calls);agent='Qwen3.8-27B' if model else 'oracle'
        score=validation.get('evaluation',{});success=bool(score.get('success'))
        frames[-1].update(terminated=success,truncated=not success)
        episode_id=folder.parent.name
        tasks.setdefault(task_id,{'id':task_id,'title':'Delivery (DeliveryBench)' if case['family']=='pickup_delivery' else 'Delivery · '+case['family'].replace('_',' '),
            'category':'Native UE courier interaction','summary':'Fixed orders, native preparation, packing, travel and deadline verification.',
            'objectives':['Accept the assigned order','Travel, wait, pick up and pack','Deliver with the required method before the deadline'],
            'environment':'simworld_delivery','episode_ids':[],'status':'recorded',
            'research':{'definition':'A fixed courier case over native DeliveryBench APIs.',
                'objective':case['public']['instruction'],'actions':' '.join(case['public']['api'].values()),
                'verification':'Native acceptance, pickup, bag placement and delivery sequence; actual UE proximity, pose synchronization, required method and case-relative deadline.',
                'difficulty':'Structural L1–L3 vary route-length sampling. `MOVE` uses the native planner. Current case difficulty is not calibrated; reference feasibility and model outcomes are separate.',
                'limits':'Development cases, not a benchmark or learning gain. Native Gym reward is constant zero; strict success comes from the fixed-case verifier. First-person images have no added object IDs.',
                'factors':[['Initial pose and battery','Fixed fixture',str(case['initial'])],['Budgets','Task constraints',str(case['budget'])]]}})
        tasks[task_id]['episode_ids'].append(episode_id)
        episodes.append({'id':episode_id,'task':task_id,'agent':agent,'scene':case['map'],'seed':case['seed'],'difficulty':case['level'],
            'budget':case['budget']['actions'],'instruction':case['public']['instruction'],'frames':frames,
            'difficulty_design':{**case['structural_difficulty'],'budget':case['budget'],'model_call_cap':16 if model else None,
                'deadline_s':[o['deadline_s'] for o in case['public']['orders']],'initial_battery_pct':case['initial']['battery_pct']},
            'result':{'success':success,'metrics':score,'reward':0,'terminated':success,'truncated':not success},
            'provenance':{'directory':str(folder.relative_to(root)),'policy':agent,'privileged_policy':not model,
                'collection':'fixed courier cases','recording_complete':True,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}})
        process_path=folder/'process/manifest.json'
        if process_path.exists():
            import shutil
            track(process_path);process=json.loads(process_path.read_text())
            samples=folder/'process'/process['samples'];track(samples)
            streams=[]
            for stream in process['streams']:
                video=folder/stream['path'];track(video);key=hashlib.sha256(video.read_bytes()).hexdigest()
                relative='static/media/'+episode_id+'-'+key[:12]+'.mp4';target=output/relative;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(video,target);assets[relative]=key;streams.append({'name':stream['name'],'url':relative})
            episodes[-1]['process']={'streams':streams,'samples':[json.loads(line) for line in samples.read_text().splitlines()],
                'time_offset_s':process['time_offset_s'],'target_interval_ms':process['target_interval_ms']}
    return list(tasks.values()),episodes,assets
