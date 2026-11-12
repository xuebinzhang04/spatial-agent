"""Build an offline research interface from public, recorded episode data."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

from PIL import Image, ImageDraw, ImageFont
from session import local_models

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SCHEMA = "spatial-agent-demo/1"
# Case cards show the same global-memory variables the generation audit checks.
sys.path.insert(0, str(ROOT / "scripts/experiments"))
from task_generation_audit import recorded_case  # noqa: E402


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def observation_video(episode, output):
    """Encode only recorded observation pixels; explanatory text stays in the DOM."""
    key = hashlib.sha256(json.dumps({'images':[f['image'] for f in episode['frames']],
                                    'fps':2,'final_hold_s':2,'max_size':640}).encode()).hexdigest()[:16]
    target = output/'static/media'/(episode['id']+'-view-'+key+'.mp4')
    target.parent.mkdir(parents=True,exist_ok=True)
    with Image.open(output/episode['frames'][0]['image']) as first:
        scale = min(1,640/max(first.size))
        size = tuple(max(2,int(axis*scale)//2*2) for axis in first.size)
    if not target.exists():
        pending = target.with_suffix('.pending.mp4')
        ffmpeg = ROOT.parent/'conda/envs/spatialagent/bin/ffmpeg'
        command = [str(ffmpeg),'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24',
                   '-video_size',str(size[0])+'x'+str(size[1]),'-framerate','2','-i','pipe:0','-an',
                   '-c:v','libx264','-threads','2','-preset','veryfast','-crf','18','-pix_fmt','yuv420p',
                   '-movflags','+faststart',str(pending)]
        process = subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            for frame in episode['frames']:
                with Image.open(output/frame['image']) as source:
                    rgb = source.convert('RGB')
                    if rgb.size != size: rgb = rgb.resize(size,Image.Resampling.LANCZOS)
                    pixels = rgb.tobytes()
                    process.stdin.write(pixels)
            for _ in range(4): process.stdin.write(pixels)
            process.stdin.close()
            error = process.stderr.read().decode()
            if process.wait() != 0: raise RuntimeError(error)
            pending.replace(target)
        finally:
            if process.poll() is None: process.kill();process.wait()
    return {'url':str(target.relative_to(output)),'presentation':'observation-only','width':size[0],'height':size[1],
            'caption':('State-verified rendering of original actions; not a new rollout. ' if episode['provenance'].get('display_render') else 'Recorded '+episode['agent']+' observations. ')+
                       '2 discrete steps/s, not wall-clock speed. Exact source episode: '+episode['id']}


def research_catalog(catalog):
    research = read(HERE / "research.json")
    catalog["environments"] = research["simulators"]
    for task in catalog["tasks"]:
        if task["id"] == "deliverybench":
            task["environment"] = "simworld_delivery"
            task["title"] = "Delivery integration cutoff (historical snapshots)"
            task["online"] = {"environment": "deliverybench", "task": "deliverybench"}
        elif task["id"] in research["tasks"]:
            task["online"] = {"environment": "shape3d" if task["environment"]=="shape3d" else "native", "task": task["id"]}
        task.setdefault("research", research["tasks"].get(task["id"], {
            "definition": task["summary"], "objective": " ".join(task["objectives"]), "factors": [],
            "verification": "No accepted recording for this native task adapter.", "difficulty": "Not calibrated.",
            "actions": "Native adapter pending acceptance.", "limits": "Task definition only; not a verified native example."}))
        task["label"] = (task['title'] if task["id"]=="deliverybench" or task['id'].startswith('delivery_')
                         else task["id"].replace("_", " ").title())


NAVIGATION = {"MoveAhead", "MoveBack", "RotateLeft", "RotateRight", "LookUp", "LookDown"}
DIRECTIONS = {("x", 1): "east of", ("x", -1): "west of", ("z", 1): "north of", ("z", -1): "south of"}


def key_steps(episode):
    """Steps whose action is a task action (not navigation or camera motion), with its name."""
    return [(i, frame["action"]["action"]) for i, frame in enumerate(episode["frames"])
            if frame.get("action") and frame["action"]["action"] not in NAVIGATION
            and not frame["action"]["action"].startswith("CAMERA_")]


def phase_label(episode, task, phase):
    return (episode.get('presentation_phase_notes', {}).get(phase) or task.get('research', {}).get('phase_labels', {}).get(phase)
            or phase.replace('_', ' ').title())


def case_gif(episode, task, output):
    """Sampled GIF of one showcase episode: key actions, phase changes and evenly spaced steps."""
    frames = episode['frames']
    changes = [i for i in range(1, len(frames)) if frames[i]['observation'].get('phase') != frames[i-1]['observation'].get('phase')]
    events = [i for i, _ in key_steps(episode)]
    indices = sorted(set([0, len(frames) - 1] + [j for i in events for j in (i - 1, i) if j >= 0] +
                         [j for i in changes for j in (i - 1, i)] + [round(i * (len(frames) - 1) / 11) for i in range(12)]))
    animation, durations = [], []
    face = ImageFont.truetype('/usr/share/fonts/cantarell/Cantarell-Bold.otf', 28)
    for index in indices:
        with Image.open(output / frames[index]['image']) as source:
            source_size = source.size
            rgb = source.convert('RGB')
        scale = min(640 / rgb.width, 640 / rgb.height)
        rgb = rgb.resize((round(rgb.width * scale), round(rgb.height * scale)), Image.Resampling.LANCZOS)
        canvas = Image.new('RGB', (640, 640), 'white')
        canvas.paste(rgb, ((640 - rgb.width) // 2, (640 - rgb.height) // 2))
        label = phase_label(episode, task, frames[index]['observation'].get('phase', 'interaction'))
        draw = ImageDraw.Draw(canvas)
        width = draw.textlength(label, font=face)
        draw.rectangle((630 - width - 16, 8, 632, 54), fill='#ffffff')
        draw.text((630 - width - 8, 11), label, font=face, fill='#222222')
        animation.append(canvas.quantize(colors=256))
        durations.append(3200 if index == 0 or index in changes or index + 1 in changes or index in events else 1600)
    durations[-1] = 4000
    pending = output / 'static/media' / (episode['task'] + '.pending.gif')
    animation[0].save(pending, save_all=True, append_images=animation[1:], duration=durations, loop=0, disposal=2)
    gif = pending.with_name(episode['task'] + '-' + digest(pending)[:12] + '.gif')
    pending.replace(gif)
    return str(gif.relative_to(output)), {
        'task': episode['task'], 'episode': episode['id'], 'status': 'recorded', 'displayed': True, 'steps': indices,
        'durations_ms': durations, 'added_annotations': True, 'annotation_kind': 'phase_only_top_right',
        'source_resolution_px': list(source_size), 'object_number_overlays': False, 'asset': str(gif.relative_to(output))}


def case_facts(templates, variables):
    """Fill the task's fact templates with this case's global-memory variables."""
    values = {}
    for name, value in variables.items():
        if isinstance(value, bool):
            value = "yes" if value else "no"
        elif isinstance(value, list) and name != "relation_axes":
            value = (" → " if name in ("landmark_types", "object_types") else ", ").join(map(str, value))
        values[name] = value
    if "relation_axes" in variables:
        values["relation"] = " and ".join(DIRECTIONS[tuple(axis)] for axis in variables["relation_axes"])
        values["near"] = f", within {variables['near_metres']} m" if variables.get("near_metres") else ""
    return [(label, template.format(**values)) for label, template in templates]


# Cards that add a before/after comparison of public frames around the evidence-adding camera path.
EVIDENCE_COMPARISON = {"shape_identification"}


def camera_path(episode):
    """Recorded camera actions before the first task action, the frame after them, and the task action."""
    frames = episode['frames']
    first_task = next((i for i, _ in key_steps(episode)), None)
    if first_task is None or first_task < 2:
        raise ValueError(f"No camera path precedes a task action: {episode['id']}")
    path = [frames[i]['action']['action'] for i in range(1, first_task)]
    if not all(name.startswith('CAMERA_') for name in path):
        raise ValueError(f"Non-camera action before the first task action: {episode['id']}")
    runs = []
    for name in path:
        if runs and runs[-1][0] == name:
            runs[-1][1] += 1
        else:
            runs.append([name, 1])
    return path, runs, first_task - 1, frames[first_task]['action']['action']


def recorded_witness(directory):
    """Build-time only: the generator's revealing camera actions from the step-0 privileged state.

    The result is compared against the public trajectory and never written to the page or catalog."""
    with (directory / 'privileged.jsonl').open() as handle:
        witness = json.loads(handle.readline())['state'].get('ambiguity_witness')
    return witness['actions'] if witness else None


def evidence_comparison(episode, variables, witness):
    """Public step 0 beside the public view reached by the recorded camera path, with provenance."""
    path, runs, after, submit = camera_path(episode)
    frames = episode['frames']
    # Exact match: every recorded action (name and arguments) equals the generator's witness, in order.
    if [frames[i]['action'] for i in range(1, after + 1)] != witness or len(path) != variables.get('witness_actions'):
        raise ValueError(f"Recorded camera path differs from the generator's revealing path: {episode['id']}")
    route = ' → '.join(f'<code>{escape(name)}</code>&nbsp;×{count}' for name, count in runs)
    twins = variables.get('initial_pixel_twins', 0)
    policy = ('Privileged oracle reference, not a VLM result: the policy replays the generator-verified camera path '
              'and submits the evaluator answer.' if episode['provenance']['privileged_policy']
              else f"Recorded {escape(episode['agent'])} policy.")
    return (f'<figure class="case-evidence"><div class="evidence-pair">'
            f'<div><img src="{escape(frames[0]["image"])}" alt="Step 0 public observation" loading="lazy">'
            f'<p><strong>Before · step 0</strong> Initial public view. The generator checked that {twins} wrong '
            f'candidate{"s" if twins != 1 else ""} placed at the target pose render{"" if twins != 1 else "s"} this '
            f'identical image, so this view alone cannot settle the answer.</p></div>'
            f'<div><img src="{escape(frames[after]["image"])}" alt="Step {after} public observation" loading="lazy">'
            f'<p><strong>After · step {after}</strong> Same recorded trajectory after {len(path)} camera actions. '
            f'The moved camera shows side faces and depth that the front projection hid.</p></div></div>'
            f'<figcaption><p class="evidence-path">Camera path, steps 1–{after}: {route} → '
            f'<code>{escape(submit)}</code> at step {after + 1}. This recorded path matches, action for action, the '
            f'camera path the generator found when it checked that, from the resulting view, the target and one '
            f'initial twin candidate differ in at least 10% of foreground pixels.</p>'
            f'<p class="evidence-label">{policy}</p></figcaption></figure>')


def case_card(episode, task, example, variables, gif, witness=None):
    """Static HTML for one task: question, design, phase timeline, this case's variables, success test."""
    frames = episode['frames']
    segments, start = [], 0
    for i in range(1, len(frames) + 1):
        if i == len(frames) or frames[i]['observation'].get('phase') != frames[start]['observation'].get('phase'):
            segments.append((phase_label(episode, task, frames[start]['observation'].get('phase', 'interaction')), start, i - 1))
            start = i
    total = max(1, len(frames) - 1)
    bar = ''.join(f'<span style="flex:{max(1, end - first)}"><strong>{escape(label)}</strong> steps {first}–{end}</span>'
                  for label, first, end in segments)
    keys = key_steps(episode)
    shown = keys if len(keys) <= 8 else keys[:4] + [(None, '…')] + keys[-3:]
    actions = ' '.join(f'<code>{escape(name)}</code>' + (f'<sub>{step}</sub>' if step is not None else '') for step, name in shown)
    navigation = total - len(keys)
    facts = ''.join(f'<dt>{escape(label)}</dt><dd>{escape(str(value))}</dd>' for label, value in case_facts(example['facts'], variables))
    level = f"L{episode['difficulty']}" + (f" · {escape(episode['scene'])}" if episode['scene'] != 'procedural' else ' · procedural')
    return (f'<article class="case-card" id="case-{escape(episode["task"])}">'
            f'<figure><img src="{escape(gif)}" alt="{escape(task["label"])} example case" loading="lazy"></figure>'
            f'<div class="case-body"><p class="case-kicker">{escape(task.get("category", ""))} · {level}</p>'
            f'<h3>{escape(task["label"])}</h3><p class="case-question">{escape(example["question"])}</p>'
            f'<p>{escape(example["design"])}</p>'
            + (evidence_comparison(episode, variables, witness) if episode['task'] in EVIDENCE_COMPARISON else '') +
            f'<div class="case-timeline">{bar}</div>'
            f'<p class="case-actions">{navigation} navigation/camera steps; task actions: {actions}</p>'
            f'<dl class="case-facts">{facts}</dl>'
            f'<p class="case-success"><strong>Success:</strong> {escape(example["verification"])}</p>'
            f'<a class="case-link" href="#explorer" data-task="{escape(episode["task"])}" data-episode="{escape(episode["id"])}">Step through this case →</a>'
            '</div></article>')


def generation_audits(config, track):
    """Bundle task-generation audits (see scripts/experiments/task_generation_audit.py)."""
    model_path = ROOT / "04_AgentTask/CustomAgentTasks/task_variables.json"
    track(model_path)
    audits = []
    for entry in config.get("generation_audits", []):
        # An entry is a run directory, or {"path": ..., "tasks": [...]} to show only some tasks of a run.
        directory, tasks = (entry, None) if isinstance(entry, str) else (entry["path"], set(entry["tasks"]))
        path = ROOT / directory / "task_generation_audit.json"
        track(path)
        audit = read(path)
        keep = lambda task: tasks is None or task in tasks  # noqa: E731
        audits.append({"run_id": Path(directory).name, "source": audit["source"], "slurm_job_id": audit.get("slurm_job_id"),
                       "generated_at": audit["generated_at"], "complete": audit.get("complete", True),
                       "summary": {t: s for t, s in audit["summary"].items() if keep(t)},
                       "shape_answer_balance": audit.get("shape_answer_balance"),
                       "cases": [{k: case.get(k) for k in ("task", "case", "source", "admitted", "reset_error",
                                                           "controls", "global_memory", "checks")}
                                 for case in audit["cases"] if keep(case["task"])]})
    return {"variable_model": read(model_path), "audits": audits}


def build(output, config_path, record):
    started_at = datetime.now(timezone.utc).isoformat()
    config = read(config_path)
    output.mkdir(parents=True, exist_ok=True)
    sources, assets = {}, {}

    def track(path):
        sources[str(path.relative_to(ROOT))] = digest(path)

    def image(source, expected_hash):
        track(source)
        key = sources[str(source.relative_to(ROOT))]
        target = Path("assets") / f"{key[:24]}.webp"
        if str(target) not in assets:
            with Image.open(source) as im:
                rgb = im.convert("RGB")
                if hashlib.sha256(rgb.tobytes()).hexdigest() != expected_hash:
                    raise ValueError(f"Recorded RGB hash mismatch: {source}")
                (output / target).parent.mkdir(exist_ok=True)
                reusable = False
                if (output / target).exists():
                    with Image.open(output / target) as cached:
                        reusable = hashlib.sha256(cached.convert("RGB").tobytes()).hexdigest() == expected_hash
                if not reusable:
                    rgb.save(output / target, "WEBP", lossless=True)
            assets[str(target)] = digest(output / target)
        return str(target)

    track(config_path)
    tasks = []
    definitions = sorted((ROOT / "04_AgentTask/CustomAgentTasks").glob("*/task.json"))
    seen = set()
    for path in definitions:
        d = read(path)
        if d.get("backend") != "spatial_interaction":
            continue
        task_id = d["name"]
        if task_id in seen:
            continue
        seen.add(task_id)
        track(path)
        presentation = config.get("tasks", {}).get(task_id, {})
        environment = "shape3d" if d.get("environment") == "shape3d" else "ai2thor"
        tasks.append({"id": task_id, "title": presentation.get("title", task_id.replace("_", " ").title()),
                      "category": presentation.get("category", "Spatial interaction"),
                      "summary": presentation.get("summary", "A custom spatial interaction task."),
                      "objectives": presentation.get("objectives", ["Complete the recorded task instruction"]),
                      "environment": environment, "definition": str(path.relative_to(ROOT)), "episode_ids": []})
    ordered = list(config.get("tasks", {}))
    tasks.sort(key=lambda t: ordered.index(t["id"]) if t["id"] in ordered else len(ordered))
    by_task = {t["id"]: t for t in tasks}

    model_root = ROOT / config.get("model_root", "01_Models")
    models = [{**m, 'kind':'model'} for m in local_models(model_root)]
    models.append({"id": "oracle", "name": "Oracle reference", "kind": "oracle", "checkpoint": None})
    model_ids = {m["id"] for m in models}
    paths = sorted({p for pattern in config["episode_globs"] for p in ROOT.glob(pattern)})
    episodes = []
    for trajectory in paths:
        folder = trajectory.parent
        if not (folder / "report.json").exists() or not (folder / "episode.json").exists():
            raise ValueError(f"Incomplete episode: {folder}")
        cfg, report = read(folder / "episode.json"), read(folder / "report.json")
        task_id = cfg["task_type"]
        if task_id not in by_task:
            raise ValueError(f"Missing task definition: {task_id}")
        for p in (trajectory, folder / "episode.json", folder / "report.json"):
            track(p)
        oracle = report.get("oracle_uses_privileged_state", report.get("policy") == "oracle")
        if oracle:
            agent_id = "oracle"
        else:
            # A model recording must name its actual model in the source run summary.
            summary = next((p / "summary.json" for p in folder.parents if (p / "summary.json").exists()), None)
            if summary is None or not read(summary).get("model"):
                raise ValueError(f"Model recording without model identity: {folder}")
            track(summary)
            agent_id = read(summary)["model"]
            if agent_id not in model_ids:
                models.append({"id": agent_id, "name": agent_id, "kind": "model", "checkpoint": None})
                model_ids.add(agent_id)
        calls = []
        if (folder / "model_calls.jsonl").exists():
            track(folder / "model_calls.jsonl")
            calls = [json.loads(line) for line in (folder / "model_calls.jsonl").read_text().splitlines() if line.strip()]
        rows = [json.loads(line) for line in trajectory.read_text().splitlines() if line.strip()]
        if not rows or rows[0]["step"] != 0 or not (rows[-1]["terminated"] or rows[-1]["truncated"]):
            raise ValueError(f"Incomplete trajectory: {trajectory}")
        if calls and len(calls) != len(rows) - 1:
            raise ValueError(f"Model call/action count mismatch: {folder}")
        frames = []
        for i, row in enumerate(rows):
            if row["step"] != i:
                raise ValueError(f"Non-contiguous steps: {trajectory}")
            observation = row["observation"]
            # The presentation adapter never loads ground_truth.json or privileged.jsonl.
            public = {key: observation.get(key, "") for key in ("instruction", "phase", "memory", "feedback", "view_step", "memory_checkpoint", "task_progress")}
            frame = {"step": i, "image": image(folder / row["frame"], row["frame_sha256"]),
                     "source_rgb_sha256": row["frame_sha256"], "action": row["action"],
                     "reward": row["reward"], "terminated": row["terminated"], "truncated": row["truncated"],
                     "observation": public}
            if i and calls:
                if calls[i - 1]["action"] != row["action"]:
                    raise ValueError(f"Model call/action mismatch: {folder} step {i}")
                frame["model_output"] = calls[i - 1].get("response", "")
                frame["latency_seconds"] = calls[i - 1].get("latency_seconds")
            frames.append(frame)
        episode_id = f"{task_id}-{agent_id}-{hashlib.sha256(str(folder.relative_to(ROOT)).encode()).hexdigest()[:8]}"
        episode = {"id": episode_id, "task": task_id, "agent": agent_id, "scene": cfg["scene_id"],
                   "seed": cfg["seed"], "difficulty": cfg["difficulty"], "budget": cfg["max_steps"],
                   "instruction": rows[0]["observation"]["instruction"], "frames": frames,
                   "result": {"success": bool(report["metrics"]["success"]), "metrics": report["metrics"],
                              "reward": report["reward"], "terminated": report["terminated"], "truncated": report["truncated"]},
                   "provenance": {"directory": str(folder.relative_to(ROOT)), "trajectory_sha256": digest(trajectory),
                                  "report_sha256": digest(folder / "report.json"), "policy": report.get("policy"),
                                  "privileged_policy": bool(oracle)}}
        controls=cfg.get('task_config',{})
        displayed=('exploration_budget','memory_protocol','min_target_pixels','max_target_pixels','max_target_distance',
                   'voxels','candidates','occlusion','rotation_axes','require_initial_pixel_ambiguity','image_size')
        episode['difficulty_design']={'level':cfg['difficulty'],'seed':cfg['seed'],'scene':cfg['scene_id'],
            'max_steps':cfg['max_steps'],'case_overrides':{k:controls[k] for k in displayed if k in controls},
            'empirically_calibrated':False,'note':'Overrides shown here; generator defaults and mechanisms are described above.'}
        episodes.append(episode)
        by_task[task_id]["episode_ids"].append(episode_id)

    for t in tasks:
        t["status"] = "recorded" if t["episode_ids"] else "no_recording"
    from imports import city_data, import_legacy
    legacy_tasks, legacy_episodes, legacy_models, legacy_assets, _ = import_legacy(ROOT, output, track)
    tasks.extend(legacy_tasks)
    episodes.extend(legacy_episodes)
    models.extend(legacy_models)
    assets.update(legacy_assets)
    from imports import import_delivery_cases
    delivery_tasks,delivery_episodes,delivery_assets=import_delivery_cases(ROOT,output,track)
    tasks.extend(delivery_tasks);episodes.extend(delivery_episodes);assets.update(delivery_assets)
    if not episodes:
        raise ValueError("No complete source episodes")
    for path in (output / "assets").glob("*.webp"):
        if str(Path("assets") / path.name) not in assets:
            path.unlink()
    vlm_ids = {m["id"] for m in models if m["kind"] == "model"}
    vlm = [e for e in episodes if e["agent"] in vlm_ids]
    catalog = {"schema": SCHEMA, "title": config.get("title", "SpatialAgent"),
               "built_at": datetime.now(timezone.utc).isoformat(), "tasks": tasks, "models": models,
               "environments": [{"id": "ai2thor", "name": "AI2-THOR", "description": "Native indoor scenes"},
                                {"id": "shape3d", "name": "Shape3D", "description": "Procedural 3D geometry"}],
               "episodes": episodes,
               "stats": {"recorded_tasks": sum(bool(t["episode_ids"]) for t in tasks), "episodes": len(episodes),
                         "model_episodes": len(vlm), "model_successes": sum(e["result"]["success"] for e in vlm)},
               "default_task": "missing_object", "default_agent": vlm[0]["agent"] if vlm else "oracle"}
    research_catalog(catalog)
    track(HERE / "research.json")
    for episode in episodes:
        if episode['task'] not in ('missing_object','moved_object'): continue
        path = ROOT/episode['provenance']['directory']/'ground_truth.json'
        track(path)
        evaluator = read(path)
        reverse = {stable:internal for internal,stable in evaluator['id_map'].items()}
        changes = evaluator['ground_truth']['changes']
        notes = []
        for change in changes:
            name = evaluator['initial_state'][reverse[change['object_id']]]['objectType']
            notes.append(name+' '+change['type'])
        # Presentation-only evaluator annotations never enter frame observations.
        episode['presentation_phase_notes'] = {'detection':'Discern: '+ '; '.join(notes)}
    clean_manifest = HERE / 'presentation/clean-frames.json'
    if clean_manifest.exists():
        track(clean_manifest)
        clean_entries = read(clean_manifest)
        for episode in episodes:
            clean = clean_entries.get(episode['id'])
            if not clean: continue
            if not clean.get('native_state_verified') or clean['source_trajectory_sha256'] != episode['provenance']['trajectory_sha256'] or len(clean['frames']) != len(episode['frames']):
                raise ValueError('Unverified or mismatched clean rendering: '+episode['id'])
            if clean.get('video') and digest(ROOT/clean['video']) != clean['video_sha256']:
                raise ValueError('Clean video hash mismatch: '+episode['id'])
            for frame, replacement in zip(episode['frames'],clean['frames']):
                frame['recorded_image'] = frame['image']
                frame['image'] = image(ROOT/replacement['path'],replacement['rgb_sha256'])
                frame['display_rgb_sha256'] = replacement['rgb_sha256']
            episode['provenance']['display_render'] = 'State-verified replay without object-number overlays; original policy observations remain archived'
            episode['clean_video'] = clean.get('video')
    videos = read(HERE / "presentation/examples.json")
    track(HERE / "presentation/examples.json")
    for item in videos["examples"]:
        episode = next((e for e in episodes if e["id"] == item["episode"]), None)
        if episode is not None:
            episode['video'] = observation_video(episode,output)
            assets[episode['video']['url']] = digest(output/episode['video']['url'])
    examples = read(HERE / "examples.json")
    track(HERE / "examples.json")
    track(ROOT / "scripts/experiments/task_generation_audit.py")
    case_cards, gif_sources = [], []
    for task_id, directory in config["showcase"].items():
        episode = next(e for e in episodes if e['provenance']['directory'] == directory)
        gif, source = case_gif(episode, by_task[task_id], output)
        assets[gif] = digest(output / gif)
        variables = recorded_case(ROOT / directory)['global_memory']
        witness = None
        if task_id in EVIDENCE_COMPARISON:
            track(ROOT / directory / 'privileged.jsonl')
            witness = recorded_witness(ROOT / directory)
        case_cards.append(case_card(episode, by_task[task_id], examples[task_id], variables, gif, witness))
        gif_sources.append(source)
    if len(case_cards) != 8:
        raise ValueError('The task overview requires eight showcase cases')
    dump(output / 'data/gif-examples.json', gif_sources)
    for stale in (output / 'static/media').glob('*.gif'):
        if str(stale.relative_to(output)) not in assets:
            stale.unlink()

    for episode in episodes:
        if not episode.get('clean_video') or episode.get('video'): continue
        episode['video'] = observation_video(episode,output)
        assets[episode['video']['url']] = digest(output/episode['video']['url'])
    delivery_episode = next(e for e in episodes if e['task']=='deliverybench')
    delivery_episode['video'] = observation_video(delivery_episode,output)
    assets[delivery_episode['video']['url']] = digest(output/delivery_episode['video']['url'])
    (output / "catalog.js").write_text("window.SPATIAL_CATALOG = " + json.dumps(catalog, ensure_ascii=False).replace("</", "<\\/") + ";\n")
    for name in ("index.html", "style.css", "app.js"):
        track(HERE / name)
        shutil.copy2(HERE / name, output / name)
    template = (output / "index.html").read_text()
    template = template.replace('<!-- TASK_CASES -->', '\n'.join(case_cards))
    (output / "index.html").write_text(template)
    generation = generation_audits(config, track)
    (output / "generation.js").write_text("window.SPATIAL_GENERATION = " + json.dumps(generation, ensure_ascii=False, default=str).replace("</", "<\\/") + ";\n")
    assets["generation.js"] = digest(output / "generation.js")
    cities = city_data(ROOT)
    (output / "cities.js").write_text("window.SPATIAL_CITIES = " + json.dumps(cities).replace("</", "<\\/") + ";\n")
    assets["cities.js"] = digest(output / "cities.js")
    task_root = ROOT / "04_AgentTask/SpatialAgentBench/DeliveryBench"
    for map_name in cities["maps"]:
        for name in ("roads.json", "buildings.json", "progen_world_enriched.json"):
            track(task_root / "maps" / map_name / name)
    track(task_root / "vlm_delivery/gameplay/action_space.py")
    track(task_root / "vlm_delivery/input/game_mechanics_config.json")
    track(ROOT / "02_Simulator/REGISTRY.md")
    track(ROOT / "05_Experiment/results/interaction-demo-345795/simulator.log")
    for name in ("imports.py", "session.py", "delivery.py", "run.py", "rollout.py", "rerender.py"):
        track(HERE / name)
    for source in sorted((HERE / "static").rglob("*")):
        if source.is_file():
            track(source)
            target = output / source.relative_to(HERE)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    for cached in (output/'static/media').iterdir():
        if cached.is_file() and str(cached.relative_to(output)) not in assets: cached.unlink()
    track(Path(__file__).resolve())
    track(HERE / "test_demo.py")
    dump(output / "data/catalog.json", catalog)
    dump(output / "data/evidence.json", {"schema": SCHEMA, "sources": sources, "web_assets": assets,
                                         "note": "Recorded RGB WebP copies are lossless; overview GIFs are sampled, palette-quantized derivatives. Target annotations are evaluator information for research presentation only, excluded from model input. No full privileged state dictionaries are included."})
    previous_record = read(record / "run_metadata.json") if (record / "run_metadata.json").exists() else {}
    dump(record / "run_metadata.json", {
        "run_id": record.name, "project_id": previous_record.get("project_id", "spatial-agent-demo-" + uuid.uuid4().hex), "project_name": "SpatialAgent research interface",
        "task_name": "Project demo interface build", "task_id": "spatial-agent-demo-v1",
        "purpose": "Build a unified research homepage and recorded interaction explorer from existing evidence",
        "dataset": {"name": "Existing SpatialAgent recorded development episodes", "tasks": [t["id"] for t in tasks],
                    "materials": [str(p.relative_to(ROOT)) for p in paths]},
        "experiment_type": "Project demo build; no new model or simulator run",
        "model": {"name": "Recorded policies only; local checkpoint catalog", "precision": "Not applicable"},
        "environment": {"name": "Offline browser replay", "version": SCHEMA, "python": sys.version},
        "source": {"entrypoint": str(Path(__file__).relative_to(ROOT)), "config": str(config_path.relative_to(ROOT)), "sha256": sources},
        "parameters": {"output": str(output.relative_to(ROOT)), "episode_count": len(episodes), "frame_count": sum(len(e["frames"]) for e in episodes)},
        "slurm": {"job_id": "Not applicable: light local build", "partition": "Not applicable", "resources": "One CPU process, no GPU",
                  "state": "COMPLETED", "exit_code": "0", "created_at": previous_record.get("slurm", {}).get("created_at", started_at),
                  "started_at": started_at, "ended_at": datetime.now(timezone.utc).isoformat(), "failure_reason": "Not applicable"},
        "artifacts": {"result_dir": str(record.relative_to(ROOT)), "report": str(output.relative_to(ROOT) / "index.html"), "primary_metrics": "validation.json"},
        "notes": "Source RGB hashes checked. Recorded model and oracle policies are separate. No RL or generalization claim."})
    print(json.dumps({"output": str(output), "tasks": catalog["stats"], "frames": sum(len(e["frames"]) for e in episodes), "assets": len(assets)}))
    return catalog


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "site")
    parser.add_argument("--config", type=Path, default=HERE / "demo.json")
    parser.add_argument("--record", type=Path, default=ROOT / "05_Experiment/results" / ("spatial-agent-demo-build-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")))
    args = parser.parse_args()
    build(args.output.resolve(), args.config.resolve(), args.record.resolve())
