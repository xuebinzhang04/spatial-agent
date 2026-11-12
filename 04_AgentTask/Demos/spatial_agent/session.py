"""Serialized online environment interaction and observation-only model inference."""
from __future__ import annotations
import base64
import hashlib
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import re
import sys
import threading
from urllib.request import Request, urlopen
from urllib.parse import urlparse
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def local_models(model_root=None):
    models = []
    for p in sorted((model_root or ROOT / '01_Models').iterdir()):
        if not p.is_dir() or not (p/'config.json').is_file(): continue
        index = p/'model.safetensors.index.json'
        shards = set(json.loads(index.read_text())['weight_map'].values()) if index.exists() else [s.name for s in p.glob('*.safetensors')]
        downloaded = bool(shards) and all((p/s).is_file() and (p/s).stat().st_size>0 for s in shards)
        models.append({'id':p.name, 'name':p.name, 'checkpoint':str(p.relative_to(ROOT)), 'downloaded':downloaded})
    return models


def png(value):
    from PIL import Image
    buffer = io.BytesIO()
    Image.fromarray(value).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def jsonable(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(type(value).__name__)


class TaskAdapter:
    def __init__(self, kind, task, seed, difficulty, budget, scene, specification=None):
        self.kind, self.task, self.seed, self.budget = kind, task, seed, budget
        self.difficulty, self.scene = difficulty, scene if kind == "native" else "procedural"
        sys.path.insert(0, str(ROOT / "04_AgentTask/TaskFrameworks"))
        from task_support import SingleTask
        tasks = {"shape3d": ("shape_identification", "pose_alignment"), "native": (
            "missing_object", "moved_object", "memory_maze", "dependency_doors", "relational_rearrangement", "dynamic_replanning")}
        if task not in tasks.get(kind, ()):
            raise ValueError("Unknown environment/task pair")
        if kind == "native" and not os.environ.get("SLURM_JOB_ID"):
            raise ValueError("Native simulator execution requires a Slurm allocation and its registered Python runtime")
        definition = SingleTask(ROOT / "04_AgentTask/CustomAgentTasks" / task)
        options = {"max_steps": budget}
        if kind == "native":
            options["scene_id"] = scene
        if specification is not None:
            expected = (definition.definition['environment'], task, seed, difficulty, budget)
            actual = (specification.environment, specification.task_type, specification.seed, specification.difficulty, specification.max_steps)
            if actual != expected or kind == "native" and specification.scene_id != scene:
                raise ValueError("Frozen recipe does not match requested task, simulator or parameters")
        spec = specification if specification is not None else definition.generate(seed=seed, difficulty=difficulty, **options)
        self.env = definition.make_env(spec)
        self.reward, self.terminated, self.truncated, self.steps = 0, False, False, 0
        try:
            self.observation, self.info = self.env.reset(seed=seed)
        except BaseException:
            self.env.close()
            raise
        self.actions = list(self.env.available_actions)

    def snapshot(self):
        obs = dict(self.observation)
        rgb = obs.pop("rgb", None)
        url = png(rgb if rgb is not None else self.env.render())
        obs = {k: obs.get(k, "") for k in ("instruction", "phase", "memory", "feedback", "action_history", "memory_checkpoint", "task_progress")}
        if self.kind == "native":
            obs["view_step"] = self.steps
        metrics = self.env.metrics if self.terminated or self.truncated else None
        return {"environment": "ai2thor" if self.kind=="native" else self.kind, "task": self.task, "seed": self.seed, "step": self.steps,
                "scene": self.scene, "difficulty": self.difficulty,
                "budget": self.budget,
                "views": [{"name": "Agent observation", "image": url}], "observation": obs, "actions": self.actions,
                "reward": self.reward, "action_valid": self.info.get("action_valid", True), "terminated": self.terminated, "truncated": self.truncated, "evaluation": metrics}

    def step(self, action):
        if self.kind == "native":
            raw = json.loads(action) if isinstance(action, str) else action
            if re.search(r'object_\d+|object_NNN', json.dumps(raw)):
                raise ValueError("Public native actions use view:STEP:X:Y visual references, not private object IDs")
        self.observation, self.reward, self.terminated, self.truncated, self.info = self.env.step(action)
        self.steps += 1
        result = self.snapshot()
        result["feedback"] = self.info.get("error") or self.observation.get("feedback") or ""
        result["action"] = action
        return result

    def close(self):
        self.env.close()


class DeliveryAdapter:
    def __init__(self, run, map_name, seed):
        if not os.environ.get("SLURM_JOB_ID"):
            raise ValueError("DeliveryBench requires a Slurm allocation and a running UE runtime")
        from delivery import Harness, TASK, isolated_checkout
        from types import SimpleNamespace
        if not (TASK / "maps" / map_name / "roads.json").is_file():
            raise ValueError("Unknown DeliveryBench map")
        checkout = isolated_checkout(run, map_name)
        sys.path.insert(0, str(checkout))
        sys.path.insert(0, str(checkout / "simworld"))
        os.chdir(checkout)
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        os.environ["DELIVERYBENCH_MULTI_AGENT"] = "0"
        from vlm_delivery.gym_like_interface.gym_like_interface import DeliveryBenchGymEnv
        self.env = DeliveryBenchGymEnv(base_dir=str(checkout), map_name=map_name, max_steps=100,
                                      action_timeout_s=900, enable_viewer=True, time_scale=1.0)
        self.env.bootstrap_qt()
        self.env.reset(seed=seed)
        self.harness = Harness(self.env, run, SimpleNamespace(map=map_name, seed=seed))
        self.harness.save(self.harness.observe())
        from imports import city_data
        self.api = city_data(ROOT)["api"]
        self.steps = 0
        self.last_transition = {}

    def normalize(self, state):
        fields = ("position_cm", "energy_pct", "battery_pct", "earnings", "transport", "sim_time_s",
                  "active_orders", "order_pool", "prompt", "action_errors")
        obs = {k: state.get(k) for k in fields}
        obs["instruction"] = "Follow the native delivery instructions and API."
        transition = state.get("transition", {})
        return {"environment": "simworld_delivery", "task": "deliverybench", "seed": state["seed"], "step": self.steps,
                "budget": 100, "scene": state["map"],
                "views": [{"name": name, "image": value} for name, value in zip(
                    ("Global map", "Local map", "First-person RGB"), state["images"]) if value],
                "observation": obs, "actions": self.api, "reward": transition.get("reward", 0),
                "terminated": transition.get("terminated", False), "truncated": transition.get("truncated", False),
                "feedback": state.get("action_errors") or "", "evaluation": None}

    def snapshot(self):
        state = self.harness.observe()
        state["transition"] = self.last_transition
        return self.normalize(state)

    def step(self, action):
        state = self.harness.step(action)
        self.last_transition = state.get("transition", {})
        self.steps += 1
        result = self.normalize(state)
        result["action"] = action
        return result

    def close(self):
        dm = self.env.dms[0] if self.env.dms else None
        if dm:
            dm._recorder.mark_end(dm.clock.now_sim())
            dm._recorder.export(dm)
        self.env.close()


class Session:
    def __init__(self, run, model_url=None, model=None, transport=None):
        self.run = Path(run).resolve()
        self.run.mkdir(parents=True, exist_ok=True)
        self.model_url, self.model, self.transport = model_url, model, transport
        self.model_verified = False
        if bool(model_url) != bool(model):
            raise ValueError("Both --model-url and --model are required for inference")
        self.lock = threading.Lock()
        self.adapter, self.last, self.history = None, None, []
        self.total_reward = 0
        self.metadata = {"run_id": self.run.name, "project_id": "demo-session-" + uuid.uuid4().hex,
            "project_name": "Unified SpatialAgent interaction", "task_name": "Online environment session", "task_id": "demo-live",
            "purpose": "Manual and model-driven interaction through the shared demo API", "dataset": {"tasks": [], "materials": []},
            "experiment_type": "Interactive session; no benchmark claim", "model": {"name": model or "Manual", "checkpoint": "Remote service; not loaded by demo", "precision": "Not recorded"},
            "environment": {"name": "Not selected", "version": "Task-declared runtime"},
            "source": {"entrypoint": "04_AgentTask/Demos/spatial_agent/run.py", "commit": "Not recorded"},
            "parameters": {"model_url": model_url}, "slurm": {"job_id": os.environ.get("SLURM_JOB_ID", "Local lightweight session"),
                "state": "RUNNING", "started_at": datetime.now(timezone.utc).isoformat()},
            "artifacts": {"result_dir": str(self.run), "trajectory": "session.jsonl", "model_calls": "model_calls.jsonl"},
            "notes": "Only public observations/actions enter inference; evaluator output is excluded. Manual reset and model actions are logged."}
        self.metadata["source"]["sha256"] = {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ("session.py", "run.py", "rollout.py", "delivery.py", "research.json")}
        previous = self.run / "run_metadata.json"
        if previous.exists():
            prepared = json.loads(previous.read_text())
            self.metadata["project_id"] = prepared.get("project_id", self.metadata["project_id"])
            self.metadata["slurm"] = {**prepared.get("slurm", {}), **self.metadata["slurm"]}
            self.metadata["parameters"] = {**prepared.get("parameters", {}), **self.metadata["parameters"]}
        self.save_metadata()

    def save_metadata(self):
        pending = self.run / 'run_metadata.pending'
        pending.write_text(json.dumps(self.metadata, indent=2) + "\n")
        pending.replace(self.run / 'run_metadata.json')

    def log(self, name, value):
        with (self.run / name).open("a") as file:
            file.write(json.dumps(value, default=jsonable) + "\n")

    def capabilities(self):
        return {"online": True, "read_only": False, "models": local_models(), "model": self.model,
                "model_url": self.model_url, "model_verified": self.model_verified,
                "environment": self.last.get("environment") if self.last else None,
                "task": self.last.get("task") if self.last else None}

    def connect_model(self, model, url):
        parsed = urlparse(url or '')
        if not isinstance(model, str) or not model or parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('Provide a model ID and an HTTP model-service base URL without credentials')
        url = url.rstrip('/')
        request = Request(url + '/models', headers={'Authorization': 'Bearer ' + os.environ.get('SPATIAL_MODEL_API_KEY', 'local-unused')})
        with urlopen(request, timeout=10) as response:
            available = json.load(response)
        if model not in [entry.get('id') for entry in available.get('data', [])]:
            raise ValueError('Selected model is not served by this endpoint')
        self.model, self.model_url, self.model_verified = model, url, True
        self.metadata['model']['name'] = model
        checkpoint = next((m['checkpoint'] for m in local_models() if m['id']==model), 'Remote service')
        self.metadata['model']['checkpoint'] = checkpoint
        self.metadata['parameters']['model_url'] = url
        self.save_metadata()
        self.log('session.jsonl', {'event':'model_connected', 'model':model, 'model_url':url})
        return self.capabilities()

    def publish(self):
        if self.last:
            pending = self.run / 'latest_state.pending'
            pending.write_text(json.dumps(self.last, default=jsonable))
            pending.replace(self.run / 'latest_state.json')

    def reset(self, config, specification=None):
        kind, task = config.get("environment", "shape3d"), config.get("task", "shape_identification")
        kind = {"ai2thor": "native", "simworld_delivery": "deliverybench"}.get(kind, kind)
        if self.adapter and isinstance(self.adapter, DeliveryAdapter):
            raise ValueError("Restart the allocated DeliveryBench session to reset or change its map")
        if self.adapter:
            self.adapter.close()
        self.adapter = None
        if kind == "deliverybench":
            raise ValueError("Start DeliveryBench with --environment deliverybench on the Qt main thread")
        self.adapter = TaskAdapter(kind, task, int(config.get("seed", 42)), int(config.get("difficulty", 1)),
                                   int(config.get("budget", 50)), config.get("scene", "FloorPlan1"), specification=specification)
        self.history = []
        self.total_reward = 0
        self.metadata["dataset"]["tasks"] = [task]
        simulator_id = "ai2thor" if kind=="native" else kind
        simulator = next(e for e in json.loads((HERE/"research.json").read_text())["simulators"] if e["id"]==simulator_id)
        self.metadata["environment"] = {key: simulator[key] for key in ("id", "name", "runtime", "revision")}
        self.metadata["task_name"] = task
        self.metadata["task_id"] = f"{simulator_id}-{task}-seed{config.get('seed',42)}"
        self.metadata["parameters"].update(config)
        if specification is not None:
            self.metadata["parameters"]["frozen_recipe"] = specification.to_dict()
        self.metadata["parameters"]["budget"] = self.adapter.budget
        self.save_metadata()
        self.last = self.adapter.snapshot()
        self.last["total_reward"] = self.total_reward
        self.log("session.jsonl", {"event": "reset", "config": config, "state": self.last})
        self.publish()
        return self.last

    def state(self):
        if not self.adapter:
            raise ValueError("Reset an environment first")
        latest = self.adapter.snapshot()
        for key in ("action", "policy", "feedback", "model_input", "model_output"):
            if self.last and key in self.last:
                latest[key] = self.last[key]
        self.last = latest
        self.last["total_reward"] = self.total_reward
        return self.last

    def step(self, action, policy="Manual API"):
        if not self.adapter:
            raise ValueError("Reset an environment first")
        if self.last and (self.last["terminated"] or self.last["truncated"]):
            raise ValueError("Episode ended; reset before another action")
        self.last = self.adapter.step(action)
        self.last["policy"] = policy
        self.total_reward += self.last["reward"]
        self.last["total_reward"] = self.total_reward
        self.history.append(action)
        self.log("session.jsonl", {"event": "step", "state": self.last})
        self.publish()
        return self.last

    def infer(self, execute=False):
        if not self.model_url or not self.model:
            raise ValueError("Server has no configured model endpoint")
        state = self.state()
        if state["terminated"] or state["truncated"]:
            raise ValueError("Episode ended; reset before inference")
        instruction = "Choose one action using only public observations. Return exactly one JSON action object."
        if state["environment"] == "simworld_delivery":
            instruction += ' For DeliveryBench return {"command":"ONE_NATIVE_API_CALL()"}.'
        public = {"observation": state["observation"], "available_actions": state["actions"], "your_actions": self.history[-12:]}
        content = [{"type": "text", "text": instruction + "\n" + json.dumps(public, default=jsonable)}]
        content += [{"type": "image_url", "image_url": {"url": view["image"]}} for view in state["views"]
                    if view["image"].startswith(("data:image/png", "data:image/jpeg"))]
        payload = {"model": self.model, "temperature": 0, "max_tokens": 1024, "messages": [{"role": "user", "content": content}]}
        if self.model.startswith('Qwen'):
            payload['chat_template_kwargs'] = {'enable_thinking': False}
        try:
            if self.transport:
                response = self.transport(payload)
            else:
                request = Request(self.model_url.rstrip("/") + "/chat/completions", data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json", "Authorization": "Bearer " + os.environ.get("SPATIAL_MODEL_API_KEY", "local-unused")})
                with urlopen(request, timeout=120) as result:
                    response = json.load(result)
            if response.get("model") and response["model"] != self.model:
                raise ValueError("Model endpoint returned a different model identity")
            text = response["choices"][0]["message"]["content"]
            clean = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
            clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", clean).strip()
            value = json.loads(clean)
            if not isinstance(value, dict):
                raise ValueError("Model response must be one action object")
            action = value.get("command") if state["environment"] == "simworld_delivery" else value
            if state["environment"] == "simworld_delivery" and (set(value) != {"command"} or not isinstance(action, str)):
                raise ValueError("Expected one native command")
            result = {"model": self.model, "response": text, "action": action, "executed": False}
        except Exception as error:
            self.log("model_calls.jsonl", {"model": self.model, "error": str(error), "executed": False})
            raise
        if execute:
            try:
                result["state"] = self.step(action, self.model)
                result["executed"] = True
            except Exception as error:
                self.log("model_calls.jsonl", {**result, "error": str(error)})
                raise
        self.log("model_calls.jsonl", {k: v for k, v in result.items() if k != "state"})
        self.model_verified = True
        self.last['model_input'] = {**public, 'environment_step':state['step'], 'image_streams':[v['name'] for v in state['views']]}
        self.last['model_output'] = text
        self.publish()
        self.log('session.jsonl', {'event':'inference', 'state':self.last})
        return result

    def close(self):
        try:
            if self.adapter:
                self.adapter.close()
        finally:
            state = "FAILED" if self.metadata["slurm"].get("state") == "FAILED" else "COMPLETED"
            self.metadata["slurm"].update(state=state, ended_at=datetime.now(timezone.utc).isoformat())
            self.save_metadata()


class ObservedSession:
    """Read a worker's public events without loading a simulator or calling a model."""
    def __init__(self, run):
        self.run = Path(run).resolve()
        if not (self.run / 'run_metadata.json').is_file():
            raise ValueError('Observed run must contain run_metadata.json')
        self.lock = threading.Lock()
        self.last = None
        self.model = None

    def capabilities(self):
        metadata = json.loads((self.run / 'run_metadata.json').read_text())
        self.model = metadata['model']['name']
        latest = self.run / 'latest_state.json'
        if latest.exists(): self.last = json.loads(latest.read_text())
        return {'online':True, 'read_only':True, 'models':local_models(), 'model':self.model,
                'model_verified':False, 'model_url':None,
                'environment':self.last.get('environment') if self.last else None,
                'task':self.last.get('task') if self.last else None, 'run_id':self.run.name}

    def state(self):
        self.capabilities()
        if not self.last: raise ValueError('Worker has not published an observation yet')
        rows = []
        path = self.run / 'session.jsonl'
        if path.exists():
            for line in path.read_text().splitlines(keepends=True):
                if not line.endswith('\n'): break
                item = json.loads(line)
                if 'state' in item: rows.append(item)
        if rows: self.last = rows[-1]['state']
        status = json.loads((self.run / 'run_metadata.json').read_text())['slurm']['state']
        return {**self.last, 'events':rows, 'run_id':self.run.name, 'run_status':status}

    def log(self, name, value):
        pass  # Observation never modifies worker evidence.

    def close(self):
        pass
