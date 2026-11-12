from __future__ import annotations

import json
import hashlib
import os
import re
import secrets
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse


ROOT = Path(os.environ.get("SPATIAL_ROOT", Path(__file__).resolve().parents[2]))
APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
STATE_FILE = APP_DIR / "monitor-state.json"
BARK_FILE = APP_DIR / "bark_url.txt"
RESULTS_DIR = ROOT / "05_Experiment" / "results"
LOGS_DIR = ROOT / "05_Experiment" / "logs"
POLL_SECONDS = max(8, int(os.environ.get("QUEUE_INSPECTOR_POLL_SECONDS", "12")))
HISTORY_DAYS = 30
MAX_HISTORY = 500
SACCT_CACHE_SECONDS = 60
sacct_cache: list[dict[str, Any]] = []
sacct_cache_time = 0.0

app = FastAPI(title="SpatialAgent Queue Inspector", docs_url=None, redoc_url=None)
lock = threading.RLock()
snapshot: dict[str, Any] = {
    "updated_at": None,
    "user": os.environ.get("USER", "unknown"),
    "jobs": [],
    "errors": [],
    "bark_configured": BARK_FILE.exists(),
    "poll_seconds": POLL_SECONDS,
}
previous_states: dict[str, str] = {}
hidden_jobs: set[str] = set()
pending_deletions: dict[str, tuple[str, str, str, float]] = {}
if STATE_FILE.exists():
    try:
        saved_state = json.loads(STATE_FILE.read_text())
        previous_states = saved_state.get("states", {})
        hidden_jobs = set(map(str, saved_state.get("hidden_jobs", [])))
    except (OSError, json.JSONDecodeError, AttributeError):
        previous_states = {}


def run_slurm(args: list[str], timeout: int = 12) -> str:
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode:
        message = (result.stderr or result.stdout).strip()
        raise RuntimeError(message or f"{args[0]} exited with {result.returncode}")
    return result.stdout


def slurm_jobs() -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    active: dict[str, dict[str, Any]] = {}
    username = os.environ.get("USER", "")
    fmt = "%i|%j|%T|%M|%l|%D|%N|%C|%R|%V|%S|%e|%P|%q|%Q|%Z|%o|%m"
    try:
        output = run_slurm(["squeue", "--user", username, "--noheader", "--format=" + fmt])
        for line in output.splitlines():
            parts = line.split("|", 17)
            if len(parts) != 18:
                continue
            (job_id, name, state, elapsed, limit, node_count, node_list, cpus, reason, submitted,
             expected_start, expected_end, partition, qos, priority, workdir,
             command, memory) = parts
            active[job_id] = {
                "id": job_id, "name": name, "state": state.upper(), "elapsed": elapsed,
                "time_limit": limit, "node_count": node_count, "node_list": node_list, "cpus": cpus,
                "reason": reason, "submitted": submitted, "expected_start": expected_start,
                "expected_end": expected_end, "partition": partition, "qos": qos,
                "priority": priority, "workdir": workdir, "command": command,
                "memory": memory,
            }
    except (OSError, subprocess.TimeoutExpired, RuntimeError) as exc:
        errors.append(f"squeue: {exc}")

    global sacct_cache, sacct_cache_time
    start = f"now-{HISTORY_DAYS}days"
    fields = "JobIDRaw,JobID,JobName,State,ExitCode,Elapsed,Timelimit,Start,End,AllocTRES,ReqTRES,Partition,Reason,StdOut,SubmitLine,WorkDir,Submit,NodeList"
    if time.time() - sacct_cache_time >= SACCT_CACHE_SECONDS:
        try:
            output = run_slurm([
                "sacct", "--user", username, "--starttime", start, "--noheader", "--parsable2",
                "--allocations", "--format=" + fields,
            ])
            recent: dict[str, dict[str, Any]] = {}
            for line in output.splitlines():
                parts = line.split("|", 17)
                if len(parts) < 18 or "." in parts[0]:
                    continue
                (job_id, array_id, name, state, exit_code, elapsed, limit, started, ended, alloc_tres,
                 req_tres, partition, reason, stdout, submit_line, workdir, submitted, node_list) = parts
                recent[job_id] = {
                    "id": job_id, "array_id": array_id, "name": name, "state": normalize_state(state),
                    "exit_code": exit_code, "elapsed": elapsed, "time_limit": limit,
                    "started": started, "ended": ended, "alloc_tres": alloc_tres,
                    "req_tres": req_tres, "partition": partition, "reason": reason,
                    "stdout": stdout, "command": submit_line, "workdir": workdir,
                    "submitted": submitted, "node_list": node_list,
                }
            sacct_cache = list(recent.values())
            sacct_cache_time = time.time()
        except (OSError, subprocess.TimeoutExpired, RuntimeError) as exc:
            errors.append(f"sacct: {exc}")
    if sacct_cache:
        # squeue is the live source of truth; account records add finished jobs and details.
        recent = {str(job["id"]): job for job in sacct_cache}
        for job_id, live in active.items():
            recent[job_id] = {**recent.get(job_id, {}), **live}
        active = recent

    jobs = [enrich_job(job) for item in active.values() for job in expand_array_job(item)]
    # squeue can show a compact pending array range while sacct already has
    # concrete records for some elements. Keep one row per benchmark task.
    deduped: dict[str, dict[str, Any]] = {}
    for job in jobs:
        key = str(job.get("task_key") or job["id"])
        current = deduped.get(key)
        if current is None or job_record_rank(job) > job_record_rank(current):
            deduped[key] = job
    jobs = list(deduped.values())
    jobs.sort(key=job_sort_key, reverse=True)
    return jobs[:MAX_HISTORY], errors


def normalize_state(state: str) -> str:
    value = state.upper().split()[0].rstrip("+") if state.strip() else "UNKNOWN"
    aliases = {"CANCELLED": "CANCELED", "COMPLETING": "COMPLETING", "CONFIGURING": "CONFIGURING"}
    return aliases.get(value, value)


def parse_duration(value: str) -> int | None:
    if not value or value in {"Unknown", "UNLIMITED", "Partition_Limit"}:
        return None
    try:
        days = 0
        if "-" in value:
            day_text, value = value.split("-", 1)
            days = int(day_text)
        fields = [int(part) for part in value.split(":")]
        if len(fields) == 3:
            hours, minutes, seconds = fields
        elif len(fields) == 2:
            hours, minutes, seconds = 0, fields[0], fields[1]
        else:
            hours, minutes, seconds = 0, 0, fields[0]
        return days * 86400 + hours * 3600 + minutes * 60 + seconds
    except (ValueError, IndexError):
        return None


def parse_tres(value: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in value.split(","):
        if "=" in item:
            key, val = item.split("=", 1)
            result[key] = val
    return result


def parse_timestamp(value: Any) -> datetime | None:
    if not value or str(value) in {"Unknown", "None", "N/A"}:
        return None
    try:
        result = datetime.fromisoformat(str(value))
        return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result
    except ValueError:
        return None


def format_duration(seconds: int | None) -> str:
    if seconds is None:
        return "—"
    seconds = max(0, seconds)
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m"
    return "<1m"


def error_from_log(job: dict[str, Any]) -> str:
    raw_path = str(job.get("stdout", ""))
    if not raw_path or raw_path in {"None", "Unknown"}:
        return ""
    path = Path(raw_path.replace("%j", str(job.get("id", ""))))
    try:
        lines = path.read_text(errors="replace")[-24000:].splitlines()
    except OSError:
        return ""
    pattern = re.compile(r"(error|exception|failed|traceback|out of memory|runtimeerror|valueerror|killed)", re.I)
    matches = [line.strip() for line in lines if pattern.search(line.strip())]
    return (matches[-1][-320:] if matches else "")


def load_run_metadata(job_id: str) -> dict[str, Any]:
    result_root = ROOT / "05_Experiment" / "results"
    if not result_root.is_dir():
        return {}
    try:
        for run_dir in result_root.glob(f"*-{job_id}"):
            metadata_file = run_dir / "run_metadata.json"
            if metadata_file.is_file():
                metadata = json.loads(metadata_file.read_text())
                if isinstance(metadata, dict):
                    return metadata
    except (OSError, json.JSONDecodeError):
        pass
    return {}


@lru_cache(maxsize=2)
def spatialbench_run_index(cache_minute: int) -> dict[str, dict[str, str]]:
    """Resolve both Slurm array aliases and concrete IDs from the run record."""
    project_dir = RESULTS_DIR / "spatialbench_qwen3.8"
    record_file = project_dir / "spatialbench_qwen3.8_full_run.json"
    try:
        record = json.loads(record_file.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    project_id = str(record.get("project_id") or "spatialbench-qwen3.8-full-20261004")
    project_name = str(record.get("project_name") or "SpatialBench")
    index: dict[str, dict[str, str]] = {}
    for batch in record.get("batches", []):
        array_id = str(batch.get("array_job_id", ""))
        single_id = str(batch.get("slurm_job_id", ""))
        if not array_id and not single_id:
            continue
        for benchmark in batch.get("benchmarks", []):
            name = str(benchmark.get("name", ""))
            task_index = benchmark.get("array_index")
            if not name or task_index is None:
                continue
            key = f"{array_id}_{task_index}" if array_id else single_id
            info = {"project_id": project_id, "project_name": project_name,
                    "task_name": name, "task_key": key,
                    "task_ids": ", ".join(benchmark.get("tasks") or [benchmark.get("task", "")])}
            index[key] = info
            if not array_id:
                continue
            final_dir = ROOT / str(record.get("final_results_directory", ""))
            slurm_file = final_dir / "benchmarks" / f"qwen3.8-{name}" / "slurm-job.txt"
            if not slurm_file.is_file():
                slurm_file = project_dir / f"qwen3.8-{name}" / "slurm-job.txt"
            try:
                with slurm_file.open() as stream:
                    first_line = stream.readline()
            except OSError:
                continue
            match = re.search(r"\bJobId=(\d+)\s+ArrayJobId=(\d+)\s+ArrayTaskId=(\d+)\b", first_line)
            if match and match.group(2) == array_id and match.group(3) == str(task_index):
                index[match.group(1)] = info
    return index


def array_identity(job: dict[str, Any]) -> str:
    job_id = str(job.get("id", ""))
    if re.fullmatch(r"\d+_\d+", job_id):
        return job_id
    account_array_id = str(job.get("array_id", ""))
    if re.fullmatch(r"\d+_\d+", account_array_id):
        return account_array_id
    for source in (job.get("stdout", ""), job.get("command", "")):
        match = re.search(r"qwen3\.8-array-(\d+_\d+)\.log", str(source))
        if match:
            return match.group(1)
    return job_id


def expand_array_job(job: dict[str, Any]) -> list[dict[str, Any]]:
    job_id = str(job.get("id", ""))
    match = re.fullmatch(r"(\d+)_\[([\d,\-]+)(?:%\d+)?\]", job_id)
    if not match:
        return [job]
    indices: list[int] = []
    for part in match.group(2).split(","):
        if "-" in part:
            start, end = (int(value) for value in part.split("-", 1))
            indices.extend(range(start, min(end, start + 499) + 1))
        else:
            indices.append(int(part))
    return [dict(job, id=f"{match.group(1)}_{index}", array_placeholder=True) for index in indices]


def job_record_rank(job: dict[str, Any]) -> tuple[int, int, int]:
    return (0 if job.get("array_placeholder") else 1,
            1 if job.get("state") in {"RUNNING", "CONFIGURING", "COMPLETING"} else 0,
            1 if job.get("started") or job.get("ended") else 0)


def classify_job(job: dict[str, Any]) -> tuple[str, str]:
    text = f"{job.get('name', '')} {job.get('command', '')}".lower()
    if "harness" in text:
        return "Agent Harness", "Model, environment interaction, and trajectory demo"
    if "delivery" in text:
        return "DeliveryBench", "Interactive delivery and navigation experiment"
    if "spatial" in text:
        return "SpatialBench", "Spatial benchmark evaluation"
    if "download" in text or "asset" in text:
        return "Data setup", "Dataset or simulator asset preparation"
    if "simulator" in text or "unreal" in text:
        return "Simulator", "Simulator runtime or environment task"
    command = str(job.get("command", "")).strip()
    return "Slurm task", command[:120] or "Task type inferred from job name and submission command"


def project_key(job: dict[str, Any]) -> str:
    metadata = job.get("run_metadata") or {}
    shared_id = metadata.get("project_id") or metadata.get("experiment_group") or metadata.get("group_id")
    if shared_id:
        return f"project:{shared_id}"
    job_id = str(job.get("id", ""))
    array_base = re.match(r"^(\d+)_\d+(?:\+\d+)?$", job_id)
    if array_base:
        return f"array:{array_base.group(1)}"
    return f"job:{job_id}"


def enrich_job(job: dict[str, Any]) -> dict[str, Any]:
    job = dict(job)
    job["state"] = normalize_state(str(job.get("state", "UNKNOWN")))
    job["type"], job["description"] = classify_job(job)
    elapsed = parse_duration(str(job.get("elapsed", ""))) or 0
    limit = parse_duration(str(job.get("time_limit", "")))
    is_live = job["state"] in {"RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED"}
    job["time_percent"] = min(100, round(elapsed / limit * 100)) if is_live and limit else (100 if job["state"] == "COMPLETED" else 0)
    job["elapsed_seconds"] = elapsed
    job["limit_seconds"] = limit
    tres = parse_tres(str(job.get("alloc_tres") or job.get("req_tres") or ""))
    job["cpus"] = job.get("cpus") or tres.get("cpu", "")
    job["memory"] = job.get("memory") or tres.get("mem", "")
    gpu_count = tres.get("gres/gpu", "")
    gpu_model = next((key.split(":", 1)[1] for key in tres if key.startswith("gres/gpu:")), "")
    if not gpu_count and gpu_model:
        gpu_count = tres.get(f"gres/gpu:{gpu_model}", "")
    if not gpu_count:
        command = str(job.get("command", ""))
        match = re.search(r"(?:^|\s)--gres(?:=|\s+)gpu(?::([^:,\s]+))?:(\d+)(?=\s|$)", command)
        if match:
            gpu_model, gpu_count = match.group(1) or gpu_model, match.group(2)
        else:
            match = re.search(r"(?:^|\s)--gpus(?:=|\s+)(?:(\w+):)?(\d+)(?=\s|$)", command)
            if match:
                gpu_model, gpu_count = match.group(1) or gpu_model, match.group(2)
    if gpu_count and not gpu_model:
        partition_model = re.search(r"(?:^|-)gpu-(h\d+|a\d+|v\d+)(?:$|-)", str(job.get("partition", "")), re.I)
        if partition_model:
            gpu_model = partition_model.group(1).lower()
    job["gpu_count"] = gpu_count or "0"
    job["gpu_model"] = gpu_model
    job["node_list"] = job.get("node_list") or ("—" if str(job.get("state", "")).upper() == "PENDING" else "")
    submitted = parse_timestamp(job.get("submitted"))
    started = parse_timestamp(job.get("started"))
    ended = parse_timestamp(job.get("ended"))
    if submitted:
        queue_end = started or ended or (datetime.now(timezone.utc) if job["state"] == "PENDING" else None)
        job["queue_wait_seconds"] = int((queue_end - submitted).total_seconds()) if queue_end else None
    else:
        job["queue_wait_seconds"] = None
    job["queue_wait"] = format_duration(job["queue_wait_seconds"])
    job["run_metadata"] = load_run_metadata(str(job.get("id", "")))
    indexed = spatialbench_run_index(int(time.time() // 60)).get(array_identity(job), {})
    if indexed:
        job["project_key"] = f"project:{indexed['project_id']}"
        job["project_name"] = indexed["project_name"]
        job["task_name"] = indexed["task_name"]
        job["task_key"] = indexed["task_key"]
        job["task_ids"] = indexed["task_ids"]
    else:
        job["project_key"] = project_key(job)
        job["project_name"] = str(job["run_metadata"].get("project_name") or job["run_metadata"].get("project_id") or job.get("name") or job["type"])
        job["task_name"] = str(job["run_metadata"].get("task_name") or job.get("name") or "Slurm task")
        job["task_key"] = str(job["run_metadata"].get("task_id") or job["id"])
    if job["state"] in {"FAILED", "TIMEOUT", "CANCELED", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}:
        job["failure_detail"] = error_from_log(job)
    else:
        job["failure_detail"] = ""
    job["active"] = is_live or job["state"] == "PENDING"
    if job["state"] == "PENDING":
        job["progress_label"] = "Waiting for allocation"
    elif is_live:
        job["progress_label"] = "Wall time used"
    else:
        job["progress_label"] = "Finished"
    return job


def job_sort_key(job: dict[str, Any]) -> str:
    submitted = str(job.get("submitted", ""))
    if submitted in {"", "Unknown", "None"}:
        submitted = str(job.get("started") or job.get("ended") or "")
    return submitted


def bark_base_url() -> str:
    value = os.environ.get("BARK_URL", "").strip()
    if not value and BARK_FILE.exists():
        value = BARK_FILE.read_text().strip()
    return value.rstrip("/")


def notify_bark(title: str, body: str, job_id: str) -> None:
    base = bark_base_url()
    if not base:
        return
    url = f"{base}/{quote(title, safe='')}/{quote(body, safe='')}?{urlencode({'group': 'SpatialAgent-Slurm', 'ttl': 600})}"
    try:
        with urlopen(Request(url, method="GET"), timeout=5) as response:
            if response.status >= 400:
                raise RuntimeError(f"HTTP {response.status}")
    except (OSError, URLError, RuntimeError) as exc:
        with lock:
            snapshot["last_bark_error"] = f"Job {job_id}: {exc}"


def notify_project_transition(project: list[dict[str, Any]], event: str) -> None:
    if not project:
        return
    lead = project[0]
    kind = str(lead.get("type", "Slurm task"))
    name = str(lead.get("project_name") or lead.get("name", "Slurm project"))
    ids = ", ".join(str(job["id"]) for job in project[:8])
    if len(project) > 8:
        ids += f", +{len(project) - 8} more"
    if event == "submitted":
        title = f"Project submitted: {kind}"
        body = f"{name}: {len(project)} Slurm task(s) submitted. Job IDs: {ids}."
    elif event == "started":
        active = [job for job in project if job["state"] in {"RUNNING", "CONFIGURING", "COMPLETING"}]
        title = f"Project started: {kind}"
        body = f"{name}: {len(active)}/{len(project)} task(s) running. Job IDs: {ids}."
    else:
        counts: dict[str, int] = {}
        for job in project:
            counts[job["state"]] = counts.get(job["state"], 0) + 1
        outcome = ", ".join(f"{state} {count}" for state, count in sorted(counts.items()))
        title = f"Project ended: {kind}"
        body = f"{name}: {len(project)} task(s) ended ({outcome}). Job IDs: {ids}."
    notify_bark(title, body, str(lead["id"]))


def save_states(states: dict[str, str]) -> None:
    STATE_FILE.write_text(json.dumps({"states": states, "hidden_jobs": sorted(hidden_jobs),
                                      "updated_at": datetime.now(timezone.utc).isoformat()}))


def poll_once(initial: bool = False) -> None:
    global previous_states
    jobs, errors = slurm_jobs()
    with lock:
        jobs = [job for job in jobs if str(job["id"]) not in hidden_jobs]
    new_states = dict(previous_states)
    current_ids = set()
    projects: dict[str, list[dict[str, Any]]] = {}
    for job in jobs:
        job_id = str(job["id"])
        current_ids.add(job_id)
        state = str(job["state"])
        projects.setdefault(str(job["project_key"]), []).append(job)
        new_states[job_id] = state
    terminal = {"COMPLETED", "FAILED", "TIMEOUT", "CANCELED", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}
    running = {"RUNNING", "CONFIGURING", "COMPLETING"}
    for key, project in projects.items():
        project.sort(key=job_sort_key)
        member_ids = [str(job["id"]) for job in project]
        marker = lambda event: f"@project:{key}:{event}"
        event_seen = lambda event: previous_states.get(marker(event)) == "sent" or any(
            previous_states.get(f"@project-event:{event}:{job_id}") == "sent" for job_id in member_ids
        )
        was_submitted = event_seen("submitted")
        was_started = event_seen("started")
        was_ended = event_seen("ended")
        newly_seen = any(previous_states.get(job_id) is None for job_id in member_ids)
        is_running = any(job["state"] in running for job in project)
        has_run = is_running or any(parse_timestamp(job.get("started")) is not None for job in project)
        is_ended = all(job["state"] in terminal for job in project)
        state_transitions = any(previous_states.get(str(job["id"])) != job["state"] for job in project)
        if initial:
            new_states[marker("submitted")] = "sent"
            for job_id in member_ids:
                new_states[f"@project-event:submitted:{job_id}"] = "sent"
            if has_run:
                new_states[marker("started")] = "sent"
                for job_id in member_ids:
                    new_states[f"@project-event:started:{job_id}"] = "sent"
            if is_ended:
                new_states[marker("ended")] = "sent"
                for job_id in member_ids:
                    new_states[f"@project-event:ended:{job_id}"] = "sent"
            continue
        if newly_seen and not was_submitted:
            notify_project_transition(project, "submitted")
            new_states[marker("submitted")] = "sent"
            for job_id in member_ids:
                new_states[f"@project-event:submitted:{job_id}"] = "sent"
        elif was_submitted:
            new_states[marker("submitted")] = "sent"
        if has_run and not was_started:
            notify_project_transition(project, "started")
            new_states[marker("started")] = "sent"
            for job_id in member_ids:
                new_states[f"@project-event:started:{job_id}"] = "sent"
        elif was_started:
            new_states[marker("started")] = "sent"
        if is_ended and state_transitions and not was_ended:
            notify_project_transition(project, "ended")
            new_states[marker("ended")] = "sent"
            for job_id in member_ids:
                new_states[f"@project-event:ended:{job_id}"] = "sent"
        elif was_ended:
            new_states[marker("ended")] = "sent"
    if len(new_states) > 500:
        keep = {str(job["id"]) for job in jobs}
        keep.update(list(new_states)[-300:])
        keep.update(f"@project:{key}:{event}" for key in projects for event in ("submitted", "started", "ended"))
        keep.update(f"@project-event:{event}:{job_id}" for job_id in current_ids for event in ("submitted", "started", "ended"))
        new_states = {key: value for key, value in new_states.items() if key in keep}
    previous_states = new_states
    try:
        save_states(new_states)
    except OSError as exc:
        errors.append(f"state file: {exc}")
    running = sum(job["state"] in {"RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED"} for job in jobs)
    queued = sum(job["state"] == "PENDING" for job in jobs)
    failed = sum(job["state"] in {"FAILED", "TIMEOUT", "CANCELED", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED"} for job in jobs)
    completed = sum(job["state"] == "COMPLETED" for job in jobs)
    with lock:
        snapshot.update({
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "jobs": jobs,
            "errors": errors,
            "user": os.environ.get("USER", "unknown"),
            "summary": {"running": running, "queued": queued, "failed": failed, "completed": completed,
                        "gpus_running": sum(int(job.get("gpu_count") or 0) for job in jobs if job["state"] in {"RUNNING", "CONFIGURING", "COMPLETING"})},
            "bark_configured": bool(bark_base_url()),
        })


def monitor_loop() -> None:
    initialized = False
    while True:
        try:
            poll_once(initial=not initialized)
            initialized = True
        except Exception as exc:
            with lock:
                snapshot["errors"] = [str(exc)]
        time.sleep(POLL_SECONDS)


@app.on_event("startup")
def startup() -> None:
    thread = threading.Thread(target=monitor_loop, daemon=True, name="slurm-monitor")
    thread.start()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/state")
def api_state() -> dict[str, Any]:
    with lock:
        return dict(snapshot)


def direct_artifact(raw: str, parent: Path) -> Path | None:
    if not raw or raw in {"None", "Unknown", "N/A"}:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    # Only a direct child of the experiment results/log directory is eligible.
    # Refuse links so metadata cannot redirect deletion outside the workspace.
    if path.is_symlink() or path.parent.resolve() != parent.resolve():
        return None
    return path


def deletion_plan(scope: str, target: str) -> tuple[str, list[str], list[Path]]:
    if scope not in {"job", "project"} or not target or len(target) > 200:
        raise HTTPException(400, "Invalid deletion target")
    jobs, errors = slurm_jobs()
    if errors:
        raise HTTPException(503, "Slurm status is unavailable; retry when the monitor is healthy")
    matched = [job for job in jobs if str(job["id"]) not in hidden_jobs and
               (str(job["id"]) == target if scope == "job" else str(job["project_key"]) == target)]
    if not matched:
        raise HTTPException(404, "This job or project is no longer in the monitor")
    terminal = {"COMPLETED", "FAILED", "TIMEOUT", "CANCELED", "OUT_OF_MEMORY",
                "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}
    selected = [job for job in matched if job["state"] in terminal]
    if not selected:
        raise HTTPException(409, "No task in this selection has ended yet")
    paths: set[Path] = set()
    run_index = spatialbench_run_index(int(time.time() // 60))
    for job in selected:
        job_id = str(job["id"])
        indexed = run_index.get(array_identity(job))
        if indexed and job.get("project_key") == f"project:{indexed['project_id']}":
            result = direct_artifact(str(RESULTS_DIR / f"qwen3.8-{indexed['task_name']}"), RESULTS_DIR)
            if result:
                paths.add(result)
            array_log = direct_artifact(str(LOGS_DIR / f"qwen3.8-array-{indexed['task_key']}.log"), LOGS_DIR)
            if array_log:
                paths.add(array_log)
        for result in RESULTS_DIR.glob(f"*-{job_id}"):
            safe = direct_artifact(str(result), RESULTS_DIR)
            if safe:
                paths.add(safe)
        metadata = job.get("run_metadata") or {}
        artifact_dir = (metadata.get("artifacts") or {}).get("result_dir") or metadata.get("result_dir")
        if artifact_dir:
            safe = direct_artifact(str(artifact_dir), RESULTS_DIR)
            if safe and safe.name.endswith(f"-{job_id}"):
                paths.add(safe)
        stdout_pattern = str(job.get("stdout") or "").replace("%j", job_id)
        array_match = re.fullmatch(r"(\d+)_(\d+)", indexed["task_key"] if indexed else array_identity(job))
        if array_match:
            stdout_pattern = stdout_pattern.replace("%A", array_match.group(1)).replace("%a", array_match.group(2))
        stdout = direct_artifact(stdout_pattern if "%" not in stdout_pattern else "", LOGS_DIR)
        if stdout and stdout.name != "queue-inspector.log":
            paths.add(stdout)
    label = selected[0].get("task_name") if scope == "job" else selected[0].get("project_name")
    return str(label or target), [str(job["id"]) for job in selected], sorted(paths)


def artifact_manifest(paths: list[Path]) -> tuple[list[dict[str, Any]], str]:
    digest = hashlib.sha256()
    entries: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            digest.update(f"{path.relative_to(ROOT)}|missing\n".encode())
            entries.append({"path": str(path.relative_to(ROOT)), "kind": "missing", "bytes": 0, "exists": False})
            continue
        size = 0
        for item in [path, *(path.rglob("*") if path.is_dir() else [])]:
            info = item.lstat()
            digest.update(f"{item.relative_to(ROOT)}|{info.st_mode}|{info.st_size}|{info.st_mtime_ns}\n".encode())
            if item.is_file() and not item.is_symlink():
                size += info.st_size
        entries.append({"path": str(path.relative_to(ROOT)), "kind": "directory" if path.is_dir() else "file", "bytes": size, "exists": True})
    return entries, digest.hexdigest()


@app.get("/api/deletion-preview")
def deletion_preview(scope: str, target: str) -> dict[str, Any]:
    label, job_ids, paths = deletion_plan(scope, target)
    excluded_jobs = 0
    if scope == "project":
        jobs, _ = slurm_jobs()
        excluded_jobs = sum(job["project_key"] == target and str(job["id"]) not in job_ids
                            and str(job["id"]) not in hidden_jobs for job in jobs)
    entries, fingerprint = artifact_manifest(paths)
    token = secrets.token_urlsafe(32)
    with lock:
        now = time.time()
        for old_token, (_, _, _, created) in list(pending_deletions.items()):
            if now - created > 300:
                del pending_deletions[old_token]
        pending_deletions[token] = (scope, target, fingerprint, now)
    return {"label": label, "scope": scope, "target": target, "job_ids": job_ids,
            "excluded_jobs": excluded_jobs, "entries": entries,
            "total_bytes": sum(entry["bytes"] for entry in entries), "token": token}


@app.post("/api/delete-artifacts")
def delete_artifacts(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    if payload.get("confirm") != "DELETE":
        raise HTTPException(400, "Type DELETE in the confirmation box")
    token = str(payload.get("token") or "")
    with lock:
        pending = pending_deletions.pop(token, None)
    if not pending or time.time() - pending[3] > 300:
        raise HTTPException(409, "Deletion preview expired; open it again")
    scope, target, fingerprint, _ = pending
    _, job_ids, paths = deletion_plan(scope, target)
    _, current_fingerprint = artifact_manifest(paths)
    if current_fingerprint != fingerprint:
        raise HTTPException(409, "Files changed after preview; review the updated list before deleting")
    selected_paths = payload.get("paths")
    if not isinstance(selected_paths, list) or any(not isinstance(value, str) for value in selected_paths):
        raise HTTPException(400, "Select the files to delete")
    allowed = {str(path.relative_to(ROOT)): path for path in paths if path.exists()}
    chosen = set(selected_paths)
    if chosen - allowed.keys():
        raise HTTPException(400, "Selection contains a path outside the preview")
    if allowed and not chosen:
        raise HTTPException(400, "Select at least one existing file or directory")
    deleted: list[str] = []
    for name in sorted(chosen):
        path = allowed[name]
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        deleted.append(name)
    newly_hidden: list[str] = []
    for job_id in job_ids:
        _, _, remaining = deletion_plan("job", job_id)
        if not any(path.exists() for path in remaining):
            newly_hidden.append(job_id)
    with lock:
        hidden_jobs.update(newly_hidden)
        save_states(previous_states)
        visible = [job for job in snapshot["jobs"] if str(job["id"]) not in hidden_jobs]
        snapshot["jobs"] = visible
        snapshot["summary"] = {
            "running": sum(job["state"] in {"RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED"} for job in visible),
            "queued": sum(job["state"] == "PENDING" for job in visible),
            "failed": sum(job["state"] in {"FAILED", "TIMEOUT", "CANCELED", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED"} for job in visible),
            "completed": sum(job["state"] == "COMPLETED" for job in visible),
            "gpus_running": sum(int(job.get("gpu_count") or 0) for job in visible if job["state"] in {"RUNNING", "CONFIGURING", "COMPLETING"}),
        }
    return {"deleted": deleted, "hidden_jobs": newly_hidden}
