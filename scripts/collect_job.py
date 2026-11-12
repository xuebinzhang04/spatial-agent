"""Save Slurm accounting and generate a demo for an existing job."""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("job_id", type=int)
parser.add_argument("--kind", choices=("smoke", "full"), default="smoke")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
run = root / "05_Experiment/results" / f"qwen3.8-{args.kind}-{args.job_id}"
run.mkdir(parents=True, exist_ok=True)
result = subprocess.run(["sacct", "-j", str(args.job_id), "--parsable2", "--format=JobID,State,ExitCode,Elapsed,NodeList,MaxRSS,AllocTRES"], capture_output=True, text=True, check=True)
(run / "slurm-accounting.txt").write_text(result.stdout)
lines = result.stdout.strip().splitlines()
rows = [dict(zip(lines[0].split("|"), line.split("|"))) for line in lines[1:]] if lines else []
record = {"job_id": args.job_id, "queried_at_utc": datetime.now(timezone.utc).isoformat(),
          "test_type": "three-step integration smoke" if args.kind == "smoke" else "single-map full lifecycle evaluation",
          "accounting": rows}
(run / "job_status.json").write_text(json.dumps(record, indent=2))
log = root / "05_Experiment/logs" / f"{args.kind}-{args.job_id}.log"
if log.exists():
    (run / "slurm.log").write_text(log.read_text(errors="replace"))
if args.kind == "full":
    reports = sorted((run / "reports").glob("run_report_agent*.json"))
    episode_log = (run / "episode.log").read_text(errors="replace") if (run / "episode.log").exists() else ""
    native_stop_reason = next((reason for reason in (
        "Simulation time reached", "VLM call limit reached", "Real time reached"
    ) if reason in episode_log), None)
    report = json.loads(reports[0].read_text()) if reports else None
    experiment = json.loads((run / "experiment_config.json").read_text()) if (run / "experiment_config.json").exists() else {}
    validation = {
        "scope": "single map, seed 0, one agent",
        "map_name": experiment.get("gym_env", {}).get("map_name"),
        "native_lifecycle_complete": bool(report and native_stop_reason and "run report exported" in episode_log),
        "native_stop_reason": native_stop_reason,
        "active_hours": report["meta"]["active_hours"] if report else None,
        "completed_orders": report["orders"]["completed_count"] if report else None,
        "scheduler_state": rows[0]["State"] if rows else None,
        "runner_completed_cleanly": bool(rows and rows[0]["State"] == "COMPLETED"),
    }
    (run / "result_validation.json").write_text(json.dumps(validation, indent=2))
    if reports:
        subprocess.run([sys.executable, str(root / "04_AgentTask/SpatialAgentBench/DeliveryBench/evaluation/agent_performance_analysis.py"),
                        str(run / "reports"), "-o", str(run / "summary")], check=True)
subprocess.run([sys.executable, str(root / "scripts/render_demo.py"), str(run)], check=True)
print(result.stdout)
