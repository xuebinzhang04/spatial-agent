# Experiment record contract

Every experiment agent, harness, benchmark runner, and simulator task must read and follow this contract. Create `05_Experiment/results/<run-id>/run_metadata.json` for each submitted run and keep it with its artifacts. Include the Slurm Job ID in `<run-id>` (for example, `spatial-eval-344069`) so the Queue Inspector can associate the manifest with the job. At the beginning of one launcher/terminal experiment session, generate a unique `project_id` and a readable `project_name`, and pass both to every child job and array in that session. Record a specific `task_name` (for example, `VSI-Bench`) and `task_id` for each child. A new independently launched session, including a simultaneous simulator run in another terminal, must receive a new `project_id`; terminal PID alone is not a durable experiment identifier. Fill in known values before or at submission, then update scheduler state, timestamps, exit status, and failure reason when they change. Never guess: use `Not recorded` and explain missing values in `notes`. Preserve failed, canceled, and partial runs with diagnostic artifacts unless the user explicitly asks to remove them.

Each record must state the research goal/question and intended conclusion; exact dataset/version/split/task IDs and input materials/asset paths; experiment type and comparison baseline; model/checkpoint/revision/precision; environment or simulator/build/map; source commit and local patches; main entrypoint, launch script, configuration, and actual result-affecting settings (including seed, task limit, and timeouts); Slurm Job ID/partition/resources/submitted/started/ended times/state/exit code/failure reason; result directory; primary metrics/report/trajectory/log paths; and limitations or deviations. Parse the selected config and record its concrete task IDs, maps, seeds, limits, and materials in `dataset` and `parameters`; a config filename alone is not an experiment description. Record values actually used by the run, not just defaults or intended settings. Store paths relative to the workspace where possible.

Run experiments in this order: read this contract and the relevant project runbook; identify the research question, data/materials, actual parameters and comparison; create the run directory and manifest; validate the selected assets/configuration and Slurm resource request; submit the job; update its manifest with actual start/end state and artifacts; then report results with valid/partial/failed status and evidence paths. Do not report a score without identifying the exact dataset tasks and input materials that produced it.

Use these JSON keys (nested objects may add domain-specific fields):

```json
{
  "run_id": "<experiment-name>-<slurm-job-id>",
  "project_id": "<unique ID shared by parallel jobs in this run group>",
  "project_name": "<human-readable project label>",
  "task_name": "<specific benchmark or sub-experiment label>",
  "task_id": "<stable identifier of this child task within the project>",
  "purpose": "<goal, research question, and intended conclusion>",
  "dataset": {"name": "<dataset>", "version": "<revision>", "split": "<split>", "tasks": ["<task ID>"], "materials": ["<input file/asset IDs or paths>"]},
  "experiment_type": "<smoke|ablation|full evaluation|...>",
  "model": {"name": "<model>", "checkpoint": "<path or revision>", "precision": "<precision>"},
  "environment": {"name": "<environment or simulator>", "version": "<build or revision>"},
  "source": {"commit": "<git revision>", "entrypoint": "<main file>", "config": "<config file>"},
  "parameters": {"seed": "<seed>", "map": "<map>", "task_limit": "<limit>"},
  "slurm": {"job_id": "<job ID>", "partition": "<partition>", "resources": "<GPU/CPU/RAM>", "state": "<state>", "created_at": "<time>", "started_at": "<time>", "ended_at": "<time>", "exit_code": "<code>", "failure_reason": "<reason or Not recorded>"},
  "artifacts": {"result_dir": "<path>", "primary_metrics": "<path>", "report": "<path>", "trajectory": "<path>", "logs": ["<path>"]},
  "notes": "<limitations, deviations, and interpretation constraints>"
}
```

The Queue Inspector reads this manifest when available and shows its purpose, dataset, model, environment, source/configuration paths, and result artifacts in separate fields inside the expanded job card. Keep the Slurm submission script and run manifest aligned with this contract.
