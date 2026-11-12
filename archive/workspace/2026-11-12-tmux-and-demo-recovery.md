# 2026-11-12 — tmux snapshot and research demo recovery

This is a point-in-time operational snapshot, not a list of guaranteed running
services. It records tmux metadata only; terminal scrollback, pairing codes,
session tokens, and credentials are intentionally omitted.

| tmux session | Pane command | Pane working directory | Use at inspection |
| --- | --- | --- | --- |
| `codex` | `bash` | `/gpfs/scrubbed/xuebiz` | Shell used for Codex remote-control setup; no active Codex foreground process shown in this pane |
| `spatial` | `claude` | `/gpfs/scrubbed/xuebiz` | Claude Code session for the spatial workspace; the pane is not inside `SpatialAgent/` |
| `eep568` | `claude` | `/gpfs/scrubbed/xuebiz/ignore/eep568` | Separate Claude Code workspace; do not treat it as the spatial implementation session |

The active Codex app chat and the short-lived Claude CLI implementation task
are not defined by these tmux panes. A tmux session can disappear while a Git
commit, run manifest, or Slurm job remains; verify each separately.

## Check before stopping or recovering

```bash
tmux list-sessions
tmux list-panes -a -F '#{session_name}:#{window_index}.#{pane_index} #{pane_current_command} #{pane_current_path}'
cd /gpfs/scrubbed/xuebiz/SpatialAgent
git status --short --branch
git log -1 --oneline
squeue --me
```

If a pane has disappeared, use the Git branch, dated record, and
`05_Experiment/results/<run-id>/run_metadata.json` as the durable state. Check
Slurm independently; losing tmux does not cancel an already submitted job.
Resume Claude Code through its session selector or the saved resume ID held by
the operator, not by committing an ID to this repository. Recreate a shell with
`tmux new-session -s spatial -c /gpfs/scrubbed/xuebiz/SpatialAgent` only when
the old `spatial` session is absent. Attach to a live session with
`tmux attach -t spatial`. Do not kill a live agent pane just to change its
working directory.

## Dedicated demo session, after the page is verified

The static/replay page can later have its own tmux session, distinct from agent
sessions. Use the registered Python environment and existing port 8768. The
launcher writes a new run manifest and stops only its own prior server on that
port. Keep stdout in `05_Experiment/logs/`, which is ignored by Git.

```bash
cd /gpfs/scrubbed/xuebiz/SpatialAgent
python 04_AgentTask/Demos/spatial_agent/build.py
tmux new-session -d -s spatial-demo -c /gpfs/scrubbed/xuebiz/SpatialAgent \
  '/gpfs/scrubbed/xuebiz/conda/envs/spatialagent/bin/python -u 04_AgentTask/Demos/spatial_agent/run.py >> 05_Experiment/logs/spatial-demo-server.log 2>&1'
curl -fsS http://127.0.0.1:8768/api/capabilities
```

Before starting a long-lived service on a cluster host, verify that the chosen
host and local policy allow it. The login node remains for light editing and
checks; GPU inference and native simulator runs belong in Slurm allocations.
If the demo session is lost, inspect the log and `run_metadata.json`, check
whether port 8768 is already served, then restart the verified command. Do not
start a second model service or billable GPU allocation just to show recorded
episodes.
