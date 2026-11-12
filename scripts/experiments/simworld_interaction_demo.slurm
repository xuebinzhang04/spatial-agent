#!/bin/bash
#SBATCH --job-name=simworld-interaction
#SBATCH --account=video
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:15:00
#SBATCH --signal=B:TERM@30
#SBATCH --output=/gpfs/scrubbed/xuebiz/SpatialAgent/05_Experiment/logs/interaction-demo-%j.log
set -euo pipefail
ulimit -c 0
ROOT=/gpfs/scrubbed/xuebiz/SpatialAgent
PY=/gpfs/scrubbed/xuebiz/conda/envs/spatialagent/bin/python
RUN="$ROOT/05_Experiment/results/interaction-demo-$SLURM_JOB_ID"
DEMO_MAP="${DEMO_MAP:-small-city-11}"
UE_ROOT="$ROOT/02_Simulator/DeliveryBench/Linux/gym_citynav"
export PATH="/gpfs/scrubbed/xuebiz/conda/envs/spatialagent/bin:$PATH"
export CONDA_PREFIX=/gpfs/scrubbed/xuebiz/conda/envs/spatialagent
export QT_QPA_PLATFORM=offscreen
export OMP_NUM_THREADS=4
export XDG_CACHE_HOME="$ROOT/.cache"
ue_pid=""
bridge_pid=""
cleanup() {
  status=$?
  trap - EXIT TERM INT
  if [[ -n "$bridge_pid" ]]; then
    kill -TERM "$bridge_pid" 2>/dev/null || true
    for _ in {1..10}; do kill -0 "$bridge_pid" 2>/dev/null || break; sleep 1; done
    kill -KILL "$bridge_pid" 2>/dev/null || true
  fi
  if [[ -n "$ue_pid" ]]; then kill "$ue_pid" 2>/dev/null || true; fi
  "$PY" "$ROOT/scripts/experiments/interaction_demo_record.py" ended --run-dir "$RUN" --exit-code "$status" || true
  exit "$status"
}
trap cleanup EXIT
trap 'exit 0' TERM INT
"$PY" "$ROOT/scripts/experiments/interaction_demo_record.py" started --run-dir "$RUN"
cp "$0" "$RUN/submitted-job.sh"
module list > "$RUN/modules.txt" 2>&1
nvidia-smi > "$RUN/gpu.txt"
[[ -x "$UE_ROOT/Binaries/Linux/gym_citynav" ]]
cd "$UE_ROOT"
(
  module unload cuda/13.0.0 gcc/13.4.0 2>/dev/null || true
  exec ./Binaries/Linux/gym_citynav "$DEMO_MAP" -RenderOffscreen -nosound
) > "$RUN/simulator.log" 2>&1 &
ue_pid=$!
"$PY" - "$RUN/simulator.log" "$DEMO_MAP" "$ue_pid" <<'PY'
import os, socket, sys, time
from pathlib import Path
log, map_name, pid = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
for _ in range(180):
    os.kill(pid, 0)
    text = log.read_text(errors='replace') if log.exists() else ''
    if 'Cannot construct DefaultSceneRoot' in text:
        raise RuntimeError('Map has missing blueprint dependencies')
    if f'LoadMap(/Game/{map_name})' in text:
        try:
            with socket.create_connection(('127.0.0.1',9000),timeout=1):
                break
        except OSError:
            pass
    time.sleep(1)
else:
    raise TimeoutError('Expected UE map/UnrealCV server did not become ready')
PY
cd "$ROOT"
MODEL_ARGS=()
if [[ -n "${DEMO_MODEL_URL:-}" && -n "${DEMO_MODEL:-}" ]]; then
  MODEL_ARGS=(--model-url "$DEMO_MODEL_URL" --model "$DEMO_MODEL")
fi
"$PY" -u "$ROOT/04_AgentTask/Demos/spatial_agent/run.py" \
  --environment deliverybench --map "$DEMO_MAP" --seed 0 --port 8767 --run-dir "$RUN" --scripted-tour "${MODEL_ARGS[@]}" \
  > "$RUN/bridge.log" 2>&1 &
bridge_pid=$!
echo "GPU demo node: $(hostname) | job: $SLURM_JOB_ID | port: 8767"
echo "From your workstation (not the cluster shell): ssh -J xuebiz@tillicum.hyak.uw.edu xuebiz@$(hostname) -L 8767:127.0.0.1:8767 -N"
echo "Then open http://127.0.0.1:8767"
while kill -0 "$bridge_pid" 2>/dev/null; do
  if ! kill -0 "$ue_pid" 2>/dev/null; then echo 'UE exited unexpectedly' >&2; exit 1; fi
  sleep 2
done
wait "$bridge_pid"
