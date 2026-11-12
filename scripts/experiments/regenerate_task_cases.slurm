#!/bin/bash
#SBATCH --job-name=interaction-cases
#SBATCH --account=video
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scrubbed/xuebiz/SpatialAgent/05_Experiment/logs/interaction-cases-%j.log
set -euo pipefail
ROOT=/gpfs/scrubbed/xuebiz/SpatialAgent
: "${CASE_PLAN:?Set CASE_PLAN to a fresh prepared case batch}"
cd "$ROOT"
export SPATIAL_THOR_SOFTWARE=1
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
export LP_NUM_THREADS=2 OMP_NUM_THREADS=2
export XDG_CONFIG_HOME="/tmp/interaction-cases-$SLURM_JOB_ID"
export XDG_CACHE_HOME="$ROOT/.cache"
mkdir -p "$XDG_CONFIG_HOME"
bash scripts/serve_qwen.sh > "$CASE_PLAN/model-service-$SLURM_JOB_ID.log" 2>&1 &
model_service_pid=$!
trap 'kill "$model_service_pid" 2>/dev/null || true' EXIT
# Poll the actual model identity; process existence is not service readiness.
"$ROOT/02_Simulator/AI2THOR/runtime/python/bin/python" - <<'PY'
import json, time
from urllib.request import urlopen
for attempt in range(120):
    try:
        with urlopen('http://127.0.0.1:8000/v1/models', timeout=5) as response:
            if any(m['id']=='Qwen3.8-27B' for m in json.load(response)['data']):break
    except OSError:pass
    time.sleep(5)
else:raise RuntimeError('Expected model service did not become ready')
PY
"$ROOT/02_Simulator/AI2THOR/runtime/python/bin/python" scripts/experiments/regenerate_task_cases.py run --output "$CASE_PLAN"
