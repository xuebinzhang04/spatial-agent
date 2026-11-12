#!/bin/bash
#SBATCH --job-name=native-task-migration
#SBATCH --account=video
#SBATCH --partition=gpu-h200
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:20:00
#SBATCH --output=/gpfs/scrubbed/xuebiz/SpatialAgent/05_Experiment/logs/native-migration-%j.log
set -euo pipefail
ROOT=/gpfs/scrubbed/xuebiz/SpatialAgent
RUN="$ROOT/05_Experiment/results/native-migration-${MIGRATION_TASK:?}-$SLURM_JOB_ID"
PY="$ROOT/02_Simulator/AI2THOR/runtime/python/bin/python"
export CUDA_VISIBLE_DEVICES=''
export SPATIAL_THOR_SOFTWARE=1
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
export LP_NUM_THREADS=2 OMP_NUM_THREADS=2
export XDG_CONFIG_HOME="$RUN/unity-config"
export XDG_CACHE_HOME="$ROOT/.cache"
mkdir -p "$XDG_CONFIG_HOME"
cd "$ROOT"
"$PY" - "$RUN" running <<'PY'
import json,sys,socket
from pathlib import Path
from datetime import datetime,timezone
p=Path(sys.argv[1])/'run_metadata.json';d=json.loads(p.read_text())
d['slurm'].update(state='RUNNING',started_at=datetime.now(timezone.utc).isoformat(),node=socket.gethostname())
p.write_text(json.dumps(d,indent=2)+'\n')
PY
trap 'status=$?; "$PY" "$ROOT/scripts/experiments/update_native_migration_state.py" "$RUN" "$status"' EXIT
validation_args=(--task "$MIGRATION_TASK" --output "$RUN" --cases "${MIGRATION_CASES:-3}")
if [[ ${MIGRATION_LEVEL:-0} != 0 ]]; then validation_args+=(--levels "$MIGRATION_LEVEL"); fi
"$PY" -u scripts/experiments/validate_native_task_migration.py "${validation_args[@]}"
