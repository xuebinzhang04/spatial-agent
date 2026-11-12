#!/bin/bash
#SBATCH --job-name=stage1-vlm-smoke
#SBATCH --account=video
#SBATCH --partition=gpu-h200
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/scrubbed/xuebiz/SpatialAgent/05_Experiment/logs/stage1-vlm-smoke-%j.log
set -euo pipefail
ulimit -c 0
ROOT=/gpfs/scrubbed/xuebiz/SpatialAgent
RUN="${STAGE1_VLM_RUN:?Set STAGE1_VLM_RUN to a fresh run directory}"
PY="$ROOT/02_Simulator/AI2THOR/runtime/python/bin/python"
mkdir -p "$RUN/unity-config"
export XDG_CONFIG_HOME="$RUN/unity-config"
export XDG_CACHE_HOME="$ROOT/.cache"
export OMP_NUM_THREADS=2
export QWEN_BASE_URL=http://127.0.0.1:8000/v1
export QWEN_MODEL=Qwen3.8-27B
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
# The model uses the assigned H200. Reuse tested lavapipe rendering for THOR;
# CUDA visibility must be cleared ONLY for the simulator worker, not vLLM.
export SPATIAL_THOR_SOFTWARE=1
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
export LP_NUM_THREADS=2
cd "$ROOT"
server_pid=''
cleanup(){
    result=$?
    trap - EXIT
    if [[ -n "$server_pid" ]]; then kill "$server_pid" 2>/dev/null || true; fi
    "$PY" - "$RUN" "$result" <<'PY'
import json,sys
from pathlib import Path
from datetime import datetime,timezone
p=Path(sys.argv[1])/'run_metadata.json';d=json.loads(p.read_text())
d['state']='COMPLETED' if sys.argv[2]=='0' else 'FAILED'
d['exit_code']=int(sys.argv[2]);d['ended_at_utc']=datetime.now(timezone.utc).isoformat()
p.write_text(json.dumps(d,indent=2)+'\n')
PY
    exit "$result"
}
trap cleanup EXIT
"$PY" - "$RUN" <<'PY'
import json,os,socket,sys
from pathlib import Path
from datetime import datetime,timezone
p=Path(sys.argv[1])/'run_metadata.json';d=json.loads(p.read_text())
d.update(state='RUNNING',node=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID'],started_at_utc=datetime.now(timezone.utc).isoformat())
p.write_text(json.dumps(d,indent=2)+'\n')
PY
nvidia-smi > "$RUN/gpu.txt"
cp "$0" "$RUN/submitted-job.sh"
bash "$ROOT/scripts/serve_qwen.sh" > "$RUN/vllm.log" 2>&1 &
server_pid=$!
ready=0
for i in {1..120}; do
    if ! kill -0 "$server_pid" 2>/dev/null; then echo 'Model server exited; see vllm.log' >&2;exit 1;fi
    if curl --noproxy '*' -fsS --max-time 2 http://127.0.0.1:8000/health >/dev/null 2>&1;then ready=1;break;fi
    sleep 5
done
[[ "$ready" == 1 ]] || { echo 'Model readiness timeout' >&2;exit 1; }
curl --noproxy '*' -fsS http://127.0.0.1:8000/v1/models > "$RUN/served-models.json"
task_args=()
if [[ "${STAGE1_VLM_NATIVE_ONLY:-0}" == 1 ]]; then task_args=(--tasks missing_object moved_object); fi
CUDA_VISIBLE_DEVICES='' "$PY" -u "$ROOT/scripts/experiments/stage1_vlm_smoke.py" --output "$RUN" --max-steps 20 "${task_args[@]}"
