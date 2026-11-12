#!/bin/bash
#SBATCH --job-name=delivery-case
#SBATCH --account=video
#SBATCH --partition=gpu-h200
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:20:00
#SBATCH --signal=B:TERM@30
#SBATCH --output=/gpfs/scrubbed/xuebiz/SpatialAgent/05_Experiment/logs/delivery-case-%j.log
set -euo pipefail
# Slurm may inherit an active environment; Lmod rejects loading conda again.
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  source /gpfs/software/miniforge3/25.3.1-3/etc/profile.d/conda.sh
else
  module load conda
fi
conda activate /gpfs/scrubbed/xuebiz/conda/envs/spatialagent
ulimit -c 0
ROOT=/gpfs/scrubbed/xuebiz/SpatialAgent
PY=/gpfs/scrubbed/xuebiz/conda/envs/spatialagent/bin/python
# Qt's bundled FreeType is older than this environment's HarfBuzz.
# Apply this only to Python; the packaged UE runtime keeps its own libraries.
PY_PRELOAD="$CONDA_PREFIX/lib/libfreetype.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
RUN="$ROOT/05_Experiment/results/delivery-case-$SLURM_JOB_ID"
if [[ "${DELIVERY_MODEL_RUN:-0}" == 1 ]]; then RUN="$ROOT/05_Experiment/results/delivery-qwen-$SLURM_JOB_ID"; fi
CASE_BATCH="${CASE_BATCH:-$ROOT/05_Experiment/results/delivery-cases-20261026}"
UE_ROOT="$ROOT/02_Simulator/DeliveryBench/Linux/gym_citynav"
export ROOT RUN CASE_BATCH
export QT_QPA_PLATFORM=offscreen
export OMP_NUM_THREADS=4
export XDG_CACHE_HOME="$ROOT/.cache"
# Node-local model traffic must not go through the inherited HTTP proxy.
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
ue_pid=""
model_pid=""
cleanup() {
  status=$?
  trap - EXIT TERM INT
  if [[ -n "$ue_pid" ]]; then kill -TERM "$ue_pid" 2>/dev/null || true; fi
  if [[ -n "$model_pid" ]]; then kill -TERM "$model_pid" 2>/dev/null || true; fi
  "$PY" - "$RUN" "$status" <<'PY'
from pathlib import Path
from datetime import datetime,timezone
import json,sys
p=Path(sys.argv[1])/'run_metadata.json'
if p.exists():
    m=json.loads(p.read_text());m['slurm'].update(state='COMPLETED' if sys.argv[2]=='0' else 'FAILED',
        ended_at=datetime.now(timezone.utc).isoformat(),exit_code=int(sys.argv[2]))
    p.write_text(json.dumps(m,indent=2)+'\n')
PY
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' TERM INT
cd "$ROOT"
"$PY" - <<'PY'
from pathlib import Path
import os,sys,json,socket
root=Path(os.environ['ROOT']);run=Path(os.environ['RUN']);batch=Path(os.environ['CASE_BATCH'])
sys.path.insert(0,str(root/'04_AgentTask/CustomAgentTasks/delivery_cases'))
from run import metadata,write
from cases import digest
rows=json.loads((batch/'summary.json').read_text())['cases']
selected=next(r for r in rows if r['family']=='pickup_delivery' and r['level']==1)
case=json.loads((batch/selected['path']).read_text())
run.mkdir(parents=True,exist_ok=False)
write(run/'selected-case.json',case)
m=metadata(run,'Native UE case admission launch')
if os.environ.get('DELIVERY_MODEL_RUN')=='1':
    m['task_name']='Qwen3.8-27B fixed courier case'
    m['model']={'name':'Qwen3.8-27B','checkpoint':'01_Models/Qwen3.8-27B','precision':'bfloat16'}
    m['parameters'].update(max_model_calls=16,model_url='http://127.0.0.1:8000/v1')
m['dataset'].update(tasks=[case['case_id']],materials=[str(batch/selected['path']),case['map']])
m['parameters'].update(map=case['map'],seed=case['seed'],level=case['level'],budget=case['budget'],ue_port=9000)
m['slurm'].update(job_id=os.environ['SLURM_JOB_ID'],partition='gpu-h200',resources='1 H200, 8 CPU, 50 GB, 20 minutes')
m['source']['sha256']['scripts/experiments/delivery_case_admission.slurm']=digest(root/'scripts/experiments/delivery_case_admission.slurm')
m['artifacts'].update(validation='episode/validation.json',logs=['simulator.log'])
write(run/'run_metadata.json',m)
# Refuse to attach this case to another task's already-running UE process.
with socket.socket() as probe:probe.bind(('127.0.0.1',9000))
PY
cp "$0" "$RUN/submitted-job.sh"
env LD_PRELOAD="$PY_PRELOAD" "$PY" - <<'PY' > "$RUN/qt-preflight.log" 2>&1
from PyQt5.QtWidgets import QApplication
app=QApplication([])
print('QT_OFFSCREEN_OK',flush=True)
app.quit()
PY
model_args=()
if [[ "${DELIVERY_MODEL_RUN:-0}" == 1 ]]; then
  bash "$ROOT/scripts/serve_qwen.sh" > "$RUN/vllm.log" 2>&1 &
  model_pid=$!
  "$PY" - "$model_pid" <<'PY'
import os,sys,time,json
from urllib.request import urlopen
for attempt in range(180):
    os.kill(int(sys.argv[1]),0)
    try:
        with urlopen('http://127.0.0.1:8000/v1/models',timeout=2) as result:models=json.load(result)
        if any(m['id']=='Qwen3.8-27B' for m in models['data']):break
    except OSError:pass
    time.sleep(2)
else:raise TimeoutError('Qwen service did not become ready')
PY
  model_args=(--model Qwen3.8-27B --max-model-calls 16 --record-process)
fi
DEMO_MAP="$($PY -c 'import json,os; print(json.load(open(os.environ["RUN"]+"/selected-case.json"))["map"])')"
nvidia-smi > "$RUN/gpu.txt"
cd "$UE_ROOT"
(
  module unload cuda/13.0.0 gcc/13.4.0 2>/dev/null || true
  exec ./Binaries/Linux/gym_citynav "$DEMO_MAP" -RenderOffscreen -nosound
) > "$RUN/simulator.log" 2>&1 &
ue_pid=$!
"$PY" - "$RUN/simulator.log" "$DEMO_MAP" "$ue_pid" <<'PY'
from pathlib import Path
import os,socket,sys,time
log=Path(sys.argv[1]);map_name=sys.argv[2];pid=int(sys.argv[3])
for attempt in range(180):
    os.kill(pid,0)
    text=log.read_text(errors='replace') if log.exists() else ''
    if 'Cannot construct DefaultSceneRoot' in text:raise RuntimeError('UE map has missing blueprint dependencies')
    if f'LoadMap(/Game/{map_name})' in text:
        try:
            with socket.create_connection(('127.0.0.1',9000),timeout=1):break
        except OSError:pass
    time.sleep(1)
else:raise TimeoutError('UE map and UnrealCV did not become ready')
PY
cd "$ROOT"
env LD_PRELOAD="$PY_PRELOAD" "$PY" -u 04_AgentTask/CustomAgentTasks/delivery_cases/run.py admit \
  --case "$RUN/selected-case.json" --output "$RUN/episode" "${model_args[@]}"
