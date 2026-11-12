#!/bin/bash
#SBATCH --job-name=demo-clean-render
#SBATCH --account=video
#SBATCH --partition=gpu-h200-mig
#SBATCH --qos=debug
# CPU (lavapipe) rendering: one MIG slice is the cheapest allocation; MIG allows 2 CPUs per GPU.
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/scrubbed/xuebiz/SpatialAgent/05_Experiment/logs/demo-clean-render-%j.log
set -euo pipefail
ROOT=/gpfs/scrubbed/xuebiz/SpatialAgent
RUN="$ROOT/05_Experiment/results/demo-clean-render-$SLURM_JOB_ID"
export CUDA_VISIBLE_DEVICES=''
export SPATIAL_THOR_SOFTWARE=1
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
export LP_NUM_THREADS=2 OMP_NUM_THREADS=2
export XDG_CONFIG_HOME="$RUN/unity-config"
export XDG_CACHE_HOME="$ROOT/.cache"
cd "$ROOT"
"$ROOT/02_Simulator/AI2THOR/runtime/python/bin/python" \
  04_AgentTask/Demos/spatial_agent/rerender.py --output "$RUN" --render-size 512 "${@:---presentation-only}"
