#!/bin/bash
# Run the official OpenEQA LLM-Match judge after exporting complete predictions.
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo 'Usage: judge_openeqa.sh /path/to/complete_openeqa_predictions.json' >&2
  exit 2
fi
if [[ -z "${OPENAI_API_KEY:-}" || "${OPENAI_API_KEY}" == local-unused ]]; then
  echo 'OPENAI_API_KEY must be set for the official GPT-4 LLM-Match judge' >&2
  exit 2
fi
ROOT=/gpfs/scrubbed/xuebiz/SpatialAgent
SOURCE="$ROOT/03_SpatialBench/benchmarks/OpenEQA/source"
export PATH=/gpfs/scrubbed/xuebiz/conda/envs/spatialagent/bin:$PATH
export PYTHONPATH="$SOURCE:${PYTHONPATH:-}"
PREDICTIONS=$1
OUTPUT_DIR=$(dirname "$PREDICTIONS")/official_judge
python "$SOURCE/evaluate-predictions.py" "$PREDICTIONS" \
  --dataset "$SOURCE/data/open-eqa-v0.json" --output-directory "$OUTPUT_DIR"
