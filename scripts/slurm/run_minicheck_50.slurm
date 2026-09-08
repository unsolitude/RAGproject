#!/usr/bin/env bash
#SBATCH --job-name=minicheck-50
#SBATCH -p normal
#SBATCH -G 1
#SBATCH -c 1
#SBATCH --mem=1M
#SBATCH --time=01:00:00
#SBATCH --output=outputs/logs/minicheck-50-%j.log

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/kangzj/RAGproject}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/Bespoke-MiniCheck-7B}"
INPUT_PATH="${INPUT_PATH:-${PROJECT_ROOT}/data/ragtruth/processed/doc_claim_pairs_50.jsonl}"
JOB_TAG="${SLURM_JOB_ID:-manual}"
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_ROOT}/outputs/minicheck_results_50-${JOB_TAG}.jsonl}"
MANIFEST_PATH="${MANIFEST_PATH:-${PROJECT_ROOT}/outputs/logs/minicheck-50-${JOB_TAG}-manifest.json}"
VENV_PATH="${VENV_PATH:-/home/kangzj/venvs/ragtruth-minicheck-cu121}"
PAIR_LIMIT="${PAIR_LIMIT:-}"
RUN_ID="${RUN_ID:-minicheck-50-${JOB_TAG}}"
BATCH_SIZE="${BATCH_SIZE:-16}"

cd "${PROJECT_ROOT}"
source "${VENV_PATH}/bin/activate"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p outputs/logs
test -f "${MODEL_PATH}/config.json"
test -f "${INPUT_PATH}"
if ! compgen -G "${MODEL_PATH}/*.safetensors" >/dev/null; then
  echo "No .safetensors weights found in ${MODEL_PATH}" >&2
  exit 1
fi

echo "run_id=${RUN_ID}"
echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

run_args=(
  --input "${INPUT_PATH}"
  --output "${OUTPUT_PATH}"
  --data-format pair
  --judge minicheck
  --run-id "${RUN_ID}"
  --manifest-output "${MANIFEST_PATH}"
  --batch-size "${BATCH_SIZE}"
  --minicheck-model-path "${MODEL_PATH}"
  --minicheck-threshold 0.5
  --minicheck-max-model-len 8192
  --minicheck-tensor-parallel-size 1
  --minicheck-enable-prefix-caching
)
if [[ -n "${PAIR_LIMIT}" ]]; then
  run_args+=(--limit "${PAIR_LIMIT}")
fi

python -B src/run_baseline.py "${run_args[@]}"
python -B src/validate_pair_results.py \
  --input-pairs "${INPUT_PATH}" \
  --results "${OUTPUT_PATH}" \
  --manifest "${MANIFEST_PATH}"

echo "minicheck_pair_run_complete output=${OUTPUT_PATH} manifest=${MANIFEST_PATH}"
