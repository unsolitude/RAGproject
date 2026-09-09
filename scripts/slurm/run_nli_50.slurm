#!/usr/bin/env bash
#SBATCH --job-name=nli-50
#SBATCH -p normal
#SBATCH -G 1
#SBATCH -c 1
#SBATCH --mem=1M
#SBATCH --time=01:00:00
#SBATCH --output=outputs/logs/nli-50-%j.log

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/kangzj/RAGproject}"
INPUT_PATH="${INPUT_PATH:-${PROJECT_ROOT}/data/ragtruth/processed/doc_claim_pairs_50.jsonl}"
JOB_TAG="${SLURM_JOB_ID:-manual}"
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_ROOT}/outputs/nli_results_50-${JOB_TAG}.jsonl}"
MANIFEST_PATH="${MANIFEST_PATH:-${PROJECT_ROOT}/outputs/logs/nli-50-${JOB_TAG}-manifest.json}"
VENV_PATH="${VENV_PATH:-/home/kangzj/venvs/ragtruth-nli}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-${PROJECT_ROOT}/models/huggingface}"
NLI_MODEL="${NLI_MODEL:-cross-encoder/nli-deberta-v3-small}"
NLI_REVISION="${NLI_REVISION:-fa2804872c3b4bd748f38c0185cc85775361e735}"
PAIR_LIMIT="${PAIR_LIMIT:-}"
RUN_ID="${RUN_ID:-nli-50-${JOB_TAG}}"
BATCH_SIZE="${BATCH_SIZE:-16}"
REVIEW_THRESHOLD="${REVIEW_THRESHOLD:-0.6}"

cd "${PROJECT_ROOT}"
source "${VENV_PATH}/bin/activate"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

mkdir -p outputs/logs
test -f "${INPUT_PATH}"

echo "run_id=${RUN_ID}"
echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

run_args=(
  --input "${INPUT_PATH}"
  --output "${OUTPUT_PATH}"
  --data-format pair
  --judge nli
  --run-id "${RUN_ID}"
  --manifest-output "${MANIFEST_PATH}"
  --batch-size "${BATCH_SIZE}"
  --device cuda
  --nli-model "${NLI_MODEL}"
  --nli-revision "${NLI_REVISION}"
  --model-cache-dir "${MODEL_CACHE_DIR}"
  --entailment-threshold 0.5
  --contradiction-threshold 0.5
  --nli-review-threshold "${REVIEW_THRESHOLD}"
)
if [[ -n "${PAIR_LIMIT}" ]]; then
  run_args+=(--limit "${PAIR_LIMIT}")
fi

python -B src/run_baseline.py "${run_args[@]}"
python -B src/validate_pair_results.py \
  --input-pairs "${INPUT_PATH}" \
  --results "${OUTPUT_PATH}" \
  --manifest "${MANIFEST_PATH}"

echo "nli_pair_run_complete output=${OUTPUT_PATH} manifest=${MANIFEST_PATH}"
