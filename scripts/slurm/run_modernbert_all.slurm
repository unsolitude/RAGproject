#!/usr/bin/env bash
#SBATCH --job-name=modernbert-nli
#SBATCH -p normal
#SBATCH -G 1
#SBATCH -c 1
#SBATCH --mem=1M
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/modernbert-%j.log

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/kangzj/RAGproject}"
VENV_PATH="${VENV_PATH:-/home/kangzj/venvs/ragtruth-modernbert}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/ModernBERT-large-nli}"
STAGE="${STAGE:-dev}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
BATCH_SIZE="${BATCH_SIZE:-4}"
REVIEW_THRESHOLD="${REVIEW_THRESHOLD:-0.7}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/modernbert-${STAGE}-${MAX_LENGTH}-${SLURM_JOB_ID}}"
cd "${PROJECT_ROOT}"
source "${VENV_PATH}/bin/activate"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
echo "python=$(command -v python) model=${MODEL_PATH} output=${OUTPUT_DIR}"
nvidia-smi
python -m pip check
args=(--model-path "${MODEL_PATH}" --output-dir "${OUTPUT_DIR}" --stage "${STAGE}"
      --max-length "${MAX_LENGTH}" --batch-size "${BATCH_SIZE}" --review-threshold "${REVIEW_THRESHOLD}")
if [[ -n "${MINICHECK_RESULTS:-}" ]]; then
    args+=(--minicheck-results "${MINICHECK_RESULTS}")
fi
python -B scripts/run_modernbert_suite.py "${args[@]}"
