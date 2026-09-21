#!/usr/bin/env bash
#SBATCH --job-name=modernbert-512-dev
#SBATCH -p normal
#SBATCH -G 1
#SBATCH -c 1
#SBATCH --mem=1M
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/modernbert-512-dev-%j.log

# Dedicated development control: no test200 and no inherited fusion inputs.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/kangzj/RAGproject}"
VENV_PATH="${VENV_PATH:-/home/kangzj/venvs/ragtruth-modernbert}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/ModernBERT-large-nli}"
cd "${PROJECT_ROOT}"
source "${VENV_PATH}/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
nvidia-smi
python -m pip check
python -B scripts/check_modernbert_control.py --model-path "${MODEL_PATH}"
python -B scripts/run_modernbert_suite.py \
    --model-path "${MODEL_PATH}" \
    --output-dir "${PROJECT_ROOT}/outputs/modernbert-dev-512-${SLURM_JOB_ID}" \
    --stage dev --max-length 512 --batch-size 4 --review-threshold 0.7
