#!/usr/bin/env bash
#SBATCH --job-name=order-dev50
#SBATCH -p normal
#SBATCH -G 1
#SBATCH -c 1
#SBATCH --mem=1M
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/order-dev50-%j.log

set -euo pipefail
trap 'code=$?; echo "ERROR: order control stopped at line ${LINENO} (exit ${code}): ${BASH_COMMAND}" >&2' ERR
PROJECT_ROOT="${PROJECT_ROOT:-/home/kangzj/RAGproject}"
NLI_VENV_PATH="${NLI_VENV_PATH:-/home/kangzj/venvs/ragtruth-modernbert}"
NLI_MODEL_PATH="${NLI_MODEL_PATH:-${PROJECT_ROOT}/models/ModernBERT-large-nli}"
SOURCE_RUN="${SOURCE_RUN:-${PROJECT_ROOT}/outputs/lesson3/evidence_runs/dev50-39484}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/lesson3/order_runs/dev50-${SLURM_JOB_ID}}"
NLI_PYTHON="${NLI_VENV_PATH}/bin/python"
echo "[preflight] source=${SOURCE_RUN} output=${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"
for path in "${NLI_PYTHON}" "${NLI_MODEL_PATH}/config.json" "${SOURCE_RUN}/experiment_manifest.json"; do
    if [[ ! -f "${path}" ]]; then echo "ERROR: missing ${path}" >&2; exit 1; fi
done
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
"${NLI_PYTHON}" -c "import torch, transformers; assert torch.cuda.is_available(), 'GPU allocation unavailable'"
nvidia-smi
args=(--source-run "${SOURCE_RUN}" --output-dir "${OUTPUT_DIR}")
prepare_args=("${args[@]}")
if [[ -n "${LIMIT:-}" ]]; then prepare_args+=(--limit "${LIMIT}"); fi
"${NLI_PYTHON}" -B scripts/run_order_control.py prepare "${prepare_args[@]}"
"${NLI_PYTHON}" -B scripts/run_order_control.py run "${args[@]}" \
    --model-path "${NLI_MODEL_PATH}" --device cuda --batch-size "${BATCH_SIZE:-4}"
"${NLI_PYTHON}" -B scripts/run_order_control.py evaluate "${args[@]}"
echo "Completed order control: ${OUTPUT_DIR}"
