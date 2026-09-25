#!/usr/bin/env bash
#SBATCH --job-name=evidence-dev50
#SBATCH -p normal
#SBATCH -G 1
#SBATCH -c 1
#SBATCH --mem=1M
#SBATCH --time=06:00:00
#SBATCH --output=outputs/logs/evidence-dev50-%j.log

set -euo pipefail
trap 'code=$?; echo "ERROR: evidence job stopped at line ${LINENO} (exit ${code}): ${BASH_COMMAND}" >&2' ERR
PROJECT_ROOT="${PROJECT_ROOT:-/home/kangzj/RAGproject}"
NLI_VENV_PATH="${NLI_VENV_PATH:-/home/kangzj/venvs/ragtruth-modernbert}"
BGE_VENV_PATH="${BGE_VENV_PATH:-${NLI_VENV_PATH}}"
NLI_MODEL_PATH="${NLI_MODEL_PATH:-${PROJECT_ROOT}/models/ModernBERT-large-nli}"
BGE_MODEL_PATH="${BGE_MODEL_PATH:-${PROJECT_ROOT}/models/bge-reranker-v2-m3}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/lesson3/evidence_runs/dev50-${SLURM_JOB_ID}}"
NLI_PYTHON="${NLI_VENV_PATH}/bin/python"
BGE_PYTHON="${BGE_VENV_PATH}/bin/python"

echo "[preflight] job=${SLURM_JOB_ID:-manual} root=${PROJECT_ROOT}"
echo "[preflight] NLI model=${NLI_MODEL_PATH} BGE model=${BGE_MODEL_PATH}"
echo "[preflight] output=${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"
require_file() {
    if [[ ! -f "$1" ]]; then
        echo "ERROR: missing required file: $1" >&2
        return 1
    fi
}
require_python() {
    if [[ ! -x "$1" ]]; then
        echo "ERROR: missing executable Python: $1" >&2
        return 1
    fi
}
require_weights() {
    if ! compgen -G "$1/*.safetensors" >/dev/null; then
        echo "ERROR: no .safetensors weights directly inside $1" >&2
        echo "       Check whether upload created a nested model folder." >&2
        return 1
    fi
}
require_python "${NLI_PYTHON}"
require_python "${BGE_PYTHON}"
require_file "${NLI_MODEL_PATH}/config.json"
require_file "${BGE_MODEL_PATH}/config.json"
require_weights "${NLI_MODEL_PATH}"
require_weights "${BGE_MODEL_PATH}"
require_file data/ragtruth/eval_dev50_assistant_v1/dev50_manifest.json
require_file data/ragtruth/processed/sample_50.jsonl
if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "ERROR: output directory already exists: ${OUTPUT_DIR}" >&2
    exit 1
fi
echo "[preflight] required files present"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
"${NLI_PYTHON}" -c "import torch, transformers; assert torch.cuda.is_available()"
"${BGE_PYTHON}" -c "import torch, transformers; assert torch.cuda.is_available()"
nvidia-smi

prepare_args=(prepare --output-dir "${OUTPUT_DIR}" --model-path "${NLI_MODEL_PATH}")
if [[ -n "${LIMIT:-}" ]]; then
    prepare_args+=(--limit "${LIMIT}")
fi
"${NLI_PYTHON}" -B scripts/run_evidence_experiment.py "${prepare_args[@]}"
"${BGE_PYTHON}" -B scripts/score_bge_chunks.py \
    --candidate-pool "${OUTPUT_DIR}/candidate_pool.jsonl" \
    --output "${OUTPUT_DIR}/bge_scores.jsonl" \
    --model-path "${BGE_MODEL_PATH}"
"${NLI_PYTHON}" -B scripts/run_evidence_experiment.py materialize-bge \
    --output-dir "${OUTPUT_DIR}" --model-path "${NLI_MODEL_PATH}"
"${NLI_PYTHON}" -B scripts/run_evidence_experiment.py run-nli \
    --output-dir "${OUTPUT_DIR}" --model-path "${NLI_MODEL_PATH}" \
    --batch-size "${BATCH_SIZE:-4}"
"${NLI_PYTHON}" -B src/evaluate_evidence_selection.py --output-dir "${OUTPUT_DIR}"
echo "Completed evidence experiment: ${OUTPUT_DIR}"
