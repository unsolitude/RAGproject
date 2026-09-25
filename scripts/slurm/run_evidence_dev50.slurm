#!/usr/bin/env bash
#SBATCH --job-name=evidence-dev50
#SBATCH -p normal
#SBATCH -G 1
#SBATCH -c 1
#SBATCH --mem=1M
#SBATCH --time=06:00:00
#SBATCH --output=outputs/logs/evidence-dev50-%j.log

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/kangzj/RAGproject}"
NLI_VENV_PATH="${NLI_VENV_PATH:-/home/kangzj/venvs/ragtruth-modernbert}"
BGE_VENV_PATH="${BGE_VENV_PATH:-${NLI_VENV_PATH}}"
NLI_MODEL_PATH="${NLI_MODEL_PATH:-${PROJECT_ROOT}/models/ModernBERT-large-nli}"
BGE_MODEL_PATH="${BGE_MODEL_PATH:-${PROJECT_ROOT}/models/bge-reranker-v2-m3}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/lesson3/evidence_runs/dev50-${SLURM_JOB_ID}}"
NLI_PYTHON="${NLI_VENV_PATH}/bin/python"
BGE_PYTHON="${BGE_VENV_PATH}/bin/python"

cd "${PROJECT_ROOT}"
test -x "${NLI_PYTHON}"
test -x "${BGE_PYTHON}"
test -f "${NLI_MODEL_PATH}/config.json"
test -f "${BGE_MODEL_PATH}/config.json"
compgen -G "${NLI_MODEL_PATH}/*.safetensors" >/dev/null
compgen -G "${BGE_MODEL_PATH}/*.safetensors" >/dev/null
test -f data/ragtruth/eval_dev50_assistant_v1/dev50_manifest.json
test -f data/ragtruth/processed/sample_50.jsonl
test ! -e "${OUTPUT_DIR}"
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
