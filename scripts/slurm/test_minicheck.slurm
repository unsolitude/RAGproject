#!/usr/bin/env bash
#SBATCH --job-name=minicheck-smoke
#SBATCH -p normal
#SBATCH -G 1
#SBATCH -c=1
#SBATCH --mem=1M
#SBATCH --time=00:30:00
#SBATCH --output=/home/kangzj/RAGproject/outputs/logs/minicheck-smoke-%j.log

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/kangzj/RAGproject}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/Bespoke-MiniCheck-7B}"
INPUT_PATH="${INPUT_PATH:-${PROJECT_ROOT}/data/ragtruth/processed/qa_one.jsonl}"
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_ROOT}/outputs/minicheck_smoke_result.jsonl}"
METRICS_PATH="${METRICS_PATH:-${PROJECT_ROOT}/outputs/minicheck_smoke_metrics.json}"
VENV_PATH="${VENV_PATH:-/home/kangzj/venvs/ragtruth-minicheck-cu121}"

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

echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi
python -c "import torch, transformers, vllm; print('torch', torch.__version__); print('transformers', transformers.__version__); print('vllm', vllm.__version__); print('gpu', torch.cuda.get_device_name(0))"

python -B src/run_baseline.py \
  --input "${INPUT_PATH}" \
  --output "${OUTPUT_PATH}" \
  --judge minicheck \
  --minicheck-model-path "${MODEL_PATH}" \
  --minicheck-threshold 0.5 \
  --minicheck-max-model-len 8192 \
  --minicheck-tensor-parallel-size 1 \
  --minicheck-enable-prefix-caching

python -B src/evaluate_results.py \
  --input "${OUTPUT_PATH}" \
  --output "${METRICS_PATH}"

echo "smoke_test_complete output=${OUTPUT_PATH} metrics=${METRICS_PATH}"
