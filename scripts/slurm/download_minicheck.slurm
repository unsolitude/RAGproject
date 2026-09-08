#!/usr/bin/env bash
#SBATCH --job-name=minicheck-download
#SBATCH -c 1
#SBATCH --mem=1M
#SBATCH -p normal
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/minicheck-download-%j.log

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/kangzj/RAGproject}"
VENV_PATH="${VENV_PATH:-/home/kangzj/venvs/ragtruth-minicheck-cu121}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/Bespoke-MiniCheck-7B}"

cd "${PROJECT_ROOT}"
source "${VENV_PATH}/bin/activate"

export HF_HOME="${PROJECT_ROOT}/models/huggingface"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_TELEMETRY=1

mkdir -p "${HF_HOME}" "${MODEL_PATH}"

echo "Starting model download"
date
python --version

huggingface-cli download \
  bespokelabs/Bespoke-MiniCheck-7B \
  --local-dir "${MODEL_PATH}" \
  --cache-dir "${HF_HOME}"

test -f "${MODEL_PATH}/config.json"
if ! compgen -G "${MODEL_PATH}/*.safetensors" >/dev/null; then
  echo "No .safetensors weights found in ${MODEL_PATH}" >&2
  exit 1
fi

echo "Download completed"
date
du -sh "${MODEL_PATH}"
