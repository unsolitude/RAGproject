#!/usr/bin/env bash
#SBATCH --job-name=minicheck-download
#SBATCH -G 1
#SBATCH -c 1
#SBATCH --mem=1M
#SBATCH -p normal
#SBATCH --time=04:00:00
#SBATCH --output=/home/kangzj/RAGproject/outputs/logs/minicheck-download-%j.log

set -euo pipefail

cd /home/kangzj/RAGproject
source /home/kangzj/venvs/ragtruth-minicheck-cu121/bin/activate

export HF_HOME=/home/kangzj/RAGproject/models/huggingface
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_TELEMETRY=1

mkdir -p "$HF_HOME"
mkdir -p /home/kangzj/RAGproject/models/minicheck

echo "Starting model download"
date
python --version

huggingface-cli download \
  bespokelabs/Bespoke-MiniCheck-7B \
  --cache-dir /home/kangzj/RAGproject/models/minicheck

echo "Download completed"
date
du -sh /home/kangzj/RAGproject/models/minicheck