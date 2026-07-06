#!/bin/bash
#SBATCH --job-name=parasitic_attack
#SBATCH --output=logs/parasitic_attack_%j.txt
#SBATCH --error=logs/parasitic_attack_%j.err
#SBATCH --nodes=1
#SBATCH --partition=nvidia
#SBATCH --cpus-per-task=4
#SBATCH --mem=80GB
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00

set -eo pipefail

source ~/.bashrc
set +u
eval "$(conda shell.bash hook)"
conda activate deeplearning
set -u

pip install "numpy<2.0" opencv-python-headless

REPO_ROOT="/scratch/ss17886/parasitic-backdoor-attack"
cd "$REPO_ROOT"
mkdir -p logs bb_export/parasitic_attack

echo "=========================================="
echo "PARASITIC ATTACK + BACKDOORBENCH EXPORT"
echo "=========================================="

python run_parasitic_attack.py \
  --seed 0 \
  --target-class 3 \
  --k-hosts 1000 \
  --poison-budget 1000 \
  --batch-size 1024 \
  --epochs-base 100 \
  --epochs-coadapt 10 \
  --data-dir "$REPO_ROOT/BackdoorBench/data" \
  --bb-root "$REPO_ROOT/BackdoorBench" \
  --export-dir "$REPO_ROOT/bb_export/parasitic_attack"

echo "=========================================="
echo "EXPORT COMPLETE"
echo "Artifact: $REPO_ROOT/bb_export/parasitic_attack/attack_result.pt"
echo "Next step: sbatch BackdoorBench/run_defenses.sh"
echo "=========================================="
