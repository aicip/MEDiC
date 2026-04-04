#!/bin/bash
#SBATCH --job-name=medic-smoke
#SBATCH -A YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=YOUR_PARTITION
#SBATCH --export=ALL,OMP_NUM_THREADS=16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/smoke_test_%j.out
#SBATCH --error=logs/smoke_test_%j.err

# MEDiC GPU Smoke Test
# Tests all 4 config variants with real CLIP teacher, bf16, synthetic data
# Usage: sbatch scripts/smoke_test.sh

set -e

# module load cuda   # Uncomment and set your CUDA module
# module load cudnn  # Uncomment and set your cuDNN module

REPO_DIR=$(dirname "$(dirname "$(readlink -f "$0")")")
cd "$REPO_DIR"

echo "================================================================"
echo "MEDiC GPU Smoke Test"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "SLURM Job: ${SLURM_JOB_ID:-none}"
echo "================================================================"

source .venv/bin/activate
export PYTHONPATH="$REPO_DIR"
export OMP_NUM_THREADS=16

python "$REPO_DIR/tests/test_gpu_smoke.py"

echo ""
echo "================================================================"
echo "GPU Smoke test completed successfully"
echo "================================================================"
