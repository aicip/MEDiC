#!/bin/bash -l
#SBATCH --job-name=eval-semseg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_%j.log

# ==============================================
# Semantic Segmentation Evaluation Script
# Usage: sbatch submit_eval.sh CHECKPOINT [CONFIG]
# ==============================================

CHECKPOINT=${1:-""}
CONFIG=${2:-"configs/medic/upernet_medic_test_ade20k.py"}

if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint path is required"
    echo "Usage: sbatch submit_eval.sh CHECKPOINT [CONFIG]"
    exit 1
fi

echo "=============================================="
echo "Semantic Segmentation Evaluation"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Config: $CONFIG"
echo "Checkpoint: $CHECKPOINT"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "=============================================="

# Change to segmentation directory
cd "$(dirname "$0")/.."

# Create log directory
mkdir -p logs

echo ""
echo "Running evaluation..."
echo "=============================================="

python tools/test.py \
    "$CONFIG" \
    "$CHECKPOINT" \
    --eval mIoU

EXIT_CODE=$?

echo "=============================================="
echo "Evaluation completed at $(date)"
echo "Exit code: $EXIT_CODE"
echo "=============================================="
