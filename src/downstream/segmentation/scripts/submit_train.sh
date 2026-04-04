#!/bin/bash -l
#SBATCH --job-name=medic_seg
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/seg_%j.log
#SBATCH --error=logs/seg_%j.err

# --- Environment Setup ---
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
nvidia-smi

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment (update this path for your environment)
# source /path/to/your/.venv/bin/activate

# Change to semantic segmentation directory
cd "$(dirname "$0")/.."

# Set environment variables
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Path to pretrained checkpoint (update this)
CHECKPOINT="${1:?ERROR: Provide checkpoint path as first argument}"

CONFIG="configs/medic/upernet_medic_base_512_160k_ade20k.py"
WORK_DIR="work_dirs/medic_seg_${SLURM_JOB_ID}"

echo "Using checkpoint: $CHECKPOINT"
echo "Config: $CONFIG"

# Run training with distributed launch (4 GPUs)
torchrun --nproc_per_node=4 \
    --master_port=$((12000 + RANDOM % 1000)) \
    tools/train.py \
    ${CONFIG} \
    --launcher pytorch \
    --options model.backbone.pretrained=${CHECKPOINT} \
    --work-dir ${WORK_DIR} \
    --seed 0

echo "Training completed at $(date)"
