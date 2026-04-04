#!/bin/bash -l
#SBATCH -J medic-smoke-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0-00:30:00
#SBATCH -A YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=YOUR_PARTITION
#SBATCH --output=logs/smoke_train_%j.out
#SBATCH --error=logs/smoke_train_%j.err

# MEDiC Real-Data Training Smoke Test
# Runs 2 epochs on 5 images/class subset (~5000 images) with pretrain_medic.yaml
# Tests: data loading, masking, multi-loss, checkpointing, W&B logging, validation

set -e

# module load cuda   # Uncomment and set your CUDA module
# module load cudnn  # Uncomment and set your cuDNN module

cd "${SLURM_SUBMIT_DIR}" || exit 1

echo "================================================================"
echo "MEDiC Real-Data Training Smoke Test"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "SLURM Job: $SLURM_JOB_ID"
echo "================================================================"

source .venv/bin/activate
export PYTHONPATH="${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Distributed setup (single GPU)
export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 20000-65000 -n 1)

CONFIG="configs/pretrain_medic_smoke.yaml"

# Override subset and batch size for smoke test
# subset=5 -> 5 images/class -> 5000 total -> ~78 steps/epoch with batch=64
# batch_per_gpu=64 (smaller to fit in memory with decoder)
srun torchrun \
  --nproc_per_node=1 \
  --nnodes=1 \
  --node_rank=0 \
  --rdzv-id=${SLURM_JOB_ID} \
  --rdzv-backend=c10d \
  --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  -m src.train \
  --cfg "$CONFIG" \
  --epochs 2 \
  2>&1

echo ""
echo "================================================================"
echo "Real-data smoke test completed at: $(date)"
echo "================================================================"
