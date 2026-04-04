#!/bin/bash -l
#SBATCH -J medic-verify-lp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0-01:00:00
#SBATCH -A YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=YOUR_PARTITION
#SBATCH --output=logs/verify/linprobe_%j.out
#SBATCH --error=logs/verify/linprobe_%j.err

# MEDiC Linear Probe Verification - 1 epoch, 1 GPU
# Verifies: checkpoint loading, feature extraction, linear head training
set -e

PRETRAINED=${1:?Usage: sbatch scripts/verify_linprobe.sh <checkpoint_path>}
DATA=${2:-/path/to/imagenet}

# module load cuda   # Uncomment and set your CUDA module
# module load cudnn  # Uncomment and set your cuDNN module

cd "${SLURM_SUBMIT_DIR}" || exit 1
source .venv/bin/activate

export PYTHONPATH="src/downstream:${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 20000-65000 -n 1)

echo "================================================================"
echo "MEDiC Linear Probe Verification (1 epoch)"
echo "Checkpoint: ${PRETRAINED}"
echo "================================================================"

cd src/downstream || exit 1

srun torchrun \
  --nproc_per_node=1 \
  --nnodes=1 \
  --node_rank=0 \
  --rdzv-id=${SLURM_JOB_ID} \
  --rdzv-backend=c10d \
  --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  run_linear_eval.py \
  --pretrained_weights "${PRETRAINED}" \
  --model_filter_name "module.student." \
  --data_path "${DATA}" \
  --rel_pos_bias \
  --epochs 1 \
  --batch_size 256 \
  --output_dir "/tmp/medic_verify_linprobe_${SLURM_JOB_ID}" \
  2>&1

echo "================================================================"
echo "Linear probe verification complete at: $(date)"
echo "================================================================"
