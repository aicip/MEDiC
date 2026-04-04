#!/bin/bash -l
#SBATCH -J medic-verify-ft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0-01:00:00
#SBATCH -A YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=YOUR_PARTITION
#SBATCH --output=logs/verify/finetune_%j.out
#SBATCH --error=logs/verify/finetune_%j.err

# MEDiC Finetune Verification - 1 epoch, 1 GPU
# Verifies: checkpoint loading, weight extraction, training loop
set -e

PRETRAINED=${1:?Usage: sbatch scripts/verify_finetune.sh <checkpoint_path>}
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
echo "MEDiC Finetune Verification (1 epoch)"
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
  run_class_finetuning.py \
  --model vit_base_patch16_224 \
  --data_path "${DATA}" \
  --finetune "${PRETRAINED}" \
  --model_filter_name "module.student." \
  --nb_classes 1000 \
  --batch_size 128 \
  --epochs 1 \
  --lr 5e-4 \
  --layer_decay 0.65 \
  --drop_path 0.1 \
  --weight_decay 0.05 \
  --mixup 0.8 \
  --cutmix 1.0 \
  --warmup_epochs 0 \
  --output_dir "/tmp/medic_verify_finetune_${SLURM_JOB_ID}" \
  --rel_pos_bias \
  --dist_eval \
  2>&1

echo "================================================================"
echo "Finetune verification complete at: $(date)"
echo "================================================================"
