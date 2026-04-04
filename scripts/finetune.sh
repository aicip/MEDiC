#!/bin/bash -l
#SBATCH -J medic-finetune
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH -A YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=YOUR_PARTITION
#SBATCH --output=logs/finetune/%x/%j.log
#SBATCH --signal=SIGUSR1@90

# MEDiC end-to-end finetuning on ImageNet-1K: 100 epochs, 4 GPUs
# Expected: ~85.0% top-1 accuracy
# Usage: sbatch scripts/finetune.sh <pretrained_checkpoint>

PRETRAINED=${1:?Usage: sbatch scripts/finetune.sh <pretrained_checkpoint_path>}
DATA=${2:-/path/to/imagenet}
OUTPUT=${3:-./output/finetune}

if [ -f .env ]; then source .env; fi
nvidia-smi
echo "Job ID: ${SLURM_JOB_ID}, Host: $(hostname)"
echo "Pretrained: ${PRETRAINED}"

cd "${SLURM_SUBMIT_DIR}" || exit 1
# module load cuda   # Uncomment and set your CUDA module
# module load cudnn  # Uncomment and set your cuDNN module
source .venv/bin/activate

unset MASTER_ADDR MASTER_PORT RANK WORLD_SIZE
export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 20000-65000 -n 1)
export NCCL_SOCKET_IFNAME=$(ip -o -4 route show to default | awk '{print $5}' | head -n1)
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TIMM_FUSED_ATTN=0

echo "Run started at: $(date)"

export PYTHONPATH="src/downstream:${PYTHONPATH}"
cd src/downstream || exit 1

srun torchrun \
  --nproc_per_node=${SLURM_GPUS_PER_NODE} \
  --nnodes=${SLURM_NNODES} \
  --node_rank=${SLURM_NODEID} \
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
  --epochs 100 \
  --lr 5e-4 \
  --layer_decay 0.65 \
  --drop_path 0.1 \
  --weight_decay 0.05 \
  --mixup 0.8 \
  --cutmix 1.0 \
  --warmup_epochs 20 \
  --output_dir "${OUTPUT}" \
  --rel_pos_bias \
  --save_ckpt \
  --dist_eval

echo "################################################################"
echo "Run completed at: $(date)"
echo "################################################################"
