#!/bin/bash -l
#SBATCH -J medic-linprobe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH -A YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=YOUR_PARTITION
#SBATCH --output=logs/linprobe/%x/%j.log
#SBATCH --signal=SIGUSR1@90

# MEDiC linear probe evaluation on ImageNet-1K: 90 epochs, 4 GPUs
# Expected: ~81.2% top-1 accuracy
# Usage: sbatch scripts/linprobe.sh <pretrained_checkpoint>

PRETRAINED=${1:?Usage: sbatch scripts/linprobe.sh <pretrained_checkpoint_path>}
DATA=${2:-/path/to/imagenet}
OUTPUT=${3:-./output/linprobe}

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
  run_linear_eval.py \
  --pretrained_weights "${PRETRAINED}" \
  --model_filter_name "module.student." \
  --data_path "${DATA}" \
  --batch_size 256 \
  --epochs 90 \
  --lr 0.1 \
  --output_dir "${OUTPUT}" \
  --rel_pos_bias

echo "################################################################"
echo "Run completed at: $(date)"
echo "################################################################"
