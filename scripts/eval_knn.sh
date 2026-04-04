#!/bin/bash -l
#SBATCH -J medic-knn
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0-02:00:00
#SBATCH -A YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=YOUR_PARTITION
#SBATCH --output=logs/knn/%x/%j.log
#SBATCH --signal=SIGUSR1@90

# ────────────────────────────────
# MEDiC k-NN Evaluation
# Expected result: ~75.0% top-1 (k=20)
# ────────────────────────────────

# Optional user secrets
if [ -f .env ]; then
  source .env
fi

# Hardware info
nvidia-smi
echo "Running on host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"

cd "${SLURM_SUBMIT_DIR}" || exit 1

# Environment
# module load cuda   # Uncomment and set your CUDA module
# module load cudnn  # Uncomment and set your cuDNN module
source .venv/bin/activate

# Distributed setup
unset MASTER_ADDR MASTER_PORT RANK WORLD_SIZE
export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 20000-65000 -n 1)
export NCCL_SOCKET_IFNAME=$(ip -o -4 route show to default | awk '{print $5}' | head -n1)
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${SLURM_SUBMIT_DIR}"

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "Run started at: $(date)"

# ────────────────────────────────
# Configuration
WEIGHTS_FOLDER="${1:?Usage: sbatch scripts/eval_knn.sh <checkpoint_folder> <epoch>}"
EPOCH=${2:-290}
CONFIG="configs/pretrain_medic.yaml"

# ────────────────────────────────
srun torchrun \
  --nproc_per_node=${SLURM_GPUS_PER_NODE} \
  --nnodes=${SLURM_NNODES} \
  --node_rank=${SLURM_NODEID} \
  --rdzv-id=${SLURM_JOB_ID} \
  --rdzv-backend=c10d \
  --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  -m src.eval_knn \
  --weights_folder "${WEIGHTS_FOLDER}" \
  --epoch ${EPOCH} \
  --cfg "${CONFIG}" \
  --checkpoint_key 'model' \
  --batch_size 256 \
  --k_values 10 20 100 200 \
  --temperature 0.07 \
  --n_last_blocks 1 \
  --avgpool_patchtokens 0

echo "################################################################"
echo "Run completed at: $(date)"
echo "################################################################"
