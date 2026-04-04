#!/bin/bash -l
#SBATCH -J medic-semseg-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0-01:00:00
#SBATCH -A YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=YOUR_PARTITION
#SBATCH --output=logs/semseg/%x/%j.log
#SBATCH --signal=SIGUSR1@90

# Evaluate MEDiC semseg checkpoint on ADE20K
# Usage: sbatch scripts/eval_semseg.sh <checkpoint_path> [ade20k_root]
# Expected: ~52.6 mIoU

CHECKPOINT=${1:?Usage: sbatch scripts/eval_semseg.sh <checkpoint_path> [ade20k_root]}
ADE20K_ROOT=${2:-/path/to/ADEChallengeData2016}

if [ -f .env ]; then source .env; fi
nvidia-smi
echo "Job ID: ${SLURM_JOB_ID}, Host: $(hostname)"
echo "Checkpoint: ${CHECKPOINT}"

cd "${SLURM_SUBMIT_DIR}" || exit 1
# module load cuda   # Uncomment and set your CUDA module
# module load cudnn  # Uncomment and set your cuDNN module
source .venv/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TIMM_FUSED_ATTN=0

# Use OUR config (type='MEDiC', no MoE params, pretrained=None)
SEMSEG_DIR="src/downstream/segmentation"
CONFIG="${SEMSEG_DIR}/configs/medic/upernet_medic_test_ade20k.py"

echo "Config: ${CONFIG}"
echo "Run started at: $(date)"

export PYTHONPATH="${SEMSEG_DIR}:${PYTHONPATH}"
cd "${SEMSEG_DIR}" || exit 1

python tools/test.py \
  configs/medic/upernet_medic_test_ade20k.py \
  "${CHECKPOINT}" \
  --eval mIoU \
  --options data_root="${ADE20K_ROOT}"

echo "################################################################"
echo "Run completed at: $(date)"
echo "################################################################"
