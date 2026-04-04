#!/bin/bash -l
#SBATCH -J medic-verify-det
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=0-01:00:00
#SBATCH -A YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=YOUR_PARTITION
#SBATCH --output=logs/verify/detection_%j.out
#SBATCH --error=logs/verify/detection_%j.err

# MEDiC Detection Verification - 1 epoch, 1 GPU
# Verifies: backbone loading, FPN features, Mask R-CNN training loop
set -e

PRETRAINED=${1:?Usage: sbatch scripts/verify_detection.sh <checkpoint_path>}
COCO_ROOT=${2:-/path/to/coco}

# module load cuda   # Uncomment and set your CUDA module
# module load cudnn  # Uncomment and set your cuDNN module

cd "${SLURM_SUBMIT_DIR}" || exit 1
source .venv/bin/activate

DET_DIR="src/downstream/detection"
export PYTHONPATH="${DET_DIR}:${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "================================================================"
echo "MEDiC Detection Verification (1 epoch)"
echo "Checkpoint: ${PRETRAINED}"
echo "COCO: ${COCO_ROOT}"
echo "================================================================"

cd "${DET_DIR}" || exit 1

python tools/train.py \
  "configs/mask_rcnn/medic_base_maskrcnn_1x_coco.py" \
  --work-dir "/tmp/medic_verify_detection_${SLURM_JOB_ID}" \
  --load-from "${PRETRAINED}" \
  --cfg-options \
    data.samples_per_gpu=2 \
    runner.max_epochs=1 \
    data_root="${COCO_ROOT}" \
  --no-validate \
  2>&1

echo "================================================================"
echo "Detection verification complete at: $(date)"
echo "================================================================"
