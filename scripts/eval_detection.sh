#!/bin/bash -l
#SBATCH -J medic-det-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0-02:00:00
#SBATCH -A YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=YOUR_PARTITION
#SBATCH --output=logs/detection/%x/%j.log
#SBATCH --signal=SIGUSR1@90

# Evaluate MEDiC detection checkpoint on COCO
# Usage: sbatch scripts/eval_detection.sh <checkpoint_path> [coco_root]
# Expected: bbox_mAP ~44.4, segm_mAP ~40.1

CHECKPOINT=${1:?Usage: sbatch scripts/eval_detection.sh <checkpoint_path> [coco_root]}
COCO_ROOT=${2:-/path/to/coco}

if [ -f .env ]; then source .env; fi
nvidia-smi
echo "Job ID: ${SLURM_JOB_ID}, Host: $(hostname)"
echo "Checkpoint: ${CHECKPOINT}"

cd "${SLURM_SUBMIT_DIR}" || exit 1
# module load cuda   # Uncomment and set your CUDA module
# module load cudnn  # Uncomment and set your cuDNN module
source .venv/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Use OUR config (type='MEDiC', no MoE, pretrained=None)
DET_DIR="src/downstream/detection"

echo "Run started at: $(date)"

export PYTHONPATH="${DET_DIR}:${PYTHONPATH}"
cd "${DET_DIR}" || exit 1

python tools/test.py \
  configs/mask_rcnn/medic_base_maskrcnn_1x_coco.py \
  "${CHECKPOINT}" \
  --eval bbox segm \
  --cfg-options data_root="${COCO_ROOT}" data.test.data_root="${COCO_ROOT}" data.test.ann_file="${COCO_ROOT}/annotations/instances_val2017.json" data.test.img_prefix="${COCO_ROOT}/val2017/"

echo "################################################################"
echo "Run completed at: $(date)"
echo "################################################################"
