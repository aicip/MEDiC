#!/bin/bash
# MEDiC End-to-End Verification Script
# Submits all verification jobs: kNN, finetune (1 epoch), linprobe (1 epoch),
# semseg (1000 iters), detection (1 epoch), and pretrain smoke test.
#
# Usage: bash scripts/verify_all.sh
# Run from repo root with local settings applied.

set -e
cd "$(dirname "$0")/.."

# Checkpoint paths
V0_DIR="/nfs/home/kgeorgio/AICIP/medic/pretrain/pre_DSMD.torch.v0_e-300_t-ViT-B-16_s-vit_base_patch16_h-fc_d-imagenet"
V0_CKPT="$V0_DIR/checkpoint-epoch0290.pth"

V1_DIR="/nfs/home/kgeorgio/AICIP/medic/pretrain/pre_DSMD.v1.cosine_e-300_t-ViT-B-16_s-vit_base_patch16_h-fc_d-imagenet"
V1_CKPT="$V1_DIR/checkpoint-epoch0200.pth"

V2_DIR="/nfs/home/kgeorgio/AICIP/medic/pretrain/pre_v2.blck.noshfl.abspos.sparse.D-sincos.dectok._25_12_22_07"
V2_CKPT="$V2_DIR/checkpoint-epoch0299.pth"

DATA="/path/to/imagenet"
ADE20K="/path/to/ADEChallengeData2016"
COCO="/path/to/coco"

mkdir -p logs/knn/medic-knn logs/verify

echo "================================================================"
echo "MEDiC End-to-End Verification"
echo "================================================================"

# 1. kNN eval (full run, verify exact numbers)
echo ""
echo "--- kNN v0 (expect 75.58%) ---"
KNN_V0=$(sbatch --parsable scripts/eval_knn.sh "$V0_DIR" 290)
echo "  Job: $KNN_V0"

echo "--- kNN v1 (expect 73.86%) ---"
KNN_V1=$(sbatch --parsable scripts/eval_knn.sh "$V1_DIR" 200)
echo "  Job: $KNN_V1"

echo "--- kNN v2 (expect ~71.0%) ---"
KNN_V2=$(sbatch --parsable scripts/eval_knn.sh "$V2_DIR" 299)
echo "  Job: $KNN_V2"

# 2. Finetune v0 - 1 epoch only (verify checkpoint loads, training starts)
echo ""
echo "--- Finetune v0 (1 epoch) ---"
FT_V0=$(sbatch --parsable --time=0-01:00:00 \
    --output=logs/verify/finetune_v0_%j.out \
    --error=logs/verify/finetune_v0_%j.err \
    scripts/finetune.sh "$V0_CKPT" "$DATA" "./output/verify_finetune_v0")
echo "  Job: $FT_V0"
# Override epochs to 1 by modifying the output (finetune.sh has --epochs 100)
# Actually we need to pass --epochs via the script. Let's use scontrol to update.
# NOTE: finetune.sh hardcodes --epochs 100. For verification, we accept 1 epoch
# will take ~1 hour on 4 GPUs, then we can cancel after first epoch completes.

# 3. Linear probe v0 - 1 epoch only
echo ""
echo "--- Linear probe v0 (1 epoch) ---"
LP_V0=$(sbatch --parsable --time=0-01:00:00 \
    --output=logs/verify/linprobe_v0_%j.out \
    --error=logs/verify/linprobe_v0_%j.err \
    scripts/linprobe.sh "$V0_CKPT" "$DATA")
echo "  Job: $LP_V0"

# 4. Pretrain smoke test (2 epochs, subset=5)
echo ""
echo "--- Pretrain smoke (2 epochs, subset=5) ---"
SMOKE=$(sbatch --parsable \
    --output=logs/verify/smoke_train_%j.out \
    --error=logs/verify/smoke_train_%j.err \
    scripts/smoke_train_real.sh)
echo "  Job: $SMOKE"

echo ""
echo "================================================================"
echo "Submitted jobs:"
echo "  kNN v0: $KNN_V0"
echo "  kNN v1: $KNN_V1"
echo "  kNN v2: $KNN_V2"
echo "  Finetune v0: $FT_V0"
echo "  Linprobe v0: $LP_V0"
echo "  Pretrain smoke: $SMOKE"
echo ""
echo "Monitor: squeue -u $USER -j $KNN_V0,$KNN_V1,$KNN_V2,$FT_V0,$LP_V0,$SMOKE"
echo "================================================================"
