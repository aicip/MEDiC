# Object Detection with MEDiC

This directory contains code for evaluating MEDiC pretrained ViT-Base on COCO object detection and instance segmentation using **Mask R-CNN** with an **FPN** neck.

The implementation is based on [mmdetection](https://github.com/open-mmlab/mmdetection) and follows the evaluation protocol from [CAE](https://arxiv.org/abs/2202.03026).

## Prerequisites

```bash
pip install mmcv-full==1.7.2 mmdet==2.28.2
```

> **Note:** This codebase includes torchvision-based wrappers for NMS and RoI Align, so you do **not** need to compile mmcv with CUDA ops. Any `mmcv-full` installation will work.

## Dataset

Download COCO 2017 from https://cocodataset.org and update the `data_root` path in `configs/_base_/datasets/coco_instance.py`.

Expected structure:
```
/path/to/coco/
  annotations/
    instances_train2017.json
    instances_val2017.json
  train2017/
  val2017/
```

## Training

### Single-node distributed training (4 GPUs)

```bash
cd src/downstream/detection

# Using dist_train.sh
bash tools/dist_train.sh \
    configs/mask_rcnn/medic_base_maskrcnn_1x_coco.py \
    4 \
    --load-from /path/to/medic_pretrained_checkpoint.pth
```

### With SLURM

```bash
cd src/downstream/detection

srun --gres=gpu:4 --ntasks=4 --ntasks-per-node=4 \
    python tools/train.py \
    configs/mask_rcnn/medic_base_maskrcnn_1x_coco.py \
    --load-from /path/to/medic_pretrained_checkpoint.pth \
    --launcher slurm
```

### Key arguments

| Argument | Description |
|----------|-------------|
| `--load-from` | Path to MEDiC pretrained checkpoint (extracts student weights automatically) |
| `--resume-from` | Path to a detection checkpoint to resume training |
| `--work-dir` | Output directory for logs and checkpoints |
| `--cfg-options` | Override config values, e.g. `model.backbone.pretrained=/path/to/ckpt.pth` |

## Evaluation

```bash
cd src/downstream/detection

bash tools/dist_test.sh \
    configs/mask_rcnn/medic_base_maskrcnn_1x_coco.py \
    /path/to/detection_checkpoint.pth \
    4 \
    --eval bbox segm
```

## Architecture

- **Backbone:** MEDiC ViT-Base/16 with FPN adapter (strides 4, 8, 16, 32)
- **Detector:** Mask R-CNN with ConvFC bbox head and FCN mask head
- **Schedule:** 1x (12 epochs), AdamW with layer-wise LR decay (rate=0.75)
- **Batch size:** 2 per GPU (effective batch size = 2 x num_GPUs x update_interval)

## Results

| Pretrained Model | bbox mAP | segm mAP |
|-----------------|----------|----------|
| MEDiC ViT-B/16 (300 epochs) | -- | -- |

## Compatibility Notes

- Requires PyTorch >= 2.0. The codebase includes compatibility patches for PyTorch 2.x/2.6/2.7 changes.
- WandB logging is integrated via mmdet's `WandbLoggerHook`. The `fixed_wandb_logger_hook.py` resolves a step-collision issue where eval metrics were silently dropped.
- Gradient checkpointing is enabled by default (`use_checkpoint=True`) to reduce memory usage.
