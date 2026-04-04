# Semantic Segmentation on ADE20K

This directory contains code for semantic segmentation evaluation on ADE20K using [mmsegmentation](https://github.com/open-mmlab/mmsegmentation).

## Architecture

- **Backbone**: MEDiC ViT-Base with FPN (Feature Pyramid Network)
- **Decoder**: UPerNet with 150-class output (ADE20K)
- **Resolution**: 512x512 with sliding window inference
- **Training**: 160K iterations, AdamW with layer decay

## Prerequisites

Install mmsegmentation v0.x and mmcv-full:

```bash
pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
pip install mmsegmentation==0.30.0
```

Adjust the CUDA and PyTorch versions in the mmcv-full URL to match your environment.

## Dataset

Download [ADE20K](http://sceneparsing.csail.mit.edu/) and organize as:

```
data/ADEChallengeData2016/
├── images/
│   ├── training/
│   └── validation/
└── annotations/
    ├── training/
    └── validation/
```

Update `data_root` in `configs/_base_/datasets/ade20k.py` if your data is in a different location.

## Training

```bash
# Single GPU
cd src/downstream/segmentation
PYTHONPATH=.:$PYTHONPATH python tools/train.py \
    configs/medic/upernet_medic_base_512_160k_ade20k.py \
    --options model.backbone.pretrained='path/to/checkpoint-epoch0290.pth'

# Multi-GPU (4 GPUs)
PYTHONPATH=.:$PYTHONPATH bash tools/dist_train.sh \
    configs/medic/upernet_medic_base_512_160k_ade20k.py 4 \
    --options model.backbone.pretrained='path/to/checkpoint-epoch0290.pth'
```

## Evaluation

```bash
PYTHONPATH=.:$PYTHONPATH python tools/test.py \
    configs/medic/upernet_medic_test_ade20k.py \
    path/to/segmentation_checkpoint.pth \
    --eval mIoU
```

## Expected Results

| Pretrained Checkpoint | mIoU  |
|-----------------------|-------|
| MEDiC v0 E290         | 52.7  |

## SLURM Submission

```bash
# Training
sbatch scripts/submit_train.sh path/to/pretrain_checkpoint.pth

# Evaluation
sbatch scripts/submit_eval.sh path/to/segmentation_checkpoint.pth
```
