#!/usr/bin/env python3
"""
Upload MEDiC checkpoints to HuggingFace Hub.

Usage:
    # Login first
    huggingface-cli login

    # Upload all checkpoints
    python scripts/upload_to_hf.py --username YOUR_HF_USERNAME

    # Upload specific checkpoint
    python scripts/upload_to_hf.py --username YOUR_HF_USERNAME --checkpoint pretrain_medic
"""

import argparse
import os
import torch
from huggingface_hub import HfApi, create_repo

CHECKPOINTS = {
    "pretrain_medic": {
        "path": "output/pretrain/checkpoint-epoch0299.pth",
        "filename": "medic_vit_base_ep299.pth",
        "description": "MEDiC ViT-Base/16 pretrained with all 3 losses (300 epochs)",
    },
    "pretrain_token_cls": {
        "path": "output/pretrain/token_cls_checkpoint.pth",
        "filename": "token_cls_vit_base.pth",
        "description": "Token + CLS distillation (ablation)",
    },
    "pretrain_token_pixel": {
        "path": "output/pretrain/token_pixel_checkpoint.pth",
        "filename": "token_pixel_vit_base.pth",
        "description": "Token + Pixel reconstruction (ablation)",
    },
    # Downstream checkpoints (paths to be filled after training):
    # "finetune": {
    #     "path": "...",
    #     "filename": "finetune_vit_base_ep100.pth",
    #     "description": "ViT-Base/16 finetuned on ImageNet-1K",
    # },
    # "semseg": {
    #     "path": "...",
    #     "filename": "semseg_upernet_ade20k_160k.pth",
    #     "description": "UPerNet on ADE20K",
    # },
}

MODEL_CARD = """---
license: apache-2.0
tags:
  - vision
  - self-supervised-learning
  - masked-image-modeling
  - knowledge-distillation
  - multi-objective-learning
  - vit
datasets:
  - ILSVRC/imagenet-1k
  - 1aurent/ADE20K
metrics:
  - accuracy
  - mIoU
pipeline_tag: image-classification
---

# MEDiC ViT-Base/16

**Multi-objective Exploration of Distillation from CLIP**

This model was trained using the [MEDiC](https://github.com/{username}/MEDiC) codebase, implementing the method from ["MEDiC: Multi-objective Exploration of Distillation from CLIP"](https://arxiv.org/abs/2603.29009).

## Model Description

MEDiC extends MaskDistill by combining three complementary training objectives:

1. **Token Distillation**: Smooth L1 loss between student and CLIP teacher patch features
2. **CLS Token Alignment**: Cosine similarity between student and teacher CLS tokens
3. **Pixel Reconstruction**: MAE-style decoder reconstructing normalized pixel patches

Architecture:
- **Student**: ViT-Base/16 (86M params)
- **Teacher**: CLIP ViT-B/16 (frozen)
- **Decoder**: 8-block transformer (512 dim, for pixel reconstruction)
- **Pretraining**: 300 epochs on ImageNet-1K, sparse encoding, block masking 40%

## Results

| Evaluation | Result |
|-----------|--------|
| k-NN (k=20) | **73.9%** top-1 |
| Linear Probe | **60.5%** top-1 |
| Finetuning (ImageNet-1K) | **85.1%** top-1 |
| Sem. Seg. (ADE20K, UPerNet) | **52.5** mIoU |

### Loss Ablation

| Configuration | k-NN |
|--------------|------|
| Token only | 68.6% |
| + Pixel | 71.4% |
| + CLS | 72.3% |
| + Pixel + CLS (MEDiC) | **73.9%** |

## Usage

```python
import torch
from src.models.vision_transformer import VisionTransformerMIM

# Load pretrained student backbone
model = VisionTransformerMIM(
    img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12,
    use_abs_pos_emb=True, use_mask_tokens=False,
)
ckpt = torch.load("medic_vit_base_ep299.pth", map_location="cpu")
state = {{k.replace("module.student.", ""): v for k, v in ckpt["model"].items() if "student" in k}}
model.load_state_dict(state, strict=False)
```

See the [GitHub repo](https://github.com/{username}/MEDiC) for full training and evaluation code.

## Citation

```bibtex
@article{{georgiou2025medic,
  title={{MEDiC: Multi-objective Exploration of Distillation from CLIP}},
  author={{Georgiou, Kostas}},
  journal={{arXiv preprint arXiv:2603.29009}},
  year={{2025}}
}}
```
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", default="drkostas", help="HuggingFace username")
    parser.add_argument("--checkpoint", default=None, help="Specific checkpoint to upload")
    parser.add_argument("--repo-name", default="MEDiC-ViT-Base", help="HF repo name")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be uploaded")
    args = parser.parse_args()

    repo_id = f"{args.username}/{args.repo_name}"
    api = HfApi()

    if not args.dry_run:
        try:
            create_repo(repo_id, repo_type="model", exist_ok=True)
            print(f"Repository {repo_id} ready")
        except Exception as e:
            print(f"Warning: {e}")

        card = MODEL_CARD.format(username=args.username)
        api.upload_file(
            path_or_fileobj=card.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"Uploaded model card to {repo_id}")

    to_upload = {args.checkpoint: CHECKPOINTS[args.checkpoint]} if args.checkpoint else CHECKPOINTS

    for name, info in to_upload.items():
        path = info["path"]
        if not os.path.exists(path):
            print(f"SKIP {name}: {path} not found")
            continue

        size_mb = os.path.getsize(path) / 1e6
        print(f"{'[DRY RUN] ' if args.dry_run else ''}Upload {name}: {path} ({size_mb:.0f} MB) -> {info['filename']}")

        if not args.dry_run:
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo=info["filename"],
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"  Uploaded {info['filename']} to {repo_id}")

    print(f"\nDone! View at: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
