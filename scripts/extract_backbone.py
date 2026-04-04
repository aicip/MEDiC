#!/usr/bin/env python3
"""
Extract the student ViT backbone from a MEDiC pretrained checkpoint.

Removes the distillation head, pixel decoder, optimizer state, and DDP
wrapper prefix to produce a clean backbone checkpoint that can be loaded
directly into VisionTransformerMIM.

Usage:
    python scripts/extract_backbone.py \
        --checkpoint output/pretrain/medic_full_*/checkpoint-epoch0299.pth \
        --output medic_vit_base_backbone.pth

    # Then load in your code:
    from src.models.vision_transformer import VisionTransformerMIM
    model = VisionTransformerMIM(img_size=224, patch_size=16, embed_dim=768,
                                  depth=12, num_heads=12, use_abs_pos_emb=True)
    model.load_state_dict(torch.load("medic_vit_base_backbone.pth"))
"""

import argparse
import torch
from pathlib import Path


def extract_backbone(checkpoint_path: str, output_path: str, verbose: bool = True):
    """Extract student backbone weights from a MEDiC checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "model" in ckpt:
        state_dict = ckpt["model"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        raise KeyError(f"No model state found. Keys: {list(ckpt.keys())}")

    # Extract student weights, strip prefixes
    backbone = {}
    skipped_prefixes = {"distill_head", "pixel_decoder", "mask_generator"}

    for key, value in state_dict.items():
        # Remove DDP "module." prefix
        clean_key = key.replace("module.", "", 1) if key.startswith("module.") else key

        # Remove "student." prefix to get bare backbone keys
        if clean_key.startswith("student."):
            bare_key = clean_key[len("student."):]
        else:
            bare_key = clean_key

        # Skip non-backbone components
        component = bare_key.split(".")[0]
        if component in skipped_prefixes:
            continue

        backbone[bare_key] = value

    if verbose:
        # Report
        print(f"Checkpoint: {checkpoint_path}")
        if "epoch" in ckpt:
            print(f"  Epoch: {ckpt['epoch']}")
        if "config" in ckpt:
            cfg = ckpt["config"]
            s = cfg.get("model", {}).get("student", {})
            print(f"  Architecture: embed_dim={s.get('embed_dim')}, depth={s.get('depth')}")
            print(f"  Position: abs_pos={s.get('use_abs_pos_emb')}, "
                  f"shared_rel_pos={s.get('use_shared_rel_pos_bias')}")
            print(f"  Mask tokens: {s.get('use_mask_tokens')}")
        print(f"  Original keys: {len(state_dict)}")
        print(f"  Backbone keys: {len(backbone)}")

        # Show what was stripped
        stripped = len(state_dict) - len(backbone)
        if stripped > 0:
            print(f"  Stripped: {stripped} non-backbone keys (distill_head, pixel_decoder, etc.)")

    torch.save(backbone, output_path)

    if verbose:
        size_mb = Path(output_path).stat().st_size / 1e6
        print(f"\nSaved: {output_path} ({size_mb:.1f} MB)")

    return backbone


def main():
    parser = argparse.ArgumentParser(description="Extract MEDiC student backbone")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to MEDiC pretrained checkpoint")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for backbone-only checkpoint")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    extract_backbone(args.checkpoint, args.output, verbose=not args.quiet)


if __name__ == "__main__":
    main()
