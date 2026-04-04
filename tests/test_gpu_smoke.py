#!/usr/bin/env python3
"""
GPU smoke test for MEDiC training pipeline.

Tests the full training loop on GPU with a real CLIP teacher,
mixed precision, and all loss variants. Uses synthetic data.

Run via SLURM:
    sbatch scripts/smoke_test.sh

Or directly on a GPU node:
    python tests/test_gpu_smoke.py
"""

import os
import sys
import time
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

# Ensure we import from THIS repo, not MEDiC_torch
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.medic_model import build_medic_model
from src.models import build_teacher
from src.utils.losses import compute_loss
from src.utils.masking_generator import create_masking_generator
from src.utils.utils import NativeScalerWithGradNormCount
from src.train import _extract_visible_teacher_features, get_config_value


def make_config(use_head=True, use_cls=False, use_decoder=False):
    """Build a full-size config for GPU testing."""
    return {
        "model": {
            "student": {
                "name": "vit_base_patch16", "img_size": 224, "patch_size": 16,
                "embed_dim": 192, "depth": 4, "num_heads": 3,  # Small but not tiny
                "drop_path_rate": 0.1, "init_values": 0.1,
                "use_abs_pos_emb": True, "use_rel_pos_bias": False,
                "use_shared_rel_pos_bias": False, "use_mask_tokens": False,
            },
            "teacher": {"name": "ViT-B/16", "embed_dim": 768},
            "decoder": {
                "decoder_embed_dim": 128, "decoder_depth": 2,
                "decoder_num_heads": 4, "replace_mask_tokens": True,
                "use_sincos_pos_emb": True,
            },
        },
        "losses": {
            "use_head_loss": use_head, "use_decoder_loss": use_decoder,
            "use_cls_loss": use_cls,
            "head_loss_weight": 1.0, "pixel_loss_weight": 1.0, "cls_loss_weight": 0.4,
            "loss_weighting_method": "literal",
            "head": {"type": "smooth_l1", "beta": 1.0},
            "decoder": {"type": "l2"},
            "cls": {"type": "cosine", "temperature": 1.0},
            "decoder_norm_pix_loss": True, "normalize_targets": True,
            "normalize_predictions": False, "normalization_method": "variance",
        },
        "mask": {"mask_ratio": 0.4, "mask_type": "block", "shuffle_patches": False},
    }


def run_gpu_smoke(name, cfg, device, dtype, num_steps=5):
    """Run a training smoke test on GPU."""
    print(f"\n{'─'*60}")
    print(f"Testing: {name}")
    print(f"{'─'*60}")

    # Build model
    mask_gen = create_masking_generator(cfg)
    model = build_medic_model(cfg, mask_generator=mask_gen).to(device)

    # Real CLIP teacher
    print("  Loading CLIP teacher...")
    teacher = build_teacher(cfg).to(device).eval()

    # Distill head needs to project to teacher dim (768 for CLIP)
    # Rebuild model with correct teacher embed_dim
    model = build_medic_model(cfg, mask_generator=mask_gen).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = NativeScalerWithGradNormCount()

    model.train()
    losses_log = []

    for step in range(num_steps):
        # Synthetic batch
        B = 4
        img = torch.randn(B, 3, 224, 224, device=device)
        N = 196
        num_mask = int(N * cfg["mask"]["mask_ratio"])
        mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        for b in range(B):
            idx = torch.randperm(N)[:num_mask]
            mask[b, idx] = True

        opt.zero_grad()

        with torch.autocast(device_type="cuda", dtype=dtype):
            # Teacher forward
            T = teacher(img, return_cls=True)

            # Student forward
            pred_tok, pred_pix, mask_out, ids_restore = model(
                img, mask, teacher=teacher, epoch=0, model_ema=None,
            )

            # Sparse mode: align teacher
            use_mask_tokens = cfg["model"]["student"]["use_mask_tokens"]
            if not use_mask_tokens and T is not None:
                T = _extract_visible_teacher_features(T, mask, model)

            # Loss
            loss, l_dict = compute_loss(
                pred_tok=pred_tok, pred_pix=pred_pix,
                teacher_features=T, img=img, mask=mask, cfg=cfg,
            )

        # Backward
        grad_norm = scaler(
            loss, opt, parameters=model.parameters(),
            clip_grad=3.0, update_grad=True,
        )
        opt.zero_grad()

        losses_log.append(l_dict)

        if step == 0 or step == num_steps - 1:
            parts = [f"loss={l_dict['total_loss']:.4f}"]
            for k in ["L_head", "L_cls", "L_pix"]:
                if k in l_dict:
                    parts.append(f"{k}={l_dict[k]:.4f}")
            parts.append(f"grad={grad_norm:.2f}")
            print(f"  Step {step}: {', '.join(parts)}")

    # Validation step
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
        val_img = torch.randn(2, 3, 224, 224, device=device)
        val_mask = torch.zeros(2, 196, dtype=torch.bool, device=device)
        val_mask[:, :num_mask] = True

        T_val = teacher(val_img, return_cls=True)
        pred_tok_v, pred_pix_v, mask_v, ids_v = model(
            val_img, val_mask, teacher=teacher, epoch=0,
        )
        if not use_mask_tokens:
            T_val = _extract_visible_teacher_features(T_val, val_mask, model)
        val_loss, val_dict = compute_loss(
            pred_tok_v, pred_pix_v, T_val, val_img, val_mask, cfg,
        )
        print(f"  Val: loss={val_dict['total_loss']:.4f}")

    # Checkpoint roundtrip
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        ckpt_path = f.name
    torch.save({"model": model.state_dict(), "config": cfg}, ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model2 = build_medic_model(cfg, mask_generator=create_masking_generator(cfg))
    missing, unexpected = model2.load_state_dict(ckpt["model"], strict=False)
    os.unlink(ckpt_path)
    assert len(missing) == 0, f"Checkpoint missing: {missing}"
    assert len(unexpected) == 0, f"Checkpoint unexpected: {unexpected}"
    print(f"  Checkpoint: OK (0 missing, 0 unexpected)")

    # Validate all losses finite
    for step_dict in losses_log:
        for k, v in step_dict.items():
            assert np.isfinite(v), f"{k} = {v} (not finite)"

    # Verify expected loss components
    expected = set()
    if cfg["losses"]["use_head_loss"]:
        expected.add("L_head")
    if cfg["losses"]["use_cls_loss"]:
        expected.add("L_cls")
    if cfg["losses"]["use_decoder_loss"]:
        expected.add("L_pix")

    for step_dict in losses_log:
        actual = {k for k in step_dict if k.startswith("L_")}
        assert actual == expected, f"Expected {expected}, got {actual}"

    print(f"  PASS")
    return True


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Run on a GPU node.")
        sys.exit(1)

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    gpu_name = torch.cuda.get_device_name(0)

    print("=" * 60)
    print("MEDiC GPU Smoke Tests")
    print(f"GPU: {gpu_name}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Precision: {dtype}")
    print("=" * 60)

    configs = {
        "baseline (head only)": make_config(use_head=True),
        "token + CLS": make_config(use_head=True, use_cls=True),
        "token + pixel": make_config(use_head=True, use_decoder=True),
        "MEDiC full (head+CLS+pixel)": make_config(use_head=True, use_cls=True, use_decoder=True),
    }

    passed = 0
    failed = 0
    start = time.time()

    for name, cfg in configs.items():
        try:
            run_gpu_smoke(name, cfg, device, dtype, num_steps=5)
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

        torch.cuda.empty_cache()

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed ({elapsed:.1f}s)")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
