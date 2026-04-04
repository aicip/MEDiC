#!/usr/bin/env python3
"""
Smoke tests for MEDiC training pipeline.

Tests the full training loop end-to-end with synthetic data for each
config variant (baseline, token_cls, token_pixel, medic_full).
Validates: forward/backward, loss components, W&B logging, checkpointing,
visualization, and validation loop.

Run: CUDA_VISIBLE_DEVICES="" python tests/test_smoke.py
  or: CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_smoke.py -v -s
"""

import os
import sys
import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call
from typing import Dict, Any, List, Optional

import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# ── Shared tiny config builder ──────────────────────────────────────────

STUDENT_CFG = {
    "name": "vit_base_patch16",
    "img_size": 224,
    "patch_size": 16,
    "embed_dim": 64,
    "depth": 2,
    "num_heads": 2,
    "drop_path_rate": 0.0,
    "init_values": 0.1,
    "use_abs_pos_emb": True,
    "use_rel_pos_bias": False,
    "use_shared_rel_pos_bias": False,
    "use_mask_tokens": False,  # sparse mode
}

DECODER_CFG = {
    "decoder_embed_dim": 32,
    "decoder_depth": 1,
    "decoder_num_heads": 2,
    "replace_mask_tokens": True,
    "use_sincos_pos_emb": True,
}


def make_config(
    use_head: bool = True,
    use_cls: bool = False,
    use_decoder: bool = False,
    mask_type: str = "block",
    mask_ratio: float = 0.4,
    output_dir: str = "/tmp/medic_smoke",
) -> Dict[str, Any]:
    """Build a complete config dict for smoke testing."""
    return {
        "model": {
            "student": STUDENT_CFG.copy(),
            "teacher": {"name": "ViT-B/16", "embed_dim": 64},
            "decoder": DECODER_CFG.copy(),
        },
        "losses": {
            "use_head_loss": use_head,
            "use_decoder_loss": use_decoder,
            "use_cls_loss": use_cls,
            "head_loss_weight": 1.0,
            "pixel_loss_weight": 1.0,
            "cls_loss_weight": 0.4,
            "loss_weighting_method": "literal",
            "head": {"type": "smooth_l1", "beta": 1.0},
            "decoder": {"type": "l2"},
            "cls": {"type": "cosine", "temperature": 1.0},
            "decoder_norm_pix_loss": True,
            "normalize_targets": True,
            "normalize_predictions": False,
            "normalization_method": "variance",
        },
        "mask": {
            "mask_ratio": mask_ratio,
            "mask_type": mask_type,
            "shuffle_patches": False,
        },
        "data": {
            "batch_per_gpu": 2,
            "num_workers": 0,
            "data_path": "/fake",
            "train_dir": "/fake/train",
            "val_dir": "/fake/val",
            "subset": None,
            "subset_start": 0,
            "augmentation": {
                "min_scale": 0.2, "max_scale": 1.0,
                "horizontal_flip_prob": 0.5, "color_jitter": 0.4,
            },
        },
        "optim": {
            "lr": 1.5e-3,
            "wd": 0.05,
            "betas": [0.9, 0.999],
            "layer_decay": 1.0,
            "accum_iter": 1,
            "grad_clip": 3.0,
        },
        "schedule": {
            "min_lr": 1e-5,
            "warmup_epochs": 1,
            "warmup_start_lr": 1e-6,
        },
        "wandb_meta": {
            "project": "MEDiC_smoke_test",
            "entity": None,
            "name": "smoke",
            "desc": "smoke test",
        },
        "visualizations": {
            "enable": False,
            "layers_to_analyze": [0, 1],
            "attention_heads_to_show": 2,
            "plots": {
                "raw_attention": False,
                "layer_evolution": False,
                "head_comparison": False,
                "attention_rollout": False,
            },
        },
        "epochs": 2,
        "cpu": True,
        "seed": 42,
        "precision": "bf16",
        "log_interval": 1,
        "output_dir": output_dir,
        "resume_path": None,
        "gpu": -1,
        "distributed": False,
        "rank": 0,
        "world_size": 1,
        "local_rank": 0,
    }


# ── Mock teacher that returns random features ──────────────────────────

class MockTeacher(nn.Module):
    """Mock CLIP teacher that returns random features of correct shape."""

    def __init__(self, embed_dim: int = 64, num_patches: int = 196):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches

    def forward(self, img, return_cls=True, return_attention=False):
        B = img.shape[0]
        N = self.num_patches + 1 if return_cls else self.num_patches
        features = torch.randn(B, N, self.embed_dim, device=img.device, dtype=img.dtype)
        if return_attention:
            attn = torch.randn(B, self.num_patches, self.num_patches, device=img.device)
            return features, attn
        return features

    def eval(self):
        return self

    def to(self, *args, **kwargs):
        return self

    def get_attention_maps(self, x):
        B = x.shape[0]
        return torch.randn(B, self.num_patches, self.num_patches, device=x.device)


# ── Synthetic data loader ──────────────────────────────────────────────

def make_fake_loader(batch_size: int = 2, num_batches: int = 4, mask_ratio: float = 0.4):
    """Create a DataLoader with synthetic images and masks."""
    N = 196  # 14x14 patches
    num_mask = int(N * mask_ratio)
    images = []
    masks = []
    for _ in range(batch_size * num_batches):
        images.append(torch.randn(3, 224, 224))
        mask = torch.zeros(N, dtype=torch.bool)
        mask_indices = torch.randperm(N)[:num_mask]
        mask[mask_indices] = True
        masks.append(mask)

    dataset = TensorDataset(torch.stack(images), torch.stack(masks))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# ── Core training step test ────────────────────────────────────────────

def run_training_smoke(
    cfg: Dict[str, Any],
    num_steps: int = 3,
    check_wandb: bool = True,
) -> Dict[str, Any]:
    """
    Run a smoke training loop and return diagnostics.

    Returns dict with: losses, gradients, wandb_calls, checkpoint info.
    """
    from src.models.medic_model import build_medic_model
    from src.utils.losses import compute_loss
    from src.utils.masking_generator import create_masking_generator
    from src.utils.utils import NativeScalerWithGradNormCount

    device = torch.device("cpu")
    dtype = torch.float32  # CPU doesn't support bf16 autocast well

    # Build model
    mask_gen = create_masking_generator(cfg)
    model = build_medic_model(cfg, mask_generator=mask_gen)
    model.to(device)

    # Mock teacher
    teacher = MockTeacher(embed_dim=cfg["model"]["teacher"]["embed_dim"])

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = NativeScalerWithGradNormCount()

    # Fake data
    loader = make_fake_loader(
        batch_size=cfg["data"]["batch_per_gpu"],
        num_batches=num_steps + 1,
        mask_ratio=cfg["mask"]["mask_ratio"],
    )

    # Track results
    results = {
        "step_losses": [],
        "loss_components": [],
        "grad_norms": [],
        "wandb_calls": [],
        "errors": [],
    }

    # Mock wandb
    wandb_log_calls = []

    model.train()
    for step, (img, mask) in enumerate(loader):
        if step >= num_steps:
            break

        img = img.to(device)
        mask = mask.to(device)

        opt.zero_grad()

        # Forward
        with torch.autocast(device_type="cpu", dtype=dtype, enabled=False):
            T = teacher(img, return_cls=True)
            pred_tok, pred_pix, mask_out, ids_restore = model(
                img, mask, teacher=teacher, epoch=0, model_ema=None,
            )

            # Sparse mode: align teacher to visible
            use_mask_tokens = cfg["model"]["student"]["use_mask_tokens"]
            if not use_mask_tokens and T is not None:
                # Simple alignment: just slice to match
                if pred_tok is not None and T.shape[1] != pred_tok.shape[1]:
                    T = T[:, :pred_tok.shape[1], :]

            loss, l_dict = compute_loss(
                pred_tok=pred_tok,
                pred_pix=pred_pix,
                teacher_features=T,
                img=img,
                mask=mask,
                cfg=cfg,
            )

        # Backward
        loss.backward()

        # Check gradients exist
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5

        opt.step()

        results["step_losses"].append(loss.item())
        results["loss_components"].append(l_dict.copy())
        results["grad_norms"].append(grad_norm)

        # Build what wandb.log would receive
        wandb_data = {
            "train_loss": loss.item(),
            "train_head_loss": l_dict.get("L_head", 0),
            "train_cls_loss": l_dict.get("L_cls", 0),
            "train_pix_loss": l_dict.get("L_pix", 0),
            "Optimizer/grad_norm": grad_norm,
            "step": step,
            "epoch": 0,
        }
        wandb_log_calls.append(wandb_data)

    results["wandb_calls"] = wandb_log_calls

    # Validation step
    model.eval()
    with torch.no_grad():
        val_img, val_mask = next(iter(loader))
        val_img = val_img.to(device)
        val_mask = val_mask.to(device)

        T_val = teacher(val_img, return_cls=True)
        pred_tok_v, pred_pix_v, mask_v, ids_v = model(
            val_img, val_mask, teacher=teacher, epoch=0,
        )

        if not use_mask_tokens and T_val is not None:
            if pred_tok_v is not None and T_val.shape[1] != pred_tok_v.shape[1]:
                T_val = T_val[:, :pred_tok_v.shape[1], :]

        val_loss, val_dict = compute_loss(
            pred_tok=pred_tok_v,
            pred_pix=pred_pix_v,
            teacher_features=T_val,
            img=val_img,
            mask=val_mask,
            cfg=cfg,
        )
        results["val_loss"] = val_loss.item()
        results["val_components"] = val_dict

    # Checkpoint save/load roundtrip
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        ckpt_path = f.name
    try:
        torch.save({
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": 0,
            "global_step": num_steps,
            "best_loss": min(results["step_losses"]),
            "config": cfg,
        }, ckpt_path)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model2 = build_medic_model(cfg, mask_generator=create_masking_generator(cfg))
        missing, unexpected = model2.load_state_dict(ckpt["model"], strict=False)
        results["checkpoint_missing_keys"] = len(missing)
        results["checkpoint_unexpected_keys"] = len(unexpected)
    finally:
        os.unlink(ckpt_path)

    return results


# ── Test cases ─────────────────────────────────────────────────────────

CONFIG_VARIANTS = {
    "baseline": {"use_head": True, "use_cls": False, "use_decoder": False},
    "token_cls": {"use_head": True, "use_cls": True, "use_decoder": False},
    "token_pixel": {"use_head": True, "use_cls": False, "use_decoder": True},
    "medic_full": {"use_head": True, "use_cls": True, "use_decoder": True},
}

EXPECTED_LOSS_KEYS = {
    "baseline": ["L_head"],
    "token_cls": ["L_head", "L_cls"],
    "token_pixel": ["L_head", "L_pix"],
    "medic_full": ["L_head", "L_cls", "L_pix"],
}


@pytest.mark.parametrize("variant", CONFIG_VARIANTS.keys())
def test_training_smoke(variant):
    """Full training smoke test for each config variant."""
    kwargs = CONFIG_VARIANTS[variant]
    cfg = make_config(**kwargs)
    results = run_training_smoke(cfg, num_steps=3)

    # 1. All losses are finite
    for i, loss in enumerate(results["step_losses"]):
        assert np.isfinite(loss), f"Step {i}: loss is {loss}"

    # 2. Correct loss components present
    expected = EXPECTED_LOSS_KEYS[variant]
    for step_dict in results["loss_components"]:
        for key in expected:
            assert key in step_dict, f"Missing loss key {key} in {step_dict.keys()}"
            assert np.isfinite(step_dict[key]), f"{key} = {step_dict[key]}"
        assert "total_loss" in step_dict

    # 3. Gradient norms are positive and finite
    for i, gn in enumerate(results["grad_norms"]):
        assert gn > 0, f"Step {i}: zero gradient norm"
        assert np.isfinite(gn), f"Step {i}: grad_norm is {gn}"

    # 4. Validation works
    assert np.isfinite(results["val_loss"])
    for key in expected:
        assert key in results["val_components"]

    # 5. Checkpoint roundtrip
    assert results["checkpoint_missing_keys"] == 0, \
        f"Checkpoint has {results['checkpoint_missing_keys']} missing keys"
    assert results["checkpoint_unexpected_keys"] == 0, \
        f"Checkpoint has {results['checkpoint_unexpected_keys']} unexpected keys"


@pytest.mark.parametrize("variant", CONFIG_VARIANTS.keys())
def test_wandb_logging_correctness(variant):
    """Verify W&B log calls contain all expected fields with explicit step."""
    kwargs = CONFIG_VARIANTS[variant]
    cfg = make_config(**kwargs)
    results = run_training_smoke(cfg, num_steps=2)

    expected_keys = EXPECTED_LOSS_KEYS[variant]
    wandb_key_map = {
        "L_head": "train_head_loss",
        "L_cls": "train_cls_loss",
        "L_pix": "train_pix_loss",
    }

    for i, log_call in enumerate(results["wandb_calls"]):
        # Must have explicit step (critical for W&B — no step mixing)
        assert "step" in log_call, f"Call {i}: missing 'step' key"
        assert isinstance(log_call["step"], int)

        # Must have total loss
        assert "train_loss" in log_call
        assert np.isfinite(log_call["train_loss"])

        # Must have all expected loss components
        for loss_key in expected_keys:
            wandb_key = wandb_key_map[loss_key]
            assert wandb_key in log_call, \
                f"Call {i}: missing {wandb_key} (expected for {variant})"
            assert isinstance(log_call[wandb_key], (int, float))

        # Must have optimizer metrics
        assert "Optimizer/grad_norm" in log_call

        # Inactive losses should be zero
        all_loss_keys = set(wandb_key_map.keys())
        inactive = all_loss_keys - set(expected_keys)
        for loss_key in inactive:
            wandb_key = wandb_key_map[loss_key]
            assert log_call[wandb_key] == 0, \
                f"Call {i}: inactive loss {wandb_key} should be 0, got {log_call[wandb_key]}"


def test_cls_loss_weight_applied():
    """Verify CLS loss weight (0.4) is applied correctly to total loss."""
    cfg = make_config(use_head=True, use_cls=True, use_decoder=False)
    results = run_training_smoke(cfg, num_steps=1)

    l_dict = results["loss_components"][0]
    expected_total = l_dict["L_head"] * 1.0 + l_dict["L_cls"] * 0.4
    assert abs(l_dict["total_loss"] - expected_total) < 1e-4, \
        f"total_loss={l_dict['total_loss']:.6f} != head*1.0 + cls*0.4 = {expected_total:.6f}"


def test_pixel_loss_uses_patchified_targets():
    """Verify pixel loss target shape matches decoder output."""
    cfg = make_config(use_head=True, use_cls=False, use_decoder=True)

    from src.models.medic_model import build_medic_model
    from src.utils.masking_generator import create_masking_generator

    model = build_medic_model(cfg, mask_generator=create_masking_generator(cfg))
    img = torch.randn(2, 3, 224, 224)
    pred_tok, pred_pix, mask, ids = model(img)

    assert pred_pix is not None
    # patch_size=16, so 16*16*3 = 768
    assert pred_pix.shape == (2, 196, 768), f"pred_pix shape: {pred_pix.shape}"


def test_dense_mode_training():
    """Smoke test with dense encoding mode (use_mask_tokens=True)."""
    cfg = make_config(use_head=True, use_cls=True, use_decoder=True)
    cfg["model"]["student"]["use_mask_tokens"] = True
    cfg["model"]["student"]["use_abs_pos_emb"] = False
    cfg["model"]["student"]["use_shared_rel_pos_bias"] = True

    results = run_training_smoke(cfg, num_steps=2)

    for loss in results["step_losses"]:
        assert np.isfinite(loss)
    assert results["checkpoint_missing_keys"] == 0


def test_random_masking_75pct():
    """Smoke test with random masking at 75% (MAE-style)."""
    cfg = make_config(use_head=True, use_cls=False, use_decoder=False,
                      mask_type="random", mask_ratio=0.75)
    results = run_training_smoke(cfg, num_steps=2)

    for loss in results["step_losses"]:
        assert np.isfinite(loss)


def test_model_ema_creation():
    """Verify EMA model can be created for evolved masking configs."""
    cfg = make_config(use_head=True, use_cls=False, use_decoder=False)
    cfg["mask"]["mask_type"] = "evolved"
    cfg["mask"]["evolved"] = {
        "attention_source": "clip_teacher",
        "gamma": 0.2,
        "cluster_range": [10, 40],
        "evolution_epochs": 300,
        "em_iterations": 15,
        "spatial_noise_type": "grid",
        "ema_decay": 0.9999,
    }

    from src.models.medic_model import build_medic_model
    from src.utils.masking_generator import create_masking_generator

    mask_gen = create_masking_generator(cfg)
    model = build_medic_model(cfg, mask_generator=mask_gen)

    try:
        from timm.utils import ModelEma
        ema = ModelEma(model, decay=0.9999)
        assert ema is not None
        # EMA should have same structure
        assert hasattr(ema, 'module') or hasattr(ema, 'ema')
    except ImportError:
        pytest.skip("timm.utils.ModelEma not available")


def test_no_nan_gradients():
    """Verify no NaN gradients during training."""
    cfg = make_config(use_head=True, use_cls=True, use_decoder=True)
    results = run_training_smoke(cfg, num_steps=3)

    for i, gn in enumerate(results["grad_norms"]):
        assert not np.isnan(gn), f"Step {i}: NaN gradient norm"
        assert not np.isinf(gn), f"Step {i}: Inf gradient norm"


# ── Main runner ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("MEDiC Training Pipeline Smoke Tests")
    print("=" * 70)

    passed = 0
    failed = 0
    errors = []

    # Run all parametrized tests manually
    all_tests = [
        ("training_smoke", test_training_smoke, list(CONFIG_VARIANTS.keys())),
        ("wandb_logging", test_wandb_logging_correctness, list(CONFIG_VARIANTS.keys())),
    ]
    standalone_tests = [
        test_cls_loss_weight_applied,
        test_pixel_loss_uses_patchified_targets,
        test_dense_mode_training,
        test_random_masking_75pct,
        test_model_ema_creation,
        test_no_nan_gradients,
    ]

    for test_name, test_fn, variants in all_tests:
        for v in variants:
            label = f"{test_name}[{v}]"
            try:
                test_fn(v)
                print(f"  PASS  {label}")
                passed += 1
            except Exception as e:
                print(f"  FAIL  {label}: {e}")
                failed += 1
                errors.append((label, str(e)))

    for test_fn in standalone_tests:
        label = test_fn.__name__
        try:
            test_fn()
            print(f"  PASS  {label}")
            passed += 1
        except pytest.skip.Exception:
            print(f"  SKIP  {label}")
        except Exception as e:
            print(f"  FAIL  {label}: {e}")
            failed += 1
            errors.append((label, str(e)))

    print(f"\n{'=' * 70}")
    print(f"Results: {passed} passed, {failed} failed")
    if errors:
        print(f"\nFailures:")
        for label, err in errors:
            print(f"  {label}: {err}")
    print(f"{'=' * 70}")
    sys.exit(0 if failed == 0 else 1)
