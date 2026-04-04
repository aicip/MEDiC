"""
MEDiC test suite — single file to avoid repeated torch import overhead on NFS (~16s).

Run: CUDA_VISIBLE_DEVICES="" python tests/test_all.py
Or:  CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_all.py -v

Uses tiny models (embed_dim=64, depth=2, num_heads=2) for speed.
"""
import sys, os, copy, time, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.medic_model import MEDiCModel, build_medic_model, build_student, build_pixel_decoder
from src.models.decoder_mae import MAEDecoder
from src.utils.losses import (
    compute_loss, masked_smooth_l1_loss, masked_mse_loss, apply_normalization,
    cls_cosine_loss, cls_ce_loss, patchify,
)
from src.utils.masking_generator import (
    BlockMaskingGenerator, RandomMaskingGenerator, create_masking_generator,
)


# ── Tiny configs ─────────────────────────────────────────────────────────────

_STUDENT_BASE = {
    "img_size": 224, "patch_size": 16, "embed_dim": 64, "depth": 2,
    "num_heads": 2, "drop_path_rate": 0.0, "init_values": 0.1,
    "use_abs_pos_emb": True, "use_shared_rel_pos_bias": False,
    "use_rel_pos_bias": False, "use_sincos_pos_emb": False,
    "use_mask_tokens": False,
}

_TEACHER_BASE = {"embed_dim": 128}

_DECODER_BASE = {
    "decoder_embed_dim": 32, "decoder_depth": 1, "decoder_num_heads": 2,
    "replace_mask_tokens": True, "use_sincos_pos_emb": True, "use_abs_pos_emb": True,
}

# Baseline: head loss only (MaskDistill reproduction)
BASELINE_CFG = {
    "model": {
        "student": {**_STUDENT_BASE},
        "teacher": {**_TEACHER_BASE},
        "decoder": {**_DECODER_BASE},
    },
    "losses": {
        "use_head_loss": True, "head_loss_weight": 1.0,
        "use_cls_loss": False, "cls_loss_weight": 0.4,
        "use_decoder_loss": False, "pixel_loss_weight": 1.0,
        "head": {"type": "smooth_l1", "beta": 1.0},
        "cls": {"type": "cosine", "temperature": 1.0},
        "decoder": {"type": "l2"},
        "decoder_norm_pix_loss": True,
        "normalize_targets": True, "normalize_predictions": False,
        "normalization_method": "variance",
    },
    "mask": {"mask_ratio": 0.4, "mask_type": "block", "shuffle_patches": False},
}

# Full MEDiC: all three losses
FULL_CFG = copy.deepcopy(BASELINE_CFG)
FULL_CFG["losses"]["use_cls_loss"] = True
FULL_CFG["losses"]["use_decoder_loss"] = True

B, N, D_s, D_t = 2, 196, 64, 128  # batch, patches, student_dim, teacher_dim
PATCH_SIZE = 16
IMG = torch.randn(B, 3, 224, 224)

# Shared objects built once at module level
model_baseline = build_medic_model(BASELINE_CFG)
model_full = build_medic_model(FULL_CFG)
mask_block = BlockMaskingGenerator(14, 14, mask_ratio=0.4)

decoder = MAEDecoder(
    in_dim=D_s, embed_dim=32, depth=1, num_heads=2,
    patch_size=PATCH_SIZE, img_size=224,
    use_abs_pos_emb=True, use_sincos_pos_emb=True,
)


# ── Helper ───────────────────────────────────────────────────────────────────

def _run_forward(model, cfg):
    """Run forward pass and return (pred_tok, pred_pix, mask_out, ids_restore)."""
    masks = torch.stack([mask_block() for _ in range(B)])
    return model(IMG, masks, mask_block)


# ══════════════════════════════════════════════════════════════════════════════
# 1. MASKING GENERATORS (6 tests)
# ══════════════════════════════════════════════════════════════════════════════


def test_block_mask_count():
    """Block masking generates correct count."""
    g = BlockMaskingGenerator(14, 14, mask_ratio=0.4)
    m = g()
    expected = int(196 * 0.4)
    assert m.sum().item() == expected, f"Expected {expected} masked, got {m.sum().item()}"


def test_random_mask_count():
    """Random masking generates correct count."""
    g = RandomMaskingGenerator(14, 14, mask_ratio=0.75)
    m = g()
    expected = int(196 * 0.75)
    assert m.sum().item() == expected, f"Expected {expected} masked, got {m.sum().item()}"


def test_block_mask_14x14_at_40pct():
    """Block masking 14x14 grid at 40%."""
    g = BlockMaskingGenerator(14, 14, mask_ratio=0.4)
    m = g()
    assert m.shape == (196,), f"Expected (196,), got {m.shape}"
    expected = int(196 * 0.4)
    assert m.sum().item() == expected, f"Expected {expected} masked, got {m.sum().item()}"


def test_random_mask_at_75pct():
    """Random masking at 75%."""
    g = RandomMaskingGenerator(14, 14, mask_ratio=0.75)
    m = g()
    assert m.shape == (196,), f"Expected (196,), got {m.shape}"
    expected = int(196 * 0.75)
    assert m.sum().item() == expected, f"Expected {expected} masked, got {m.sum().item()}"


def test_mask_is_boolean():
    """Mask is boolean tensor."""
    g_block = BlockMaskingGenerator(14, 14, mask_ratio=0.4)
    m_block = g_block()
    assert m_block.dtype == torch.bool, f"Block mask dtype {m_block.dtype}, expected bool"

    g_random = RandomMaskingGenerator(14, 14, mask_ratio=0.75)
    m_random = g_random()
    assert m_random.dtype == torch.bool, f"Random mask dtype {m_random.dtype}, expected bool"


def test_evolved_masking_instantiation():
    """Evolved masking generator can be instantiated."""
    from src.utils.evolved_masking_generator import EvolvedMaskingGenerator

    dummy_model = nn.Linear(10, 10)
    emg = EvolvedMaskingGenerator(
        student_model=dummy_model,
        mask_ratio=0.4,
        cfg={"attention_source": "student_ema", "cluster_range": [10, 40]},
        grid_h=14, grid_w=14,
    )
    assert emg.num_patches == 196, f"Expected 196 patches, got {emg.num_patches}"
    assert emg.mask_ratio == 0.4
    assert hasattr(emg, "attention_source")


# ══════════════════════════════════════════════════════════════════════════════
# 2. MODEL CONSTRUCTION (6 tests)
# ══════════════════════════════════════════════════════════════════════════════


def test_build_baseline():
    """MEDiCModel builds with head loss only (baseline)."""
    m = build_medic_model(BASELINE_CFG)
    assert isinstance(m, MEDiCModel)
    assert m.distill_head is not None, "distill_head should exist for head loss"
    assert m.pixel_decoder is None, "pixel_decoder should be None for baseline"


def test_build_full():
    """MEDiCModel builds with all 3 losses (full MEDiC)."""
    m = build_medic_model(FULL_CFG)
    assert isinstance(m, MEDiCModel)
    assert m.distill_head is not None, "distill_head should exist"
    assert m.pixel_decoder is not None, "pixel_decoder should exist for full MEDiC"


def test_forward_baseline_returns():
    """Forward pass baseline returns (pred_tok, None, mask, ids_restore)."""
    masks = torch.stack([mask_block() for _ in range(B)])
    pred_tok, pred_pix, mask_out, ids_restore = model_baseline(IMG, masks, mask_block)
    assert pred_tok is not None, "pred_tok should not be None for baseline"
    assert pred_pix is None, "pred_pix should be None for baseline (no decoder loss)"
    assert mask_out.shape == (B, N), f"mask shape {mask_out.shape}, expected ({B}, {N})"


def test_forward_full_returns():
    """Forward pass full returns (pred_tok, pred_pix, mask, ids_restore)."""
    masks = torch.stack([mask_block() for _ in range(B)])
    pred_tok, pred_pix, mask_out, ids_restore = model_full(IMG, masks, mask_block)
    assert pred_tok is not None, "pred_tok should not be None"
    assert pred_pix is not None, "pred_pix should not be None for full MEDiC"
    assert pred_pix.shape == (B, N, PATCH_SIZE**2 * 3), (
        f"pred_pix shape {pred_pix.shape}, expected ({B}, {N}, {PATCH_SIZE**2 * 3})"
    )
    assert mask_out.shape == (B, N)


def test_pixel_decoder_none_when_disabled():
    """pixel_decoder is None when use_decoder_loss=False."""
    m = build_medic_model(BASELINE_CFG)
    assert m.pixel_decoder is None, "pixel_decoder should be None when use_decoder_loss=False"


def test_pixel_decoder_exists_when_enabled():
    """pixel_decoder exists when use_decoder_loss=True."""
    m = build_medic_model(FULL_CFG)
    assert m.pixel_decoder is not None, "pixel_decoder should exist when use_decoder_loss=True"
    assert isinstance(m.pixel_decoder, MAEDecoder)


# ══════════════════════════════════════════════════════════════════════════════
# 3. DECODER TESTS (3 tests)
# ══════════════════════════════════════════════════════════════════════════════


def test_decoder_construction():
    """MAEDecoder constructs correctly."""
    assert decoder.num_patches == N, f"Expected {N} patches, got {decoder.num_patches}"
    assert decoder.embed.in_features == D_s
    assert decoder.embed.out_features == 32
    assert decoder.pred.out_features == PATCH_SIZE**2 * 3


def test_decoder_forward_shape():
    """MAEDecoder forward produces correct output shape [B, N, patch_size^2*3]."""
    encoder_out = torch.randn(B, N + 1, D_s)
    mask = torch.zeros(B, N, dtype=torch.bool)
    mask[:, :int(N * 0.4)] = True

    out = decoder(encoder_out, mask)
    expected_shape = (B, N, PATCH_SIZE**2 * 3)
    assert out.shape == expected_shape, f"Got {out.shape}, expected {expected_shape}"


def test_decoder_no_weight_decay():
    """MAEDecoder no_weight_decay returns mask_token and pos_embed."""
    nwd = decoder.no_weight_decay()
    assert "mask_token" in nwd, f"mask_token not in no_weight_decay: {nwd}"
    assert "pos_embed" in nwd, f"pos_embed not in no_weight_decay: {nwd}"


# ══════════════════════════════════════════════════════════════════════════════
# 4. LOSS COMPUTATION (6 tests)
# ══════════════════════════════════════════════════════════════════════════════


def test_loss_head_only():
    """compute_loss with head only returns L_head."""
    pred_tok, pred_pix, mask_out, ids_restore = _run_forward(model_baseline, BASELINE_CFG)
    T = torch.randn(B, pred_tok.shape[1], D_t)
    loss, ld = compute_loss(pred_tok, pred_pix, T, IMG, mask_out, BASELINE_CFG)
    assert torch.isfinite(loss), f"Loss not finite: {loss}"
    assert "L_head" in ld, f"L_head not in loss_dict: {ld.keys()}"
    assert "L_cls" not in ld, f"L_cls should not be in loss_dict for baseline"
    assert "L_pix" not in ld, f"L_pix should not be in loss_dict for baseline"


def test_loss_head_cls():
    """compute_loss with head+cls returns L_head, L_cls."""
    cfg = copy.deepcopy(BASELINE_CFG)
    cfg["losses"]["use_cls_loss"] = True
    m = build_medic_model(cfg)
    pred_tok, pred_pix, mask_out, ids_restore = _run_forward(m, cfg)
    T = torch.randn(B, pred_tok.shape[1], D_t)
    loss, ld = compute_loss(pred_tok, pred_pix, T, IMG, mask_out, cfg)
    assert torch.isfinite(loss)
    assert "L_head" in ld, f"L_head missing: {ld.keys()}"
    assert "L_cls" in ld, f"L_cls missing: {ld.keys()}"
    assert "L_pix" not in ld, f"L_pix should not be present"


def test_loss_head_pixel():
    """compute_loss with head+pixel returns L_head, L_pix."""
    cfg = copy.deepcopy(BASELINE_CFG)
    cfg["losses"]["use_decoder_loss"] = True
    m = build_medic_model(cfg)
    pred_tok, pred_pix, mask_out, ids_restore = _run_forward(m, cfg)
    T = torch.randn(B, pred_tok.shape[1], D_t)
    loss, ld = compute_loss(pred_tok, pred_pix, T, IMG, mask_out, cfg)
    assert torch.isfinite(loss)
    assert "L_head" in ld
    assert "L_pix" in ld, f"L_pix missing: {ld.keys()}"
    assert "L_cls" not in ld


def test_loss_all_three():
    """compute_loss with all 3 returns L_head, L_cls, L_pix."""
    pred_tok, pred_pix, mask_out, ids_restore = _run_forward(model_full, FULL_CFG)
    T = torch.randn(B, pred_tok.shape[1], D_t)
    loss, ld = compute_loss(pred_tok, pred_pix, T, IMG, mask_out, FULL_CFG)
    assert torch.isfinite(loss)
    assert "L_head" in ld, f"L_head missing: {ld.keys()}"
    assert "L_cls" in ld, f"L_cls missing: {ld.keys()}"
    assert "L_pix" in ld, f"L_pix missing: {ld.keys()}"


def test_cls_cosine_differentiable():
    """CLS cosine loss is differentiable."""
    pred = torch.randn(B, D_t, requires_grad=True)
    target = torch.randn(B, D_t)
    loss = cls_cosine_loss(pred, target, temperature=1.0)
    assert torch.isfinite(loss), f"Loss not finite: {loss}"
    loss.backward()
    assert pred.grad is not None, "Gradient not computed"
    assert torch.isfinite(pred.grad).all(), "Gradient contains non-finite values"


def test_patchify_shape():
    """Patchify produces correct shape."""
    imgs = torch.randn(B, 3, 224, 224)
    patches = patchify(imgs, PATCH_SIZE)
    expected = (B, N, PATCH_SIZE**2 * 3)
    assert patches.shape == expected, f"Got {patches.shape}, expected {expected}"


# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAINING STEP (2 tests)
# ══════════════════════════════════════════════════════════════════════════════


def test_backward_baseline():
    """Full forward+backward works for baseline config."""
    m = build_medic_model(BASELINE_CFG)
    masks = torch.stack([mask_block() for _ in range(B)])
    pred_tok, pred_pix, mask_out, ids_restore = m(IMG, masks, mask_block)
    T = torch.randn(B, pred_tok.shape[1], D_t)
    loss, ld = compute_loss(pred_tok, pred_pix, T, IMG, mask_out, BASELINE_CFG)
    loss.backward()
    grads = sum(1 for p in m.parameters() if p.grad is not None)
    total = sum(1 for p in m.parameters() if p.requires_grad)
    assert grads == total, f"Only {grads}/{total} params have gradients"


def test_backward_full():
    """Full forward+backward works for full MEDiC config."""
    m = build_medic_model(FULL_CFG)
    masks = torch.stack([mask_block() for _ in range(B)])
    pred_tok, pred_pix, mask_out, ids_restore = m(IMG, masks, mask_block)
    T = torch.randn(B, pred_tok.shape[1], D_t)
    loss, ld = compute_loss(pred_tok, pred_pix, T, IMG, mask_out, FULL_CFG)
    loss.backward()
    grads = sum(1 for p in m.parameters() if p.grad is not None)
    total = sum(1 for p in m.parameters() if p.requires_grad)
    assert grads == total, f"Only {grads}/{total} params have gradients"


# ══════════════════════════════════════════════════════════════════════════════
# 6. CHECKPOINT COMPATIBILITY (2 tests)
# ══════════════════════════════════════════════════════════════════════════════


def test_save_load_roundtrip():
    """Model state dict save/load roundtrip."""
    state = model_full.state_dict()
    path = os.path.join(tempfile.gettempdir(), "test_medic_ckpt.pth")
    torch.save({"model": state}, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    m2 = build_medic_model(FULL_CFG)
    msg = m2.load_state_dict(loaded["model"], strict=True)
    assert len(msg.missing_keys) == 0, f"Missing: {msg.missing_keys}"
    assert len(msg.unexpected_keys) == 0, f"Unexpected: {msg.unexpected_keys}"
    os.remove(path)


def test_build_student_from_config():
    """Build student from config."""
    student = build_student(BASELINE_CFG)
    assert student.embed_dim == D_s, f"Expected embed_dim={D_s}, got {student.embed_dim}"
    assert student.use_mask_tokens == False
    masks = torch.stack([mask_block() for _ in range(B)])
    x, ids_restore = student(IMG, masks, mask_block)
    assert x.shape[0] == B
    assert x.shape[2] == D_s


# ══════════════════════════════════════════════════════════════════════════════
# Manual runner (for: python tests/test_all.py)
# ══════════════════════════════════════════════════════════════════════════════

ALL_TESTS = [
    # 1. Masking (6)
    test_block_mask_count, test_random_mask_count,
    test_block_mask_14x14_at_40pct, test_random_mask_at_75pct,
    test_mask_is_boolean, test_evolved_masking_instantiation,
    # 2. Model (6)
    test_build_baseline, test_build_full,
    test_forward_baseline_returns, test_forward_full_returns,
    test_pixel_decoder_none_when_disabled, test_pixel_decoder_exists_when_enabled,
    # 3. Decoder (3)
    test_decoder_construction, test_decoder_forward_shape,
    test_decoder_no_weight_decay,
    # 4. Loss (6)
    test_loss_head_only, test_loss_head_cls, test_loss_head_pixel,
    test_loss_all_three, test_cls_cosine_differentiable, test_patchify_shape,
    # 5. Training step (2)
    test_backward_baseline, test_backward_full,
    # 6. Checkpoint (2)
    test_save_load_roundtrip, test_build_student_from_config,
]

if __name__ == "__main__":
    t0 = time.time()
    _passed, _failed, _errors = 0, 0, []

    for fn in ALL_TESTS:
        name = fn.__name__
        try:
            fn()
            _passed += 1
            print(f"  PASS {name}")
        except AssertionError as e:
            _failed += 1
            _errors.append((name, str(e)))
            print(f"  FAIL {name}: {e}")
        except Exception as e:
            _failed += 1
            _errors.append((name, f"{type(e).__name__}: {e}"))
            print(f"  ERROR {name}: {type(e).__name__}: {e}")

    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Results: {_passed} passed, {_failed} failed in {total_time:.1f}s")
    if _errors:
        print(f"\nFailed tests:")
        for name, msg in _errors:
            print(f"  {name}: {msg}")
    print(f"{'='*60}")
    sys.exit(1 if _failed else 0)
