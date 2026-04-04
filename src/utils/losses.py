"""
Loss functions for MEDiC multi-objective distillation.

Implements three loss components:
  - L_head: Token distillation (Smooth L1 between student/teacher patch features)
  - L_cls:  CLS token alignment (cosine similarity or cross-entropy)
  - L_pix:  Pixel reconstruction (L2 on normalized patches from MAE decoder)

Combined with literal (fixed) weighting:
    L_total = w_head * L_head + w_cls * L_cls + w_pix * L_pix

Supports two encoder modes:
  - Dense mode (use_mask_tokens=True): Loss computed on masked positions only
  - Sparse mode (use_mask_tokens=False): All student outputs contribute to loss

Reference: "MEDiC: Multi-objective Exploration of Distillation from CLIP" (arXiv:2603.29009)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple


# ======================================================================
# Normalization
# ======================================================================


def apply_normalization(tensor: torch.Tensor, method: str = "variance") -> torch.Tensor:
    """Apply normalization to teacher features (or optionally to student predictions).

    Args:
        tensor: Input tensor [B, N, D]. Normalization applied along last dim.
        method: "variance" (LayerNorm without affine), "l2", or "none".

    Returns:
        Normalized tensor with same shape.
    """
    if method == "variance":
        mean = tensor.mean(dim=-1, keepdim=True)
        var = tensor.var(dim=-1, keepdim=True)
        var = torch.where(torch.isnan(var), torch.ones_like(var), var)
        return (tensor - mean) / (var + 1.0e-6) ** 0.5
    elif method == "l2":
        return F.normalize(tensor, p=2, dim=-1)
    elif method == "none":
        return tensor
    else:
        raise ValueError(f"Unknown normalization method: {method}")


# ======================================================================
# Token distillation losses (L_head)
# ======================================================================


def masked_smooth_l1_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
    beta: float = 2.0, eps: float = 1e-6,
) -> torch.Tensor:
    """Smooth L1 loss on masked positions only.

    Args:
        pred: Student predictions [B, N, D].
        target: Teacher targets [B, N, D].
        mask: Binary mask [B, N]. True = compute loss here.
        beta: Smooth L1 transition threshold.

    Returns:
        Scalar loss averaged over masked elements.
    """
    assert mask.shape == pred.shape[:2]
    assert pred.shape == target.shape

    if pred.numel() == 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    mask_expanded = mask.unsqueeze(-1).expand_as(pred)

    with torch.autocast(
        device_type="cuda" if pred.device.type == "cuda" else "cpu", enabled=True
    ):
        loss = F.smooth_l1_loss(pred, target, beta=beta, reduction="none")
        masked_loss = loss * mask_expanded
        result = masked_loss.sum() / (mask_expanded.sum() + eps)

        if not torch.isfinite(result):
            result = masked_loss.float().sum() / (mask_expanded.float().sum() + eps)

        return result


def masked_mse_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """MSE (L2) loss on masked positions only.

    Averages per-patch (mean over features first, then over masked patches).

    Args:
        pred: Student predictions [B, N, D].
        target: Teacher targets [B, N, D].
        mask: Binary mask [B, N]. True = compute loss here.

    Returns:
        Scalar loss averaged over masked patches.
    """
    assert mask.shape == pred.shape[:2]
    assert pred.shape == target.shape

    mask_expanded = mask.unsqueeze(-1).expand_as(pred)

    with torch.autocast(
        device_type="cuda" if pred.device.type == "cuda" else "cpu", enabled=True
    ):
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [B, N]
        mask_2d = mask_expanded[:, :, 0]
        result = (loss * mask_2d).sum() / (mask_2d.sum() + eps)

        if not torch.isfinite(result):
            result = (loss.float() * mask_2d.float()).sum() / (mask_2d.float().sum() + eps)

        return result


def _masked_l1_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """L1 loss on masked positions only."""
    assert mask.shape == pred.shape[:2]
    assert pred.shape == target.shape

    mask_expanded = mask.unsqueeze(-1).expand_as(pred)

    with torch.autocast(
        device_type="cuda" if pred.device.type == "cuda" else "cpu", enabled=True
    ):
        loss = torch.abs(pred - target)
        masked_loss = loss * mask_expanded
        result = masked_loss.sum() / (mask_expanded.sum() + eps)

        if not torch.isfinite(result):
            result = masked_loss.float().sum() / (mask_expanded.float().sum() + eps)

        return result


# ======================================================================
# CLS distillation losses (L_cls)
# ======================================================================


def cls_cosine_loss(
    pred_cls: torch.Tensor, target_cls: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Cosine similarity loss for CLS token alignment.

    L = mean((1 - cos(pred, target)) / temperature)

    Args:
        pred_cls: Student CLS features [B, D].
        target_cls: Teacher CLS features [B, D].
        temperature: Scaling factor (lower = sharper gradients).

    Returns:
        Scalar loss.
    """
    pred_norm = F.normalize(pred_cls, dim=-1, p=2)
    target_norm = F.normalize(target_cls, dim=-1, p=2)

    cosine_sim = (pred_norm * target_norm).sum(dim=-1)
    cosine_sim = torch.clamp(cosine_sim, -1.0, 1.0)

    loss = (1 - cosine_sim) / temperature
    return loss.mean()


def cls_ce_loss(
    pred_cls: torch.Tensor, target_cls: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy loss for CLS token using soft targets.

    Converts teacher features to probability distribution and computes
    KL-divergence style loss: L = -sum(softmax(target) * log_softmax(pred)).

    Args:
        pred_cls: Student CLS features [B, D].
        target_cls: Teacher CLS features [B, D].

    Returns:
        Scalar loss.
    """
    target_prob = F.softmax(target_cls, dim=-1)
    pred_log_prob = F.log_softmax(pred_cls, dim=-1)

    loss = -(target_prob * pred_log_prob).sum(dim=-1).mean()
    return loss


# ======================================================================
# Pixel reconstruction helpers
# ======================================================================


def patchify(imgs: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Convert images to patch sequences.

    Args:
        imgs: Input images [B, 3, H, W].
        patch_size: Size of each patch.

    Returns:
        Patches [B, N, patch_size^2 * 3] where N = (H/p) * (W/p).
    """
    B, C, H, W = imgs.shape
    assert H % patch_size == 0 and W % patch_size == 0
    h = H // patch_size
    w = W // patch_size

    x = imgs.reshape(B, C, h, patch_size, w, patch_size)
    x = x.permute(0, 2, 4, 3, 5, 1)  # [B, h, w, p, p, C]
    x = x.reshape(B, h * w, patch_size ** 2 * C)
    return x


# ======================================================================
# Master loss function
# ======================================================================


def compute_loss(
    pred_tok: Optional[torch.Tensor],
    pred_pix: Optional[torch.Tensor],
    teacher_features: Optional[torch.Tensor],
    img: torch.Tensor,
    mask: torch.Tensor,
    cfg: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute MEDiC multi-objective loss with literal weighting.

    L_total = w_head * L_head + w_cls * L_cls + w_pix * L_pix

    Args:
        pred_tok: Student features after distill head [B, N+1, D] or
                  [B, num_visible+1, D]. Includes CLS at position 0.
                  None if distill_head is disabled.
        pred_pix: Pixel decoder output [B, N, patch_size^2*3].
                  None if pixel_decoder is disabled.
        teacher_features: CLIP teacher features [B, N+1, D] or
                          [B, num_visible+1, D]. Includes CLS at position 0.
        img: Input images [B, 3, H, W] (for pixel targets).
        mask: Boolean mask [B, N].
        cfg: Configuration dictionary.

    Returns:
        total_loss: Scalar weighted loss.
        loss_dict: Individual loss values for logging.
    """
    losses_cfg = cfg.get("losses", {})
    model_cfg = cfg.get("model", {}).get("student", {})
    use_mask_tokens = model_cfg.get("use_mask_tokens", True)

    # Normalization
    norm_method = losses_cfg.get("normalization_method", "variance")
    T = teacher_features
    if T is not None and losses_cfg.get("normalize_targets", True):
        T = apply_normalization(T, norm_method)
    if pred_tok is not None and losses_cfg.get("normalize_predictions", False):
        pred_tok = apply_normalization(pred_tok, norm_method)

    loss_dict: Dict[str, float] = {}
    total_loss = torch.tensor(0.0, device=img.device, dtype=img.dtype)

    # ------------------------------------------------------------------
    # 1. Head loss (token distillation on patch features)
    # ------------------------------------------------------------------
    if losses_cfg.get("use_head_loss", True) and pred_tok is not None and T is not None:
        pred_patches = pred_tok[:, 1:, :]
        target_patches = T[:, 1:, :]

        head_cfg = losses_cfg.get("head", {})
        loss_type = head_cfg.get("type", "smooth_l1")
        beta = head_cfg.get("beta", 1.0)

        if use_mask_tokens:
            # Dense mode: loss on masked positions only
            if loss_type == "smooth_l1":
                L_head = masked_smooth_l1_loss(pred_patches, target_patches, mask, beta=beta)
            elif loss_type == "l1":
                L_head = _masked_l1_loss(pred_patches, target_patches, mask)
            elif loss_type == "l2":
                L_head = masked_mse_loss(pred_patches, target_patches, mask)
            else:
                raise ValueError(f"Unknown head loss type: {loss_type}")
        else:
            # Sparse mode: all outputs contribute
            if loss_type == "smooth_l1":
                L_head = F.smooth_l1_loss(pred_patches, target_patches, beta=beta, reduction="mean")
            elif loss_type == "l1":
                L_head = F.l1_loss(pred_patches, target_patches, reduction="mean")
            elif loss_type == "l2":
                L_head = F.mse_loss(pred_patches, target_patches, reduction="mean")
            else:
                raise ValueError(f"Unknown head loss type: {loss_type}")

        w = losses_cfg.get("head_loss_weight", 1.0)
        total_loss = total_loss + w * L_head
        loss_dict["L_head"] = L_head.item()

    # ------------------------------------------------------------------
    # 2. CLS loss (CLS token alignment)
    # ------------------------------------------------------------------
    if losses_cfg.get("use_cls_loss", False) and pred_tok is not None and T is not None:
        pred_cls = pred_tok[:, 0, :]    # Student CLS
        target_cls = T[:, 0, :]          # Teacher CLS

        cls_cfg = losses_cfg.get("cls", {})
        cls_type = cls_cfg.get("type", "cross_entropy")

        if cls_type == "cosine":
            temperature = cls_cfg.get("temperature", 0.1)
            L_cls = cls_cosine_loss(pred_cls, target_cls, temperature=temperature)
        elif cls_type == "cross_entropy":
            L_cls = cls_ce_loss(pred_cls, target_cls)
        else:
            raise ValueError(f"Unknown CLS loss type: {cls_type}")

        w = losses_cfg.get("cls_loss_weight", 1.0)
        total_loss = total_loss + w * L_cls
        loss_dict["L_cls"] = L_cls.item()

    # ------------------------------------------------------------------
    # 3. Pixel loss (reconstruction from decoder)
    # ------------------------------------------------------------------
    if losses_cfg.get("use_decoder_loss", False) and pred_pix is not None:
        patch_size = cfg["model"]["student"]["patch_size"]

        with torch.no_grad():
            pixel_target = patchify(img, patch_size=patch_size)

            if losses_cfg.get("decoder_norm_pix_loss", True):
                mean = pixel_target.mean(dim=-1, keepdim=True)
                var = pixel_target.var(dim=-1, keepdim=True)
                pixel_target = (pixel_target - mean) / (var + 1.0e-6) ** 0.5

        decoder_cfg = losses_cfg.get("decoder", {})
        pix_loss_type = decoder_cfg.get("type", "l2")

        if pix_loss_type == "l2":
            L_pix = masked_mse_loss(pred_pix, pixel_target, mask)
        elif pix_loss_type == "l1":
            L_pix = _masked_l1_loss(pred_pix, pixel_target, mask)
        elif pix_loss_type == "smooth_l1":
            pix_beta = decoder_cfg.get("beta", 1.0)
            L_pix = masked_smooth_l1_loss(pred_pix, pixel_target, mask, beta=pix_beta)
        else:
            raise ValueError(f"Unknown pixel loss type: {pix_loss_type}")

        w = losses_cfg.get("pixel_loss_weight", 1.0)
        total_loss = total_loss + w * L_pix
        loss_dict["L_pix"] = L_pix.item()

    loss_dict["total_loss"] = total_loss.item()
    return total_loss, loss_dict
