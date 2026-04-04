"""
MEDiC Model — unified wrapper for student ViT + distillation head + pixel decoder.

Implements the MEDiC framework (Multi-objective Exploration of Distillation from CLIP):
    L_total = w_head * L_head(H(S(I_masked)), N(T(I_full)))
            + w_cls  * L_cls(S_cls, T_cls)
            + w_pix  * L_pix(D(S(I_masked)), I)

where:
    T = frozen CLIP teacher (external, not wrapped here)
    S = student ViT encoder (sparse or dense mode)
    H = distillation head (linear projection to teacher dim)
    D = pixel decoder (MAE-style, reconstructs masked patches)
    N = layer normalization on teacher features
    L_head = Smooth-L1 loss on masked token positions
    L_cls = CLS token alignment (cosine or cross-entropy)
    L_pix = Pixel reconstruction loss (L2 on normalized patches)

Reference: "MEDiC: Multi-objective Exploration of Distillation from CLIP" (arXiv:2603.29009)
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple

from .vision_transformer import VisionTransformerMIM


class MEDiCModel(nn.Module):
    """
    MEDiC model: student ViT encoder + distillation head + optional pixel decoder.

    Supports both dense mode (BEiT-style, mask tokens replace masked patches)
    and sparse mode (MAE-style, masked patches dropped). The distill_head
    projects student token features to match the CLIP teacher's embedding
    dimension (used for both token and CLS distillation). The pixel_decoder
    reconstructs masked patches for pixel-level self-supervision.
    """

    def __init__(self, cfg: Dict[str, Any], mask_generator=None):
        """
        Args:
            cfg: Full configuration dictionary. Expected keys:
                 - model.student.*  (ViT architecture params)
                 - model.teacher.embed_dim  (target projection dim)
                 - model.decoder.*  (pixel decoder params, optional)
                 - losses.use_head_loss, use_cls_loss, use_decoder_loss
            mask_generator: A masking generator instance (block, random, or evolved).
        """
        super().__init__()
        self.cfg = cfg
        self.student = build_student(cfg)
        self.mask_generator = mask_generator

        losses_cfg = cfg.get("losses", {})
        in_dim = cfg["model"]["student"]["embed_dim"]
        out_dim = cfg.get("model", {}).get("teacher", {}).get("embed_dim", in_dim)

        # Distillation head: project student embed_dim -> teacher embed_dim
        # Used for both token distillation (L_head) and CLS distillation (L_cls)
        self.distill_head: Optional[nn.Module] = None
        if losses_cfg.get("use_head_loss", True) or losses_cfg.get("use_cls_loss", False):
            self.distill_head = nn.Linear(in_dim, out_dim)

        # Pixel decoder: MAE-style transformer decoder for reconstruction
        self.pixel_decoder: Optional[nn.Module] = None
        if losses_cfg.get("use_decoder_loss", False):
            self.pixel_decoder = build_pixel_decoder(cfg)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        img: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mask_generator=None,
        teacher=None,
        epoch: Optional[int] = None,
        model_ema=None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Forward pass: generate mask -> encode -> project -> decode.

        Args:
            img:            Input images [B, 3, H, W].
            mask:           Pre-generated boolean mask [B, N] (True = masked).
                            If None, self.mask_generator is used.
            mask_generator: Legacy override; ignored if mask is provided.
            teacher:        Teacher model (for evolved masking with CLIP attention).
            epoch:          Current epoch (for evolved masking alpha schedule).
            model_ema:      EMA model (for evolved masking with student attention).

        Returns:
            pred_tok:    Distillation head output [B, N_vis+1, D_teacher]
                         (includes CLS token at position 0), or None if
                         distill_head is disabled.
            pred_pix:    Pixel decoder output [B, N, patch_size^2 * 3], or None
                         if pixel_decoder is disabled.
            mask_out:    The boolean mask that was applied [B, N].
            ids_restore: Indices to unshuffle / restore full patch order
                         [B, N], needed for loss computation.
        """
        # ----- Mask generation ----------------------------------------
        if mask is None:
            mask = self._generate_mask(img, epoch=epoch, model_ema=model_ema, teacher=teacher)

        # ----- Student forward ----------------------------------------
        z, ids_restore = self.student(img, mask, mask_generator=mask_generator)

        # ----- Distillation head (token + CLS features) ---------------
        pred_tok = None
        if self.distill_head is not None:
            pred_tok = self.distill_head(z)  # [B, N_vis+1, D_teacher]

        # ----- Pixel decoder ------------------------------------------
        pred_pix = None
        if self.pixel_decoder is not None:
            visible_positions_shuffled = getattr(self.student, 'visible_positions_shuffled', None)
            pred_pix = self.pixel_decoder(
                z, mask, ids_restore=ids_restore,
                visible_positions_shuffled=visible_positions_shuffled,
            )  # [B, N, patch_size^2 * 3]

        return pred_tok, pred_pix, mask, ids_restore

    # ------------------------------------------------------------------
    # Mask generation helpers
    # ------------------------------------------------------------------

    def _generate_mask(
        self, img: torch.Tensor,
        epoch: Optional[int] = None,
        model_ema=None,
        teacher=None,
    ) -> torch.Tensor:
        """
        Generate a batch of masks using self.mask_generator.

        For evolved masking, passes epoch and model_ema/teacher for
        attention-based semantic part discovery.
        """
        if self.mask_generator is None:
            raise ValueError("Either mask or mask_generator must be provided")

        B = img.shape[0]

        # Check if this is an evolved masking generator
        is_evolved = hasattr(self.mask_generator, 'attention_source')

        if is_evolved:
            # Evolved masking needs model features for attention extraction
            attention_source = getattr(self.mask_generator, 'attention_source', 'student_ema')

            if attention_source == 'clip_teacher':
                # CLIP teacher provides attention from raw images
                mask = self.mask_generator(
                    x=None, epoch=epoch, model_ema=model_ema,
                    clip_teacher=teacher, images=img,
                )
            else:
                # Student/EMA provides attention from patch embeddings
                x = self.student.patch_embed(img)
                mask = self.mask_generator(
                    x=x, epoch=epoch, model_ema=model_ema,
                )
            return mask

        # Standard masking (block or random)
        has_shuffle = (
            hasattr(self.mask_generator, "shuffle_patches")
            and self.mask_generator.shuffle_patches
        )

        if has_shuffle:
            first_mask = self.mask_generator()
            stored_shuffle = self.mask_generator.ids_shuffle
            stored_restore = self.mask_generator.ids_restore

            masks = [first_mask]
            for _ in range(1, B):
                masks.append(self.mask_generator())

            self.mask_generator.ids_shuffle = stored_shuffle
            self.mask_generator.ids_restore = stored_restore
        else:
            masks = [self.mask_generator() for _ in range(B)]

        return torch.stack(masks, dim=0).to(img.device)

    def get_masking_info(self, epoch: Optional[int] = None) -> Dict[str, Any]:
        """Get masking generator info (e.g., evolution alpha for evolved masking)."""
        info = {}
        if hasattr(self.mask_generator, '_compute_evolution_alpha') and epoch is not None:
            info['alpha'] = self.mask_generator._compute_evolution_alpha(epoch)
        return info

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def no_weight_decay(self) -> set:
        """Parameters that should be excluded from weight decay."""
        no_decay = {"student." + k for k in self.student.no_weight_decay()}
        if self.pixel_decoder is not None and hasattr(self.pixel_decoder, 'no_weight_decay'):
            no_decay.update({"pixel_decoder." + k for k in self.pixel_decoder.no_weight_decay()})
        return no_decay


# ======================================================================
# Factory functions
# ======================================================================


def build_student(cfg: Dict[str, Any]) -> VisionTransformerMIM:
    """
    Build the student ViT from configuration.

    Supports both dense mode (BEiT-style, mask tokens replace masked patches)
    and sparse mode (MAE-style, masked patches dropped).
    """
    s = cfg["model"]["student"]
    use_mask_tokens = s.get("use_mask_tokens", True)

    return VisionTransformerMIM(
        img_size=s["img_size"],
        patch_size=s["patch_size"],
        embed_dim=s["embed_dim"],
        depth=s["depth"],
        num_heads=s["num_heads"],
        mlp_ratio=s.get("mlp_ratio", 4.0),
        drop_path_rate=s.get("drop_path_rate", 0.1),
        init_values=s.get("init_values", 0.1),
        use_abs_pos_emb=s.get("use_abs_pos_emb", False),
        use_sincos_pos_emb=s.get("use_sincos_pos_emb", False),
        use_shared_rel_pos_bias=s.get("use_shared_rel_pos_bias", True),
        use_rel_pos_bias=s.get("use_rel_pos_bias", False),
        use_mask_tokens=use_mask_tokens,
    )


def build_pixel_decoder(cfg: Dict[str, Any]):
    """Build the MAE-style pixel reconstruction decoder from config."""
    from .decoder_mae import MAEDecoder

    s = cfg["model"]["student"]
    d = cfg.get("model", {}).get("decoder", {})

    return MAEDecoder(
        in_dim=s["embed_dim"],
        embed_dim=d.get("decoder_embed_dim", 512),
        depth=d.get("decoder_depth", 8),
        num_heads=d.get("decoder_num_heads", 16),
        patch_size=s["patch_size"],
        replace_mask_tokens=d.get("replace_mask_tokens", True),
        img_size=s["img_size"],
        use_abs_pos_emb=d.get("use_abs_pos_emb", True),
        use_sincos_pos_emb=d.get("use_sincos_pos_emb", True),
        use_rel_pos_bias=False,
        use_shared_rel_pos_bias=False,
    )


def build_medic_model(
    cfg: Dict[str, Any],
    mask_generator=None,
) -> MEDiCModel:
    """
    Build a MEDiCModel from a configuration dictionary.

    Args:
        cfg:            Full config dict with model.student, model.teacher, losses.
        mask_generator: Optional masking generator (block, random, or evolved).

    Returns:
        Configured MEDiCModel instance.
    """
    return MEDiCModel(cfg, mask_generator=mask_generator)
