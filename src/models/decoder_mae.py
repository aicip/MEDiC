"""
MAE-style pixel reconstruction decoder for MEDiC.

Lightweight transformer decoder that reconstructs normalized pixel patches
from encoder features. Supports both sparse (encoder drops masked patches)
and dense (encoder keeps mask tokens) operating modes.

Architecture:
    - Linear projection: encoder_dim -> decoder_dim
    - Learnable mask tokens for missing patches (sparse mode)
    - N transformer blocks with positional embeddings
    - Pixel prediction head: decoder_dim -> patch_size^2 * 3

Reference: "Masked Autoencoders Are Scalable Vision Learners" (arXiv:2111.06377)
"""

import torch
import torch.nn as nn
from functools import partial
from typing import Set, Optional

from .vision_transformer import Block, RelativePositionBias
from ..utils.pos_embed import get_2d_sincos_pos_embed


class MAEDecoder(nn.Module):
    """
    MAE-style decoder for pixel reconstruction.

    Supports both sparse and dense encoder modes:
    - Sparse: encoder outputs visible patches only; decoder adds mask tokens
    - Dense: encoder outputs all patches with mask tokens; decoder can
             optionally replace encoder mask tokens with its own

    Args:
        in_dim: Input dimension from encoder (default: 768)
        embed_dim: Decoder embedding dimension (default: 512)
        depth: Number of decoder transformer blocks (default: 8)
        num_heads: Number of attention heads (default: 16)
        patch_size: Size of image patches (default: 16)
        replace_mask_tokens: In dense mode, replace encoder mask tokens
                             with decoder's own learnable tokens
        img_size: Input image size (default: 224)
        use_abs_pos_emb: Use absolute position embeddings
        use_sincos_pos_emb: Initialize pos embeddings with sincos (fixed)
        use_rel_pos_bias: Use per-layer relative position bias
        use_shared_rel_pos_bias: Use shared relative position bias
    """

    def __init__(
        self,
        in_dim=768,
        embed_dim=512,
        depth=8,
        num_heads=16,
        patch_size=16,
        replace_mask_tokens=True,
        img_size=224,
        use_abs_pos_emb=True,
        use_rel_pos_bias=False,
        use_shared_rel_pos_bias=False,
        use_sincos_pos_emb=False,
        **kwargs,
    ):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.embed = nn.Linear(in_dim, embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.replace_mask_tokens = replace_mask_tokens
        self.use_sincos_pos_emb = use_sincos_pos_emb

        grid_size = img_size // patch_size

        # Absolute positional embeddings
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(
                torch.zeros((1, self.num_patches + 1, embed_dim)), requires_grad=True
            )
        else:
            self.pos_embed = None

        # Shared relative position bias
        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(
                window_size=(grid_size, grid_size), num_heads=num_heads,
            )
        else:
            self.rel_pos_bias = None

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                window_size=(grid_size, grid_size) if use_rel_pos_bias and not use_shared_rel_pos_bias else None,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.pred = nn.Linear(embed_dim, patch_size ** 2 * 3, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize decoder weights."""
        if self.pos_embed is not None:
            if self.use_sincos_pos_emb:
                pos_embed = get_2d_sincos_pos_embed(
                    self.pos_embed.shape[-1], int(self.num_patches ** 0.5), cls_token=True,
                )
                self.pos_embed.data.copy_(torch.tensor(pos_embed, dtype=torch.float32).unsqueeze(0))
                self.pos_embed.requires_grad = False
            else:
                torch.nn.init.normal_(self.pos_embed, std=0.02)

        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        encoder_out: torch.Tensor,
        mask: torch.Tensor,
        ids_restore: Optional[torch.Tensor] = None,
        visible_positions_shuffled: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decoder forward pass with automatic sparse/dense mode detection.

        Args:
            encoder_out: Encoder output with CLS token.
                - Sparse: [B, num_visible+1, D]
                - Dense: [B, N+1, D]
            mask: Boolean mask where True = masked patch [B, N].
            ids_restore: Indices for unshuffling patches back to spatial order.
            visible_positions_shuffled: Positions of visible patches in shuffled
                                         order (for scatter-based reconstruction).

        Returns:
            Pixel predictions for all N patches [B, N, patch_size^2 * 3].
        """
        batch_size = encoder_out.shape[0]
        is_sparse_mode = encoder_out.shape[1] < self.num_patches + 1

        # Project to decoder dimension
        x = self.embed(encoder_out)

        if is_sparse_mode:
            x = self._handle_sparse_mode(
                x, mask, batch_size, ids_restore, visible_positions_shuffled,
            )
        elif self.replace_mask_tokens:
            x = self._handle_dense_replace(x, mask, batch_size)

        # Add positional embeddings
        if self.pos_embed is not None:
            x = x + self.pos_embed

        # Shared relative position bias
        shared_rel_pos_bias = None
        if self.rel_pos_bias is not None:
            shared_rel_pos_bias = self.rel_pos_bias()

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x, shared_rel_pos_bias=shared_rel_pos_bias)

        x = self.norm(x)
        x = self.pred(x)

        # Return patch predictions only (skip CLS token)
        return x[:, 1:, :]

    def _handle_sparse_mode(
        self, x, mask, batch_size, ids_restore, visible_positions_shuffled,
    ):
        """Add mask tokens and unshuffle for sparse encoder mode."""
        cls_token = x[:, :1, :]
        x_visible = x[:, 1:, :]
        decoder_dim = x.shape[-1]
        num_patches = mask.shape[1]

        if ids_restore is not None:
            if visible_positions_shuffled is not None:
                # Scatter-based reconstruction (block masking)
                x_full_shuffled = self.mask_token.expand(batch_size, num_patches, decoder_dim).clone().type_as(x_visible)
                num_visible = x_visible.shape[1]
                batch_indices = torch.arange(batch_size, device=x_visible.device).unsqueeze(1).expand(-1, num_visible)
                x_full_shuffled[batch_indices, visible_positions_shuffled] = x_visible

                if ids_restore.dim() == 1:
                    ids_restore_batch = ids_restore.unsqueeze(0).expand(batch_size, -1)
                else:
                    ids_restore_batch = ids_restore

                x_full = torch.gather(
                    x_full_shuffled, dim=1,
                    index=ids_restore_batch.unsqueeze(-1).expand(-1, -1, decoder_dim),
                )
            else:
                # Concat-based reconstruction (random masking)
                num_mask = num_patches - x_visible.shape[1]
                mask_tokens = self.mask_token.repeat(batch_size, num_mask, 1).to(dtype=x_visible.dtype)
                x_full_shuffled = torch.cat([x_visible, mask_tokens], dim=1)

                if ids_restore.dim() == 1:
                    ids_restore_batch = ids_restore.unsqueeze(0).expand(batch_size, -1)
                else:
                    ids_restore_batch = ids_restore

                x_full = torch.gather(
                    x_full_shuffled, dim=1,
                    index=ids_restore_batch.unsqueeze(-1).expand(-1, -1, decoder_dim),
                )
        else:
            # No shuffling: scatter visible patches at original positions
            x_full = self.mask_token.expand(batch_size, num_patches, decoder_dim).clone().type_as(x_visible)
            visible_mask = ~mask
            for b in range(batch_size):
                vis_idx = visible_mask[b].nonzero(as_tuple=True)[0]
                n_vis = min(len(vis_idx), x_visible.shape[1])
                x_full[b, vis_idx[:n_vis]] = x_visible[b, :n_vis]

        return torch.cat([cls_token, x_full], dim=1)

    def _handle_dense_replace(self, x, mask, batch_size):
        """Replace encoder mask tokens with decoder mask tokens (dense mode)."""
        cls_token = x[:, :1, :]
        x_patches = x[:, 1:, :]

        mask_expanded = mask.unsqueeze(-1).expand_as(x_patches)
        decoder_mask = self.mask_token.expand_as(x_patches).type_as(x_patches)

        # Replace masked positions with decoder mask tokens
        x_patches = torch.where(mask_expanded.bool(), decoder_mask, x_patches)

        return torch.cat([cls_token, x_patches], dim=1)

    def no_weight_decay(self) -> Set[str]:
        """Parameters that should skip weight decay."""
        no_decay = {"mask_token"}
        if self.pos_embed is not None:
            no_decay.add("pos_embed")
        if self.rel_pos_bias is not None:
            no_decay.add("rel_pos_bias.relative_position_bias_table")
        return no_decay
