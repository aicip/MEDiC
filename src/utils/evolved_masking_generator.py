"""
Evolved Part Masking Generator for MEDiC Framework

This module implements the evolved masking strategy from "Evolved Part Masking for Self-Supervised Learning"
(CVPR 2023) integrated into the MEDiC framework. The evolved masking strategy progressively transitions
from spatial (grid-based) to semantic (attention-based part) masking during training.

Key Features:
- EMA model for stable attention extraction
- EM clustering for semantic part discovery
- Progressive spatial→semantic evolution
- Configurable evolution parameters
- Compatible with MEDiC's multi-loss training framework

Reference:
Feng, Zhanzhou, and Shiliang Zhang. "Evolved Part Masking for Self-Supervised Learning."
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.
"""

import math
import random
from typing import Optional, Tuple, List, Dict, Any
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from timm.utils import ModelEma
except ImportError:
    warnings.warn("timm.utils.ModelEma not available. Please install timm>=0.6.0")
    ModelEma = None


def EM_clustering(x: torch.Tensor, stage_num: int = 10, k: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Expectation-Maximization clustering - EXACT copy from original EPM implementation.

    Args:
        x: Input tensor [batch, channels, num_tokens] - attention features
        stage_num: Number of EM iterations
        k: Number of clusters

    Returns:
        mu: Cluster centers [batch, k, channels]
        z: Soft assignments [batch, num_tokens, k]
    """
    def l2norm(inp, dim):
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))

    def Kernel(x, mu):
        x_t = x.permute(0, 2, 1)  # b * n * c
        z = kappa * torch.bmm(x_t, mu)  # b * n * k
        return z

    mu = torch.Tensor(1, x.size(1), k).to(x.device)
    mu.normal_(0, math.sqrt(2. / k))
    mu = l2norm(mu, dim=1)
    kappa = 40.0
    b = x.shape[0]
    mu = mu.repeat(b, 1, 1)  # b * c * k

    with torch.no_grad():
        for i in range(stage_num):
            # E STEP:
            z = Kernel(x, mu)
            z = F.softmax(z, dim=2)  # b * n * k
            # M STEP:
            z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))  # CRITICAL: This was missing!
            mu = torch.bmm(x, z_)  # b * c * k
            mu = l2norm(mu, dim=1)

    z = Kernel(x, mu)
    z = F.softmax(z, dim=2)  # b * n * k
    z = z / (1e-6 + z.sum(dim=1, keepdim=True))  # CRITICAL: This normalization was missing!
    mu = mu.permute(0, 2, 1)  # b * k * c

    return mu, z


class EvolvedMaskingGenerator:
    """
    Evolved masking generator that progressively transitions from spatial to semantic masking.

    This generator implements the evolved masking strategy where the masking pattern
    evolves during training from purely spatial (grid-based) to semantic (attention-based
    part discovery) through an alpha parameter that increases over epochs.

    Key components:
    1. EMA model for stable attention extraction
    2. EM clustering for semantic part discovery
    3. Progressive evolution parameter (alpha)
    4. Spatial + semantic noise combination
    """

    def __init__(
        self,
        student_model: nn.Module,
        mask_ratio: float,
        cfg: Dict[str, Any],
        grid_h: int = 14,
        grid_w: int = 14,
        shuffle_patches: bool = False,
        per_sample_shuffling: bool = False,
    ):
        """
        Initialize the evolved masking generator.

        Args:
            student_model: The student ViT model (no longer used for internal EMA)
            mask_ratio: Ratio of patches to mask (0.0 to 1.0)
            cfg: Configuration dictionary with evolved masking parameters
            grid_h: Height of the patch grid (default: 14 for 224x224 images with 16x16 patches)
            grid_w: Width of the patch grid (default: 14 for 224x224 images with 16x16 patches)
        """
        self.mask_ratio = mask_ratio
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_patches = grid_h * grid_w
        self.len_keep = int(self.num_patches * (1 - mask_ratio))

        # Evolution parameters from config
        self.ema_decay = cfg.get('ema_decay', 0.9999)
        self.cluster_range = cfg.get('cluster_range', [10, 40])
        self.evolution_epochs = cfg.get('evolution_epochs', 300)
        self.em_iterations = cfg.get('em_iterations', 15)
        self.spatial_noise_type = cfg.get('spatial_noise_type', 'grid')
        self.validation_alpha_scale = cfg.get('validation_alpha_scale', 0.5)
        self.gamma = cfg.get('gamma', 0.5)  # Evolution speed parameter - default to 0.5 (sqrt)

        # Attention source configuration
        self.attention_source = cfg.get('attention_source', 'student_ema')  # Default to student EMA
        if self.attention_source not in ['student_ema', 'student_online', 'clip_teacher']:
            raise ValueError(f"Invalid attention_source: {self.attention_source}. "
                           f"Must be one of: 'student_ema', 'student_online', 'clip_teacher'")

        # Debug settings
        self.debug = cfg.get('debug', False)
        if self.debug:
            print(f"[EvolvedMasking] Debug mode enabled for attention_source: {self.attention_source}")

        # Store shuffling configuration
        self.shuffle_patches = shuffle_patches
        self.per_sample_shuffling = per_sample_shuffling  # NEW: per-sample vs consistent shuffling
        self.ids_shuffle = None  # Will store shuffle indices if shuffling enabled
        self.ids_restore = None  # Will store restore indices if shuffling enabled

        # Validate parameters
        if not (0.0 <= mask_ratio <= 1.0):
            raise ValueError(f"mask_ratio must be between 0.0 and 1.0, got {mask_ratio}")
        if len(self.cluster_range) != 2 or self.cluster_range[0] >= self.cluster_range[1]:
            raise ValueError(f"cluster_range must be [low, high] with low < high, got {self.cluster_range}")

        # No longer create internal EMA - will use passed model_ema from train.py
        # This follows EPM exactly where EMA is managed externally

    def __repr__(self) -> str:
        """String representation of the generator."""
        return (
            f"EvolvedMaskingGenerator(H={self.grid_h}, W={self.grid_w}, "
            f"N={self.num_patches}, N_keep={self.len_keep}, "
            f"mask_ratio={self.mask_ratio:.3f}, "
            f"cluster_range={self.cluster_range}, "
            f"evolution_epochs={self.evolution_epochs}, "
            f"gamma={self.gamma}, "
            f"attention_source={self.attention_source}, "
            f"no_shuffling=True)"
        )

    def get_shape(self) -> Tuple[int, int]:
        """Returns the shape of the grid (height, width)."""
        return self.grid_h, self.grid_w


    def _get_features(self, input_x: torch.Tensor, model_or_ema, clip_teacher=None) -> torch.Tensor:
        """
        Extract attention features - Fixed version with proper relative position bias handling.

        Args:
            input_x: Input patch embeddings [B, N, D] or raw images [B, 3, H, W] for CLIP
            model_or_ema: Either EMA model from train.py or direct online model
            clip_teacher: CLIP teacher model (required when attention_source='clip_teacher')

        Returns:
            Aggregated attention maps [B, N, N] summed across all heads and layers
        """
        # Handle CLIP teacher attention extraction
        if self.attention_source == 'clip_teacher':
            if clip_teacher is None:
                raise ValueError("CLIP teacher model is required when attention_source='clip_teacher'")

            # Validate input shape and type for CLIP
            if input_x is None:
                raise ValueError("Input images are required when using CLIP teacher attention")

            if not isinstance(input_x, torch.Tensor):
                raise TypeError(f"Expected input_x to be torch.Tensor, got {type(input_x)}")

            if input_x.dim() != 4:
                raise ValueError(f"CLIP teacher expects 4D input [B, C, H, W], got shape {input_x.shape}")

            B, C, H, W = input_x.shape
            if C != 3:
                raise ValueError(f"CLIP teacher expects 3-channel RGB images [B, 3, H, W], got {C} channels")

            # Validate image dimensions are compatible with expected patch size
            expected_size = int(math.sqrt(self.num_patches) * 16)  # Assuming patch_size=16
            if H != expected_size or W != expected_size:
                warnings.warn(f"Image size {H}x{W} may not match expected size {expected_size}x{expected_size} "
                            f"for {self.num_patches} patches. This could cause issues.")

            with torch.no_grad():
                # For CLIP, input_x should be raw images [B, 3, H, W]
                # Extract attention maps directly from CLIP
                attention_maps = clip_teacher.get_attention_maps(input_x)

                if self.debug:
                    print(f"[EvolvedMasking] CLIP attention maps shape: {attention_maps.shape}")
                    print(f"[EvolvedMasking] CLIP attention stats - min: {attention_maps.min():.4f}, "
                          f"max: {attention_maps.max():.4f}, mean: {attention_maps.mean():.4f}, "
                          f"std: {attention_maps.std():.4f}")

                return attention_maps

        # Handle student model attention extraction (EMA or online)
        if model_or_ema is None:
            warnings.warn("Model not provided. Using random attention patterns.")
            if len(input_x.shape) == 4:  # Images
                B = input_x.shape[0]
                N = self.num_patches
            else:  # Patch embeddings
                B, N, _ = input_x.shape
            return torch.rand(B, N, N, device=input_x.device)

        with torch.no_grad():
            input_x = input_x.clone().detach()
            x = input_x

            # Handle both EMA model and direct online model
            # Extract the actual student model from various wrappers
            if hasattr(model_or_ema, 'ema'):
                # EMA model case
                base_model = model_or_ema.ema
            else:
                # Direct online model case
                base_model = model_or_ema

            # Handle DDP wrapper first
            if hasattr(base_model, 'module'):
                base_model = base_model.module

            # Now extract student from MEDiCModel if needed
            if hasattr(base_model, 'student'):
                # MEDiCModel case - get the student ViT
                actual_model = base_model.student
            else:
                # Direct ViT model case
                actual_model = base_model

            # Add position embeddings to patches
            if hasattr(actual_model, 'pos_embed') and actual_model.pos_embed is not None:
                x = x + actual_model.pos_embed[:, 1:, :]

            # append cls token
            cls_token = actual_model.cls_token
            if hasattr(actual_model, 'pos_embed') and actual_model.pos_embed is not None:
                cls_token = cls_token + actual_model.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

            # Apply position dropout
            if hasattr(actual_model, 'pos_drop'):
                x = actual_model.pos_drop(x)

            rel_pos_bias = None
            if hasattr(actual_model, 'rel_pos_bias') and actual_model.rel_pos_bias is not None:
                rel_pos_bias = actual_model.rel_pos_bias()

            # apply Transformer blocks with correct relative position bias
            attn_abs_set = None
            for blk in actual_model.blocks:
                if hasattr(blk, 'get_attn_graph'):
                    x, attn_abs, attn_softmax = blk.get_attn_graph(x, shared_rel_pos_bias=rel_pos_bias)

                    if attn_abs_set is None:
                        attn_abs_set = attn_softmax
                    else:
                        attn_abs_set = attn_abs_set + attn_softmax
                else:
                    # Fallback for blocks without get_attn_graph (shouldn't happen)
                    x = blk(x)
                    # Use random attention as fallback
                    B, N, C = x.shape
                    fake_attn = torch.rand(B, 12, N, N, device=x.device)  # 12 heads for ViT-Base
                    fake_attn = fake_attn.softmax(dim=-1)

                    if attn_abs_set is None:
                        attn_abs_set = fake_attn
                    else:
                        attn_abs_set = attn_abs_set + fake_attn

            # Remove CLS token interactions: [B, H, N+1, N+1] -> [B, H, N, N]
            attn_abs_set = attn_abs_set[:, :, 1:, 1:]

            # Sum across attention heads: [B, H, N, N] -> [B, N, N]
            attn_abs_set = attn_abs_set.sum(dim=1)

            return attn_abs_set

    def _unsupervised_token_classification(
        self,
        x: torch.Tensor,
        cluster_num: List[int],
        model_or_ema,
        clip_teacher=None
    ) -> torch.Tensor:
        """
        EXACT copy from original EPM unsupervised_token_classification method.

        Args:
            x: Input patch embeddings [B, N, D]
            cluster_num: [min_clusters, max_clusters] range
            model_or_ema: Either EMA model or direct online model

        Returns:
            Token part assignments [B, N] with values in [0, num_clusters-1]
        """
        with torch.no_grad():
            attn_abs_set = self._get_features(x, model_or_ema, clip_teacher)  # [B, N, N]

            # Transpose attention tensor for EM clustering
            # EM_clustering expects [B, channels, num_tokens] format
            # We have [B, N_query, N_key] from attention maps
            # Transpose to [B, N_key, N_query] to treat attention patterns as features
            attn_abs_set = attn_abs_set.permute(0, 2, 1)  # [B, N_query, N_key] -> [B, N_key, N_query]

            cluster_num_val = random.randint(cluster_num[0], cluster_num[1])
            meta, meta_map = EM_clustering(attn_abs_set, stage_num=15, k=cluster_num_val)
            # serial num starts from 0
            classified = torch.max(meta_map, dim=-1)[1]

            return classified

    def _generate_spatial_noise(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Generate spatial noise pattern.

        Args:
            batch_size: Number of samples in batch
            device: Device to create tensors on

        Returns:
            Spatial noise [B, N] where N = grid_h * grid_w
        """
        if self.spatial_noise_type == 'grid':
            # Checkerboard/grid pattern
            noise_grid = torch.rand(2, 2, device=device)
            noise = torch.ones(self.num_patches, device=device)

            for i in range(self.num_patches):
                row = (i // self.grid_w) % 2
                col = i % 2
                noise[i] = noise_grid[row, col]

            noise = noise.repeat(batch_size, 1)

        elif self.spatial_noise_type == 'random':
            # Pure random noise
            noise = torch.rand(batch_size, self.num_patches, device=device)

        else:
            raise ValueError(f"Unknown spatial_noise_type: {self.spatial_noise_type}")

        return noise

    def _generate_semantic_noise(
        self,
        x: torch.Tensor,
        batch_size: int,
        device: torch.device,
        epoch: Optional[int] = None,
        model_or_ema = None,
        clip_teacher = None
    ) -> torch.Tensor:
        """
        Generate semantic noise based on attention-based part discovery.

        Args:
            x: Input patch embeddings [B, N, D]
            batch_size: Number of samples in batch
            device: Device to create tensors on
            epoch: Current training epoch
            model_or_ema: Either EMA model or direct online model

        Returns:
            Semantic noise [B, N] based on discovered parts
        """
        # Dynamic cluster count that decreases over training (per EPM paper)
        low, high = self.cluster_range
        if epoch is not None:
            # Linearly decrease cluster count over training: high→low
            progress = min(epoch + 1, self.evolution_epochs) / self.evolution_epochs
            cluster_idx = (high - low) * (1 - progress) + low
            cluster_num = [int(cluster_idx), int(cluster_idx + 2)]
        else:
            # Use full range for validation/testing
            cluster_num = [low, high]

        # Classify tokens into semantic parts
        part_assignments = self._unsupervised_token_classification(x, cluster_num, model_or_ema, clip_teacher)  # [B, N]

        # Generate random noise for each part
        max_parts = high + 1
        parts_noise_assign = torch.rand(batch_size, max_parts, device=device)

        # Gather noise values according to part assignments
        semantic_noise = torch.gather(parts_noise_assign, dim=1, index=part_assignments)

        return semantic_noise

    def _compute_evolution_alpha(self, epoch: Optional[int]) -> float:
        """
        Compute the evolution parameter alpha that controls spatial→semantic transition.

        Args:
            epoch: Current training epoch (None for random alpha, rarely used)

        Returns:
            Alpha value in [0, 1] where 0=spatial, 1=semantic

        Note: During validation, the current training epoch is passed, so alpha
        continues to evolve. This allows validation to track training progress.
        The validation_alpha_scale parameter is only used when epoch=None.
        """
        if epoch is None:
            # Random alpha (rarely used - only for special testing scenarios)
            # Note: Normal validation passes the current epoch
            return random.random() * self.validation_alpha_scale
        else:
            # Progressive evolution: α = (epoch / total_epochs)^gamma
            # Used for both training and validation
            alpha =  math.pow(1.0 / self.evolution_epochs * (epoch + 1), self.gamma)
            return alpha

    def __call__(
        self,
        x: Optional[torch.Tensor] = None,
        epoch: Optional[int] = None,
        model_ema = None,
        clip_teacher = None,
        images = None
    ) -> torch.Tensor:
        """
        Generate evolved mask based on semantic parts (simplified for MEDiC).

        This simplified version removes shuffling since MEDiC processes all patches
        anyway. The mask generation logic remains the same - using attention patterns
        to discover semantic parts and progressively evolving from spatial to semantic.

        Args:
            x: Input patch embeddings [B, N, D] (required for student attention) or None for CLIP
            epoch: Current training epoch (None for random evolution)
            model_ema: Either EMA model from train.py or direct online model (for no-EMA mode)
            clip_teacher: CLIP teacher model (required when attention_source='clip_teacher')
            images: Raw images [B, 3, H, W] (required when using CLIP attention)

        Returns:
            torch.Tensor: Boolean mask [B, N] where True=masked, False=visible
        """
        # Determine input based on attention source
        if self.attention_source == 'clip_teacher':
            if images is None:
                raise ValueError("Raw images are required when using CLIP teacher attention")
            if clip_teacher is None:
                raise ValueError("CLIP teacher model is required when attention_source='clip_teacher'")

            # Validate image tensor shape and type
            if not isinstance(images, torch.Tensor):
                raise TypeError(f"Expected images to be torch.Tensor when using CLIP, got {type(images)}")
            if images.dim() != 4:
                raise ValueError(f"CLIP expects 4D images tensor [B, 3, H, W], got shape {images.shape}")
            if images.shape[1] != 3:
                raise ValueError(f"CLIP expects 3-channel RGB images, got {images.shape[1]} channels")

            # Use images for CLIP attention extraction
            input_for_classification = images
        else:
            if x is None:
                raise ValueError("Input patch embeddings x are required for student attention masking")
            input_for_classification = x

        low = self.cluster_range[0]
        high = self.cluster_range[1]
        total_epoch = float(self.evolution_epochs)

        with torch.no_grad():
            # Use the dedicated method to compute alpha (avoiding redundancy)
            alpha = self._compute_evolution_alpha(epoch)

            # Dynamic cluster range that decreases over training
            cluster_idx = (high-low)*(total_epoch-epoch)/total_epoch + low if epoch is not None else (high+low)/2
            cluster_num = [int(cluster_idx), int(cluster_idx+2)]
            parts_num = cluster_num[1] + 1

            # Get dimensions based on input type
            if self.attention_source == 'clip_teacher':
                N = images.shape[0]  # batch size
                L = self.num_patches  # number of patches
            else:
                N, L, D = x.shape  # batch, length, dim

            H, W = int(math.sqrt(L)), int(math.sqrt(L))
            len_keep = int(L * (1 - self.mask_ratio))

            # Classify tokens into semantic parts using attention patterns
            tk_parts_dic = self._unsupervised_token_classification(
                input_for_classification, cluster_num, model_ema, clip_teacher
            )

            if self.debug:
                print(f"\n[EvolvedMasking] Epoch {epoch}:")
                print(f"  - Alpha (evolution): {alpha:.3f}")
                print(f"  - Spatial weight: {1-alpha:.3f}, Semantic weight: {alpha:.3f}")
                print(f"  - Cluster range: {cluster_num}")
                print(f"  - Token assignments shape: {tk_parts_dic.shape}")
                unique_parts = torch.unique(tk_parts_dic)
                print(f"  - Unique parts discovered: {len(unique_parts)} - {unique_parts.tolist()}")

            # Grid noise for spatial masking (early training)
            device = input_for_classification.device
            noise_grid = torch.rand(2, 2, device=device)
            noise = torch.ones((L), device=device)
            for i in range(0, L):
                noise[i] = noise_grid[(i//W)%2, i%2]
            noise = noise.repeat(N, 1)

            # Semantic noise for part-based masking (late training)
            parts_noise_asign = torch.rand(N, parts_num, device=device)
            parts_noise = torch.gather(parts_noise_asign, dim=1, index=tk_parts_dic)

            # Blend spatial and semantic noise based on evolution parameter alpha
            noise = (1-alpha)*noise + alpha*parts_noise
            noise = noise.detach()

            # Sort noise to determine which patches to mask
            ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
            ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=device)
        mask[:, :len_keep] = 0
        # Handle shuffling configuration
        if self.shuffle_patches:
            # Store shuffle and restore indices for encoder/decoder
            self.ids_shuffle = ids_shuffle
            self.ids_restore = ids_restore
            # Still return mask in spatial order (encoder will handle shuffling)
            mask = torch.gather(mask, dim=1, index=ids_restore)
        else:
            # No shuffling - return mask in spatial order
            self.ids_shuffle = None
            self.ids_restore = None
            mask = torch.gather(mask, dim=1, index=ids_restore)

        if self.debug:
            print(f"  - Final mask shape: {mask.shape}")
            print(f"  - Masked patches: {mask.sum(dim=1).float().mean():.1f} / {L}")
            print(f"  - Mask ratio: {mask.float().mean():.3f}")
            if self.shuffle_patches:
                print(f"  - Shuffling enabled: ids_shuffle shape {self.ids_shuffle.shape}")

        # Convert to torch.bool tensor for consistency
        # This ensures ~mask works correctly throughout the codebase
        mask = mask.bool()
        return mask

    def get_evolution_info(self, epoch: Optional[int]) -> Dict[str, Any]:
        """
        Get information about the current evolution state using EXACT original formula.

        Args:
            epoch: Current training epoch

        Returns:
            Dictionary with evolution information
        """
        low = self.cluster_range[0]
        high = self.cluster_range[1]
        total_epoch = float(self.evolution_epochs)

        # Use the dedicated method to compute alpha (avoiding redundancy)
        alpha = self._compute_evolution_alpha(epoch)

        # Compute current cluster range (decreases over time) - EXACT original
        if epoch is not None:
            cluster_idx = (high-low)*(total_epoch-epoch)/total_epoch + low
            current_cluster_range = [int(cluster_idx), int(cluster_idx+2)]
        else:
            current_cluster_range = [low, high]

        return {
            'alpha': alpha,
            'spatial_weight': 1 - alpha,
            'semantic_weight': alpha,
            'current_cluster_range': current_cluster_range,
            'evolution_progress': alpha if epoch is not None else None,
        }


def create_evolved_masking_generator(
    student_model: nn.Module,
    mask_ratio: float,
    cfg: Dict[str, Any],
    grid_h: int = 14,
    grid_w: int = 14,
) -> EvolvedMaskingGenerator:
    """
    Factory function to create an evolved masking generator.

    Args:
        student_model: The student ViT model for EMA
        mask_ratio: Ratio of patches to mask
        cfg: Configuration dictionary for evolved masking
        grid_h: Height of patch grid
        grid_w: Width of patch grid

    Returns:
        Initialized EvolvedMaskingGenerator
    """
    return EvolvedMaskingGenerator(
        student_model=student_model,
        mask_ratio=mask_ratio,
        cfg=cfg,
        grid_h=grid_h,
        grid_w=grid_w
    )
