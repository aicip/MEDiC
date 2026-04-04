"""
Originally inspired by impl at https://github.com/zhunzhong07/Random-Erasing, Apache 2.0
Copyright Zhun Zhong & Liang Zheng

Hacked together by / Copyright 2020 Ross Wightman

Modified by Hangbo Bao, for generating the masked position for visual image transformer
"""

# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, DINO and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# Originally inspired by impl at https://github.com/zhunzhong07/Random-Erasing,
# Apache 2.0
# Copyright Zhun Zhong & Liang Zheng
#
# Hacked together by / Copyright 2020 Ross Wightman
#
# Modified by Hangbo Bao, for generating the masked position
# for visual image transformer
# --------------------------------------------------------'
import random
import math
import numpy as np
from typing import Tuple, Optional
import torch


class BlockMaskingGenerator:
    """
    Generates a block-based mask for an image grid.

    This generator creates a boolean mask where `True` indicates a masked patch,
    following the block masking strategy described in the BEiT paper.
    """
    def __init__(
        self,
        grid_h: int,
        grid_w: int,
        mask_ratio: float,
        min_num_patches: int = 16,
        min_aspect: float = 0.3,
        seed: Optional[int] = None,
        shuffle_patches: bool = False,
        per_sample_shuffling: bool = False,
    ):
        """
        Initializes the BlockMaskingGenerator.

        Args:
            grid_h (int): The height of the patch grid.
            grid_w (int): The width of the patch grid.
            mask_ratio (float): The ratio of patches to mask.
            min_num_patches (int, optional): The minimum number of patches in a
                                             single mask block. Defaults to 16.
            min_aspect (float, optional): The minimum aspect ratio for a mask block.
                                          Defaults to 0.3.
            seed (int, optional): Random seed for deterministic mask generation.
                                  If None, uses current random state. Defaults to None.
            shuffle_patches (bool, optional): Whether to shuffle patch order (for sparse mode).
                                             Defaults to False.
            per_sample_shuffling (bool, optional): Whether to use different shuffle indices
                                                  for each sample in batch (MAE-style) or
                                                  consistent shuffling (more efficient).
                                                  Defaults to False.
        """
        self.height, self.width = grid_h, grid_w
        self.num_patches = self.height * self.width
        self.num_masking_patches = int(self.num_patches * mask_ratio)

        self.min_num_patches = min_num_patches
        self.log_aspect_ratio = (math.log(min_aspect), math.log(1 / min_aspect))

        # Store seed and create a local random state if seed is provided
        self.seed = seed
        self.rng = random.Random(seed) if seed is not None else None

        # Store shuffling configuration
        self.shuffle_patches = shuffle_patches
        self.per_sample_shuffling = per_sample_shuffling  # NEW: per-sample vs consistent shuffling
        self.ids_shuffle = None  # Will store shuffle indices if shuffling enabled
        self.ids_restore = None  # Will store restore indices if shuffling enabled

    def __repr__(self) -> str:
        """Returns a string representation of the generator's configuration."""
        repr_str = (
            f"Generator(H={self.height}, W={self.width}, N={self.num_patches}, "
            f"N_mask={self.num_masking_patches}, "
            f"min_num_patches={self.min_num_patches}, "
            f"log_aspect_ratio=({self.log_aspect_ratio[0]:.3f}, "
            f"{self.log_aspect_ratio[1]:.3f}))"
        )
        return repr_str

    def get_shape(self) -> Tuple[int, int]:
        """Returns the shape of the grid (height, width)."""
        return self.height, self.width

    def _mask(self, mask: np.ndarray, max_mask_patches: int) -> int:
        """Generates a single random block mask
           and applies it to the given mask array."""
        delta = 0
        # Use local RNG if seed was provided, otherwise use global random
        rand_func = self.rng if self.rng is not None else random

        for _ in range(10):
            target_area = rand_func.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(rand_func.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = rand_func.randint(0, self.height - h)
                left = rand_func.randint(0, self.width - w)

                num_masked = mask[top : top + h, left : left + w].sum()
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1
                if delta > 0:
                    break
        return delta

    def __call__(self) -> torch.Tensor:
        """
        Generates and returns a new boolean mask as a torch tensor.

        Returns:
            torch.Tensor: A 1D boolean tensor of shape (H*W,)
                         where True indicates a masked patch, False indicates visible.
        """
        mask = np.zeros(shape=self.get_shape(), dtype=np.int32)
        mask_count = 0
        # Use numpy RandomState for deterministic post-processing
        np_rng = np.random.RandomState(self.seed) if self.seed is not None else np.random
        while mask_count < self.num_masking_patches:
            max_mask_patches = self.num_masking_patches - mask_count
            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta
        # --- PATCH: match BEiT2 reference logic for exact mask count, deterministic ---
        if mask_count > self.num_masking_patches:
            delta = mask_count - self.num_masking_patches
            mask_x, mask_y = np.nonzero(mask)
            to_unmask = np_rng.choice(mask_x.shape[0], delta, replace=False)
            mask[mask_x[to_unmask], mask_y[to_unmask]] = 0
        elif mask_count < self.num_masking_patches:
            delta = self.num_masking_patches - mask_count
            mask_x, mask_y = np.nonzero(mask == 0)
            to_mask = np_rng.choice(mask_x.shape[0], delta, replace=False)
            mask[mask_x[to_mask], mask_y[to_mask]] = 1
        # --- END PATCH ---
        assert mask.sum() == self.num_masking_patches, f"mask: {mask}, mask count {mask.sum()}"
        mask = mask.flatten()

        # Generate shuffle indices if shuffling is enabled
        if self.shuffle_patches:
            L = self.num_patches
            # Generate random noise for shuffling
            rand_func = self.rng if self.rng is not None else random
            noise = [rand_func.random() for _ in range(L)]
            noise_tensor = torch.tensor(noise)

            # Create shuffle and restore indices
            self.ids_shuffle = torch.argsort(noise_tensor)
            self.ids_restore = torch.argsort(self.ids_shuffle)
        else:
            self.ids_shuffle = None
            self.ids_restore = None

        # Convert to torch.bool tensor for consistency
        # This ensures ~mask works correctly throughout the codebase
        mask_tensor = torch.from_numpy(mask).bool()
        return mask_tensor


class RandomMaskingGenerator:
    """
    Generates random masks where each patch has equal probability of being masked.
    
    This is the simplest masking strategy - pure random selection without any
    spatial structure or semantic awareness. Useful as a baseline comparison.
    
    Args:
        grid_h: Height of the patch grid
        grid_w: Width of the patch grid
        mask_ratio: Fraction of patches to mask (0.0 to 1.0)
        seed: Optional random seed for reproducibility
    """
    
    def __init__(
        self,
        grid_h: int,
        grid_w: int,
        mask_ratio: float,
        seed: Optional[int] = None,
        shuffle_patches: bool = False,
        per_sample_shuffling: bool = False,
    ):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_patches = grid_h * grid_w
        self.mask_ratio = mask_ratio
        self.num_masking_patches = int(self.mask_ratio * self.num_patches)
        self.seed = seed

        # Create torch generator for reproducibility if seed is provided
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

        # Store shuffling configuration
        self.shuffle_patches = shuffle_patches
        self.per_sample_shuffling = per_sample_shuffling  # NEW: per-sample vs consistent shuffling
        self.ids_shuffle = None  # Will store shuffle indices if shuffling enabled
        self.ids_restore = None  # Will store restore indices if shuffling enabled
        
        # Validate parameters
        if not (0.0 <= mask_ratio <= 1.0):
            raise ValueError(f"mask_ratio must be between 0.0 and 1.0, got {mask_ratio}")
        if self.num_masking_patches > self.num_patches:
            raise ValueError(
                f"Number of masking patches ({self.num_masking_patches}) "
                f"exceeds total patches ({self.num_patches})"
            )
    
    def __repr__(self) -> str:
        """String representation of the generator."""
        return (
            f"RandomMaskingGenerator(H={self.grid_h}, W={self.grid_w}, "
            f"N={self.num_patches}, N_mask={self.num_masking_patches}, "
            f"mask_ratio={self.mask_ratio:.3f})"
        )
    
    def get_shape(self) -> Tuple[int, int]:
        """Returns the shape of the grid (height, width)."""
        return self.grid_h, self.grid_w
    
    def __call__(self) -> torch.Tensor:
        """
        Generates a random mask and returns as a torch boolean tensor.

        Returns:
            torch.Tensor: Shape (num_patches,), dtype torch.bool,
                         where True=masked, False=visible
        """
        L = self.num_patches
        # Use our mask count to match BlockMaskingGenerator behavior
        len_keep = L - self.num_masking_patches
        
        # MAE-inspired random masking logic with proper seeding
        if self.seed is not None:
            # Use the seeded generator for reproducibility
            noise = torch.rand(L, generator=self.generator)  # noise in [0, 1]
        else:
            # Use default random state
            noise = torch.rand(L)  # noise in [0, 1]
            
        ids_shuffle = torch.argsort(noise)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle)
        
        # Create mask in shuffled order: 0 is keep, 1 is remove
        mask_shuffled = torch.ones(L)
        mask_shuffled[:len_keep] = 0
        
        # Handle shuffling configuration
        if self.shuffle_patches:
            # Keep the shuffled order and store indices for encoder
            self.ids_shuffle = ids_shuffle
            self.ids_restore = ids_restore
            # Still return mask in spatial order (encoder will shuffle)
            mask = mask_shuffled[ids_restore]
        else:
            # No shuffling - return mask in spatial order
            self.ids_shuffle = None
            self.ids_restore = None
            mask = mask_shuffled[ids_restore]
        
        # Convert to torch.bool tensor for consistency
        # This ensures ~mask works correctly throughout the codebase
        mask_bool = mask.bool()

        # Validate mask count
        assert mask_bool.sum() == self.num_masking_patches, (
            f"RandomMaskingGenerator: Expected {self.num_masking_patches} masked patches, "
            f"but got {mask_bool.sum()}"
        )

        return mask_bool


def create_masking_generator(cfg, student_model=None):
    """
    Factory function to create appropriate masking generator based on configuration.
    
    Args:
        cfg: Configuration dictionary containing mask settings
        student_model: Student ViT model (unused, kept for API compatibility)

    Returns:
        Masking generator instance (BlockMaskingGenerator or RandomMaskingGenerator)

    Example config:
        mask:
          mask_type: "block"  # or "random"
          mask_ratio: 0.40
          block:              # Block masking options
            min_num_patches: 16
            min_aspect: 0.3
          random:             # Random masking options
            seed: 42          # Optional, for reproducibility
    """
    from typing import Dict, Any, Optional
    import torch.nn as nn
    
    # Get mask configuration
    mask_cfg = cfg.get('mask', {})
    mask_type = mask_cfg.get('mask_type', 'block')
    mask_ratio = mask_cfg.get('mask_ratio', 0.4)
    shuffle_patches = mask_cfg.get('shuffle_patches', False)  # Global shuffling config
    
    # Validate mask_ratio
    if not (0.0 <= mask_ratio <= 1.0):
        raise ValueError(f"mask_ratio must be between 0.0 and 1.0, got {mask_ratio}")
    
    if mask_type == 'block':
        # Create block masking generator (existing behavior)
        block_cfg = mask_cfg.get('block', {})
        
        # Extract grid dimensions (assumes 224x224 images with 16x16 patches by default)
        img_size = cfg.get('model', {}).get('student', {}).get('img_size', 224)
        patch_size = cfg.get('model', {}).get('student', {}).get('patch_size', 16)
        grid_h = grid_w = img_size // patch_size
        
        return BlockMaskingGenerator(
            grid_h=grid_h,
            grid_w=grid_w,
            mask_ratio=mask_ratio,
            min_num_patches=block_cfg.get('min_num_patches', 16),
            min_aspect=block_cfg.get('min_aspect', 0.3),
            seed=block_cfg.get('seed', None),
            shuffle_patches=shuffle_patches
        )
        
    elif mask_type == 'random':
        # Create random masking generator
        # Extract grid dimensions
        img_size = cfg.get('model', {}).get('student', {}).get('img_size', 224)
        patch_size = cfg.get('model', {}).get('student', {}).get('patch_size', 16)
        grid_h = grid_w = img_size // patch_size
        
        # Get random-specific config if available
        random_cfg = mask_cfg.get('random', {})
        
        return RandomMaskingGenerator(
            grid_h=grid_h,
            grid_w=grid_w,
            mask_ratio=mask_ratio,
            seed=random_cfg.get('seed', None),
            shuffle_patches=shuffle_patches
        )
        
    elif mask_type == 'evolved':
        # Create evolved part masking generator
        from .evolved_masking_generator import EvolvedMaskingGenerator

        img_size = cfg.get('model', {}).get('student', {}).get('img_size', 224)
        patch_size = cfg.get('model', {}).get('student', {}).get('patch_size', 16)
        grid_h = grid_w = img_size // patch_size

        return EvolvedMaskingGenerator(
            student_model=student_model,
            mask_ratio=mask_ratio,
            cfg=mask_cfg.get('evolved', {}),
            grid_h=grid_h,
            grid_w=grid_w,
            shuffle_patches=shuffle_patches,
        )

    else:
        available_types = ['block', 'random', 'evolved']
        raise ValueError(
            f"Unknown mask_type: '{mask_type}'. Available types: {available_types}"
        )


def get_masking_generator_info(cfg):
    """
    Get information about the masking configuration without creating the generator.
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        Dictionary with masking configuration info
    """
    mask_cfg = cfg.get('mask', {})
    mask_type = mask_cfg.get('mask_type', 'block')
    mask_ratio = mask_cfg.get('mask_ratio', 0.4)
    
    # Extract grid dimensions
    img_size = cfg.get('model', {}).get('student', {}).get('img_size', 224)
    patch_size = cfg.get('model', {}).get('student', {}).get('patch_size', 16)
    grid_h = grid_w = img_size // patch_size
    num_patches = grid_h * grid_w
    num_masked = int(num_patches * mask_ratio)
    
    info = {
        'mask_type': mask_type,
        'mask_ratio': mask_ratio,
        'img_size': img_size,
        'patch_size': patch_size,
        'grid_size': (grid_h, grid_w),
        'num_patches': num_patches,
        'num_masked': num_masked,
        'num_keep': num_patches - num_masked,
    }
    
    if mask_type == 'block':
        block_cfg = mask_cfg.get('block', {})
        info['block_config'] = {
            'min_num_patches': block_cfg.get('min_num_patches', 16),
            'min_aspect': block_cfg.get('min_aspect', 0.3),
            'seed': block_cfg.get('seed', None),
        }
    elif mask_type == 'random':
        random_cfg = mask_cfg.get('random', {})
        info['random_config'] = {
            'seed': random_cfg.get('seed', None),
        }
    elif mask_type == 'evolved':
        evolved_cfg = mask_cfg.get('evolved', {})
        info['evolved_config'] = {
            'attention_source': evolved_cfg.get('attention_source', 'clip_teacher'),
            'gamma': evolved_cfg.get('gamma', 0.5),
            'cluster_range': evolved_cfg.get('cluster_range', [10, 40]),
            'evolution_epochs': evolved_cfg.get('evolution_epochs', 300),
        }

    return info


if __name__ == "__main__":
    generator = BlockMaskingGenerator(
        grid_h=14, grid_w=14, mask_ratio=0.5, min_num_patches=16
    )
    for i in range(100):
        mask = generator()
        assert mask.sum() > 0, f"Empty mask at iteration {i}"
    print(f"Generated 100 masks successfully, last mask sum: {mask.sum()}")
