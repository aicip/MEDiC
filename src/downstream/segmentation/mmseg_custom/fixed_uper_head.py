"""Fixed UPerHead that handles tensor division correctly for newer PyTorch versions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.decode_heads import UPerHead
from mmseg.models.builder import HEADS
from mmseg.ops import resize


@HEADS.register_module()
class FixedUPerHead(UPerHead):
    """UPerHead with fixed pooling size calculation for newer PyTorch versions.

    The issue: In newer PyTorch, x.size()[2:] // pool_scale can return a scalar
    instead of a tuple when dimensions are equal, causing resize operations to fail.
    This happens in the PPM module's forward method.
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        """Initialize FixedUPerHead with proper parent initialization."""
        # Store pool_scales before calling parent init
        self.pool_scales = pool_scales
        super(FixedUPerHead, self).__init__(pool_scales=pool_scales, **kwargs)

    def psp_forward(self, inputs):
        """Forward function of PSP module with fixed pooling size calculation.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: PSP module output.
        """
        # Handle list input - take the last level features like the parent does
        x = inputs[-1]
        psp_outs = [x]

        # Process through PPM modules with fixed resize
        for ppm in self.psp_modules:
            ppm_out = ppm(x)
            # Fixed: Ensure size is always a tuple
            h, w = x.size()[2:]
            resize_shape = (h, w)  # Explicitly create tuple
            upsampled_ppm_out = resize(
                ppm_out,
                size=resize_shape,
                mode='bilinear',
                align_corners=self.align_corners)
            psp_outs.append(upsampled_ppm_out)

        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output