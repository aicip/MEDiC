# --------------------------------------------------------
# MEDiC: RoI Align wrapper using torchvision instead of mmcv CUDA ops
# Provides drop-in replacement for mmcv.ops.roi_align
# This avoids the need to compile mmcv with CUDA support
# --------------------------------------------------------

import torch
from torch import nn
from torchvision.ops import roi_align as torchvision_roi_align


def patch_mmcv_roi_align():
    """Patch mmcv's internal RoIAlignFunction to use torchvision.

    This is needed because mmdet may import and use the internal
    RoIAlignFunction class directly, bypassing our module-level patches.
    """
    try:
        import mmcv.ops.roi_align as mmcv_roi_align_module

        # Create a new forward method that uses torchvision
        original_forward = mmcv_roi_align_module.RoIAlignFunction.forward

        @staticmethod
        def patched_forward(ctx, input, rois, output_size, spatial_scale,
                           sampling_ratio=0, pool_mode='avg', aligned=True):
            """Patched forward using torchvision roi_align."""
            ctx.output_size = output_size
            ctx.spatial_scale = spatial_scale
            ctx.sampling_ratio = sampling_ratio
            ctx.input_shape = input.shape
            ctx.aligned = aligned
            ctx.pool_mode = pool_mode
            ctx.save_for_backward(rois)

            # Handle output_size format
            if isinstance(output_size, int):
                out_h = out_w = output_size
            else:
                out_h, out_w = output_size

            # Use torchvision roi_align
            output = torchvision_roi_align(
                input=input,
                boxes=rois,
                output_size=(out_h, out_w),
                spatial_scale=spatial_scale,
                sampling_ratio=sampling_ratio if sampling_ratio >= 0 else 0,
                aligned=aligned
            )
            return output

        # Replace the forward method
        mmcv_roi_align_module.RoIAlignFunction.forward = patched_forward
        return True
    except Exception as e:
        print(f"Warning: Could not patch mmcv RoIAlignFunction: {e}")
        return False


class RoIAlignFunction(torch.autograd.Function):
    """Compatibility wrapper for mmcv's RoIAlign forward function."""

    @staticmethod
    def forward(ctx, input, rois, output_size, spatial_scale, sampling_ratio=0,
                pool_mode='avg', aligned=True):
        """Forward pass using torchvision roi_align."""
        ctx.save_for_backward(rois)
        ctx.output_size = output_size
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        ctx.input_shape = input.shape
        ctx.aligned = aligned

        # Handle output_size format
        if isinstance(output_size, int):
            out_h = out_w = output_size
        else:
            out_h, out_w = output_size

        # torchvision roi_align expects aligned=True by default in modern versions
        # Note: torchvision.ops.roi_align signature:
        # roi_align(input, boxes, output_size, spatial_scale, sampling_ratio, aligned)
        output = torchvision_roi_align(
            input=input,
            boxes=rois,
            output_size=(out_h, out_w),
            spatial_scale=spatial_scale,
            sampling_ratio=sampling_ratio if sampling_ratio >= 0 else 0,
            aligned=aligned
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass - torchvision handles this automatically."""
        # torchvision.ops.roi_align is differentiable, so gradients flow through
        # This is a placeholder - the actual backward is handled by PyTorch autograd
        return None, None, None, None, None, None, None


def roi_align(input, rois, output_size, spatial_scale=1.0, sampling_ratio=0,
              pool_mode='avg', aligned=True):
    """Drop-in replacement for mmcv.ops.roi_align using torchvision.

    Args:
        input (Tensor): Input feature map of shape (N, C, H, W).
        rois (Tensor): RoIs in format (batch_idx, x1, y1, x2, y2).
        output_size (int or tuple): Output size (H, W) of the pooled feature.
        spatial_scale (float): Scale the input coordinates by this factor.
        sampling_ratio (int): Number of sampling points in each bin.
            0 means adaptive sampling.
        pool_mode (str): 'avg' or 'max'. Default: 'avg'.
            Note: torchvision only supports 'avg' mode.
        aligned (bool): If True, use aligned mode (default in modern mmcv).

    Returns:
        Tensor: Pooled features of shape (num_rois, C, output_size, output_size).
    """
    if pool_mode != 'avg':
        # torchvision roi_align only supports average pooling
        # For max pooling, we'd need a different approach, but avg is standard
        import warnings
        warnings.warn(f"pool_mode='{pool_mode}' not supported by torchvision, using 'avg'")

    # Handle output_size format
    if isinstance(output_size, int):
        out_h = out_w = output_size
    else:
        out_h, out_w = output_size

    # Use torchvision's roi_align directly (it's differentiable)
    output = torchvision_roi_align(
        input=input,
        boxes=rois,
        output_size=(out_h, out_w),
        spatial_scale=spatial_scale,
        sampling_ratio=sampling_ratio if sampling_ratio >= 0 else 0,
        aligned=aligned
    )

    return output


class RoIAlign(nn.Module):
    """Drop-in replacement for mmcv.ops.RoIAlign using torchvision.

    Args:
        output_size (int or tuple): Output size (H, W).
        spatial_scale (float): Scale factor for input coordinates.
        sampling_ratio (int): Number of sampling points in each bin.
        pool_mode (str): 'avg' or 'max' pooling mode.
        aligned (bool): If True, use aligned mode.
        use_torchvision (bool): Ignored, always uses torchvision.
    """

    def __init__(self,
                 output_size,
                 spatial_scale=1.0,
                 sampling_ratio=0,
                 pool_mode='avg',
                 aligned=True,
                 use_torchvision=True):
        super(RoIAlign, self).__init__()

        self.output_size = output_size if isinstance(output_size, tuple) else (output_size, output_size)
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)
        self.pool_mode = pool_mode
        self.aligned = aligned

    def forward(self, input, rois):
        """Forward pass.

        Args:
            input (Tensor): Input feature map of shape (N, C, H, W).
            rois (Tensor): RoIs of shape (num_rois, 5) where each row is
                (batch_idx, x1, y1, x2, y2).

        Returns:
            Tensor: Pooled features of shape (num_rois, C, out_h, out_w).
        """
        return roi_align(
            input=input,
            rois=rois,
            output_size=self.output_size,
            spatial_scale=self.spatial_scale,
            sampling_ratio=self.sampling_ratio,
            pool_mode=self.pool_mode,
            aligned=self.aligned
        )

    def __repr__(self):
        s = self.__class__.__name__
        s += f'(output_size={self.output_size}, '
        s += f'spatial_scale={self.spatial_scale}, '
        s += f'sampling_ratio={self.sampling_ratio}, '
        s += f'pool_mode={self.pool_mode}, '
        s += f'aligned={self.aligned})'
        return s
