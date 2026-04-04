# Custom mmseg modules with fixes for compatibility

# First, monkey-patch the resize function before importing anything else
def _patch_resize():
    """Patch the resize function to handle PyTorch tensor size compatibility."""
    import torch.nn.functional as F

    def fixed_resize(input,
                     size=None,
                     scale_factor=None,
                     mode='nearest',
                     align_corners=None,
                     warning=True):
        """Fixed resize function that ensures size is always a proper tuple."""
        if size is not None:
            # Handle tensor.shape[2:] returning torch.Size([512]) instead of (512, 512)
            if hasattr(size, '__len__'):
                # Convert to list to check length
                size_list = list(size)
                if len(size_list) == 1:
                    # Assume square dimensions for single value
                    size = (size_list[0], size_list[0])
                else:
                    # Ensure it's a tuple
                    size = tuple(size_list)
            else:
                # It's a scalar, make it a square tuple
                size = (size, size)

        return F.interpolate(input, size, scale_factor, mode, align_corners)

    # Patch the resize function in mmseg.ops.wrappers
    import mmseg.ops.wrappers as wrappers
    wrappers.resize = fixed_resize

# Apply the patch immediately when this module is imported
_patch_resize()

from .fixed_uper_head import FixedUPerHead
from .safe_cross_entropy_loss import SafeCrossEntropyLoss

__all__ = ['FixedUPerHead', 'SafeCrossEntropyLoss']