"""Fixed resize function that handles PyTorch version incompatibility."""

import torch.nn.functional as F


def fixed_resize(input,
                 size=None,
                 scale_factor=None,
                 mode='nearest',
                 align_corners=None,
                 warning=True):
    """Fixed resize function that ensures size is always a tuple.

    In newer PyTorch versions, tensor.size()[2:] can return a single value
    instead of a tuple when dimensions are equal, causing interpolation to fail.
    This wrapper ensures the size parameter is always a proper tuple.
    """
    if size is not None:
        # Handle torch.Size objects that might be scalar
        if hasattr(size, '__len__'):
            # It's already iterable, but might be torch.Size([512])
            if len(size) == 1:
                # Assume square dimensions for single value
                size = (size[0], size[0])
            else:
                # Ensure it's a tuple
                size = tuple(size)
        else:
            # It's a scalar, make it a square tuple
            size = (size, size)

    return F.interpolate(input, size, scale_factor, mode, align_corners)