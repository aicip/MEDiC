# --------------------------------------------------------
# MEDiC: Custom MMCV utilities
# --------------------------------------------------------

import torch

# Fix PyTorch 2.6+ weights_only default change
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

from .checkpoint import load_checkpoint
from .layer_decay_optimizer_constructor import LayerDecayOptimizerConstructor

__all__ = ['load_checkpoint', 'LayerDecayOptimizerConstructor']
