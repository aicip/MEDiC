# --------------------------------------------------------
# MEDiC: Masked Enhanced Distillation with Contrastive Learning
# Custom MMCV utilities for detection
# --------------------------------------------------------

import torch

# --------------------------------------------------------
# Fix PyTorch 2.6 weights_only default change
# PyTorch 2.6 changed torch.load default from weights_only=False to True
# This breaks mmcv's checkpoint loading which doesn't pass the argument
# --------------------------------------------------------
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

# --------------------------------------------------------
# Fix PyTorch 2.7 compatibility with MMDistributedDataParallel
# PyTorch 2.7 removed the `_use_replicated_tensor_module` attribute
# from DistributedDataParallel, but mmcv 1.7.2 still references it
# in the _run_ddp_forward method. This patch fixes the AttributeError.
# --------------------------------------------------------
try:
    from mmcv.parallel import MMDistributedDataParallel

    _original_run_ddp_forward = MMDistributedDataParallel._run_ddp_forward

    def _patched_run_ddp_forward(self, *inputs, **kwargs):
        """Patched _run_ddp_forward that handles missing _use_replicated_tensor_module."""
        # Use getattr with default False for PyTorch 2.7+ compatibility
        use_replicated = getattr(self, '_use_replicated_tensor_module', False)
        if use_replicated and hasattr(self, '_replicated_tensor_module'):
            module_to_run = self._replicated_tensor_module
        else:
            module_to_run = self.module

        if self.device_ids:
            inputs, kwargs = self.to_kwargs(inputs, kwargs, self.device_ids[0])
            return module_to_run(*inputs[0], **kwargs[0])
        else:
            return module_to_run(*inputs, **kwargs)

    MMDistributedDataParallel._run_ddp_forward = _patched_run_ddp_forward
    print("Applied PyTorch 2.7 compatibility patch for MMDistributedDataParallel._run_ddp_forward()")
except ImportError:
    pass  # mmcv not installed or MMDistributedDataParallel not available

from .checkpoint import load_checkpoint
from .layer_decay_optimizer_constructor import LayerDecayOptimizerConstructor

# Custom AMP runner and optimizer hook (CAE-style with PyTorch native AMP)
from .runner import EpochBasedRunnerAmp
from .optimizer_hook import DistOptimizerHook

# Custom hook to explicitly log epoch/iter as WandB table columns
from .epoch_logger_hook import EpochIterLoggerHook

__all__ = [
    'load_checkpoint',
    'LayerDecayOptimizerConstructor',
    'EpochBasedRunnerAmp',
    'DistOptimizerHook',
    'EpochIterLoggerHook',
]
