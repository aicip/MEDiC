"""Native PyTorch AMP optimizer hook for mmseg v0.x.

NOTE: Tested in semseg Run 6 (SLURM 5035546) — 52.36 mIoU, WORSE than FP32 52.68.
Kept for future experiments but NOT used in canonical config.

Replaces the default OptimizerHook with GradScaler-based mixed precision.
Equivalent to BEiT v2's Apex O1 DistOptimizerHook but using native PyTorch AMP.

Usage:
    In config or via --options:
        optimizer_config = dict(type='NativeAmpOptimizerHook', grad_clip=None)

    The model's train_step must be wrapped with torch.cuda.amp.autocast()
    separately (done in tools/train.py when --use-fp16 is set).
"""

import torch
from mmcv.runner import HOOKS
from mmcv.runner.hooks import OptimizerHook


@HOOKS.register_module()
class NativeAmpOptimizerHook(OptimizerHook):
    """Optimizer hook with native PyTorch AMP GradScaler.

    Handles:
    - Scaled loss backward (prevents FP16 underflow)
    - Gradient unscaling before clipping
    - Dynamic loss scale adjustment
    """

    def __init__(self, grad_clip=None, coalesce=True, bucket_size_mb=-1,
                 loss_scale='dynamic', **kwargs):
        super().__init__(grad_clip)
        if loss_scale == 'dynamic':
            self._scaler = torch.amp.GradScaler('cuda')
        else:
            self._scaler = torch.amp.GradScaler('cuda', init_scale=float(loss_scale))

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        self._scaler.scale(runner.outputs['loss']).backward()
        if self.grad_clip is not None:
            self._scaler.unscale_(runner.optimizer)
            self.clip_grads(runner.model.parameters())
        self._scaler.step(runner.optimizer)
        self._scaler.update()
