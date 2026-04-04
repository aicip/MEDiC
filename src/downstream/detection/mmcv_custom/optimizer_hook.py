# --------------------------------------------------------
# MEDiC: Custom optimizer hook with PyTorch native AMP support
# Adapted from CAE's DistOptimizerHook for PyTorch native AMP
# --------------------------------------------------------

import torch
from mmcv.runner import OptimizerHook, HOOKS


@HOOKS.register_module()
class DistOptimizerHook(OptimizerHook):
    """Optimizer hook for distributed training with PyTorch native AMP.

    This hook handles:
    - Mixed precision training using torch.cuda.amp.GradScaler
    - Gradient accumulation with update_interval
    - Gradient clipping (optional)

    Args:
        update_interval (int): Gradient accumulation steps. Default: 1.
        grad_clip (dict): Config for gradient clipping. Default: None.
        coalesce (bool): Coalesce gradients for allreduce. Default: True.
        bucket_size_mb (int): Bucket size for gradient allreduce. Default: -1.
        use_fp16 (bool): Whether to use FP16 mixed precision. Default: False.
    """

    def __init__(self,
                 update_interval=1,
                 grad_clip=None,
                 coalesce=True,
                 bucket_size_mb=-1,
                 use_fp16=False):
        self.grad_clip = grad_clip
        self.coalesce = coalesce
        self.bucket_size_mb = bucket_size_mb
        self.update_interval = update_interval
        self.use_fp16 = use_fp16
        self.scaler = None

    def before_run(self, runner):
        """Initialize GradScaler and zero gradients."""
        runner.optimizer.zero_grad()

        if self.use_fp16:
            # Create GradScaler for PyTorch native AMP
            self.scaler = torch.cuda.amp.GradScaler()
            # Share scaler with runner for checkpoint saving
            runner.scaler = self.scaler
            runner.logger.info('Initialized PyTorch native AMP GradScaler')
        else:
            runner.scaler = None

    def after_train_iter(self, runner):
        """Perform backward pass with optional AMP loss scaling.

        This method:
        1. Divides loss by update_interval for gradient accumulation
        2. If use_fp16: scales loss and performs scaled backward
        3. If update_interval reached: performs optimizer step
        4. Handles gradient clipping if configured
        """
        # Divide loss by update_interval for gradient accumulation
        loss = runner.outputs['loss'] / self.update_interval

        if self.use_fp16:
            # Scale loss and backward with GradScaler
            self.scaler.scale(loss).backward()

            if self.every_n_iters(runner, self.update_interval):
                # Unscale gradients for clipping
                if self.grad_clip is not None:
                    self.scaler.unscale_(runner.optimizer)
                    self.clip_grads(runner.model.parameters())

                # Optimizer step with scaler
                self.scaler.step(runner.optimizer)
                self.scaler.update()
                runner.optimizer.zero_grad()
        else:
            # Standard backward without AMP
            loss.backward()

            if self.every_n_iters(runner, self.update_interval):
                if self.grad_clip is not None:
                    self.clip_grads(runner.model.parameters())
                runner.optimizer.step()
                runner.optimizer.zero_grad()
