"""
Fixed WandbLoggerHook that resolves step collision in EpochBasedRunner + EvalHook.

Problem:
    mmdet's EvalHook.after_train_epoch() calls all LoggerHooks to flush training
    metrics before running evaluation. This causes WandbLoggerHook to commit
    training metrics at step=N. When eval results are then logged at the same
    step=N, WandB silently drops them (steps must be monotonically increasing):

        WARNING Tried to log to step N that is less than the current step N+1.
        Steps must be monotonically increasing, so this data will be ignored.

    Result: bbox_mAP, segm_mAP, and all eval metrics are silently lost in WandB
    while still appearing in SLURM logs. This affected ALL detection runs.

Fix:
    Track the last logged step. If a new log call would use a step that was
    already committed, auto-increment by 1. The offset is negligible (1 step
    out of ~175K total iterations) and ensures both training and eval metrics
    are captured.
"""

from mmcv.runner import HOOKS
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks.logger.wandb import WandbLoggerHook as _WandbLoggerHook


@HOOKS.register_module(force=True)
class WandbLoggerHook(_WandbLoggerHook):
    """WandbLoggerHook with step collision fix for epoch-based eval hooks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_logged_step = -1

    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner)
        if tags:
            step = self.get_iter(runner)

            # Fix: if this step was already committed (training metrics),
            # increment to avoid collision (eval metrics at same iter)
            if step <= self._last_logged_step:
                step = self._last_logged_step + 1

            self._last_logged_step = step

            if self.with_step:
                self.wandb.log(tags, step=step, commit=self.commit)
            else:
                tags['global_step'] = step
                self.wandb.log(tags, commit=self.commit)
