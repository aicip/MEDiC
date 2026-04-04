# --------------------------------------------------------
# MEDiC: Custom hook to log iteration/epoch explicitly to WandB
# This ensures iter appears as a column in WandB tables, not just x-axis
# --------------------------------------------------------

import os
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class IterLoggerHook(Hook):
    """Hook to explicitly log current iteration to WandB as a metric.

    WandB's default step counter is internal and can't be used as a table column.
    This hook logs 'iter' and 'progress' as explicit metrics that appear in tables.

    Only logs on rank 0 to avoid duplicate logging in DDP.
    """

    def __init__(self, interval=50):
        self.interval = interval
        self._rank = int(os.environ.get('RANK', 0))

    def after_train_iter(self, runner):
        # Only log on rank 0
        if self._rank != 0:
            return

        if runner.iter % self.interval == 0:
            try:
                import wandb
                if wandb.run is not None:
                    # Get max_iters from runner config
                    max_iters = runner.max_iters if hasattr(runner, 'max_iters') else 160000
                    progress = runner.iter / max_iters * 100

                    # Log iter and progress as explicit metrics
                    # Use step=runner.iter to ensure proper ordering
                    wandb.log({
                        'train/iter': runner.iter,
                        'train/progress_pct': progress,
                        'train/max_iters': max_iters,
                    }, step=runner.iter, commit=False)  # explicit step to match WandbLoggerHook
            except Exception as e:
                # Log error on first occurrence only
                if not hasattr(self, '_logged_error'):
                    print(f"IterLoggerHook warning: {e}")
                    self._logged_error = True

    def after_val_epoch(self, runner):
        """Also log iter after validation."""
        # Only log on rank 0
        if self._rank != 0:
            return

        try:
            import wandb
            if wandb.run is not None:
                max_iters = runner.max_iters if hasattr(runner, 'max_iters') else 160000
                progress = runner.iter / max_iters * 100

                wandb.log({
                    'val/iter': runner.iter,
                    'val/progress_pct': progress,
                }, step=runner.iter, commit=False)
        except Exception:
            pass
