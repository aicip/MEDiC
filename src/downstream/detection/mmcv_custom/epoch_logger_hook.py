# --------------------------------------------------------
# MEDiC: Custom hook to log epoch/iteration explicitly to WandB
# This ensures epoch/iter appear as columns in WandB tables, not just x-axis
# --------------------------------------------------------

import os
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class EpochIterLoggerHook(Hook):
    """Hook to explicitly log current epoch and iteration to WandB as metrics.

    WandB's default step counter is internal and can't be used as a table column.
    This hook logs 'epoch', 'iter', and 'progress' as explicit metrics that appear in tables.

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
                    # Get max values from runner
                    max_epochs = runner.max_epochs if hasattr(runner, 'max_epochs') else 12
                    max_iters = runner.max_iters if hasattr(runner, 'max_iters') else None

                    # Calculate progress
                    if max_iters:
                        progress = runner.iter / max_iters * 100
                    else:
                        progress = runner.epoch / max_epochs * 100

                    # Log epoch and iter as explicit metrics
                    log_dict = {
                        'train/epoch': runner.epoch,
                        'train/iter': runner.iter,
                        'train/progress_pct': progress,
                        'train/max_epochs': max_epochs,
                    }

                    # Also log inner_iter if available (iteration within epoch)
                    if hasattr(runner, 'inner_iter'):
                        log_dict['train/inner_iter'] = runner.inner_iter

                    wandb.log(log_dict, commit=False)  # commit=False to batch with other metrics
            except Exception as e:
                # Log error on first occurrence only
                if not hasattr(self, '_logged_error'):
                    print(f"EpochIterLoggerHook warning: {e}")
                    self._logged_error = True

    def after_val_epoch(self, runner):
        """Also log epoch/iter after validation."""
        # Only log on rank 0
        if self._rank != 0:
            return

        try:
            import wandb
            if wandb.run is not None:
                max_epochs = runner.max_epochs if hasattr(runner, 'max_epochs') else 12
                progress = runner.epoch / max_epochs * 100

                wandb.log({
                    'val/epoch': runner.epoch,
                    'val/iter': runner.iter,
                    'val/progress_pct': progress,
                }, commit=False)
        except Exception:
            pass
