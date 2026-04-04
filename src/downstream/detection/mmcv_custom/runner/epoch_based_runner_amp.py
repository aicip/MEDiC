# --------------------------------------------------------
# MEDiC: Epoch-based Runner with PyTorch native AMP support
# Adapted from CAE's EpochBasedRunnerAmp for PyTorch native AMP
# --------------------------------------------------------

import os.path as osp
import platform
import shutil
import torch

import mmcv
from torch.optim import Optimizer
from mmcv.runner import RUNNERS, EpochBasedRunner
from .checkpoint import save_checkpoint, load_checkpoint_amp


@RUNNERS.register_module()
class EpochBasedRunnerAmp(EpochBasedRunner):
    """Epoch-based Runner with PyTorch native AMP support.

    This runner trains models epoch by epoch with automatic mixed precision
    using PyTorch's native torch.cuda.amp.GradScaler.

    The GradScaler state is saved in checkpoints and restored on resume,
    ensuring consistent training across interruptions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # GradScaler will be set by DistOptimizerHook
        self.scaler = None

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        """Save checkpoint with AMP GradScaler state.

        Args:
            out_dir (str): Directory to save checkpoints.
            filename_tmpl (str): Checkpoint filename template.
            save_optimizer (bool): Whether to save optimizer state.
            meta (dict): Metadata to save.
            create_symlink (bool): Whether to create 'latest.pth' symlink.
        """
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        elif isinstance(meta, dict):
            meta.update(epoch=self.epoch + 1, iter=self.iter)
        else:
            raise TypeError(f'meta should be a dict or None, but got {type(meta)}')

        if self.meta is not None:
            meta.update(self.meta)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None

        # Save with GradScaler state
        save_checkpoint(
            self.model,
            filepath,
            optimizer=optimizer,
            scaler=self.scaler,
            meta=meta
        )

        # Create symlink to latest checkpoint
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)

    def resume(self,
               checkpoint,
               resume_optimizer=True,
               map_location='default'):
        """Resume training from checkpoint with AMP state restoration.

        Args:
            checkpoint: Path to checkpoint or loaded checkpoint dict.
            resume_optimizer: Whether to restore optimizer state.
            map_location: Device mapping for checkpoint loading.
        """
        if map_location == 'default':
            if torch.cuda.is_available():
                device_id = torch.cuda.current_device()
                checkpoint = self.load_checkpoint(
                    checkpoint,
                    map_location=lambda storage, loc: storage.cuda(device_id))
            else:
                checkpoint = self.load_checkpoint(checkpoint)
        else:
            checkpoint = self.load_checkpoint(checkpoint, map_location=map_location)

        self._epoch = checkpoint['meta']['epoch']
        self._iter = checkpoint['meta']['iter']

        # Restore optimizer and GradScaler state
        if resume_optimizer:
            load_checkpoint_amp(self, checkpoint, resume_optimizer=True)

        self.logger.info(f'Resumed from epoch {self.epoch}, iter {self.iter}')
