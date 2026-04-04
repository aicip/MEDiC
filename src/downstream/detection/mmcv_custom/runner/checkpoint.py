# --------------------------------------------------------
# MEDiC: Checkpoint utilities with GradScaler state support
# Adapted from CAE for PyTorch native AMP
# --------------------------------------------------------

import os.path as osp
import torch
from mmcv.runner import save_checkpoint as mmcv_save_checkpoint


def save_checkpoint(model,
                    filename,
                    optimizer=None,
                    scaler=None,
                    meta=None):
    """Save checkpoint with optional GradScaler state for AMP.

    Args:
        model: The model to save.
        filename: Path to save checkpoint.
        optimizer: Optional optimizer to save.
        scaler: Optional GradScaler for AMP state.
        meta: Optional metadata dict.
    """
    checkpoint = {
        'meta': meta or {},
        'state_dict': model.state_dict() if hasattr(model, 'state_dict') else model.module.state_dict(),
    }

    if optimizer is not None:
        if isinstance(optimizer, dict):
            checkpoint['optimizer'] = {k: v.state_dict() for k, v in optimizer.items()}
        else:
            checkpoint['optimizer'] = optimizer.state_dict()

    # Save GradScaler state for PyTorch native AMP
    if scaler is not None:
        checkpoint['amp_scaler'] = scaler.state_dict()

    torch.save(checkpoint, filename)


def load_checkpoint_amp(runner, checkpoint, resume_optimizer=True):
    """Load checkpoint and restore GradScaler state if available.

    Args:
        runner: The runner instance.
        checkpoint: Loaded checkpoint dict.
        resume_optimizer: Whether to restore optimizer state.

    Returns:
        The checkpoint dict.
    """
    # Restore optimizer
    if 'optimizer' in checkpoint and resume_optimizer:
        if isinstance(runner.optimizer, dict):
            for k in runner.optimizer.keys():
                runner.optimizer[k].load_state_dict(checkpoint['optimizer'][k])
        else:
            runner.optimizer.load_state_dict(checkpoint['optimizer'])

    # Restore GradScaler state for PyTorch native AMP
    if 'amp_scaler' in checkpoint and hasattr(runner, 'scaler') and runner.scaler is not None:
        runner.scaler.load_state_dict(checkpoint['amp_scaler'])
        runner.logger.info('Loaded AMP GradScaler state')

    # Legacy support: APEX amp state
    if 'amp' in checkpoint:
        runner.logger.info('Found legacy APEX amp state (not loaded, using PyTorch native AMP)')

    return checkpoint
