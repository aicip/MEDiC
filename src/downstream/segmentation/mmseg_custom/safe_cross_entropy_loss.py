"""
Custom CrossEntropyLoss that handles PyTorch 2.x CUDA assertion issues with ignore_index.

PyTorch 2.x CUDA kernel performs bounds check (cur_target >= 0 && cur_target < n_classes)
BEFORE applying ignore_index filtering. This causes assertion failures when labels contain
values like 255 (ADE20K ignore label) but num_classes=150.

This wrapper pre-filters labels by clamping them to valid range before F.cross_entropy sees them,
while properly handling ignore_index semantics.
"""

import torch
import torch.nn as nn
from mmseg.models.losses import CrossEntropyLoss
from mmseg.models.builder import LOSSES


@LOSSES.register_module()
class SafeCrossEntropyLoss(nn.Module):
    """
    Safe wrapper around CrossEntropyLoss that prevents CUDA assertion failures.

    Args:
        Same as CrossEntropyLoss from mmseg
    """

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 loss_name='loss_ce',
                 avg_non_ignore=False):
        super(SafeCrossEntropyLoss, self).__init__()

        # Create the underlying CrossEntropyLoss
        self.ce_loss = CrossEntropyLoss(
            use_sigmoid=use_sigmoid,
            use_mask=use_mask,
            reduction=reduction,
            class_weight=class_weight,
            loss_weight=loss_weight,
            loss_name=loss_name,
            avg_non_ignore=avg_non_ignore
        )

        self._loss_name = loss_name

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=-100,
                **kwargs):
        """
        Forward pass with label pre-filtering to prevent CUDA assertion.

        Args:
            cls_score (torch.Tensor): Predictions with shape (N, C, H, W)
            label (torch.Tensor): Ground truth labels with shape (N, H, W)
            weight (torch.Tensor, optional): Sample-wise loss weight
            avg_factor (int, optional): Average factor
            reduction_override (str, optional): Reduction method override
            ignore_index (int): Label value to ignore (default: -100, but mmseg uses 255 for ADE20K)
            **kwargs: Additional arguments

        Returns:
            torch.Tensor: Computed loss
        """

        # Get number of classes from prediction shape
        num_classes = cls_score.shape[1]

        # Create a mask for ignore_index values
        ignore_mask = (label == ignore_index)

        # Clamp labels to valid range [0, num_classes-1]
        # This prevents CUDA assertion by ensuring all labels are in bounds
        # Ignore_index values will be clamped to 0, but we'll restore them after
        safe_label = torch.clamp(label, 0, num_classes - 1)

        # Restore ignore_index values
        # This ensures ignore_index semantics are preserved for F.cross_entropy
        safe_label = torch.where(ignore_mask, torch.full_like(safe_label, ignore_index), safe_label)

        # Call the underlying CrossEntropyLoss with safe labels
        return self.ce_loss(
            cls_score,
            safe_label,
            weight=weight,
            avg_factor=avg_factor,
            reduction_override=reduction_override,
            ignore_index=ignore_index,
            **kwargs
        )

    @property
    def loss_name(self):
        """Loss name for identification."""
        return self._loss_name
