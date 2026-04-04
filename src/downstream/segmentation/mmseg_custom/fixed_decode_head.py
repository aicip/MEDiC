"""Fixed DecodeHead base class that handles PyTorch tensor shape compatibility."""

import torch
from mmseg.models.decode_heads import BaseDecodeHead
from mmseg.models.builder import HEADS
from mmcv.runner import force_fp32
from mmseg.ops import resize


class FixedBaseDecodeHead(BaseDecodeHead):
    """Base class with fixed losses method for PyTorch compatibility."""

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss with fixed size handling."""
        loss = dict()

        # Fixed: Ensure size is always a tuple
        h, w = seg_label.shape[2], seg_label.shape[3]
        target_size = (h, w)  # Explicitly create tuple

        seg_logit = resize(
            input=seg_logit,
            size=target_size,
            mode='bilinear',
            align_corners=self.align_corners)

        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)

        loss['acc_seg'] = accuracy(
            seg_logit, seg_label, ignore_index=self.ignore_index)
        return loss