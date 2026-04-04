# --------------------------------------------------------
# MEDiC: NMS wrapper using torchvision instead of mmcv CUDA ops
# Provides drop-in replacement for mmcv.ops.nms and batched_nms
# This avoids the need to compile mmcv with CUDA support
# --------------------------------------------------------

import torch
from torchvision.ops import nms as torch_nms
from torchvision.ops import batched_nms as torch_batched_nms


class NMSop(torch.autograd.Function):
    """Compatibility wrapper for mmcv's NMSop interface."""

    @staticmethod
    def forward(ctx, bboxes, scores, iou_threshold, offset, score_threshold, max_num):
        """Forward pass using torchvision NMS."""
        # Filter by score threshold
        if score_threshold > 0:
            valid_mask = scores > score_threshold
            valid_inds = torch.nonzero(valid_mask).squeeze(1)
            bboxes = bboxes[valid_mask]
            scores = scores[valid_mask]
        else:
            valid_inds = torch.arange(scores.size(0), device=scores.device)

        if bboxes.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=bboxes.device)

        # Apply torchvision NMS
        keep = torch_nms(bboxes, scores, iou_threshold)

        # Limit to max_num
        if max_num > 0 and keep.numel() > max_num:
            keep = keep[:max_num]

        # Map back to original indices
        return valid_inds[keep]


def nms(boxes, scores, iou_threshold, offset=0, score_threshold=0, max_num=-1):
    """Drop-in replacement for mmcv.ops.nms using torchvision.

    Args:
        boxes (Tensor): boxes in shape (N, 4).
        scores (Tensor): scores in shape (N, ).
        iou_threshold (float): IoU threshold for NMS.
        offset (int): offset (0 or 1). Ignored in torchvision implementation.
        score_threshold (float): score threshold for filtering boxes.
        max_num (int): maximum number of boxes after NMS. -1 means no limit.

    Returns:
        tuple: (dets, inds) where dets is (K, 5) [x1, y1, x2, y2, score]
               and inds is (K,) indices into original boxes.
    """
    if boxes.numel() == 0:
        dets = torch.cat([boxes, scores[:, None]], dim=-1)
        return dets, torch.empty(0, dtype=torch.long, device=boxes.device)

    # Filter by score threshold
    if score_threshold > 0:
        valid_mask = scores > score_threshold
        valid_inds = torch.nonzero(valid_mask).squeeze(1)
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
    else:
        valid_inds = torch.arange(scores.size(0), device=scores.device)

    if boxes.numel() == 0:
        dets = torch.cat([boxes, scores[:, None]], dim=-1)
        return dets, torch.empty(0, dtype=torch.long, device=boxes.device)

    # Apply torchvision NMS
    keep = torch_nms(boxes, scores, iou_threshold)

    # Limit to max_num
    if max_num > 0 and keep.numel() > max_num:
        keep = keep[:max_num]

    # Map back to original indices
    inds = valid_inds[keep]

    # Return (dets, inds) where dets includes scores — matches mmcv interface
    dets = torch.cat([boxes[keep], scores[keep, None]], dim=-1)
    return dets, inds


def batched_nms(boxes, scores, idxs, nms_cfg, class_agnostic=False):
    """Drop-in replacement for mmcv.ops.batched_nms using torchvision.

    Args:
        boxes (Tensor): boxes in shape (N, 4).
        scores (Tensor): scores in shape (N, ).
        idxs (Tensor): class indices in shape (N, ).
        nms_cfg (dict): NMS configuration dict with 'iou_threshold' key.
        class_agnostic (bool): if True, do class-agnostic NMS.

    Returns:
        tuple: (dets, keep) where dets is (K, 5) with [x1, y1, x2, y2, score]
               and keep is the indices of kept boxes.
    """
    if boxes.numel() == 0:
        dets = torch.cat([boxes, scores[:, None]], dim=-1)
        return dets, torch.empty(0, dtype=torch.long, device=boxes.device)

    # Extract NMS parameters
    iou_threshold = nms_cfg.get('iou_threshold', 0.5)
    score_threshold = nms_cfg.get('score_threshold', 0)
    max_num = nms_cfg.get('max_num', -1)

    # Filter by score threshold
    if score_threshold > 0:
        valid_mask = scores > score_threshold
        valid_inds = torch.nonzero(valid_mask).squeeze(1)
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        idxs = idxs[valid_mask]
    else:
        valid_inds = torch.arange(scores.size(0), device=scores.device)

    if boxes.numel() == 0:
        dets = torch.cat([boxes, scores[:, None]], dim=-1)
        return dets, torch.empty(0, dtype=torch.long, device=boxes.device)

    # Apply torchvision batched_nms
    if class_agnostic:
        # For class-agnostic, treat all boxes as same class
        keep = torch_nms(boxes, scores, iou_threshold)
    else:
        keep = torch_batched_nms(boxes, scores, idxs, iou_threshold)

    # Limit to max_num
    if max_num > 0 and keep.numel() > max_num:
        keep = keep[:max_num]

    # Map back to original indices
    kept_inds = valid_inds[keep]

    # Build output: [x1, y1, x2, y2, score]
    dets = torch.cat([boxes[keep], scores[keep, None]], dim=-1)

    return dets, kept_inds
