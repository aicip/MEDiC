# --------------------------------------------------------
# MEDiC: Masked Enhanced Distillation with Contrastive Learning
# Custom mmdet modules with PyTorch 2.x compatibility fixes
# Provides torchvision wrappers to avoid mmcv CUDA compilation
# --------------------------------------------------------

# NMS wrapper using torchvision (avoids mmcv CUDA compilation requirement)
from .nms_wrapper import nms, batched_nms, NMSop

# RoI Align wrapper using torchvision
from .roi_align_wrapper import roi_align, RoIAlign, patch_mmcv_roi_align

__all__ = ['nms', 'batched_nms', 'NMSop', 'roi_align', 'RoIAlign', 'patch_mmcv_roi_align']
