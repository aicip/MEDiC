# --------------------------------------------------------
# MEDiC: Masked Enhanced Distillation with Contrastive Learning
# Object Detection Downstream Task
# --------------------------------------------------------

# Import custom modules to register with mmdet/mmcv registries
from . import mmcv_custom
from . import mmdet_custom
from . import backbone

__all__ = ['mmcv_custom', 'mmdet_custom', 'backbone']
