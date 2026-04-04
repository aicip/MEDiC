__all__ = ["build_teacher", "build_student", "build_medic_model"]

from typing import Dict, Any

from .vision_transformer import VisionTransformerMIM
from .medic_model import build_student, build_medic_model


def build_teacher(cfg: Dict[str, Any]):
    """Builds the frozen CLIP teacher model. Import is lazy to avoid slow CLIP loading at import time."""
    from .clip_teacher import ClipTeacher
    teacher_cfg = cfg["model"]["teacher"]
    return ClipTeacher(
        model_name=teacher_cfg["name"],
        layer_extraction=teacher_cfg.get("layer_extraction", "last"),
        num_layers_to_extract=teacher_cfg.get("num_layers_to_extract", 1),
    )
