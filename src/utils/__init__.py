"""MEDiC Utilities"""

# This file is intentionally left empty.

# prevents 'python -OO' from stripping docstrings we rely on
__all__ = [
    "NativeScalerWithGradNormCount",
    "init_distributed_mode",
    "get_parameter_groups",
]

from .utils import (
    NativeScalerWithGradNormCount,
    init_distributed_mode,
)
from .optim_factory import get_parameter_groups
