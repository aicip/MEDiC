# --------------------------------------------------------
# MEDiC: Masked Encoder Distillation Components
# Default Runtime Configuration (mmseg v0.x format)
# --------------------------------------------------------

# Logging configuration
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
    ])

# Distributed settings
dist_params = dict(backend='nccl')

# Logging level
log_level = 'INFO'

# Checkpoint loading
load_from = None
resume_from = None

# Workflow: train only or train + val
# Use [('train', 1)] for train-only, [('train', 1), ('val', 1)] for train+val
workflow = [('train', 1)]

# Enable cudnn benchmark for faster training
cudnn_benchmark = True
