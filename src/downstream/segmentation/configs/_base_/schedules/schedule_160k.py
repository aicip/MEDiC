# --------------------------------------------------------
# MEDiC: Masked Encoder Distillation Components
# Training Schedule Configuration (mmseg v0.x format)
# 160K iterations with polynomial LR decay
# Based on BEiT2 semantic segmentation config
# --------------------------------------------------------

# Optimizer - will be overridden by command line args
# Default uses AdamW with layer decay (set via paramwise_cfg in main config)
optimizer = dict(
    type='AdamW',
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.05)

# Optimizer config - gradient clipping
optimizer_config = dict(grad_clip=dict(max_norm=1.0, norm_type=2))

# Learning rate scheduler - polynomial decay
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# Runtime settings
runner = dict(type='IterBasedRunner', max_iters=160000)

# Checkpoint settings
checkpoint_config = dict(by_epoch=False, interval=5000, max_keep_ckpts=3)

# Evaluation settings - validate every 5000 iterations
evaluation = dict(interval=5000, metric='mIoU', save_best='mIoU')
