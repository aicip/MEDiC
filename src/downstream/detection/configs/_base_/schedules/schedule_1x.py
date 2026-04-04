# --------------------------------------------------------
# MEDiC: 1x Training Schedule (12 epochs)
# Based on CAE detection schedule
# Note: optimizer_config and runner are defined in default_runtime.py
# --------------------------------------------------------

# Optimizer with layer-wise learning rate decay
optimizer = dict(
    type='AdamW',
    lr=0.0003,  # Base learning rate
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(
        num_layers=12,  # ViT-Base depth
        layer_decay_rate=0.75,  # Decay rate per layer
    ))

# Learning rate schedule (step decay at epochs 9 and 11)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[9, 11])
