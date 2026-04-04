# --------------------------------------------------------
# MEDiC: Masked Encoder Distillation Components
# Semantic Segmentation on ADE20K (mmseg v0.x format)
# Model: ViT-Base + UperNet
# Resolution: 512x512
# Iterations: 160K
# Expected result: ~52.7 mIoU
# --------------------------------------------------------

_base_ = [
    '../_base_/models/upernet_medic.py',
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

# Custom imports to override mmseg's LayerDecayOptimizerConstructor
custom_imports = dict(
    imports=['mmcv_custom', 'mmseg_custom'],
    allow_failed_imports=False)

crop_size = (512, 512)

# Model configuration
model = dict(
    backbone=dict(
        type='MEDiC',
        img_size=512,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        use_abs_pos_emb=False,
        use_rel_pos_bias=True,
        use_shared_rel_pos_bias=True,
        init_values=0.1,
        drop_path_rate=0.1,
        out_indices=[3, 5, 7, 11],
    ),
    decode_head=dict(
        in_channels=[768, 768, 768, 768],
        num_classes=150,
        channels=768,
    ),
    auxiliary_head=dict(
        in_channels=768,
        num_classes=150
    ),
    # Use sliding window for inference (higher accuracy but slower)
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341))
)

# AdamW optimizer with layer decay following BEiT2 paper specifications
# Layer decay prevents catastrophic forgetting of pretrained features
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=8e-5,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(
        num_layers=12,
        layer_decay_rate=0.75,
    ))

# Optimizer config - no gradient clipping (matches v0 training)
optimizer_config = dict(grad_clip=None)

# Learning rate schedule: polynomial decay with linear warmup (v0.x format)
lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# Runtime settings
runner = dict(type='IterBasedRunner', max_iters=160000)

# Checkpoint settings - save every 5000 iters, keep max 3
checkpoint_config = dict(by_epoch=False, interval=5000, max_keep_ckpts=3)

# Evaluation settings - validate every 5000 iterations
evaluation = dict(interval=5000, metric='mIoU', save_best='mIoU')

# Override batch size per GPU
data = dict(samples_per_gpu=2)

# IMPORTANT: Set this to your pretrained MEDiC checkpoint path
# Example: 'checkpoints/pretrain/checkpoint-epoch0290.pth'
# Set via command line: --options model.backbone.pretrained='path/to/checkpoint.pth'
