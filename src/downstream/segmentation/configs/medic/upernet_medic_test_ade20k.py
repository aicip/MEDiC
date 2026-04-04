# Standalone v0.x format config for evaluation with test.py
# Does NOT inherit from base configs to avoid format conflicts

# Dataset root
data_root = '/path/to/ADEChallengeData2016'

# ImageNet normalization
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

# Model config - must match the trained model architecture
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='MEDiC',
        img_size=512,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        init_values=0.1,
        use_abs_pos_emb=False,
        use_sincos_pos_emb=False,
        use_rel_pos_bias=True,
        use_shared_rel_pos_bias=True,
        use_checkpoint=False,
        out_indices=[3, 5, 7, 11],
        out_with_norm=False,
    ),
    decode_head=dict(
        type='FixedUPerHead',
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=768,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='SafeCrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='SafeCrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
    ),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341))
)

# Test pipeline in v0.x format
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]
    )
]

# v0.x format data config that test.py expects
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    test=dict(
        type='ADE20KDataset',
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline,
        test_mode=True,
    )
)

# Distributed settings
dist_params = dict(backend='nccl')
