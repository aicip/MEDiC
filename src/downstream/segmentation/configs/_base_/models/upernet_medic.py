# --------------------------------------------------------
# MEDiC: Masked Encoder Distillation Components
# Semantic Segmentation Configuration
# --------------------------------------------------------
norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    pretrained=None,  # Will be loaded via backbone.init_weights()
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
        # Output parameters
        out_indices=[3, 5, 7, 11],
        out_with_norm=False
    ),
    decode_head=dict(
        type='FixedUPerHead',
        in_channels=[768, 768, 768, 768],  # Same embed_dim for all levels
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),  # Pyramid pooling scales
        channels=768,  # Match embed_dim
        dropout_ratio=0.1,
        num_classes=150,  # ADE20K has 150 classes
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='SafeCrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,
        in_index=2,  # Layer 7 features
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='SafeCrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # Model training and testing settings (v0.x format)
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)))
