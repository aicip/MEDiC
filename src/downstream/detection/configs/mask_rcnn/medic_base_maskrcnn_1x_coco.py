# --------------------------------------------------------
# MEDiC: MaskRCNN with ViT-Base Backbone, 1x Schedule, COCO
# --------------------------------------------------------

_base_ = [
    '../_base_/models/mask_rcnn_medic_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# Override backbone settings for ViT-Base
model = dict(
    backbone=dict(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        drop_path_rate=0.2,  # CAE paper: 0.2 for ViT-B
        init_values=0.1,
        use_sincos_pos_emb=True,
        use_checkpoint=True,
        # Pretrained path (override with --cfg-options model.backbone.pretrained=...)
        pretrained=None,
    ),
    neck=dict(in_channels=[768, 768, 768, 768]),
)

# Override optimizer for ViT-Base
optimizer = dict(
    lr=0.0003,
    paramwise_cfg=dict(
        num_layers=12,
        layer_decay_rate=0.75,
    ))

# Custom imports to register modules
custom_imports = dict(
    imports=['mmcv_custom', 'mmdet_custom', 'backbone'],
    allow_failed_imports=False)
