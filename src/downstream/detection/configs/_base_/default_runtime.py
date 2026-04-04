# --------------------------------------------------------
# MEDiC: Default Runtime Configuration
# Uses CAE-style custom AMP runner with PyTorch native AMP
# Includes WandB logging for experiment tracking
# --------------------------------------------------------

# Device configuration (required by mmdet)
device = 'cuda'

# Checkpoint configuration - save only current epoch to save disk space
checkpoint_config = dict(
    interval=1,
    max_keep_ckpts=1,  # Keep only current checkpoint (saves ~12 GB per job)
    save_optimizer=True,
)

# NOTE: evaluation config is defined in coco_instance.py (dataset config)
# Do NOT define it here to avoid duplicate key error in mmdet config inheritance

# Logging configuration with WandB
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='MEDiC_detection',
                entity=None,  # Set to your W&B entity
                # name will be set by train.py based on checkpoint
            ),
            interval=50,
            by_epoch=True,  # Log epoch number
            with_step=True,  # Log global step/iteration
            # Define metrics to track best values on WandB
            log_artifact=False,  # Prevent crash on finish (debug-internal.log conflict)
            define_metric_cfg=dict(
                **{f'val/bbox_mAP': 'max'},
                **{f'val/bbox_mAP_50': 'max'},
                **{f'val/bbox_mAP_75': 'max'},
                **{f'val/segm_mAP': 'max'},
                **{f'val/segm_mAP_50': 'max'},
                **{f'val/segm_mAP_75': 'max'},
            ),
        ),
    ])

# Custom hooks
# Note: NumClassCheckHook removed - it incorrectly checks backbone.num_classes instead of head.num_classes
custom_hooks = []

# Distributed settings
dist_params = dict(backend='nccl')

# Logging level
log_level = 'INFO'

# Load from and resume from paths (will be overridden by CLI args)
load_from = None
resume_from = None

# Workflow: train and validate
workflow = [('train', 1), ('val', 1)]

# CUDNN benchmark
cudnn_benchmark = True

# ========================================================================
# Mixed precision training - CAE-style with PyTorch native AMP
# ========================================================================
# Do NOT use mmdet's built-in fp16 - it causes issues with GIoU loss
fp16 = None

# AMP disabled: GradScaler was running without autocast (no-op).
# EpochBasedRunnerAmp doesn't override run_iter() to add autocast context,
# so use_fp16=True was just scaling FP32 losses without any actual FP16 compute.
# CAE uses Apex O1 which properly handles per-op casting; our PyTorch native
# AMP needs autocast wrapping in the runner to be functional.
# Disabled until proper autocast integration is implemented.
optimizer_config = dict(
    type='DistOptimizerHook',
    update_interval=1,
    grad_clip=None,
    use_fp16=False,
)

# Standard runner (AMP disabled -- see optimizer_config comment above)
runner = dict(type='EpochBasedRunner', max_epochs=12)
