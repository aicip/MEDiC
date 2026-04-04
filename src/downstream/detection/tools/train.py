# --------------------------------------------------------
# MEDiC: Masked Enhanced Distillation with Contrastive Learning
# Object Detection Training Script
# Adapted from CAE detection with PyTorch 2.x compatibility, WandB logging,
# and checkpoint config auto-detection
# --------------------------------------------------------

import argparse
import copy
import os
import os.path as osp
import sys
import time
import glob
import yaml
import warnings
import signal
import atexit
import wandb

# CRITICAL: Monkey-patch PyTorch BEFORE importing mmcv
# PyTorch 2.x changed _get_stream() to expect torch.device objects instead of integers
# This MUST be done before mmcv imports PyTorch, otherwise mmcv will get the unpatched version
import torch
import torch.nn.parallel._functions as pytorch_parallel_functions
_original_get_stream = pytorch_parallel_functions._get_stream

def _patched_get_stream(device):
    """Convert integer device to torch.device object for PyTorch 2.x compatibility"""
    if isinstance(device, int):
        device = torch.device(f'cuda:{device}')
    return _original_get_stream(device)

pytorch_parallel_functions._get_stream = _patched_get_stream
print("Applied PyTorch 2.x compatibility patch for mmcv _get_stream()")

# NOW import mmcv and related modules - they will use the patched _get_stream
import mmcv
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

# ========================================================================
# CRITICAL: Patch mmcv NMS BEFORE importing mmdet
# mmdet modules import from mmcv.ops at import time, so we must patch first.
# mmcv-full 1.7.2 wasn't compiled with CUDA ops, but torchvision provides
# equivalent NMS functionality that works with any CUDA version.
# ========================================================================

# Add detection directory to path for mmdet_custom imports
import sys
_det_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _det_dir not in sys.path:
    sys.path.insert(0, _det_dir)

try:
    from mmdet_custom.nms_wrapper import nms, batched_nms, NMSop
    from mmdet_custom.roi_align_wrapper import roi_align, RoIAlign, patch_mmcv_roi_align
    import mmcv.ops

    # Import the actual modules (not functions) via sys.modules
    # Note: mmcv.ops.nms and mmcv.ops.roi_align in the namespace are FUNCTIONS
    # but the actual module objects are in sys.modules
    import mmcv.ops.nms  # This loads the module
    import mmcv.ops.roi_align  # This loads the module

    nms_module = sys.modules['mmcv.ops.nms']
    roi_align_module = sys.modules['mmcv.ops.roi_align']

    # CRITICAL: Patch the internal RoIAlignFunction class
    # This patches the class method so that even direct calls to
    # RoIAlignFunction.apply() will use torchvision
    patch_mmcv_roi_align()

    # Patch NMS functions at module level
    nms_module.nms = nms
    nms_module.batched_nms = batched_nms
    nms_module.NMSop = NMSop
    mmcv.ops.nms = nms
    mmcv.ops.batched_nms = batched_nms

    # Patch RoI Align functions at module level
    roi_align_module.roi_align = roi_align
    roi_align_module.RoIAlign = RoIAlign
    mmcv.ops.roi_align = roi_align
    mmcv.ops.RoIAlign = RoIAlign

    print("Patched mmcv.ops with torchvision implementations (no CUDA compilation required):")
    print("  - NMS: nms, batched_nms, NMSop")
    print("  - RoI Align: roi_align, RoIAlign, RoIAlignFunction.forward")
except ImportError as e:
    print(f"Warning: Could not patch mmcv ops with torchvision: {e}")
    print("Detection training may fail if mmcv CUDA ops are not compiled")

# NOW import mmdet - it will use the patched NMS functions
from mmdet import __version__
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger

# Import custom modules to register with mmcv/mmdet
import mmcv_custom
import mmdet_custom

# Import backbone to register MEDiC with mmdet
import backbone

# Add parent directory to path to import downstream utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
try:
    import utils as downstream_utils
    WANDB_AVAILABLE = True
except ImportError:
    print("Warning: downstream utils not available. WandB logging disabled.")
    WANDB_AVAILABLE = False


# ========================================================================
# Graceful Shutdown Handler for SLURM Time Limits
# ========================================================================
# SLURM sends SIGTERM before killing jobs that hit time limits
# This handler ensures wandb.finish() is called before process termination
# preventing runs from being marked as "crashed" in WandB

_shutdown_initiated = False

def graceful_shutdown_handler(signum, frame):
    """
    Handle SLURM SIGTERM/SIGINT signals gracefully.

    When SLURM jobs hit their time limit (e.g., 24 hours), SLURM sends SIGTERM
    before killing the process. Without this handler, wandb doesn't get a chance
    to call finish(), causing runs to be marked as "crashed" instead of "finished".

    This handler:
    1. Catches the termination signal
    2. Calls wandb.finish() to properly close the WandB run
    3. Exits cleanly with status 0
    """
    global _shutdown_initiated

    if _shutdown_initiated:
        return
    _shutdown_initiated = True

    signal_names = {signal.SIGTERM: 'SIGTERM', signal.SIGINT: 'SIGINT'}
    signal_name = signal_names.get(signum, f'signal {signum}')

    print(f"\n{'='*80}")
    print(f"[GRACEFUL SHUTDOWN] Received {signal_name} (likely SLURM time limit)")
    print(f"[GRACEFUL SHUTDOWN] Finishing WandB run and exiting cleanly...")
    print(f"{'='*80}\n")

    try:
        # Try to import and finish wandb
        import wandb
        if wandb.run is not None:
            print("[GRACEFUL SHUTDOWN] Calling wandb.finish()...")
            wandb.finish(exit_code=0)
            print("[GRACEFUL SHUTDOWN] WandB run finished successfully")
        else:
            print("[GRACEFUL SHUTDOWN] No active WandB run to finish")
    except Exception as e:
        print(f"[GRACEFUL SHUTDOWN] Error finishing WandB: {e}")

    print("[GRACEFUL SHUTDOWN] Exiting with status 0 (success)")
    sys.exit(0)


def setup_graceful_shutdown():
    """
    Register signal handlers for graceful shutdown.

    This should be called early in main() to ensure signals are caught
    before training begins.
    """
    signal.signal(signal.SIGTERM, graceful_shutdown_handler)
    signal.signal(signal.SIGINT, graceful_shutdown_handler)

    # Also register an atexit handler as a fallback
    def atexit_handler():
        try:
            import wandb
            if wandb.run is not None and not _shutdown_initiated:
                print("[ATEXIT] Finishing WandB run...")
                wandb.finish(exit_code=0)
        except:
            pass

    atexit.register(atexit_handler)

    print("[SIGNAL HANDLER] Registered graceful shutdown handlers for SIGTERM and SIGINT")
    print("[SIGNAL HANDLER] WandB will be properly finished if SLURM hits time limit")


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector with MEDiC backbone')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--load-from', help='the checkpoint file to load weights from')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()

    # Setup graceful shutdown handlers FIRST to catch SLURM time limit signals
    setup_graceful_shutdown()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./checkpoints/object_det',
                                osp.splitext(osp.basename(args.config))[0])

    if args.load_from is not None:
        # CRITICAL FIX: For MEDiC pretraining checkpoints, we need to load weights
        # via backbone.pretrained (which calls init_weights), NOT via cfg.load_from
        # (which tries to load a full detector checkpoint with wrong key format).
        #
        # cfg.load_from: Used by mmdet runner to load FULL detector checkpoints
        # cfg.model.backbone.pretrained: Used by backbone to load PRETRAIN checkpoints
        #
        # The pretrain checkpoint has format: {'model': {'module.student.*': ...}}
        # The detector checkpoint has format: {'state_dict': {'backbone.*': ...}}
        #
        # Setting backbone.pretrained ensures init_weights() extracts student weights correctly.
        cfg.model.backbone.pretrained = args.load_from
        # Don't set cfg.load_from - let backbone handle its own weight loading
        print(f"[MEDiC] Setting backbone.pretrained = {args.load_from}")
        print(f"[MEDiC] Weights will be loaded via backbone.init_weights(), not cfg.load_from")
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # ========================================================================
    # Load checkpoint config for position embedding auto-detection
    # ========================================================================
    checkpoint_config = None
    checkpoint_path = args.load_from or (cfg.model.backbone.get('pretrained') if hasattr(cfg.model.backbone, 'pretrained') else None)

    if checkpoint_path:
        checkpoint_folder = os.path.dirname(checkpoint_path)
        config_files = glob.glob(os.path.join(checkpoint_folder, "config_*.yaml"))
        if config_files:
            config_file = config_files[0]
            print(f"Loading checkpoint config from: {config_file}")
            with open(config_file, 'r') as f:
                checkpoint_config = yaml.safe_load(f)

            # Update model config with checkpoint settings
            if checkpoint_config and 'model' in checkpoint_config:
                student_cfg = checkpoint_config['model'].get('student', {})

                # Update backbone settings from checkpoint
                if 'backbone' in cfg.model:
                    # Update position embedding settings
                    cfg.model.backbone.use_abs_pos_emb = student_cfg.get('use_abs_pos_emb', False)
                    cfg.model.backbone.use_sincos_pos_emb = student_cfg.get('use_sincos_pos_emb', True)

                    print(f"Position embedding configuration from checkpoint:")
                    print(f"  use_abs_pos_emb: {cfg.model.backbone.use_abs_pos_emb}")
                    print(f"  use_sincos_pos_emb: {cfg.model.backbone.use_sincos_pos_emb}")

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        # Make sure dist_params exists in config
        if not hasattr(cfg, 'dist_params'):
            cfg.dist_params = dict(backend='nccl')
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    # ========================================================================
    # Configure WandB logging via mmdet's WandbLoggerHook
    # ========================================================================
    global_rank = int(os.environ.get('RANK', 0))
    if global_rank == 0 and cfg.work_dir is not None:
        # Derive descriptive WandB run name from checkpoint
        wandb_name = None
        if args.load_from:
            wandb_name = downstream_utils.derive_wandb_name_from_checkpoint(args.load_from, "det")
        elif args.resume_from:
            wandb_name = downstream_utils.derive_wandb_name_from_checkpoint(args.resume_from, "det")
        elif hasattr(cfg.model.backbone, 'pretrained') and cfg.model.backbone.pretrained:
            wandb_name = downstream_utils.derive_wandb_name_from_checkpoint(cfg.model.backbone.pretrained, "det")

        # Find and update WandbLoggerHook in log_config
        if hasattr(cfg, 'log_config') and 'hooks' in cfg.log_config:
            for hook in cfg.log_config.hooks:
                if hook.get('type') == 'WandbLoggerHook':
                    # Update init_kwargs with run name and config
                    if 'init_kwargs' not in hook:
                        hook['init_kwargs'] = {}
                    hook['init_kwargs']['name'] = wandb_name
                    hook['init_kwargs']['config'] = {
                        'checkpoint': args.load_from or args.resume_from,
                        'lr': cfg.optimizer.lr if hasattr(cfg, 'optimizer') else 0.0003,
                        'batch_size': cfg.data.samples_per_gpu if hasattr(cfg.data, 'samples_per_gpu') else 2,
                        'update_interval': cfg.get('optimizer_config', {}).get('update_interval', 1),
                        'layer_decay_rate': cfg.optimizer.get('paramwise_cfg', {}).get('layer_decay_rate', 0.75),
                        'work_dir': cfg.work_dir,
                        'seed': args.seed,
                    }
                    logger.info(f'Configured WandB run name: {wandb_name}')
                    break

    # Build model
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))

    logger.info(f'Model:\n{model}')

    # Build datasets
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))

    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES)

    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    # ========================================================================
    # DDP configuration
    # ========================================================================
    cfg.find_unused_parameters = True
    logger.info("Set find_unused_parameters=True for DDP")

    # ========================================================================
    # Add EpochIterLoggerHook for explicit epoch/iter logging
    # ========================================================================
    if not hasattr(cfg, 'custom_hooks') or cfg.custom_hooks is None:
        cfg.custom_hooks = []
    cfg.custom_hooks.append(dict(type='EpochIterLoggerHook', interval=50))
    logger.info('Added EpochIterLoggerHook to log epoch/iter/progress_pct as explicit WandB metrics')

    # Train the detector
    # WandB logging is handled automatically by mmdet's WandbLoggerHook
    # which logs training losses, evaluation metrics, and best mAP scores
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)

    # ========================================================================
    # Properly finish WandB run to avoid "crashed" status
    # ========================================================================
    global_rank = int(os.environ.get('RANK', 0))
    if global_rank == 0 and wandb.run is not None:
        try:
            logger.info("Finishing WandB run...")
            wandb.finish()
            logger.info("WandB run finished successfully")
        except Exception as e:
            logger.warning(f"WandB finish failed (non-fatal): {e}")
            try:
                wandb.finish(exit_code=1)
            except Exception:
                pass


if __name__ == '__main__':
    main()
