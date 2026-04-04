import argparse
import copy
import os
import os.path as osp
import sys
import time
import glob

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
from mmcv.runner import init_dist
from mmcv.utils import Config, DictAction, get_git_hash

from mmseg import __version__
from mmseg.apis import set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_root_logger

# Import custom modules AFTER mmseg to override built-in modules with force=True
import mmcv_custom  # Overrides LayerDecayOptimizerConstructor
import mmseg_custom  # Import our custom mmseg modules with fixes

# Import backbone to register MEDiC with mmseg
import backbone


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
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
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    # Force re-register our custom LayerDecayOptimizerConstructor
    # (mmseg's config loading may override the registration)
    from mmcv_custom.layer_decay_optimizer_constructor import LayerDecayOptimizerConstructor  # noqa: F811
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        # Make sure dist_params exists in config
        if not hasattr(cfg, 'dist_params'):
            cfg.dist_params = dict(backend='nccl')
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: '
                    f'{args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))

    logger.info(model)

    # Handle both old format (cfg.data.train) and new format (cfg.train_dataloader)
    if hasattr(cfg, 'data') and hasattr(cfg.data, 'train'):
        # v0.x format with cfg.data.train/val/test
        logger.info('Using mmseg v0.x config format with cfg.data.train')
        datasets = [build_dataset(cfg.data.train)]
        logger.info(f'Built training dataset: {len(datasets[0])} samples')

        # Build validation dataset if validation is enabled
        if hasattr(cfg.data, 'val') and not args.no_validate:
            val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
            datasets.append(val_dataset)
            logger.info(f'Built validation dataset: {len(val_dataset)} samples')
        else:
            logger.info('Validation dataset not built (--no-validate or cfg.data.val missing)')
    else:
        raise ValueError(
            "Config must use mmseg v0.x format with cfg.data.train. "
            "See configs/medic/upernet_medic_base_512_160k_ade20k.py for reference."
        )

    if hasattr(cfg, 'checkpoint_config') and cfg.checkpoint_config is not None:
        # save mmseg version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    # Add device field for train_segmentor compatibility
    if not hasattr(cfg, 'device'):
        cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Add gpu_ids for old mmseg v0.x DDP compatibility
    if not hasattr(cfg, 'gpu_ids'):
        if distributed:
            world_size = int(os.environ.get('WORLD_SIZE', 1))
            cfg.gpu_ids = list(range(world_size))
        else:
            cfg.gpu_ids = [0]
        logger.info(f'Added cfg.gpu_ids={cfg.gpu_ids} for mmseg v0.x DDP compatibility')

    # Add resume_from attribute if not present (mmseg v0.x expects this field)
    if not hasattr(cfg, 'resume_from'):
        cfg.resume_from = None
        logger.info(f'Added cfg.resume_from=None for mmseg v0.x compatibility')

    # Convert cfg.data.train and cfg.data.val to dicts for compatibility with old mmseg build_dataset
    if hasattr(cfg.data, 'train') and not isinstance(cfg.data.train, dict):
        cfg.data.train = dict(cfg.data.train)
        logger.info(f'Converted cfg.data.train to dict for mmseg compatibility')
    if hasattr(cfg.data, 'val') and not isinstance(cfg.data.val, dict):
        cfg.data.val = dict(cfg.data.val)
        logger.info(f'Converted cfg.data.val to dict for mmseg compatibility')

    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)

    # =========================================================================
    # AUTOMATIC POST-TRAINING EVALUATION
    # Run evaluation after training completes to get mIoU
    # Only runs on rank 0 to avoid duplicate evaluations
    # =========================================================================
    rank = int(os.environ.get('RANK', 0))
    if rank == 0:
        logger.info("=" * 60)
        logger.info("Training completed. Running automatic evaluation...")
        logger.info("=" * 60)

        # Find the latest/best checkpoint
        latest_ckpt = osp.join(cfg.work_dir, 'latest.pth')
        best_ckpt = osp.join(cfg.work_dir, 'best_mIoU_*.pth')
        best_ckpt_files = glob.glob(best_ckpt)

        checkpoint_to_eval = None
        if best_ckpt_files:
            checkpoint_to_eval = best_ckpt_files[0]
            logger.info(f"Found best checkpoint: {checkpoint_to_eval}")
        elif osp.exists(latest_ckpt):
            checkpoint_to_eval = latest_ckpt
            logger.info(f"Using latest checkpoint: {checkpoint_to_eval}")
        else:
            # Find any checkpoint
            all_ckpts = glob.glob(osp.join(cfg.work_dir, 'iter_*.pth'))
            if all_ckpts:
                checkpoint_to_eval = sorted(all_ckpts)[-1]  # Latest by iteration
                logger.info(f"Using checkpoint: {checkpoint_to_eval}")

        if checkpoint_to_eval:
            import subprocess

            test_config = 'configs/medic/upernet_medic_test_ade20k.py'

            eval_cmd = [
                'python', 'tools/test.py',
                test_config,
                checkpoint_to_eval,
                '--eval', 'mIoU'
            ]

            logger.info(f"Running evaluation command: {' '.join(eval_cmd)}")

            try:
                result = subprocess.run(
                    eval_cmd,
                    capture_output=True,
                    text=True,
                    timeout=3600,  # 1 hour timeout
                    cwd=osp.dirname(osp.dirname(osp.abspath(__file__)))  # segmentation dir
                )

                # Log stdout (contains mIoU results)
                if result.stdout:
                    logger.info("Evaluation output:")
                    for line in result.stdout.split('\n'):
                        if line.strip():
                            logger.info(f"  {line}")

                if result.returncode != 0:
                    logger.error(f"Evaluation failed with code {result.returncode}")
                    if result.stderr:
                        logger.error(f"Stderr: {result.stderr}")
                else:
                    logger.info("Evaluation completed successfully!")

            except subprocess.TimeoutExpired:
                logger.error("Evaluation timed out after 1 hour")
            except Exception as e:
                logger.error(f"Evaluation failed with exception: {e}")
        else:
            logger.warning(f"No checkpoint found in {cfg.work_dir}, skipping evaluation")


if __name__ == '__main__':
    main()
