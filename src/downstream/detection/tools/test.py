# --------------------------------------------------------
# MEDiC: Masked Enhanced Distillation with Contrastive Learning
# Object Detection Test Script
# --------------------------------------------------------

import argparse
import os
import os.path as osp
import sys
import glob
import yaml
import warnings

# CRITICAL: Monkey-patch PyTorch BEFORE importing mmcv
import torch
import torch.nn.parallel._functions as pytorch_parallel_functions
_original_get_stream = pytorch_parallel_functions._get_stream

def _patched_get_stream(device):
    if isinstance(device, int):
        device = torch.device(f'cuda:{device}')
    return _original_get_stream(device)

pytorch_parallel_functions._get_stream = _patched_get_stream
print("Applied PyTorch 2.x compatibility patch for mmcv _get_stream()")

import mmcv
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector

# Import custom modules
import mmcv_custom
import mmdet_custom
import backbone

# Patch mmcv NMS and RoI Align with torchvision implementations
try:
    from mmdet_custom.nms_wrapper import nms, batched_nms, NMSop
    from mmdet_custom.roi_align_wrapper import roi_align, RoIAlign, patch_mmcv_roi_align
    import mmcv.ops
    import mmcv.ops.nms
    import mmcv.ops.roi_align

    nms_module = sys.modules['mmcv.ops.nms']
    roi_align_module = sys.modules['mmcv.ops.roi_align']

    patch_mmcv_roi_align()

    nms_module.nms = nms
    nms_module.batched_nms = batched_nms
    nms_module.NMSop = NMSop
    mmcv.ops.nms = nms
    mmcv.ops.batched_nms = batched_nms

    roi_align_module.roi_align = roi_align
    roi_align_module.RoIAlign = RoIAlign
    mmcv.ops.roi_align = roi_align
    mmcv.ops.RoIAlign = RoIAlign
    print("Patched mmcv.ops with torchvision implementations for evaluation")
except ImportError as e:
    print(f"Warning: Could not patch mmcv ops: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description='Test a detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation.')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config')
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
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    # Load checkpoint config for position embedding auto-detection
    checkpoint_folder = os.path.dirname(args.checkpoint)
    config_files = glob.glob(os.path.join(checkpoint_folder, "config_*.yaml"))
    if config_files:
        with open(config_files[0], 'r') as f:
            checkpoint_config = yaml.safe_load(f)
        if checkpoint_config and 'model' in checkpoint_config:
            student_cfg = checkpoint_config['model'].get('student', {})
            if 'backbone' in cfg.model:
                cfg.model.backbone.use_abs_pos_emb = student_cfg.get('use_abs_pos_emb', False)
                cfg.model.backbone.use_sincos_pos_emb = student_cfg.get('use_sincos_pos_emb', True)

    # init distributed env
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        if not hasattr(cfg, 'dist_params'):
            cfg.dist_params = dict(backend='nccl')
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    if samples_per_gpu > 1:
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))

    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        from mmcv.cnn import fuse_conv_bn
        model = fuse_conv_bn(model)

    # old versions did not save class info in checkpoints
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  args.show_score_thr)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=True)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            for key in ['interval', 'tmpdir', 'start', 'gpu_collect']:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))


def replace_ImageToTensor(pipelines):
    """Replace ImageToTensor transform with DefaultFormatBundle for multi-gpu testing."""
    pipelines = copy.deepcopy(pipelines)
    for i, pipeline in enumerate(pipelines):
        if pipeline['type'] == 'MultiScaleFlipAug':
            assert 'transforms' in pipeline
            pipeline['transforms'] = replace_ImageToTensor(pipeline['transforms'])
        elif pipeline['type'] == 'ImageToTensor':
            pipelines[i] = {'type': 'DefaultFormatBundle'}
    return pipelines


if __name__ == '__main__':
    import copy
    main()
