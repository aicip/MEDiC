# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on DINO code bases
# https://github.com/facebookresearch/dino/blob/main/eval_linear.py
# --------------------------------------------------------'
import os
import argparse
import json
import yaml
import glob
from pathlib import Path
from collections import OrderedDict
from pprint import pprint

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms

import utils
import modeling_finetune
from timm.models import create_model

# Import VisionTransformerMIM from training code (same as k-NN uses)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.vision_transformer import VisionTransformerMIM




def load_model(model, checkpoint_file, model_key, model_filter_name, is_cmae=False):
    if checkpoint_file.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
            checkpoint_file, map_location="cpu", check_hash=True, weights_only=False
        )
    else:
        checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)

    print("Load ckpt from %s" % checkpoint_file)
    checkpoint_model = None
    for model_key in model_key.split("|"):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print("Load state_dict by model_key = %s" % model_key)
            break
    if checkpoint_model is None:
        checkpoint_model = checkpoint

    if is_cmae:
        checkpoint_model = {k.replace('layers', 'blocks'): v for k, v in checkpoint_model.items()}
        checkpoint_model = {k.replace('patch_embed.projection', 'patch_embed.proj'): v for k, v in checkpoint_model.items()}
        checkpoint_model = {k.replace('ln1', 'norm') if 'blocks' not in k else k: v for k, v in checkpoint_model.items()}
        checkpoint_model = {k.replace('ln', 'norm') if 'blocks' in k else k: v for k, v in checkpoint_model.items()}
        checkpoint_model = {k.replace('ffn.blocks.0.0', 'mlp.fc1') if 'blocks' in k else k: v for k, v in checkpoint_model.items()}
        checkpoint_model = {k.replace('ffn.blocks.1', 'mlp.fc2') if 'blocks' in k else k: v for k, v in checkpoint_model.items()}

    if (checkpoint_model is not None):
        all_keys = list(checkpoint_model.keys())
        new_dict = OrderedDict()
        for key in all_keys:
            if key.startswith(model_filter_name):
                # Remove the filter prefix (e.g., "module.student.")
                new_key = key.replace(model_filter_name, "")
                if new_key.startswith("encoder.") or new_key.startswith("student."):
                    new_key = new_key[8:]
                    # it just so happens that encoder and student have the same length
                new_dict[new_key] = checkpoint_model[key]
            else:
                # Keep keys that don't match the filter pattern
                new_dict[key] = checkpoint_model[key]
        if new_dict:
            checkpoint_model = new_dict

    # CRITICAL: Handle norm vs fc_norm naming difference
    # VisionTransformerMIM uses "norm", VisionTransformer uses "fc_norm"
    if 'norm.weight' in checkpoint_model and 'norm.weight' not in model.state_dict():
        checkpoint_model['fc_norm.weight'] = checkpoint_model.pop('norm.weight')
        checkpoint_model['fc_norm.bias'] = checkpoint_model.pop('norm.bias')
        print("Renamed norm → fc_norm for compatibility")

    state_dict = model.state_dict()
    for k in ["head.weight", "head.bias", "head.fc.weight", "head.fc.bias"]:
        if (
            k in checkpoint_model
            and checkpoint_model[k].shape != state_dict[k].shape
        ):
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    utils.load_state_dict(model, checkpoint_model, prefix="")

    # Validate weight loading
    model_state_keys = set(model.state_dict().keys())
    checkpoint_keys = set(checkpoint_model.keys())
    loaded = model_state_keys & checkpoint_keys
    missing = model_state_keys - checkpoint_keys
    unexpected = checkpoint_keys - model_state_keys

    print(f"\n{'='*60}")
    print(f"Weight Loading: {len(loaded)} loaded, {len(missing)} missing, {len(unexpected)} unexpected")
    if missing:
        print(f"  Missing (first 5): {list(missing)[:5]}")
    if unexpected:
        print(f"  Unexpected (first 5): {list(unexpected)[:5]}")
    print(f"{'='*60}\n")

    # raise ValueError("Testing")


def eval_linear(args):
    utils.init_distributed_mode(args)
    # print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # Initialize WandB logging
    if utils.get_rank() == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        wandb_name = utils.derive_wandb_name_from_checkpoint(args.pretrained_weights, "lin")
        log_writer = utils.WandbLogger(
            project_name="MEDiC_linear",
            entity=None,  # Set to your W&B entity
            config=vars(args),
            name=wandb_name
        )
    else:
        log_writer = None

    mean = (0.485, 0.456, 0.406) if args.imagenet_default_mean_and_std else (0.5, 0.5, 0.5)
    std = (0.229, 0.224, 0.225) if args.imagenet_default_mean_and_std else (0.5, 0.5, 0.5)

    # ============ preparing data ... ============
    train_transform = pth_transforms.Compose([
        pth_transforms.RandomResizedCrop(224),
        pth_transforms.RandomHorizontalFlip(),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize(mean, std),
    ])
    val_transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize(mean, std),
    ])
    print("train_transform = %s" % str(train_transform))
    print("val_transform = %s" % str(val_transform))
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, "train"), transform=train_transform)
    dataset_val = datasets.ImageFolder(os.path.join(args.data_path, "val"), transform=val_transform)
    global_rank = utils.get_rank()
    world_size = utils.get_world_size()
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset_train, num_replicas=world_size, rank=global_rank, shuffle=True)

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # ============ Load config from checkpoint folder ... ============
    use_abs_pos_emb = args.abs_pos_emb
    use_rel_pos_bias = args.rel_pos_bias
    student_cfg = {}
    checkpoint_folder = os.path.dirname(args.pretrained_weights)
    config_files = glob.glob(os.path.join(checkpoint_folder, "config_*.yaml"))
    if config_files:
        config_path = config_files[0]
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        student_cfg = cfg.get("model", {}).get("student", {})

        # CRITICAL: Load position embedding config from checkpoint to match training
        use_abs_pos_emb = student_cfg.get("use_abs_pos_emb", args.abs_pos_emb)
        use_rel_pos_bias = student_cfg.get("use_rel_pos_bias", args.rel_pos_bias)
        use_shared_rel_pos_bias = student_cfg.get("use_shared_rel_pos_bias", args.rel_pos_bias)

        # CRITICAL: Load layer scale init_values from checkpoint config
        init_values = student_cfg.get("init_values", args.layer_scale_init_value)

        # CRITICAL: Infer qkv_bias from checkpoint weights (qkv_bias not in config)
        # If checkpoint has q_bias/v_bias, then qkv_bias=True was used during training
        checkpoint = torch.load(args.pretrained_weights, map_location='cpu', weights_only=False)
        checkpoint_model = checkpoint['model'] if 'model' in checkpoint else checkpoint
        has_qkv_bias = any('q_bias' in k for k in checkpoint_model.keys())
        qkv_bias = has_qkv_bias  # True if checkpoint has q_bias weights

        if utils.get_rank() == 0:
            print(f"Position embedding configuration:")
            print(f"  Absolute pos emb: {use_abs_pos_emb}")
            print(f"  Relative pos bias: {use_rel_pos_bias}")
            print(f"  Shared rel pos bias: {use_shared_rel_pos_bias}")
            print(f"Architecture parameters:")
            print(f"  Layer scale init_values: {init_values}")
            print(f"  QKV bias: {qkv_bias}")

    # ============ building network ... ============
    # CRITICAL: Default values for when no checkpoint config exists
    if 'init_values' not in locals():
        init_values = args.layer_scale_init_value
    if 'qkv_bias' not in locals():
        qkv_bias = args.qkv_bias

    # CRITICAL FIX: Use VisionTransformerMIM from training code (same as k-NN)
    # This ensures identical Block, Attention, and MLP implementations
    model = VisionTransformerMIM(
        img_size=student_cfg.get("img_size", 224),
        patch_size=student_cfg.get("patch_size", 16),
        embed_dim=student_cfg.get("embed_dim", 768),
        depth=student_cfg.get("depth", 12),
        num_heads=student_cfg.get("num_heads", 12),
        mlp_ratio=student_cfg.get("mlp_ratio", 4.0),
        qkv_bias=qkv_bias,  # Inferred from checkpoint weights
        drop_path_rate=student_cfg.get("drop_path_rate", 0.0),  # No drop path during eval
        init_values=init_values,  # From checkpoint config
        # Position embedding config
        use_abs_pos_emb=use_abs_pos_emb,
        use_sincos_pos_emb=student_cfg.get("use_sincos_pos_emb", False),
        use_rel_pos_bias=use_rel_pos_bias,
        use_shared_rel_pos_bias=use_shared_rel_pos_bias,
    )

    model.cuda()
    model.eval()
    print(f"Model {args.model} built.")
    # load weights to evaluate
    load_model(model=model,
               checkpoint_file=args.pretrained_weights,
               model_key=args.checkpoint_key,
               model_filter_name=args.model_filter_name,
               is_cmae=args.is_cmae)

    nan_params = []
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            nan_params.append(name)
    if nan_params:
        print(f"\n⚠️ WARNING: Found NaN in {len(nan_params)} model parameters:")
        for name in nan_params[:10]:  # Print first 10
            print(f"  - {name}")
    else:
        print("\n✅ All model parameters are finite (no NaN)")

    if hasattr(model, 'pos_embed') and model.pos_embed is not None:
        print(f"  Shape: {model.pos_embed.shape}")
        print(f"  min={model.pos_embed.min():.6f}, max={model.pos_embed.max():.6f}")
        print(f"  mean={model.pos_embed.mean():.6f}, std={model.pos_embed.std():.6f}")
        if torch.allclose(model.pos_embed, torch.zeros_like(model.pos_embed)):
            print(f"  ⚠️ WARNING: pos_embed is all zeros!")
    else:
        print(f"  No pos_embed attribute")

    linear_classifier = LinearClassifier(
        dim=model.embed_dim * (1 + int(args.avgpool_patchtokens)),
        num_labels=args.num_labels, num_layers=model.get_num_layers())
    linear_classifier = linear_classifier.cuda()
    if world_size > 1:
        linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[args.gpu])

    print("Model = %s" % str(linear_classifier))

    # set optimizer
    learning_rate = args.lr or args.base_lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256
    # use absolute or linear scaled learning rate
    if args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(
            linear_classifier.parameters(), learning_rate, momentum=0.9,
            weight_decay=0, # we do not apply weight decay
        )
    else:
        optimizer = torch.optim.AdamW(
            linear_classifier.parameters(), learning_rate, weight_decay=1e-4,
        )
    print(f"Optimizer = %s" % str(optimizer))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_acc": 0.}
    # utils.restart_from_checkpoint(
    #     os.path.join(args.output_dir, "checkpoint.pth.tar"),
    #     run_variables=to_restore,
    #     state_dict=linear_classifier,
    #     optimizer=optimizer,
    #     scheduler=scheduler,
    # )
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]

    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)

        train_stats = train(
            model, linear_classifier, optimizer, train_loader, epoch, args.avgpool_patchtokens, args.amp_forward)
        scheduler.step()
        
        # Process train stats to aggregate per-layer metrics
        train_acc1_layers = {}
        train_acc5_layers = {}
        for key, value in train_stats.items():
            if 'acc1_layer' in key:
                layer_idx = key.split('layer')[1]
                train_acc1_layers[f'layer{layer_idx}'] = value
            elif 'acc5_layer' in key:
                layer_idx = key.split('layer')[1]
                train_acc5_layers[f'layer{layer_idx}'] = value

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            test_stats = validate_network(
                val_loader, model, linear_classifier, args.avgpool_patchtokens, args.amp_forward)
            # Find max accuracy across all layers
            current_max_acc = 0.0
            best_layer = None
            for classifier_key in test_stats:
                classifier = test_stats[classifier_key]
                print(f"Validation accuracy at epoch {epoch} on {len(dataset_val)} images: {classifier['acc1']:.1f}% (layer: {classifier_key})")
                if classifier["acc1"] > current_max_acc:
                    current_max_acc = classifier["acc1"]
                    best_layer = classifier_key
            best_acc = max(best_acc, current_max_acc)
            print(f'Max accuracy so far: {best_acc:.2f}% (from {best_layer})')
            # Flatten test_stats for WandB logging
            flattened_test_stats = {}
            for layer_name, layer_stats in test_stats.items():
                if isinstance(layer_stats, dict):
                    for metric_name, metric_value in layer_stats.items():
                        # Use 'val' prefix instead of 'test' to indicate validation metrics
                        flattened_test_stats[f'val_{layer_name}_{metric_name}'] = metric_value
                else:
                    flattened_test_stats[f'val_{layer_name}'] = layer_stats

            log_stats = {**{k: v for k, v in log_stats.items()},
                         **flattened_test_stats}
        else:
            # Even without validation, log train accuracy per layer
            if log_writer is not None:
                for layer_name, acc in train_acc1_layers.items():
                    log_writer.update(**{f'train_acc1_{layer_name}': acc}, step=epoch)
                for layer_name, acc in train_acc5_layers.items():
                    log_writer.update(**{f'train_acc5_{layer_name}': acc}, step=epoch)
                    
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # Log to WandB
            if log_writer is not None and (epoch % args.val_freq == 0 or epoch == args.epochs - 1):
                # Log all metrics with proper organization
                log_writer.update(step=epoch, **log_stats)
                
                # Log train accuracy per layer
                for layer_name, acc in train_acc1_layers.items():
                    log_writer.update(**{f'train_acc1_{layer_name}': acc}, step=epoch)
                for layer_name, acc in train_acc5_layers.items():
                    log_writer.update(**{f'train_acc5_{layer_name}': acc}, step=epoch)
                
                # Log the max accuracy across all layers
                log_writer.update(val_max_accuracy=current_max_acc, head="best", step=epoch)
                log_writer.update(val_best_accuracy_so_far=best_acc, head="best", step=epoch)
                # Extract numeric layer ID from string like "layer0" for WandB compatibility
                if best_layer is not None:
                    layer_num = int(best_layer.replace('layer', ''))
                    log_writer.update(val_best_layer=layer_num, head="best", step=epoch)

                # Log with alternative metric names for compatibility
                log_writer.update(best_accuracy_so_far=best_acc, head="best", step=epoch)
                log_writer.update(max_accuracy=current_max_acc, head="best", step=epoch)
                # Also log as test_best_acc1 (validation is same as test for linear eval)
                log_writer.update(test_best_acc1=best_acc, head="scalar", step=epoch)
            # Only save if this is the best accuracy so far
            if current_max_acc >= best_acc:
                save_dict = {
                    "epoch": epoch + 1,
                    "state_dict": linear_classifier.state_dict(),
                    "best_acc": current_max_acc,
                }
                torch.save(save_dict, os.path.join(args.output_dir, "checkpoint-best.pth.tar"), _use_new_zipfile_serialization=False)
    print("Training of the supervised linear classifier on frozen features completed.\n"
                "Top-1 validation accuracy: {acc:.1f}".format(acc=best_acc))


def train(model, linear_classifier, optimizer, loader, epoch, avgpool, amp_forward):
    linear_classifier.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    assert avgpool
    module = linear_classifier.module if hasattr(linear_classifier, 'module') else linear_classifier

    first_batch = True
    for (inp, target) in metric_logger.log_every(loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        if first_batch and torch.distributed.get_rank() == 0:
            print(f"  Shape: {inp.shape}")
            print(f"  min={inp.min():.4f}, max={inp.max():.4f}, mean={inp.mean():.4f}")
            print(f"  Has NaN: {torch.isnan(inp).any()}")
            print(f"  Has Inf: {torch.isinf(inp).any()}")

        # forward
        with torch.no_grad():
            if amp_forward:
                with torch.amp.autocast('cuda'):
                    intermediate_output = model.get_intermediate_layers(inp, use_last_norm=True)
            else:
                intermediate_output = model.get_intermediate_layers(inp, use_last_norm=True)

            output = []
            for i, each_layer in enumerate(intermediate_output):
                cls_rep = each_layer[:, 0]
                mean_rep = torch.mean(each_layer[:, 1:], dim=1)
                concat_features = torch.cat((cls_rep, mean_rep), dim=-1).float()
                output.append(concat_features)

                if first_batch and i == 0 and torch.distributed.get_rank() == 0:
                    print(f"  CLS: min={cls_rep.min():.4f}, max={cls_rep.max():.4f}, mean={cls_rep.mean():.4f}")
                    print(f"  Mean: min={mean_rep.min():.4f}, max={mean_rep.max():.4f}, mean={mean_rep.mean():.4f}")
                    print(f"  Concat: min={concat_features.min():.4f}, max={concat_features.max():.4f}")
                    print(f"  Has NaN: {torch.isnan(concat_features).any()}")
                    print(f"  Has Inf: {torch.isinf(concat_features).any()}")

        output = linear_classifier(output)

        if first_batch and torch.distributed.get_rank() == 0:
            print(f"  min={output[0].min():.4f}, max={output[0].max():.4f}, mean={output[0].mean():.4f}")
            print(f"  Has NaN: {torch.isnan(output[0]).any()}")
            print(f"  Has Inf: {torch.isinf(output[0]).any()}")

        # compute cross entropy loss and accuracy for each layer
        loss = 0
        batch_size = inp.shape[0]
        for i, each_output in enumerate(output):
            layer_loss = nn.CrossEntropyLoss()(each_output, target)

            if first_batch and i == 0 and torch.distributed.get_rank() == 0:
                print(f"  Is NaN: {torch.isnan(layer_loss)}")
            loss += layer_loss
            
            # Calculate accuracy for this layer
            if module.num_labels >= 5:
                acc1, acc5 = utils.accuracy(each_output, target, topk=(1, 5))
                metric_logger.meters[f'acc1_layer{i}'].update(acc1.item(), n=batch_size)
                metric_logger.meters[f'acc5_layer{i}'].update(acc5.item(), n=batch_size)
            else:
                acc1, = utils.accuracy(each_output, target, topk=(1,))
                metric_logger.meters[f'acc1_layer{i}'].update(acc1.item(), n=batch_size)

        first_batch = False

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # log
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_network(val_loader, model, linear_classifier, avgpool, amp_forward):
    linear_classifier.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    assert avgpool
    module = linear_classifier.module if hasattr(linear_classifier, 'module') else linear_classifier
    for inp, target in metric_logger.log_every(val_loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            if amp_forward:
                with torch.amp.autocast('cuda'):
                    intermediate_output = model.get_intermediate_layers(inp, use_last_norm=True)
            else:
                intermediate_output = model.get_intermediate_layers(inp, use_last_norm=True)

            output = []
            for each_layer in intermediate_output:
                cls_rep = each_layer[:, 0]
                mean_rep = torch.mean(each_layer[:, 1:], dim=1)
                output.append(torch.cat((cls_rep, mean_rep), dim=-1).float())
            all_output = linear_classifier(output)

        for i, output in enumerate(all_output):
            loss = nn.CrossEntropyLoss()(output, target)

            if module.num_labels >= 5:
                acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            else:
                acc1, = utils.accuracy(output, target, topk=(1,))

            batch_size = inp.shape[0]
            post_str = '_layer%d' % i
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1' + post_str].update(acc1.item(), n=batch_size)
            if module.num_labels >= 5:
                metric_logger.meters['acc5' + post_str].update(acc5.item(), n=batch_size)

    eval_results = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    updated_results = {}
    for key in eval_results:
        if '_' in key:
            this_key, classifier_idx = key.split('_')

            if classifier_idx not in updated_results:
                updated_results[classifier_idx] = {}

            updated_results[classifier_idx][this_key] = eval_results[key]

    print("Eval result = %s" % json.dumps(updated_results, indent=2))
    return updated_results


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, num_layers, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.ModuleList()
        self.num_classifier = num_layers
        for i in range(self.num_classifier):
            linear = nn.Linear(dim, num_labels)
            linear.weight.data.normal_(mean=0.0, std=0.01)
            linear.bias.data.zero_()
            self.linear.append(linear)

    def forward(self, x_list):
        results = []
        for i, linear in enumerate(self.linear):
            results.append(linear(x_list[i]))
        return results


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--avgpool_patchtokens', default=True, type=bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to True for BEiT pretrained models. """)
    parser.add_argument('--model', default='beit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--qkv_bias', action='store_true')
    parser.set_defaults(rel_pos_bias=False)
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.set_defaults(rel_pos_bias=False)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=False)
    parser.add_argument('--is_cmae', action='store_true')
    parser.set_defaults(is_cmae=False)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float,
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--optimizer', default="adamw", type=str, help='optimizer type')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="model|module|teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--model_filter_name", default="", type=str, help='Filter key')
    parser.add_argument('--epochs', default=50, type=int, help='Number of epochs of training.')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument("--base_lr", default=0.001, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')
    parser.add_argument('--log_dir', default=None, type=str, help='Path to save logs')
    parser.add_argument('--num_labels', default=1000, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--imagenet_default_mean_and_std', default=False, type=bool_flag,
        help="""Set True to use the imagenet default mean and std, Set False will use the mean and std in Inception.
        We recommand keep it same to the pre-training stage. """)
    parser.add_argument('--amp_forward', default=True, type=bool_flag, help='Use amp to inference the pre-trained model, which can speed up the evaluation. ')

    args = parser.parse_args()
    pprint(vars(args))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    eval_linear(args)
