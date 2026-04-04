# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------'
import torch
from torch import optim
from torch.optim.optimizer import Optimizer # pylint: disable=unused-import
from typing import Optional, Tuple, Dict, Any, List


def get_num_layer_for_vit(var_name: str, num_max_layer: int) -> int:
    """
    Assigns a layer ID to a variable name for Vision Transformers.
    This is used for layer-wise learning rate decay.

    Updated to handle parameter names with prefixes (student., distill_head., etc.)
    """
    # Strip common prefixes to get the actual parameter name
    if var_name.startswith("student."):
        var_name = var_name[8:]  # Remove "student." prefix
    elif var_name.startswith("distill_head."):
        # Distillation head gets assigned to the last layer
        return num_max_layer - 1

    if var_name in ("cls_token", "mask_token", "pos_embed"):
        return 0
    elif var_name.startswith("patch_embed"):
        return 0
    elif var_name.startswith("rel_pos_bias"):
        return num_max_layer - 1
    elif var_name.startswith("blocks"):
        parts = var_name.split(".")
        if len(parts) < 2:
            raise ValueError(f"Invalid blocks parameter name: {var_name}")
        try:
            layer_id = int(parts[1])
        except ValueError:
            raise ValueError(f"Invalid layer id in parameter name: {var_name}")
        return layer_id + 1
    else:
        return num_max_layer - 1


class LayerDecayValueAssigner(object):
    """
    Helper class to assign layer-wise decay values.
    """
    def __init__(self, values: List[float]):
        self.values = values

    def get_scale(self, layer_id: int) -> float:
        # Bounds checking to prevent IndexError
        if layer_id < 0:
            return self.values[0]
        elif layer_id >= len(self.values):
            return self.values[-1]
        else:
            return self.values[layer_id]

    def get_layer_id(self, var_name: str) -> int:
        return get_num_layer_for_vit(var_name, len(self.values))


def get_parameter_groups(
    model: torch.nn.Module, weight_decay: float = 1e-5, skip_list=(),
    get_num_layer=None, get_layer_scale=None
) -> List[Dict[str, Any]]:
    """
    Assigns model parameters to different groups for optimization.
    This is used for applying layer-wise learning rate decay and
    excluding certain parameters (biases, norm layers) from weight decay.
    """
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.0
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = f"layer_{layer_id}_{group_name}"
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.0

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    return list(parameter_group_vars.values())


def create_optimizer(
    model: torch.nn.Module,
    opt: str,
    lr: float,
    weight_decay: float,
    momentum: float = 0.9,
    opt_eps: Optional[float] = None,
    opt_betas: Optional[Tuple[float, float]] = None,
    get_num_layer=None,
    get_layer_scale=None,
    filter_bias_and_bn=True,
    skip_list=None,
) -> Optimizer:
    """
    Creates an optimizer for the given model.
    Supports 'sgd' and 'adamw'.
    """
    opt_lower = opt.lower()
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if skip_list is not None:
            skip = skip_list
        elif hasattr(model, "no_weight_decay"):
            skip = model.no_weight_decay()
        print(f"Skip weight decay name marked in model: {skip}")
        parameters = get_parameter_groups(
            model, weight_decay, skip, get_num_layer, get_layer_scale
        )
        weight_decay = 0.0
    else:
        parameters = model.parameters()

    opt_args: Dict[str, Any] = dict(lr=lr, weight_decay=weight_decay)
    if opt_eps is not None:
        opt_args["eps"] = opt_eps
    if opt_betas is not None:
        opt_args["betas"] = opt_betas

    print("Optimizer config:", opt_args)

    if opt_lower == "sgd" or opt_lower == "nesterov":
        opt_args.pop("eps", None)
        opt_args.pop("betas", None)
        optimizer = optim.SGD(parameters,
                              momentum=momentum,
                              nesterov=True,
                              **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    else:
        raise ValueError(f"Invalid optimizer: {opt_lower}")

    return optimizer
