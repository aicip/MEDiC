# --------------------------------------------------------
# MEDiC: Masked Enhanced Distillation with Contrastive Learning
# Layer Decay Optimizer Constructor for ViT Detection
# Implements per-layer learning rate decay for Vision Transformers
# --------------------------------------------------------
import json
from mmcv.runner import OPTIMIZER_BUILDERS, DefaultOptimizerConstructor
from mmcv.runner import get_dist_info


def get_num_layer_for_vit(var_name, num_max_layer):
    """Get layer index for a parameter name in ViT.

    Layer-wise learning rate decay assigns different learning rates to different
    layers. Early layers (closer to input) get smaller learning rates, later
    layers get larger learning rates.

    Args:
        var_name (str): Parameter name
        num_max_layer (int): Total number of layers (depth + 2)

    Returns:
        int: Layer index (0 = embeddings, 1-depth = transformer blocks, depth+1 = head)
    """
    # CLS token, mask token, position embedding -> layer 0 (lowest LR)
    if var_name in ("backbone.cls_token", "backbone.mask_token", "backbone.pos_embed",
                    "backbone.backbone.cls_token", "backbone.backbone.mask_token", "backbone.backbone.pos_embed"):
        return 0
    # Patch embedding -> layer 0
    elif var_name.startswith("backbone.patch_embed") or var_name.startswith("backbone.backbone.patch_embed"):
        return 0
    # Transformer blocks -> layers 1 to depth
    elif var_name.startswith("backbone.blocks") or var_name.startswith("backbone.backbone.blocks"):
        # Extract block index: "backbone.blocks.5.attn.qkv.weight" -> 5
        parts = var_name.split('.')
        for i, part in enumerate(parts):
            if part == 'blocks':
                layer_id = int(parts[i + 1])
                return layer_id + 1
    # Everything else (FPN, neck, head) -> last layer (highest LR)
    return num_max_layer - 1


@OPTIMIZER_BUILDERS.register_module()
class LayerDecayOptimizerConstructor(DefaultOptimizerConstructor):
    """Optimizer constructor with layer-wise learning rate decay.

    This constructor assigns different learning rates to different layers
    using layer decay rate. The formula is:
        lr_layer = base_lr * (layer_decay_rate ** (num_layers - layer_id - 1))

    Example with layer_decay_rate=0.75 and 12 transformer blocks (num_layers=14):
        - Layer 0 (embeddings):    0.75^13 ~ 0.025 * base_lr
        - Layer 1 (block 0):       0.75^12 ~ 0.032 * base_lr
        - Layer 6 (block 5):       0.75^7  ~ 0.133 * base_lr
        - Layer 12 (block 11):     0.75^1  ~ 0.750 * base_lr
        - Layer 13 (head):         0.75^0  ~ 1.000 * base_lr

    Args in paramwise_cfg:
        num_layers (int): Number of transformer blocks (default: 12)
        layer_decay_rate (float): Decay rate per layer (default: 0.75)
    """

    def add_params(self, params, module, prefix='', is_dcn_module=None):
        """Add all parameters of module to the params list with layer decay.

        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
            prefix (str): The prefix of the module
            is_dcn_module (int|float|None): If the current module is a
                submodule of DCN, `is_dcn_module` will be passed to
                control conv_offset layer's learning rate. Defaults to None.
        """
        parameter_groups = {}

        # Get configuration
        num_layers = self.paramwise_cfg.get('num_layers', 12) + 2  # +2 for embed and head
        layer_decay_rate = self.paramwise_cfg.get('layer_decay_rate', 0.75)
        weight_decay = self.base_wd

        rank, _ = get_dist_info()
        if rank == 0:
            print(f"Building LayerDecayOptimizerConstructor:")
            print(f"  layer_decay_rate={layer_decay_rate}")
            print(f"  num_layers={num_layers} (includes embed + head)")
            print(f"  base_lr={self.base_lr}")
            print(f"  base_wd={weight_decay}")

        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights

            # Determine weight decay (no decay for 1D params, biases, and special tokens)
            if len(param.shape) == 1 or name.endswith(".bias") or \
               any(token in name for token in ['pos_embed', 'cls_token', 'mask_token']):
                group_name = "no_decay"
                this_weight_decay = 0.
            else:
                group_name = "decay"
                this_weight_decay = weight_decay

            # Get layer index for this parameter
            layer_id = get_num_layer_for_vit(name, num_layers)
            group_name = f"layer_{layer_id}_{group_name}"

            if group_name not in parameter_groups:
                # Calculate learning rate scale
                scale = layer_decay_rate ** (num_layers - layer_id - 1)

                parameter_groups[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "param_names": [],
                    "lr_scale": scale,
                    "group_name": group_name,
                    "lr": scale * self.base_lr,
                }

            parameter_groups[group_name]["params"].append(param)
            parameter_groups[group_name]["param_names"].append(name)

        # Log parameter groups
        if rank == 0:
            to_display = {}
            for key in sorted(parameter_groups.keys()):
                group = parameter_groups[key]
                to_display[key] = {
                    "num_params": len(group["param_names"]),
                    "lr_scale": f"{group['lr_scale']:.4f}",
                    "lr": f"{group['lr']:.6f}",
                    "weight_decay": group["weight_decay"],
                }
            print(f"Parameter groups:\n{json.dumps(to_display, indent=2)}")

        params.extend(parameter_groups.values())
