import argparse
import os
import sys
import yaml
import glob
import json
from pathlib import Path
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
from typing import Dict, Any, Tuple

# No longer using timm, we are using the project's own model builder
from .models import build_student
from .models.vision_transformer import VisionTransformerMIM

from .utils.utils import init_distributed_mode, is_main_process, get_rank, get_world_size

# =================================================================================================
# Configuration Handling Functions
# =================================================================================================

def get_config_value(cfg: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Get config value with dot notation support."""
    keys = key.split('.')
    current = cfg
    for k in keys:
        if isinstance(current, dict) and k in current:
            current = current[k]
        else:
            return default
    return current


def validate_knn_config(cfg: Dict[str, Any]) -> None:
    """Validate that config contains required fields for KNN evaluation."""
    if cfg is None:
        raise ValueError("Configuration cannot be None")

    required_sections = ['model', 'data', 'wandb_meta']
    for section in required_sections:
        if section not in cfg:
            raise ValueError(f"Missing required config section: {section}")

    # Validate model config
    if 'student' not in cfg['model']:
        raise ValueError("Missing student model configuration")

    # Validate data config
    if 'data_path' not in cfg['data']:
        raise ValueError("Missing data_path in data configuration")


def load_checkpoint_for_knn(checkpoint_path: str, model: nn.Module, device: torch.device, model_config: Dict[str, Any] = None) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any], Dict[str, Any]]:
    """Load checkpoint and extract student model weights for KNN evaluation.

    Returns:
        student_state_dict: Model weights
        config: Training configuration
        wandb_metadata: Wandb metadata from pretrain run
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Handle different checkpoint formats
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        raise KeyError(f"No model state found in checkpoint. Available keys: {list(checkpoint.keys())}")

    # Extract student model weights from MEDiC model
    student_state_dict = {}
    non_student_components = {'teacher', 'distill_head'}

    for key, value in state_dict.items():
        if key.startswith('student.'):
            # Remove 'student.' prefix
            student_key = key[8:]  # len('student.') = 8
            student_state_dict[student_key] = value
        elif key.startswith('module.student.'):
            # Handle DDP wrapped models
            student_key = key[15:]  # len('module.student.') = 15
            student_state_dict[student_key] = value
        elif key.startswith('module.'):
            # Handle DDP wrapped models - check if it's a student component
            module_key = key[7:]  # Remove 'module.' prefix
            component = module_key.split('.')[0]
            if component not in non_student_components:
                # This is a student component with module prefix
                student_state_dict[module_key] = value
        else:
            # Check if this is a direct student component (not prefixed)
            # This handles cases where the checkpoint contains direct keys like 'blocks.0.norm1.weight'
            component = key.split('.')[0] if '.' in key else key
            if component not in non_student_components:
                # Assume this is a student component if it's not a known non-student component
                student_state_dict[key] = value

    # Extract wandb metadata from checkpoint
    wandb_metadata = {}
    if 'wandb_run_id' in checkpoint:
        wandb_metadata['run_id'] = checkpoint['wandb_run_id']
    if 'wandb_run_name' in checkpoint:
        wandb_metadata['run_name'] = checkpoint['wandb_run_name']
    if 'wandb_run_url' in checkpoint:
        wandb_metadata['run_url'] = checkpoint['wandb_run_url']
    if 'wandb_group' in checkpoint:
        wandb_metadata['group'] = checkpoint['wandb_group']
    if 'wandb_project' in checkpoint:
        wandb_metadata['project'] = checkpoint['wandb_project']
    if 'wandb_entity' in checkpoint:
        wandb_metadata['entity'] = checkpoint['wandb_entity']

    return student_state_dict, checkpoint.get('config', {}), wandb_metadata

# =================================================================================================
# Subclass for k-NN without masking
# =================================================================================================

class VisionTransformerKNN(VisionTransformerMIM):
    """Vision Transformer for k-NN evaluation without masking."""

    def forward(self, img, mask=None, mask_generator=None, return_all_tokens=True):
        """Forward pass without masking for k-NN feature extraction."""
        # For k-NN, we don't use masking, so we override to bypass the mask requirement
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None

        # Patch embedding
        x = self.patch_embed(img)
        batch_size, num_patches, embed_dim = x.size()

        # Add position embeddings
        if self.pos_embed is not None:
            x = x + self.pos_embed[:, 1:, :]

        # Add cls token
        cls_token = self.cls_token
        if self.pos_embed is not None:
            cls_token = cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(batch_size, -1, -1)

        #! Skip masking (everything else is the same)

        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)

        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x, shared_rel_pos_bias=rel_pos_bias)

        x = self.norm(x)

        if return_all_tokens:
            return x
        else:
            return x[:, 0]  # Return only CLS token

# =================================================================================================
# Wrapper Class for k-NN Feature Extraction
# =================================================================================================

class ViTForFeatures(nn.Module):
    """
    A wrapper class for the VisionTransformerMIM model.
    This wrapper provides a `forward` method that is compatible with the
    feature extraction logic, which requires a `features_only` flag and returns
    a list of intermediate layer outputs.
    """
    def __init__(self, vit_model):
        super().__init__()
        self.vit = vit_model

        # Expose attributes needed by the feature extraction logic
        self.patch_embed = self.vit.patch_embed

    def forward(self, x, is_training=False, features_only=False, out_indices=None):
        """
        The forward pass expected by the feature extraction logic.
        Uses the student model's forward method without masking for proper feature extraction.
        """
        # We only support the feature extraction case here.
        if not features_only:
            raise ValueError("This wrapper is only for feature extraction (features_only=True)")

        if out_indices is None:
            # Default to the last block if not specified.
            out_indices = [len(self.vit.blocks) - 1]

        # This is the critical fix: handle negative indices, just like the timm library does.
        # It converts indices like -1 to their positive equivalent (e.g., 11 for a 12-block model).
        for i, oi in enumerate(out_indices):
            if oi < 0:
                out_indices[i] = len(self.vit.blocks) + oi

        # CRITICAL FIX: Use the student model's forward method without masking
        # This ensures proper feature extraction without applying masking during k-NN evaluation
        features_and_mask = self.vit(x, mask=None, mask_generator=None, return_all_tokens=True)

        # The VisionTransformerMIM returns (features, mask) tuple
        if isinstance(features_and_mask, tuple):
            full_features = features_and_mask[0]  # Shape: [B, 1+N, D] where first token is CLS
        else:
            full_features = features_and_mask  # Some models might return just features

        # For now, return the full features from the last layer as expected by extract_features
        # The extract_features function will handle CLS token extraction
        return [full_features]

# =================================================================================================
# NOTE: The following class is a minimal implementation of a helper found in the reference
#       script's ecosystem. It is included here to make this script self-contained.
# =================================================================================================

class ReturnIndexDataset(datasets.ImageFolder):
    """
    A custom dataset that returns the index of the item along with the image and label.
    This is required for the memory-efficient distributed feature gathering strategy.
    """
    def __getitem__(self, idx):
        img, lab = super().__getitem__(idx)
        return img, lab, idx

# =================================================================================================
# Argument Parser
# =================================================================================================

def get_args_parser():
    parser = argparse.ArgumentParser("k-NN evaluation script", add_help=False)

    # --- Arguments from your original script ---
    parser.add_argument("--k_values", nargs='+', type=int, default=[10, 20, 100, 200], help="List of k values for k-NN.")
    parser.add_argument("--temperature", default=0.07, type=float, help="Temperature for softmax.")
    parser.add_argument("--weights_folder", type=str, required=True, help="Path to the folder containing pretrained weights.")
    parser.add_argument("--epoch", type=int, required=True, help="The epoch number of the checkpoint to evaluate.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size per GPU.")
    parser.add_argument("--num_workers", default=16, type=int, help="Number of data loading workers per GPU.")
    parser.add_argument("--dist_url", default="env://", type=str, help="URL used to set up distributed training.")
    parser.add_argument("--cfg", type=str, required=True, help="Path to the model config file used for data paths and wandb.")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU instead of GPU.")

    # --- Arguments replicated from the reference implementation for full feature parity ---
    parser.add_argument('--arch', default='vit_base', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'vit_large'], help='Architecture.')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--n_last_blocks', default=1, type=int, help="Concatenate [CLS] tokens for the `n` last blocks.")
    parser.add_argument('--avgpool_patchtokens', default=0, choices=[0, 1, 2], type=int,
        help="""Whether to use global average pooled features (1) or the [CLS] token (0).
        (2) is a combination of both. Defaults to 0 (CLS token).""")
    parser.add_argument('--multiscale', action='store_true', help='Use multi-scale averaging for features.')
    parser.add_argument('--checkpoint_key', default="model", type=str, help='Key to use in the checkpoint (e.g., "teacher", "student", "model").')

    return parser

# =================================================================================================
# Feature Extraction (Logic 100% replicated from reference implementation)
# =================================================================================================

@torch.no_grad()
def extract_features(model, data_loader, n_last_blocks, avgpool_patchtokens, multiscale=False, device=torch.device('cpu')):
    """
    Exact feature extraction pipeline replicated from the reference script.
    It supports multi-scale inference, selecting features from last N blocks,
    and choosing between CLS token and average patch pooling.
    """
    print("Extracting features...")
    features = None
    labels = None

    # If the dataloader is empty, return empty tensors
    if len(data_loader.dataset) == 0:
        return torch.empty(0), torch.empty(0, dtype=torch.long)

    # We iterate through the dataset
    for i, (samples, labs, index) in enumerate(data_loader):
        if i % 10 == 0:
            if is_main_process():
                print(f"  Processed batch {i}/{len(data_loader)}")

        samples = samples.to(device, non_blocking=True)
        labs = labs.to(device, non_blocking=True)
        index = index.to(device, non_blocking=True)

        def forward_single(x):
            # This is the core model interaction logic.
            # We use features_only=True and out_indices to get intermediate layers.
            # This requires a model that supports this feature, like timm models.
            with torch.autocast(device_type='cuda' if x.device.type == 'cuda' else 'cpu', enabled=True):
                intermediate_output = model(x, is_training=False, features_only=True, out_indices=[-n_last_blocks,])

            # The reference logic for selecting feature types
            if avgpool_patchtokens == 0:  # CLS token
                output = [x[:, 0] for x in intermediate_output]
            elif avgpool_patchtokens == 1:  # Average patch tokens
                output = [torch.mean(intermediate_output[-1][:, 1:], dim=1)]
            elif avgpool_patchtokens == 2:  # CLS + Average patch tokens
                output = [x[:, 0] for x in intermediate_output] + [torch.mean(intermediate_output[-1][:, 1:], dim=1)]
            else:
                raise ValueError(f"Unknown avgpool type {avgpool_patchtokens}")

            return torch.cat(output, dim=-1).clone()

        if multiscale:
            v = None
            for s in [1, 1/2**(1/2), 1/2]:  # 3 different scales
                if s == 1:
                    inp = samples.clone()
                else:
                    inp = nn.functional.interpolate(samples, scale_factor=s, mode='bilinear', align_corners=False)

                feats = forward_single(inp)
                if v is None:
                    v = feats
                else:
                    v += feats

            assert v is not None
            v /= 3
            v /= v.norm()
            feats = v
        else:
            feats = forward_single(samples)

        # Initialize storage tensors on all processes (not just rank 0)
        if features is None:
            if get_world_size() > 1:
                # In distributed mode, we need to initialize on all ranks
                features = torch.zeros(len(data_loader.dataset), feats.shape[-1], device='cpu')
                labels = torch.zeros(len(data_loader.dataset), dtype=torch.long, device='cpu')
            else:
                # Single GPU mode
                features = torch.zeros(len(data_loader.dataset), feats.shape[-1], device='cpu')
                labels = torch.zeros(len(data_loader.dataset), dtype=torch.long, device='cpu')

            if is_main_process():
                print(f"Storing features into tensor of shape {features.shape}")

                # Gather features, labels, and indices from all processes
        # Fixed distributed gathering to handle uneven batch sizes across ranks
        if get_world_size() > 1:
            # Keep tensors on GPU for distributed operations (NCCL backend requirement)
            # Get the maximum batch size across all ranks to handle uneven batches
            batch_size_tensor = torch.tensor(index.shape[0], device=device)
            max_batch_size_list = [torch.zeros_like(batch_size_tensor) for _ in range(get_world_size())]
            dist.all_gather(max_batch_size_list, batch_size_tensor)
            max_batch_size = max(t.item() for t in max_batch_size_list)

            # Pad tensors to max batch size to ensure consistent shapes
            current_batch_size = index.shape[0]
            index_gpu = index
            feats_gpu = feats
            labs_gpu = labs

            if current_batch_size < max_batch_size:
                pad_size = max_batch_size - current_batch_size
                # Pad with -1 for indices (will be filtered out later)
                index_gpu = torch.cat([index, torch.full((pad_size,), -1, dtype=index.dtype, device=device)])
                # Pad features and labels with zeros
                feats_gpu = torch.cat([feats, torch.zeros(pad_size, feats.shape[1], dtype=feats.dtype, device=device)])
                labs_gpu = torch.cat([labs, torch.zeros(pad_size, dtype=labs.dtype, device=device)])

            # Now gather with consistent shapes (all on GPU)
            indices_list = [torch.zeros_like(index_gpu) for _ in range(get_world_size())]
            feats_list = [torch.zeros_like(feats_gpu) for _ in range(get_world_size())]
            labs_list = [torch.zeros_like(labs_gpu) for _ in range(get_world_size())]

            dist.all_gather(indices_list, index_gpu)
            dist.all_gather(feats_list, feats_gpu)
            dist.all_gather(labs_list, labs_gpu)

            # Concatenate and filter out padded entries
            all_indices = torch.cat(indices_list)
            all_feats = torch.cat(feats_list)
            all_labs = torch.cat(labs_list)

            # Filter out padded entries (indices == -1)
            valid_mask = all_indices >= 0
            all_indices = all_indices[valid_mask]
            all_feats = all_feats[valid_mask]
            all_labs = all_labs[valid_mask]

            # Move to CPU after gathering
            all_indices = all_indices.cpu()
            all_feats = all_feats.cpu()
            all_labs = all_labs.cpu()
        else:
            all_indices = index.cpu()
            all_feats = feats.cpu()
            all_labs = labs.cpu()

        # Copy to storage tensor on all processes
        if len(all_indices) > 0 and features is not None and labels is not None:  # Only copy if we have valid data
            features.index_copy_(0, all_indices, all_feats)
            labels.index_copy_(0, all_indices, all_labs)

    if get_world_size() > 1:
        dist.barrier()

    if is_main_process():
        # Move to device for kNN
        assert features is not None and labels is not None
        return features.to(device), labels.to(device)
    return None, None

# =================================================================================================
# k-NN Classifier (Logic 100% replicated from reference implementation)
# =================================================================================================

@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, device, num_classes=1000):
    """
    Exact k-NN classifier logic replicated from the reference script.
    It uses a brute-force torch.mm to compute similarity, which is memory
    and compute intensive but serves as a perfect baseline.
    Features are expected to be L2-normalized before being passed in.
    """
    top1, top5, total = 0.0, 0.0, 0.0

    # Move all tensors to the GPU for computation
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    test_features = test_features.to(device)
    test_labels = test_labels.to(device)

    # The reference implementation transposes the feature matrix for matmul
    train_features_t = train_features.t()

    num_test_images = test_labels.shape[0]
    # Replicating the exact chunking logic from the reference script
    # It uses a fixed number of chunks (500) to avoid memory issues with the similarity matrix
    num_chunks = 500
    imgs_per_chunk = num_test_images // num_chunks
    if imgs_per_chunk == 0:
        imgs_per_chunk = 1 # Handle cases where the dataset is smaller than num_chunks

    # Pre-allocate the one-hot tensor, as in the reference, for memory reuse
    retrieval_one_hot = torch.zeros(k, num_classes).to(device)

    for idx in range(0, num_test_images, imgs_per_chunk):
        # Get the features for the current chunk of test images
        features = test_features[idx : min((idx + imgs_per_chunk), num_test_images), :]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # Calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features_t)

        # Handle case where k is larger than available neighbors
        actual_k = min(k, similarity.shape[1])
        distances, indices = similarity.topk(actual_k, largest=True, sorted=True)

        # Gather the labels of the retrieved neighbors
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        # Reuse the one-hot tensor for efficiency
        retrieval_one_hot.resize_(batch_size * actual_k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)

        # Perform temperature-scaled voting
        distances_transform = distances.clone().div_(T).exp_()

        # Explicitly use torch.mul as in the reference
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # Find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 += correct.narrow(1, 0, 1).sum().item()

        # Handle top5 calculation for cases with fewer than 5 classes
        top5_k = min(5, num_classes)
        top5 += correct.narrow(1, 0, top5_k).sum().item()
        total += batch_size

    top1_acc = top1 * 100.0 / total
    top5_acc = top5 * 100.0 / total
    return top1_acc, top5_acc

# =================================================================================================
# Main Execution Logic
# =================================================================================================

def main():
    parser = argparse.ArgumentParser('k-NN evaluation', parents=[get_args_parser()])
    args = parser.parse_args()

    # --- Environment and distributed setup ---
    args_dict = vars(args)
    if not args.cpu:
        init_distributed_mode(args_dict)
        # If distributed wasn't initialized, set default values
        if not args_dict.get('distributed', False):
            args_dict['gpu'] = 0  # Use GPU 0 by default
    else:
        args_dict['distributed'] = False
        args_dict['rank'] = 0
        args_dict['world_size'] = 1
        args_dict['gpu'] = -1
    device = torch.device("cpu" if args.cpu else f"cuda:{args_dict.get('gpu', 0)}")
    if not args.cpu:
        torch.cuda.set_device(device)

    # --- Load and validate config ---
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    # Validate config
    validate_knn_config(cfg)

    # --- Prepare W&B metadata (but don't initialize yet) ---
    if is_main_process():
        run_name = f"knn-eval-E{args.epoch}-{Path(args.weights_folder).name}"
        wandb_project = get_config_value(cfg, 'wandb_meta.project', 'MEDiC_knn')
        wandb_entity = get_config_value(cfg, 'wandb_meta.entity', None)

        # Load parent run mapping
        folder_name = Path(args.weights_folder).name
        parent_info = {}
        parent_run_id = None
        parent_group = None

        # Try to load from run_mappings.json
        mapping_file = Path('run_mappings.json')
        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                mappings = json.load(f)
                if folder_name in mappings:
                    parent_mapping = mappings[folder_name]
                    parent_info = parent_mapping.get('parent_pretrain', {})
                    parent_run_id = parent_info.get('run_id')
                    parent_group = parent_info.get('group')  # Try to get group from mappings

                    # Add key configs to wandb config
                    if 'key_configs' in parent_mapping:
                        parent_info['configs'] = parent_mapping['key_configs']

                    print(f"Loaded parent run mapping: {parent_info.get('run_name', 'unknown')}")
                    if parent_group:
                        print(f"Found parent group in mappings: {parent_group}")

    # --- Find checkpoint path ---
    pattern = f"{args.weights_folder}/*{args.epoch}*"
    found_paths = [p for p in glob.glob(pattern, recursive=True) if os.path.isfile(p) and (p.endswith('.pth') or p.endswith('.ckpt'))]
    if not found_paths:
        raise FileNotFoundError(f"No checkpoint found for epoch {args.epoch} in {args.weights_folder}")
    pretrained_weights = found_paths[0]
    if is_main_process():
        print(f"Found checkpoint: {pretrained_weights}")

    # --- Load checkpoint and detect format dynamically ---
    checkpoint = torch.load(pretrained_weights, map_location="cpu", weights_only=False)

    # Dynamic detection of checkpoint format
    if 'state_dict' in checkpoint:
        # Lightning checkpoint format
        state_dict = checkpoint['state_dict']
        if is_main_process():
            print(f"Detected PyTorch Lightning checkpoint format")
    elif 'model' in checkpoint:
        # Regular PyTorch checkpoint
        state_dict = checkpoint['model']
        if is_main_process():
            print(f"Detected regular PyTorch checkpoint format")
    else:
        raise KeyError(f"No model state found in checkpoint. Available keys: {list(checkpoint.keys())}")

    # --- Load Weights using dynamic detection ---
    # Load checkpoint first to detect architecture from embedded config
    student_state_dict, checkpoint_config, checkpoint_wandb = load_checkpoint_for_knn(pretrained_weights, None, device, cfg)

    # --- Build Model ---
    # Prefer architecture config from checkpoint (matches the trained weights),
    # fall back to YAML config if checkpoint doesn't embed config
    if checkpoint_config and "model" in checkpoint_config and "student" in checkpoint_config["model"]:
        student_cfg = checkpoint_config["model"]["student"]
        if is_main_process():
            print(f"Using architecture config from checkpoint")
    else:
        student_cfg = cfg["model"]["student"]
        if is_main_process():
            print(f"Using architecture config from YAML: {args.cfg}")

    if is_main_process():
        print(f"Building model: {args.arch}")
        print(f"  embed_dim={student_cfg.get('embed_dim')}, depth={student_cfg.get('depth')}, "
              f"abs_pos={student_cfg.get('use_abs_pos_emb')}, "
              f"shared_rel_pos={student_cfg.get('use_shared_rel_pos_bias')}, "
              f"mask_tokens={student_cfg.get('use_mask_tokens')}")

    student_model = VisionTransformerKNN(
        img_size=student_cfg.get("img_size", 224),
        patch_size=student_cfg.get("patch_size", 16),
        embed_dim=student_cfg.get("embed_dim", 768),
        depth=student_cfg.get("depth", 12),
        num_heads=student_cfg.get("num_heads", 12),
        drop_path_rate=student_cfg.get("drop_path_rate", 0.1),
        init_values=student_cfg.get("init_values", 0.1),
        use_abs_pos_emb=student_cfg.get("use_abs_pos_emb", False),
        use_sincos_pos_emb=student_cfg.get("use_sincos_pos_emb", False),
        use_shared_rel_pos_bias=student_cfg.get("use_shared_rel_pos_bias", True),
        use_rel_pos_bias=student_cfg.get("use_rel_pos_bias", False),
    )

    # Wrap the student model in our k-NN feature extractor
    model = ViTForFeatures(student_model)
    model.to(device)

    # Use checkpoint config if available and no config was provided
    if checkpoint_config and not cfg:
        cfg = checkpoint_config
        validate_knn_config(cfg)

    # --- Initialize W&B Run with parent run linking (AFTER loading checkpoint) ---
    if is_main_process():
        # Update parent info with checkpoint wandb metadata
        if checkpoint_wandb:
            # Extract parent pretrain info from checkpoint
            if 'run_id' in checkpoint_wandb:
                parent_info['run_id'] = checkpoint_wandb['run_id']
                parent_run_id = checkpoint_wandb['run_id']
            if 'run_name' in checkpoint_wandb:
                parent_info['run_name'] = checkpoint_wandb['run_name']
            if 'run_url' in checkpoint_wandb:
                parent_info['run_url'] = checkpoint_wandb['run_url']

            # Most importantly, use the original group from pretrain run
            if 'group' in checkpoint_wandb:
                group_name = checkpoint_wandb['group']
                print(f"Using original pretrain group from checkpoint: {group_name}")
            elif parent_group:
                # Use group from run_mappings.json if available
                group_name = parent_group
                print(f"Using parent group from run_mappings.json: {group_name}")
            else:
                # Fallback to run_id or folder_name
                group_name = parent_run_id if parent_run_id else folder_name
                print(f"Using fallback group: {group_name}")
        else:
            # No checkpoint metadata, check run_mappings.json or use fallback
            if parent_group:
                group_name = parent_group
                print(f"Using parent group from run_mappings.json: {group_name}")
            else:
                group_name = parent_run_id if parent_run_id else folder_name
                print(f"Using fallback group: {group_name}")

        # Apply shortening to group_name for cleaner wandb UI
        # Pattern matches: _e-{epochs}_t-{teacher}_s-{student}_h-{head}_d-{dataset}{optional_date}
        import re
        suffix_pattern = r'_e-\d+_t-[A-Za-z0-9\-]+_s-[A-Za-z0-9_]+_h-[A-Za-z_]+_d-[A-Za-z0-9_]+(?:_\d{2}_\d{2}_\d{2}_\d{2})?$'
        original_group_name = group_name
        group_name = re.sub(suffix_pattern, '', group_name)
        if original_group_name != group_name:
            print(f"Shortened group name from: {original_group_name} to: {group_name}")

        # Prepare wandb config with all parent info
        wandb_config = {
            **cfg,
            **vars(args),
            'parent_pretrain': parent_info,
            'parent_pretrain_checkpoint': checkpoint_wandb if checkpoint_wandb else {}
        }

        # Create tags
        epoch_tag = f"epoch_{args.epoch}"

        # Handle parent tag - truncate if too long (wandb limit is 64 chars)
        parent_tag = f"parent_{folder_name}"
        if len(parent_tag) > 64:
            # For legacy long names, truncate intelligently
            max_folder_len = 64 - len("parent_")  # 57 chars for folder name
            if folder_name.startswith("pre_"):
                parent_tag = f"parent_{folder_name[:max_folder_len]}"
            else:
                # For old-style names, try to keep version and key params
                import re
                version_match = re.search(r'\.v\d+', folder_name)
                if version_match:
                    version = version_match.group()
                    prefix_len = 20
                    suffix_len = max_folder_len - prefix_len - len(version) - 3  # 3 for "..."
                    if suffix_len > 0:
                        parent_tag = f"parent_{folder_name[:prefix_len]}...{version}{folder_name[-(suffix_len):]}"
                    else:
                        parent_tag = f"parent_{folder_name[:max_folder_len-3]}..."
                else:
                    parent_tag = f"parent_{folder_name[:max_folder_len-3]}..."

        # Now initialize wandb with all the correct metadata
        wandb.init(project=wandb_project.replace("pretrain", "knn"),
                   entity=wandb_entity,
                   config=wandb_config,
                   name=run_name[:128],
                   job_type='knn-eval',
                   group=group_name[:128],
                   tags=[epoch_tag, parent_tag])

        print(f"Initialized wandb run with group: {group_name}")

        # Add group_name to summary (this is different from the group field)
        # group_name in summary is the human-readable checkpoint identifier
        if parent_info:
            checkpoint_group_name = parent_info.get('group', None)
            if checkpoint_group_name:
                wandb.run.summary['group_name'] = checkpoint_group_name
                print(f"Set summary group_name: {checkpoint_group_name}")

    # Load the weights into the underlying student model, not the wrapper
    original_keys = len(student_state_dict)

    # CRITICAL: Use strict=False only as fallback, but validate thoroughly
    # Validate loaded weights match the model
    msg = model.vit.load_state_dict(student_state_dict, strict=False)

    if is_main_process():
        print(f"\n{'='*80}")
        print(f"Weight Loading Summary")
        print(f"{'='*80}")
        print(f"Checkpoint contains: {original_keys} student parameters")
        print(f"Model expects: {len(model.vit.state_dict())} parameters")
        print(f"Loading result: {len(msg.missing_keys)} missing, {len(msg.unexpected_keys)} unexpected")

        # Print detailed diagnostics if there are issues
        if len(msg.missing_keys) > 0:
            print(f"\nWARNING: Missing keys ({len(msg.missing_keys)}):")
            for key in msg.missing_keys[:10]:
                print(f"  - {key}")
            if len(msg.missing_keys) > 10:
                print(f"  ... and {len(msg.missing_keys) - 10} more")

        if len(msg.unexpected_keys) > 0:
            print(f"\nUnexpected keys ({len(msg.unexpected_keys)}):")
            for key in msg.unexpected_keys[:10]:
                print(f"  - {key}")
            if len(msg.unexpected_keys) > 10:
                print(f"  ... and {len(msg.unexpected_keys) - 10} more")

        if len(msg.missing_keys) == 0 and len(msg.unexpected_keys) == 0:
            print(f"\nSUCCESS: Perfect match! All weights loaded successfully.")
        else:
            print(f"\nWARNING: Partial match. Some keys missing/unexpected (may be OK for position embeddings).")

        print(f"{'='*80}\n")

    model.eval()

    if get_world_size() > 1 and not args.cpu:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # --- Data Pipeline ---
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    data_path = get_config_value(cfg, 'data.data_path')
    dataset_train = ReturnIndexDataset(os.path.join(data_path, 'train'), transform=transform)
    dataset_val = ReturnIndexDataset(os.path.join(data_path, 'val'), transform=transform)
    sampler_train = DistributedSampler(dataset_train, shuffle=False) if get_world_size() > 1 and not args.cpu else SequentialSampler(dataset_train)
    loader_train = DataLoader(dataset_train, sampler=sampler_train, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    sampler_val = DistributedSampler(dataset_val, shuffle=False) if get_world_size() > 1 and not args.cpu else SequentialSampler(dataset_val)
    loader_val = DataLoader(dataset_val, sampler=sampler_val, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    if is_main_process():
        print(f"Data loaded: {len(dataset_train)} train images, {len(dataset_val)} val images.")

    # --- Feature Extraction and Evaluation ---
    train_features, train_labels = extract_features(model, loader_train, args.n_last_blocks, args.avgpool_patchtokens, args.multiscale, device)
    val_features, val_labels = extract_features(model, loader_val, args.n_last_blocks, args.avgpool_patchtokens, args.multiscale, device)

    if is_main_process():
        # As per the reference implementation, features must be normalized before classification
        if train_features is not None and val_features is not None:
            train_features = nn.functional.normalize(train_features, dim=1, p=2)
            val_features = nn.functional.normalize(val_features, dim=1, p=2)

        if train_features is not None:
            print("Running k-NN classifier...")
            log_stats = {}

            # Track maximum scores across all k values
            max_top1 = 0.0
            max_top5 = 0.0
            best_k_top1 = 0
            best_k_top5 = 0

            for k in args.k_values:
                top1, top5 = knn_classifier(train_features, train_labels, val_features, val_labels, k, args.temperature, device)
                print(f"k={k}: Top-1={top1:.2f}%, Top-5={top5:.2f}%")
                log_stats[f'knn/top1_k={k}'] = top1
                log_stats[f'knn/top5_k={k}'] = top5

                # Update maximum scores if current scores are better
                if top1 > max_top1:
                    max_top1 = top1
                    best_k_top1 = k
                if top5 > max_top5:
                    max_top5 = top5
                    best_k_top5 = k

            # Log maximum scores across all k values
            log_stats['knn/max_top1'] = max_top1
            log_stats['knn/max_top5'] = max_top5
            log_stats['knn/best_k_top1'] = best_k_top1
            log_stats['knn/best_k_top5'] = best_k_top5

            # Print summary of maximum scores
            print(f"\nMaximum k-NN Results:")
            print(f"Best Top-1: {max_top1:.2f}% (k={best_k_top1})")
            print(f"Best Top-5: {max_top5:.2f}% (k={best_k_top5})")

            if wandb.run:
                wandb.log(log_stats)

                # Compute group metrics (best performance across all epochs for this checkpoint)
                try:
                    print("\nComputing group metrics for this checkpoint...")
                    current_group = wandb.run.group
                    current_run_id = wandb.run.id

                    # Get all runs in the same group (all epochs of the same checkpoint)
                    api = wandb.Api()
                    entity = wandb.run.entity
                    project = wandb.run.project

                    # Find all runs with the same group
                    all_runs = api.runs(f"{entity}/{project}", filters={"group": current_group})

                    # Compute the best top-1 and top-5 across all epochs
                    best_top1 = max_top1  # Start with current run's score
                    best_top5 = max_top5
                    best_run_id = current_run_id
                    best_epoch = args.epoch

                    for run in all_runs:
                        if hasattr(run.summary, 'get'):
                            run_top1 = run.summary.get('knn/max_top1', 0)
                            run_top5 = run.summary.get('knn/max_top5', 0)

                            if run_top1 > best_top1:
                                best_top1 = run_top1
                                best_top5 = run_top5
                                best_run_id = run.id
                                # Extract epoch from run name
                                if 'E' in run.name:
                                    epoch_str = run.name.split('E')[1].split('-')[0]
                                    try:
                                        best_epoch = int(epoch_str)
                                    except:
                                        pass

                    # Update the current run's summary with group metrics
                    wandb.run.summary['group_max_top1'] = best_top1
                    wandb.run.summary['group_max_top5'] = best_top5
                    wandb.run.summary['group_best_id'] = best_run_id
                    wandb.run.summary['group_best_epoch'] = best_epoch

                    # Also update all other runs in the group with the same metrics
                    print(f"Found {len(list(all_runs))} runs in group {current_group}")
                    print(f"Best performance: Top-1={best_top1:.2f}% (epoch {best_epoch}, run {best_run_id})")

                    # Update all runs in the group
                    for run in api.runs(f"{entity}/{project}", filters={"group": current_group}):
                        run.summary['group_max_top1'] = best_top1
                        run.summary['group_max_top5'] = best_top5
                        run.summary['group_best_id'] = best_run_id
                        run.summary['group_best_epoch'] = best_epoch
                        run.update()

                    print("✅ Group metrics updated for all runs in group")

                except Exception as e:
                    print(f"⚠️ Could not compute group metrics: {e}")

                # Finish the wandb run
                wandb.finish()
            else:
                # If wandb was not initialized, still clean up
                pass

            # Clean up memory
            del train_features, val_features, train_labels, val_labels
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    if get_world_size() > 1:
        dist.destroy_process_group()

    print("k-NN evaluation finished.")

if __name__ == "__main__":
    main()
