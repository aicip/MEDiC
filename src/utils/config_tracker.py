"""
Config usage tracking and validation system.

This module provides tools to track configuration parameter usage throughout
the training pipeline and validate that all parameters are properly used.
"""

import copy
import json
import traceback
from collections import defaultdict, deque
from typing import Dict, Any, Set, List, Optional, Tuple, Union
from pathlib import Path
import yaml


class ConfigUsageTracker:
    """
    Tracks configuration parameter usage and validates parameter types and ranges.

    This class monitors all config key accesses, validates parameter types and ranges,
    and reports unused or invalid config keys. It helps ensure that all configuration
    parameters are properly used and propagated through the training pipeline.
    """

    def __init__(self, config: Dict[str, Any], strict_mode: bool = True):
        """
        Initialize the config usage tracker.

        Args:
            config: The configuration dictionary to track
            strict_mode: If True, raise errors for invalid parameters. If False, only warn.
        """
        self.original_config = copy.deepcopy(config)
        self.config = config
        self.accessed_keys: Set[str] = set()
        self.usage_stack: deque = deque(maxlen=100)  # Track recent usage
        self.strict_mode = strict_mode
        self.validation_errors: List[str] = []
        self.type_validations: Dict[str, Dict[str, Any]] = {}
        self.range_validations: Dict[str, Dict[str, Any]] = {}

        # Define parameter type validations
        self._setup_type_validations()
        self._setup_range_validations()

    def _setup_type_validations(self):
        """Setup type validation rules for config parameters."""
        self.type_validations = {
            # Model configuration
            "model.student.img_size": {"type": int, "description": "Image size must be integer"},
            "model.student.patch_size": {"type": int, "description": "Patch size must be integer"},
            "model.student.embed_dim": {"type": int, "description": "Embedding dimension must be integer"},
            "model.student.depth": {"type": int, "description": "Model depth must be integer"},
            "model.student.num_heads": {"type": int, "description": "Number of heads must be integer"},
            "model.student.drop_path_rate": {"type": (int, float), "description": "Drop path rate must be numeric"},
            "model.student.init_values": {"type": (int, float), "description": "Init values must be numeric"},
            "model.student.use_abs_pos_emb": {"type": bool, "description": "Use absolute pos emb must be boolean"},
            "model.student.use_rel_pos_bias": {"type": bool, "description": "Use relative pos bias must be boolean"},
            "model.student.use_shared_rel_pos_bias": {"type": bool, "description": "Use shared relative pos bias must be boolean"},
            "model.student.name": {"type": str, "description": "Student model name must be string"},

            # Teacher model configuration
            "model.teacher.name": {"type": str, "description": "Teacher model name must be string"},
            "teacher_model": {"type": str, "description": "Teacher model must be string"},

            # Decoder configuration
            "model.decoder.decoder_embed_dim": {"type": int, "description": "Decoder embed dim must be integer"},
            "model.decoder.decoder_depth": {"type": int, "description": "Decoder depth must be integer"},
            "model.decoder.decoder_num_heads": {"type": int, "description": "Decoder num heads must be integer"},
            "decoder_embed_dim": {"type": int, "description": "Decoder embed dim must be integer"},
            "decoder_depth": {"type": int, "description": "Decoder depth must be integer"},
            "decoder_heads": {"type": int, "description": "Decoder heads must be integer"},

            # Data configuration
            "data.batch_per_gpu": {"type": int, "description": "Batch per GPU must be integer"},
            "data.num_workers": {"type": int, "description": "Num workers must be integer"},
            "data.data_path": {"type": str, "description": "Data path must be string"},
            "data.train_dir": {"type": str, "description": "Train directory must be string"},
            "data.val_dir": {"type": str, "description": "Validation directory must be string"},
            "data.subset": {"type": (int, type(None)), "description": "Data subset must be integer or None"},
            "data.subset_start": {"type": int, "description": "Data subset start must be integer"},
            "batch_per_gpu": {"type": int, "description": "Batch per GPU must be integer"},
            "num_workers": {"type": int, "description": "Num workers must be integer"},

            # Masking configuration
            "mask.mask_ratio": {"type": (int, float), "description": "Mask ratio must be numeric"},
            "mask.mask_type": {"type": str, "description": "Mask type must be string"},

            # Optimization configuration
            "optim.lr": {"type": (int, float), "description": "Learning rate must be numeric"},
            "optim.wd": {"type": (int, float), "description": "Weight decay must be numeric"},
            "optim.betas": {"type": list, "description": "Betas must be list"},
            "optim.layer_decay": {"type": (int, float), "description": "Layer decay must be numeric"},
            "optim.grad_clip": {"type": (int, float), "description": "Gradient clipping must be numeric"},
            "lr": {"type": (int, float), "description": "Learning rate must be numeric"},
            "wd": {"type": (int, float), "description": "Weight decay must be numeric"},
            "betas": {"type": list, "description": "Betas must be list"},
            "layer_decay": {"type": (int, float), "description": "Layer decay must be numeric"},
            "grad_clip": {"type": (int, float), "description": "Gradient clipping must be numeric"},

            # Schedule configuration
            "schedule.min_lr": {"type": (int, float), "description": "Min LR must be numeric"},
            "schedule.warmup_epochs": {"type": int, "description": "Warmup epochs must be integer"},
            "schedule.warmup_start_lr": {"type": (int, float), "description": "Warmup start LR must be numeric"},
            "min_lr": {"type": (int, float), "description": "Min LR must be numeric"},
            "warmup_epochs": {"type": int, "description": "Warmup epochs must be integer"},
            "warmup_start_lr": {"type": (int, float), "description": "Warmup start LR must be numeric"},

            # Loss configuration
            "losses.use_head_loss": {"type": bool, "description": "Use head loss must be boolean"},
            "losses.head_loss_weight": {"type": (int, float), "description": "Head loss weight must be numeric"},
            "losses.head.type": {"type": str, "description": "Head loss type must be string"},
            "losses.head.beta": {"type": (int, float), "description": "Head loss beta must be numeric"},
            "losses.normalize_targets": {"type": bool, "description": "Normalize targets must be boolean"},
            "losses.normalize_predictions": {"type": bool, "description": "Normalize predictions must be boolean"},
            "losses.normalization_method": {"type": str, "description": "Normalization method must be string"},

            # Training configuration
            "epochs": {"type": int, "description": "Epochs must be integer"},
            "gpu": {"type": int, "description": "GPU must be integer"},
            "seed": {"type": int, "description": "Seed must be integer"},
            "precision": {"type": str, "description": "Precision must be string"},
            "log_interval": {"type": int, "description": "Log interval must be integer"},
            "output_dir": {"type": str, "description": "Output directory must be string"},

            # Visualization configuration
            "visualizations.enable": {"type": bool, "description": "Visualizations enable must be boolean"},
            "visualizations.layers_to_analyze": {"type": list, "description": "Layers to analyze must be list"},
            "visualizations.attention_heads_to_show": {"type": int, "description": "Attention heads to show must be integer"},
            "visualizations.plots.raw_attention": {"type": bool, "description": "Raw attention plot must be boolean"},
            "visualizations.plots.layer_evolution": {"type": bool, "description": "Layer evolution plot must be boolean"},
            "visualizations.plots.head_comparison": {"type": bool, "description": "Head comparison plot must be boolean"},
            "visualizations.plots.attention_rollout": {"type": bool, "description": "Attention rollout plot must be boolean"},

            # WandB configuration
            "wandb_meta.project": {"type": str, "description": "WandB project must be string"},
            "wandb_meta.entity": {"type": str, "description": "WandB entity must be string"},
            "wandb_meta.name": {"type": str, "description": "WandB name must be string"},
            "wandb_meta.desc": {"type": (str, type(None)), "description": "WandB description must be string or None"},

            # Distributed configuration
            "distributed": {"type": bool, "description": "Distributed must be boolean"},
            "dist_backend": {"type": str, "description": "Dist backend must be string"},
            "dist_url": {"type": str, "description": "Dist URL must be string"},
            "dist_on_itp": {"type": bool, "description": "Dist on ITP must be boolean"},
            "rank": {"type": int, "description": "Rank must be integer"},
            "world_size": {"type": int, "description": "World size must be integer"},
            "local_rank": {"type": int, "description": "Local rank must be integer"},

            # Evaluation configuration
            "k_values": {"type": list, "description": "K values must be list"},
            "temperature": {"type": (int, float), "description": "Temperature must be numeric"},
            "weights_folder": {"type": str, "description": "Weights folder must be string"},
            "epoch": {"type": int, "description": "Epoch must be integer"},
            "checkpoint_key": {"type": str, "description": "Checkpoint key must be string"},
            "batch_size": {"type": int, "description": "Batch size must be integer"},
            "pretrained_weights": {"type": str, "description": "Pretrained weights must be string"},

            # Legacy/alternative keys
            "student_model": {"type": str, "description": "Student model must be string"},
            "student_embed_dim": {"type": int, "description": "Student embed dim must be integer"},
            "depth": {"type": int, "description": "Depth must be integer"},
            "num_heads": {"type": int, "description": "Num heads must be integer"},
            "drop_path_rate": {"type": (int, float), "description": "Drop path rate must be numeric"},
            "use_abs_pos_emb": {"type": bool, "description": "Use absolute pos emb must be boolean"},
            "use_rel_pos_bias": {"type": bool, "description": "Use relative pos bias must be boolean"},
            "use_shared_rel_pos_bias": {"type": bool, "description": "Use shared relative pos bias must be boolean"},
            "init_values": {"type": (int, float), "description": "Init values must be numeric"},
        }

    def _setup_range_validations(self):
        """Setup range validation rules for config parameters."""
        self.range_validations = {
            # Model configuration
            "model.student.img_size": {"min": 32, "max": 1024, "description": "Image size must be between 32 and 1024"},
            "model.student.patch_size": {"min": 4, "max": 64, "description": "Patch size must be between 4 and 64"},
            "model.student.embed_dim": {"min": 64, "max": 4096, "description": "Embed dim must be between 64 and 4096"},
            "model.student.depth": {"min": 1, "max": 100, "description": "Depth must be between 1 and 100"},
            "model.student.num_heads": {"min": 1, "max": 128, "description": "Num heads must be between 1 and 128"},
            "model.student.drop_path_rate": {"min": 0.0, "max": 1.0, "description": "Drop path rate must be between 0 and 1"},
            "model.student.init_values": {"min": 0.0, "max": 1.0, "description": "Init values must be between 0 and 1"},

            # Decoder configuration
            "model.decoder.decoder_embed_dim": {"min": 64, "max": 4096, "description": "Decoder embed dim must be between 64 and 4096"},
            "model.decoder.decoder_depth": {"min": 1, "max": 100, "description": "Decoder depth must be between 1 and 100"},
            "model.decoder.decoder_num_heads": {"min": 1, "max": 128, "description": "Decoder num heads must be between 1 and 128"},
            "decoder_embed_dim": {"min": 64, "max": 4096, "description": "Decoder embed dim must be between 64 and 4096"},
            "decoder_depth": {"min": 1, "max": 100, "description": "Decoder depth must be between 1 and 100"},
            "decoder_heads": {"min": 1, "max": 128, "description": "Decoder heads must be between 1 and 128"},

            # Data configuration
            "data.batch_per_gpu": {"min": 1, "max": 1024, "description": "Batch per GPU must be between 1 and 1024"},
            "data.num_workers": {"min": 0, "max": 64, "description": "Num workers must be between 0 and 64"},
            "data.subset_start": {"min": 0, "max": 1000000, "description": "Data subset start must be between 0 and 1000000"},
            "data.augmentation.min_scale": {"min": 0.0, "max": 1.0, "description": "Min scale must be between 0 and 1"},
            "data.augmentation.max_scale": {"min": 0.0, "max": 2.0, "description": "Max scale must be between 0 and 2"},
            "data.augmentation.horizontal_flip_prob": {"min": 0.0, "max": 1.0, "description": "Horizontal flip probability must be between 0 and 1"},
            "data.augmentation.color_jitter": {"min": 0.0, "max": 1.0, "description": "Color jitter must be between 0 and 1"},
            "batch_per_gpu": {"min": 1, "max": 1024, "description": "Batch per GPU must be between 1 and 1024"},
            "num_workers": {"min": 0, "max": 64, "description": "Num workers must be between 0 and 64"},

            # Masking configuration
            "mask.mask_ratio": {"min": 0.0, "max": 1.0, "description": "Mask ratio must be between 0 and 1"},
            "mask_ratio": {"min": 0.0, "max": 1.0, "description": "Mask ratio must be between 0 and 1"},

            # Optimization configuration
            "optim.lr": {"min": 1e-8, "max": 1.0, "description": "Learning rate must be between 1e-8 and 1.0"},
            "optim.wd": {"min": 0.0, "max": 1.0, "description": "Weight decay must be between 0 and 1"},
            "optim.layer_decay": {"min": 0.0, "max": 1.0, "description": "Layer decay must be between 0 and 1"},
            "optim.grad_clip": {"min": 0.0, "max": 100.0, "description": "Gradient clipping must be between 0 and 100"},
            "lr": {"min": 1e-8, "max": 1.0, "description": "Learning rate must be between 1e-8 and 1.0"},
            "wd": {"min": 0.0, "max": 1.0, "description": "Weight decay must be between 0 and 1"},
            "layer_decay": {"min": 0.0, "max": 1.0, "description": "Layer decay must be between 0 and 1"},
            "grad_clip": {"min": 0.0, "max": 100.0, "description": "Gradient clipping must be between 0 and 100"},

            # Schedule configuration
            "schedule.min_lr": {"min": 0.0, "max": 1.0, "description": "Min LR must be between 0 and 1"},
            "schedule.warmup_epochs": {"min": 0, "max": 1000, "description": "Warmup epochs must be between 0 and 1000"},
            "schedule.warmup_start_lr": {"min": 1e-10, "max": 1.0, "description": "Warmup start LR must be between 1e-10 and 1.0"},
            "min_lr": {"min": 0.0, "max": 1.0, "description": "Min LR must be between 0 and 1"},
            "warmup_epochs": {"min": 0, "max": 1000, "description": "Warmup epochs must be between 0 and 1000"},
            "warmup_start_lr": {"min": 1e-10, "max": 1.0, "description": "Warmup start LR must be between 1e-10 and 1.0"},

            # Loss configuration
            "losses.head_loss_weight": {"min": 0.0, "max": 100.0, "description": "Head loss weight must be between 0 and 100"},
            "losses.head.beta": {"min": 0.1, "max": 10.0, "description": "Head loss beta must be between 0.1 and 10.0"},

            # Training configuration
            "epochs": {"min": 1, "max": 10000, "description": "Epochs must be between 1 and 10000"},
            "gpu": {"min": -1, "max": 128, "description": "GPU must be between -1 (CPU) and 128"},
            "seed": {"min": 0, "max": 2**32-1, "description": "Seed must be between 0 and 2^32-1"},
            "log_interval": {"min": 1, "max": 10000, "description": "Log interval must be between 1 and 10000"},

            # Visualization configuration
            "visualizations.attention_heads_to_show": {"min": 1, "max": 128, "description": "Attention heads to show must be between 1 and 128"},

            # Distributed configuration
            "rank": {"min": 0, "max": 10000, "description": "Rank must be between 0 and 10000"},
            "world_size": {"min": 1, "max": 10000, "description": "World size must be between 1 and 10000"},
            "local_rank": {"min": 0, "max": 10000, "description": "Local rank must be between 0 and 10000"},

            # Evaluation configuration
            "temperature": {"min": 0.01, "max": 10.0, "description": "Temperature must be between 0.01 and 10.0"},
            "epoch": {"min": 0, "max": 10000, "description": "Epoch must be between 0 and 10000"},
            "batch_size": {"min": 1, "max": 1024, "description": "Batch size must be between 1 and 1024"},

            # Legacy/alternative keys
            "student_embed_dim": {"min": 64, "max": 4096, "description": "Student embed dim must be between 64 and 4096"},
            "depth": {"min": 1, "max": 100, "description": "Depth must be between 1 and 100"},
            "num_heads": {"min": 1, "max": 128, "description": "Num heads must be between 1 and 128"},
            "drop_path_rate": {"min": 0.0, "max": 1.0, "description": "Drop path rate must be between 0 and 1"},
            "init_values": {"min": 0.0, "max": 1.0, "description": "Init values must be between 0 and 1"},
        }

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a config value and track its usage.

        Args:
            key_path: Dot-separated path to the config key
            default: Default value if key doesn't exist

        Returns:
            The config value

        Raises:
            KeyError: If key doesn't exist and no default provided
            ValueError: If parameter validation fails
        """
        # Check if key exists before tracking
        key_exists = self._key_exists_in_nested(key_path)

        # Only track usage if key exists or if we're using a default value
        if key_exists or default is not None:
            self.accessed_keys.add(key_path)
            self.usage_stack.append({
                "key": key_path,
                "timestamp": None,  # Could add timestamp if needed
                "stack_trace": traceback.format_stack()[-3:-1]  # Last few stack frames
            })

        # Get value from nested dict
        value = self._get_nested_value(key_path, default)

        # Validate if key exists
        if key_exists:
            self._validate_parameter(key_path, value)

        return value

    def _get_nested_value(self, key_path: str, default: Any) -> Any:
        """Get value from nested dictionary using dot notation."""
        keys = key_path.split('.')
        current = self.config

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                if default is not None:
                    return default
                raise KeyError(f"Config key '{key_path}' not found")

        return current

    def _key_exists_in_nested(self, key_path: str) -> bool:
        """Check if key exists in nested dictionary."""
        try:
            self._get_nested_value(key_path, None)
            return True
        except KeyError:
            return False

    def _validate_parameter(self, key_path: str, value: Any):
        """Validate parameter type and range."""
        # Type validation
        if key_path in self.type_validations:
            expected_type = self.type_validations[key_path]["type"]
            if not isinstance(value, expected_type):
                error_msg = f"Type validation failed for '{key_path}': expected {expected_type}, got {type(value)}. {self.type_validations[key_path]['description']}"
                self.validation_errors.append(error_msg)
                if self.strict_mode:
                    raise ValueError(error_msg)

        # Range validation
        if key_path in self.range_validations:
            validation = self.range_validations[key_path]
            if isinstance(value, (int, float)):
                if value < validation["min"] or value > validation["max"]:
                    error_msg = f"Range validation failed for '{key_path}': value {value} not in range [{validation['min']}, {validation['max']}]. {validation['description']}"
                    self.validation_errors.append(error_msg)
                    if self.strict_mode:
                        raise ValueError(error_msg)

    def get_unused_keys(self) -> Set[str]:
        """Get all config keys that were not accessed."""
        all_keys = self._get_all_keys(self.config)
        return all_keys - self.accessed_keys

    def _get_all_keys(self, config: Dict[str, Any], prefix: str = "") -> Set[str]:
        """Recursively get all keys from nested dictionary."""
        keys = set()
        for key, value in config.items():
            current_key = f"{prefix}.{key}" if prefix else key
            keys.add(current_key)
            if isinstance(value, dict):
                keys.update(self._get_all_keys(value, current_key))
        return keys

    def get_usage_report(self) -> Dict[str, Any]:
        """Generate a comprehensive usage report."""
        unused_keys = self.get_unused_keys()

        return {
            "total_keys": len(self._get_all_keys(self.config)),
            "accessed_keys": len(self.accessed_keys),
            "unused_keys": len(unused_keys),
            "unused_key_list": sorted(list(unused_keys)),
            "validation_errors": self.validation_errors,
            "recent_usage": list(self.usage_stack)[-10:],  # Last 10 usages
            "coverage_percentage": (len(self.accessed_keys) / len(self._get_all_keys(self.config))) * 100 if self._get_all_keys(self.config) else 0
        }

    def save_report(self, filepath: Union[str, Path]):
        """Save usage report to file."""
        report = self.get_usage_report()

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)

    def print_report(self):
        """Print usage report to console."""
        report = self.get_usage_report()

        # Check for unused keys in strict mode (warn instead of raising)
        if self.strict_mode and report['unused_keys'] > 0:
            print(f"WARNING: Unused config keys detected: {report['unused_keys']} keys were not accessed")
            print("Consider removing unused keys or using --config-tracker without --strict-config")

        print("=" * 60)
        print("CONFIG USAGE TRACKING REPORT")
        print("=" * 60)
        print(f"Total config keys: {report['total_keys']}")
        print(f"Accessed keys: {report['accessed_keys']}")
        print(f"Unused keys: {report['unused_keys']}")
        print(f"Coverage: {report['coverage_percentage']:.1f}%")

        if report['unused_key_list']:
            print("\nUnused keys:")
            for key in report['unused_key_list']:
                print(f"  - {key}")

        if report['validation_errors']:
            print("\nValidation errors:")
            for error in report['validation_errors']:
                print(f"  - {error}")

        if report['recent_usage']:
            print("\nRecent usage (last 10):")
            for usage in report['recent_usage']:
                print(f"  - {usage['key']}")

        print("=" * 60)


def track_config_usage(config: Dict[str, Any], strict_mode: bool = True) -> ConfigUsageTracker:
    """
    Create a config usage tracker for the given configuration.

    Args:
        config: The configuration dictionary to track
        strict_mode: If True, raise errors for invalid parameters

    Returns:
        ConfigUsageTracker instance
    """
    return ConfigUsageTracker(config, strict_mode)


def validate_config_file(filepath: Union[str, Path], strict_mode: bool = True) -> Tuple[Dict[str, Any], ConfigUsageTracker]:
    """
    Load and validate a config file.

    Args:
        filepath: Path to the config file
        strict_mode: If True, raise errors for invalid parameters

    Returns:
        Tuple of (config_dict, tracker)
    """
    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)

    tracker = ConfigUsageTracker(config, strict_mode)
    return config, tracker
