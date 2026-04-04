"""
MEDiC pretraining script.

Implements the MEDiC multi-objective training loop:
    L_total = w_head * L_head(H(S(I_masked)), N(T(I_full)))
            + w_cls  * L_cls(S_cls, T_cls)
            + w_pix  * L_pix(D(S(I_masked)), I)

Features:
  - Multi-objective loss: token distillation + CLS alignment + pixel reconstruction
  - Literal (static) loss weighting
  - Distributed training (DDP) with SLURM integration
  - Mixed precision (bf16/fp16) with GradScaler
  - Cosine LR schedule with warmup
  - Checkpoint saving and resuming
  - W&B logging and visualization
  - EMA model for evolved part masking

Reference: "MEDiC: Multi-objective Exploration of Distillation from CLIP" (arXiv:2603.29009)
"""

import argparse
import datetime
import os
import time
import random
import yaml
import json
import shutil
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any, cast
import glob
from io import BytesIO
from PIL import Image

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from torch.utils.data import DistributedSampler, DataLoader
import wandb
import matplotlib.pyplot as plt
from torchvision.transforms import Compose

# Local imports
from .utils.utils import (
    NativeScalerWithGradNormCount,
    is_main_process,
    cosine_scheduler,
    init_distributed_mode,
    get_rank,
    get_world_size,
)
from .utils.optim_factory import (
    get_parameter_groups,
    get_num_layer_for_vit,
    LayerDecayValueAssigner,
)
from .utils.losses import compute_loss
from .utils.viz import get_reconstruction_fig, plot_scheduler
from .data.loader import build_loader, MEDiCDataset
from .data.transforms import build_transform
from .models.medic_model import build_medic_model
from .models import build_teacher
from .utils.masking_generator import (
    BlockMaskingGenerator,
    create_masking_generator,
    get_masking_generator_info,
)

import numpy as np
from .utils.config_tracker import ConfigUsageTracker



def get_args() -> Dict[str, Any]:
    """Parse CLI arguments and load the YAML config file."""
    parser = argparse.ArgumentParser("MEDiC pre-train")
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--config-tracker", action="store_true",
                        help="Enable config usage tracking")
    parser.add_argument("--strict-config", action="store_true",
                        help="Raise on unused/invalid config keys")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--test-mode", action="store_true",
                        help="Quick verification (5 steps)")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)

    args = parser.parse_args()
    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["_config_path"] = args.cfg
    cfg["_config_filename"] = Path(args.cfg).name

    if args.resume:
        cfg["resume_path"] = args.resume

    if args.test_mode:
        cfg["test_mode"] = True
        cfg["max_steps"] = 5
        cfg["epochs"] = 1
        if is_main_process():
            print("Test mode enabled: 5 steps maximum")
    elif args.max_steps:
        cfg["max_steps"] = args.max_steps

    if args.epochs:
        cfg["epochs"] = args.epochs

    if args.config_tracker:
        cfg["_config_tracker"] = ConfigUsageTracker(cfg, strict_mode=args.strict_config)

    return cfg


def get_config_value(cfg: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Get a config value by dot-separated key path, with optional tracking."""
    if cfg is None:
        return default
    if "_config_tracker" in cfg:
        return cfg["_config_tracker"].get(key, default)
    keys = key.split(".")
    current = cfg
    for k in keys:
        if isinstance(current, dict) and k in current:
            current = current[k]
        else:
            return default
    return current



def setup_environment() -> Dict[str, Any]:
    """Set up DDP, output directories, seeds, and broadcast the config."""
    cfg = get_args()
    init_distributed_mode(cfg)

    local_gpu = get_config_value(cfg, "gpu")
    local_rank = get_config_value(cfg, "rank")

    cfg_list_to_broadcast: List[Optional[Dict[str, Any]]] = [None]
    if is_main_process():
        base_output_dir = Path(get_config_value(cfg, "output_dir", "output"))
        accum_iter = get_config_value(cfg, "optim.accum_iter", 1)
        batch_per_gpu = get_config_value(cfg, "data.batch_per_gpu")
        effective_batch_size = batch_per_gpu * accum_iter * get_world_size()
        print(f"GPUs: {get_world_size()}, Batch/GPU: {batch_per_gpu}, "
              f"Accum: {accum_iter}, Effective batch: {effective_batch_size}")

        data_root_suffix = Path(get_config_value(cfg, "data.data_path")).name
        try:
            teacher_name = get_config_value(cfg, "model.teacher.name").replace("/", "-")
        except (KeyError, AttributeError):
            teacher_name = "no-teacher"

        full_components = "_".join([
            f"e-{get_config_value(cfg, 'epochs')}", f"t-{teacher_name}",
            f"s-{get_config_value(cfg, 'model.student.name')}",
            "h-fc", f"d-{data_root_suffix}",
        ])
        base_name = get_config_value(cfg, "wandb_meta.name")
        timestamp = datetime.datetime.now().strftime("%y_%m_%d_%H")
        checkpoint_folder_name = f"{base_name}_{timestamp}"

        output_dir = base_output_dir / checkpoint_folder_name
        output_dir.mkdir(parents=True, exist_ok=True)
        cfg["output_dir"] = str(output_dir)
        cfg["_checkpoint_folder_name"] = checkpoint_folder_name
        cfg["_full_reference_name"] = f"{base_name}_{full_components}_{timestamp}"
        cfg["_base_name"] = base_name

        # Copy config to checkpoint directory
        shutil.copy2(Path(cfg["_config_path"]),
                      output_dir / cfg.get("_config_filename", "config.yaml"))

        # Scale LR linearly with effective batch size (base = 2048)
        cfg["optim"]["lr"] = get_config_value(cfg, "optim.lr") * effective_batch_size / 2048.0
        if "schedule" not in cfg:
            cfg["schedule"] = {}
        cfg["schedule"]["min_lr"] = float(get_config_value(cfg, "schedule.min_lr")) * effective_batch_size / 2048.0
        print(f"Scaled LR: {cfg['optim']['lr']:.6f}, min LR: {cfg['schedule']['min_lr']:.6f}")

        cfg_list_to_broadcast = [cfg]

    if cfg.get("distributed"):
        dist.broadcast_object_list(cfg_list_to_broadcast, src=0)

    cfg = cfg_list_to_broadcast[0]
    assert cfg is not None

    cfg["gpu"] = local_gpu
    cfg["rank"] = local_rank
    if not cfg.get("cpu", False) and torch.cuda.is_available():
        torch.cuda.set_device(cfg["gpu"])

    seed = get_config_value(cfg, "seed", 42) + get_rank()
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = True

    return cfg



def fig_to_pil(fig) -> Image.Image:
    """Convert a matplotlib Figure to a PIL Image.

    Respects the figure's DPI and size so logged images are not blurry.
    """
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=fig.dpi, bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf)


def init_wandb(
    cfg: Dict[str, Any], output_dir: Path,
    lr_schedule: np.ndarray, niter_per_ep: int,
):
    """Initialise W&B, save run mapping, and log the LR schedule plot."""
    if not is_main_process():
        return

    wandb_cfg = get_config_value(cfg, "wandb_meta")
    wandb_run_name = cfg.get("_base_name", output_dir.name)
    data_name = Path(get_config_value(cfg, "data.data_path")).name

    tags = [
        f"E:{get_config_value(cfg, 'epochs')}",
        f"B:{get_config_value(cfg, 'data.batch_per_gpu') * get_world_size()}",
        f"LR:{get_config_value(cfg, 'optim.lr'):.1e}",
        f"TEACH:{get_config_value(cfg, 'model.teacher.name', 'no-teacher')}",
        f"STDNT:{get_config_value(cfg, 'model.student.name')}",
        f"MASK:{get_config_value(cfg, 'mask.mask_ratio')}",
        f"DATA:{data_name}",
    ]

    import tempfile
    wandb.init(
        project=wandb_cfg["project"], entity=wandb_cfg.get("entity"),
        config=cfg, name=wandb_run_name, dir=tempfile.gettempdir(),
        tags=tags, job_type=cfg.get("mask", {}).get("mask_type", "block"),
        group=data_name,
    )

    # Save run mapping for linking eval runs (e.g. k-NN) to this pretrain run
    _update_run_mappings(cfg, output_dir)

    # Log LR schedule
    caption = f"Iters per epoch: {niter_per_ep}, Total steps: {len(lr_schedule)}"
    fig = plot_scheduler(lr_schedule.tolist(), niter_per_ep, caption)
    wandb.log({"Scheduler": wandb.Image(fig_to_pil(fig))}, step=0)
    plt.close(fig)


def _update_run_mappings(cfg: Dict[str, Any], output_dir: Path):
    """Persist wandb run ID to run_mappings.json for eval script linking."""
    if wandb.run is None:
        return

    mapping_file = Path("run_mappings.json")
    mappings = {}
    if mapping_file.exists():
        try:
            with open(mapping_file, "r") as f:
                mappings = json.load(f)
        except (json.JSONDecodeError, IOError):
            mappings = {}

    folder_name = output_dir.name
    base_name = cfg.get("_base_name", folder_name)
    mappings[folder_name] = {
        "folder_name": folder_name,
        "full_reference_name": cfg.get("_full_reference_name", folder_name),
        "parent_pretrain": {
            "run_id": wandb.run.id, "run_name": wandb.run.name,
            "run_url": wandb.run.url, "group": base_name,
            "project": wandb.run.project, "entity": wandb.run.entity,
        },
        "key_configs": {
            "mask_ratio": cfg.get("mask", {}).get("mask_ratio", 0.4),
            "mask_type": cfg.get("mask", {}).get("mask_type", "block"),
            "head_loss_weight": cfg.get("losses", {}).get("head_loss_weight", 1.0),
        },
    }

    try:
        temp_file = mapping_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(mappings, f, indent=2)
        temp_file.replace(mapping_file)
        print(f"Updated run_mappings.json: {folder_name} -> {wandb.run.id}")
    except Exception as e:
        print(f"Warning: Could not update run_mappings.json: {e}")



def build_models_optimizer_and_schedules(
    cfg: Dict[str, Any], device: torch.device,
    num_training_steps_per_epoch: int,
):
    """Build student, teacher, EMA model, optimiser, and LR/WD schedules."""
    # Teacher (frozen CLIP)
    teacher = build_teacher(cfg).to(device).eval()

    # Mask generator
    mask_info = get_masking_generator_info(cfg)
    print(f"Masking: {mask_info}")
    mask_generator = create_masking_generator(cfg)

    # MEDiC model (ViT + distill head + optional pixel decoder)
    model = build_medic_model(cfg, mask_generator=mask_generator)
    if cfg.get("cpu", False) or cfg["gpu"] == -1:
        model = model.to("cpu")
    else:
        model = model.to(get_config_value(cfg, "gpu"))

    # EMA model for evolved masking (wraps full MEDiCModel, not just student)
    model_ema = None
    mask_type = get_config_value(cfg, "mask.mask_type", "block")
    if mask_type == "evolved":
        ema_decay = get_config_value(cfg, "mask.evolved.ema_decay", 0.9999)
        try:
            from timm.utils import ModelEma
            model_ema = ModelEma(model, decay=ema_decay, device=device)
            print(f"EMA model created (decay={ema_decay}) for evolved masking")
        except ImportError:
            attn_source = get_config_value(cfg, "mask.evolved.attention_source", "clip_teacher")
            if attn_source == "student_ema":
                raise ImportError(
                    "timm.utils.ModelEma required for attention_source='student_ema'. "
                    "Install timm>=0.6.0 or use attention_source='clip_teacher'."
                )
            print("Warning: timm.utils.ModelEma not available. "
                  "EMA disabled (only clip_teacher attention source will work).")

    model_without_ddp = model
    if cfg.get("distributed"):
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[cfg["gpu"]], find_unused_parameters=False
        )
        model_without_ddp = model.module

    # Optimiser with layer-wise LR decay
    num_layers = model_without_ddp.student.get_num_layers()
    layer_decay = get_config_value(cfg, "optim.layer_decay", 1.0)
    assigner = LayerDecayValueAssigner(
        [layer_decay ** (num_layers - i) for i in range(num_layers + 1)]
    )
    param_groups = get_parameter_groups(
        model_without_ddp, get_config_value(cfg, "optim.wd"),
        get_num_layer=lambda name: get_num_layer_for_vit(name, num_layers),
        get_layer_scale=assigner.get_scale,
    )

    # Mirror BEiT v2: top-level weight_decay=0.0, actual WD per param group
    opt = AdamW(
        param_groups, lr=get_config_value(cfg, "optim.lr"),
        betas=tuple(get_config_value(cfg, "optim.betas")), weight_decay=0.0,
    )

    # LR and WD schedules
    max_steps = cfg.get("max_steps")
    if max_steps is not None:
        total_steps = max_steps + 1
        lr_schedule = np.full(total_steps, get_config_value(cfg, "optim.lr"), dtype=np.float32)
        wd_schedule = np.full(total_steps, get_config_value(cfg, "optim.wd"), dtype=np.float32)
    else:
        lr_schedule = cosine_scheduler(
            get_config_value(cfg, "optim.lr"), get_config_value(cfg, "schedule.min_lr"),
            get_config_value(cfg, "epochs"), num_training_steps_per_epoch,
            warmup_epochs=get_config_value(cfg, "schedule.warmup_epochs"),
            start_warmup_value=get_config_value(cfg, "schedule.warmup_start_lr", 1e-6),
        )
        wd_schedule = cosine_scheduler(
            get_config_value(cfg, "optim.wd"), 0,
            get_config_value(cfg, "epochs"), num_training_steps_per_epoch,
        )

    return model, model_without_ddp, teacher, model_ema, opt, lr_schedule, wd_schedule



def load_visualization_images(
    cfg: Dict[str, Any], device: torch.device,
) -> Optional[torch.Tensor]:
    """Load images from ``images/`` for epoch-level masking visualisation."""
    if not is_main_process():
        return None

    viz_transform = cast(Compose, build_transform(is_train=False, cfg=cfg))
    img_paths = sorted(glob.glob("images/*.JPEG"))
    if not img_paths:
        print("No images found in images/ for visualization.")
        return None

    try:
        images = [Image.open(p).convert("RGB") for p in img_paths]
        return torch.stack(
            cast(List[torch.Tensor], [viz_transform(im) for im in images])
        ).to(device)
    except Exception as e:
        print(f"Error loading visualization images: {e}")
        return None


def _generate_viz_masks(
    model_without_ddp: nn.Module, n_images: int, device: torch.device,
    cfg: Dict[str, Any],
) -> torch.Tensor:
    """Generate deterministic masks for visualization."""
    if hasattr(model_without_ddp, "mask_generator") and model_without_ddp.mask_generator is not None:
        gen = model_without_ddp.mask_generator
    else:
        grid = get_config_value(cfg, "model.student.img_size") // get_config_value(cfg, "model.student.patch_size")
        gen = BlockMaskingGenerator(grid_h=grid, grid_w=grid,
                                    mask_ratio=get_config_value(cfg, "mask.mask_ratio"))

    masks = []
    for _ in range(n_images):
        m = gen()
        masks.append(torch.from_numpy(m) if isinstance(m, np.ndarray) else m)
    return torch.stack(masks).to(device)


def log_visualizations(
    model_without_ddp: nn.Module, viz_images: Optional[torch.Tensor],
    epoch: int, device: torch.device, global_step: int, cfg: Dict[str, Any],
):
    """Generate and log masking + attention visualizations to W&B."""
    if not is_main_process() or viz_images is None:
        return

    # 1. Masking visualization
    rng_state = np.random.get_state()
    try:
        np.random.seed(42)
        viz_mask = _generate_viz_masks(model_without_ddp, viz_images.shape[0], device, cfg)
        fig = get_reconstruction_fig(model_without_ddp, (viz_images, None),
                                     viz_mask, epoch, n_images=viz_images.shape[0])
        wandb.log({"Masking": wandb.Image(fig_to_pil(fig))}, step=global_step)
        plt.close(fig)
    except Exception as e:
        print(f"ERROR generating visualization: {e}")
        import traceback; traceback.print_exc()
    finally:
        np.random.set_state(rng_state)

    # 2. Attention visualization (optional)
    try:
        from .utils.viz import log_attention_plots_to_wandb
        log_attention_plots_to_wandb(model_without_ddp, viz_images, cfg, epoch, global_step)
    except ImportError:
        pass
    except Exception as e:
        print(f"ERROR generating attention visualizations: {e}")



def handle_checkpointing(
    current_loss: float, epoch: int, model: nn.Module, opt: Optimizer,
    cfg: Dict[str, Any], output_dir: Path, best_loss: float,
    top_k_checkpoints: List[Tuple[float, Path]], specific_save_epochs: set,
    scaler: NativeScalerWithGradNormCount, global_step: int,
    teacher: Optional[nn.Module] = None,
    lr_schedule: Optional[np.ndarray] = None,
    wd_schedule: Optional[np.ndarray] = None,
    model_ema=None,
) -> Tuple[float, List[Tuple[float, Path]]]:
    """Save epoch-specific and best checkpoints. Returns updated (best_loss, top_k)."""
    saved: set = set()

    def _save(path: Path):
        if path in saved:
            return
        to_save: Dict[str, Any] = {
            "model": model.state_dict(), "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict(), "epoch": epoch,
            "global_step": global_step, "best_loss": best_loss,
            "top_k_checkpoints": top_k_checkpoints, "config": cfg,
        }
        if model_ema is not None:
            to_save["model_ema"] = model_ema.state_dict()
        if lr_schedule is not None:
            to_save["lr_schedule"] = lr_schedule
        if wd_schedule is not None:
            to_save["wd_schedule"] = wd_schedule
        if wandb.run is not None:
            to_save.update({
                "wandb_run_id": wandb.run.id, "wandb_run_url": wandb.run.url,
                "wandb_run_name": wandb.run.name, "wandb_group": output_dir.name,
                "wandb_project": wandb.run.project, "wandb_entity": wandb.run.entity,
            })
        torch.save(to_save, path, _use_new_zipfile_serialization=False)
        saved.add(path)

    if epoch in specific_save_epochs:
        _save(output_dir / f"checkpoint-epoch{epoch:04d}.pth")

    if current_loss < best_loss:
        best_loss = current_loss
        if epoch not in specific_save_epochs:
            new_path = output_dir / f"checkpoint-best-epoch{epoch:04d}-loss{current_loss:.4f}.pth"
            _save(new_path)
            top_k_checkpoints.append((current_loss, new_path))
            top_k_checkpoints.sort(key=lambda x: x[0])
            if len(top_k_checkpoints) > 3:
                _, old = top_k_checkpoints.pop()
                old.unlink(missing_ok=True)

    return best_loss, top_k_checkpoints


def load_checkpoint(
    checkpoint_path: str, model: nn.Module, optimizer: Optimizer,
    scaler: NativeScalerWithGradNormCount,
    teacher: Optional[nn.Module] = None,
    device: torch.device = torch.device("cpu"),
    model_ema=None,
) -> Dict[str, Any]:
    """Load checkpoint and restore training state (including EMA model if present)."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load to CPU first; weights_only=False for numpy schedule arrays
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = ["model", "optimizer", "scaler", "epoch", "global_step", "best_loss", "config"]
    missing = [k for k in required if k not in ckpt]
    if missing:
        raise ValueError(f"Checkpoint missing keys: {missing}")

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])

    # Restore EMA model if available (for evolved masking resume)
    if model_ema is not None and "model_ema" in ckpt:
        model_ema.load_state_dict(ckpt["model_ema"])
        print(f"Restored EMA model from checkpoint")

    state = {
        "epoch": ckpt["epoch"], "global_step": ckpt["global_step"],
        "best_loss": ckpt["best_loss"],
        "top_k_checkpoints": ckpt.get("top_k_checkpoints", []),
        "config": ckpt["config"],
        "lr_schedule": ckpt.get("lr_schedule"),
        "wd_schedule": ckpt.get("wd_schedule"),
    }
    print(f"Resumed from epoch {state['epoch']}, step {state['global_step']}, "
          f"best_loss={state['best_loss']:.6f}")
    return state


def validate_checkpoint_completeness(checkpoint_path: str) -> Dict[str, Any]:
    """Check that a checkpoint has all required components for resuming."""
    if not os.path.exists(checkpoint_path):
        return {"file_exists": False}
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        result = {"file_exists": True}
        for key in ["model", "optimizer", "scaler", "epoch", "global_step",
                     "best_loss", "config", "lr_schedule", "wd_schedule"]:
            result[key] = key in ckpt
        return result
    except Exception as e:
        return {"file_exists": True, "loadable": False, "error": str(e)}



def train_one_epoch(
    model: nn.Module, teacher: Optional[nn.Module],
    train_loader: DataLoader, opt: Optimizer,
    lr_schedule_values: np.ndarray, wd_schedule_values: np.ndarray,
    scaler: NativeScalerWithGradNormCount, epoch: int, global_step: int,
    dtype: torch.dtype, device: torch.device, cfg: Dict[str, Any],
    model_ema=None,
) -> Tuple[int, Dict[str, float]]:
    """Run one training epoch. Returns (updated_global_step, epoch_stats)."""
    model.train()

    epoch_start_time = time.time()
    iter_times = []
    epoch_samples = 0
    accum_iter = get_config_value(cfg, "optim.accum_iter", 1)

    opt.zero_grad()

    for i, (img, mask) in enumerate(train_loader):
        batch_start_time = time.time()
        it = len(train_loader) * epoch + i

        # Test mode early exit
        max_steps = cfg.get("max_steps")
        if max_steps is not None and global_step >= max_steps:
            if is_main_process():
                print(f"Reached maximum steps limit: {max_steps}")
            break

        # Update LR and WD at accumulation boundaries
        if i % accum_iter == 0:
            accum_step = it // accum_iter
            for param_group in opt.param_groups:
                param_group["lr"] = (
                    lr_schedule_values[min(accum_step, len(lr_schedule_values) - 1)]
                    * param_group.get("lr_scale", 1.0)
                )
                if param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[
                        min(accum_step, len(wd_schedule_values) - 1)
                    ]

        img = img.to(device, non_blocking=True)
        if mask is not None:
            mask = mask.to(device, non_blocking=True)
        epoch_samples += img.shape[0]

        assert isinstance(train_loader.dataset, MEDiCDataset)

        with torch.autocast(device_type=device.type, dtype=dtype):
            # Teacher forward: full image through frozen CLIP
            T = None
            if teacher is not None:
                T = teacher(img, return_cls=True)

            # MEDiC forward: encode + distill head + pixel decoder
            pred_tok, pred_pix, mask_out, ids_restore = model(
                img, mask, teacher=teacher, epoch=epoch, model_ema=model_ema,
            )
            if mask is None:
                mask = mask_out

            # In sparse mode, align teacher features to visible positions
            use_mask_tokens = get_config_value(
                cfg, "model.student.use_mask_tokens", True
            )
            if not use_mask_tokens and T is not None:
                T = _extract_visible_teacher_features(T, mask, model)

            # Shape assertions
            batch_size = img.shape[0]
            if T is not None and pred_tok is not None:
                assert T.shape[0] == batch_size
                assert T.shape[1] == pred_tok.shape[1], (
                    f"Teacher/student seq len mismatch: "
                    f"{T.shape[1]} vs {pred_tok.shape[1]}"
                )

            # Compute MEDiC multi-objective loss
            loss, l_dict = compute_loss(
                pred_tok=pred_tok,
                pred_pix=pred_pix,
                teacher_features=T,
                img=img,
                mask=mask,
                cfg=cfg,
            )

            # Scale loss for gradient accumulation
            loss = loss / accum_iter

        # Backward and optimise
        should_update_grad = (
            (i + 1) % accum_iter == 0 or (i + 1) == len(train_loader)
        )

        grad_norm = scaler(
            loss,
            opt,
            parameters=model.parameters(),
            clip_grad=get_config_value(cfg, "optim.grad_clip"),
            update_grad=should_update_grad,
            create_graph=False,
        )

        if should_update_grad:
            opt.zero_grad()
            # Update EMA model after optimizer step (for evolved masking)
            if model_ema is not None:
                model_ema.update(model)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        iter_times.append(time.time() - batch_start_time)

        # Logging (at accumulation boundaries)
        log_interval = cfg.get("log_interval", 25)
        if is_main_process() and should_update_grad and (i % log_interval == 0 or i == 0):
            cur_loss = loss.item() * accum_iter
            cur_lr = opt.param_groups[0]["lr"]
            speed = len(img) / (time.time() - batch_start_time) if (time.time() - batch_start_time) > 0 else 0
            print(
                f"  Epoch [{epoch}] Step [{i}/{len(train_loader)}] "
                f"loss={cur_loss:.4f} lr={cur_lr:.2e} "
                f"speed={speed:.0f} img/s"
            )

        if is_main_process() and should_update_grad and (it % 20 == 0 or it == 0):
            log_data = {
                "train_loss": loss.item() * accum_iter,
                "train_head_loss": l_dict.get("L_head", 0),
                "train_cls_loss": l_dict.get("L_cls", 0),
                "train_pix_loss": l_dict.get("L_pix", 0),
                "Optimizer/grad_norm": grad_norm,
                "step": global_step,
                "epoch": epoch,
            }

            for ind, param_group in enumerate(opt.param_groups):
                log_data[f"Optimizer/lr_scale_{ind}"] = param_group["lr_scale"]
                log_data[f"Optimizer/lr_{ind}"] = param_group["lr"]
                log_data[f"Optimizer/weight_decay_{ind}"] = param_group[
                    "weight_decay"
                ]

            # CRITICAL: always use explicit step= to avoid W&B step mixing
            wandb.log(log_data, step=global_step)

        if should_update_grad:
            global_step += 1

    epoch_time = time.time() - epoch_start_time
    throughput = epoch_samples / epoch_time if epoch_time > 0 else 0
    avg_iter_time = sum(iter_times) / len(iter_times) if iter_times else 0

    if is_main_process():
        print(
            f"  Epoch [{epoch}] complete — "
            f"time={epoch_time:.1f}s, "
            f"throughput={throughput:.0f} img/s, "
            f"avg_iter={avg_iter_time*1000:.0f}ms"
        )

    epoch_stats = {
        "train/epoch_time": epoch_time,
        "train/throughput": throughput,
        "train/avg_iter_time": avg_iter_time,
    }
    return global_step, epoch_stats



@torch.no_grad()
def validate_one_epoch(
    model: nn.Module, teacher: Optional[nn.Module],
    val_loader: DataLoader, device: torch.device, dtype: torch.dtype,
    cfg: Dict[str, Any], epoch: int = 0, model_ema=None,
) -> Dict[str, float]:
    """Run one validation epoch. Returns averaged val metrics."""
    model.eval()

    val_stats: Dict[str, torch.Tensor] = {
        "val_loss": torch.tensor(0.0, device=device),
        "val_head_loss": torch.tensor(0.0, device=device),
        "val_cls_loss": torch.tensor(0.0, device=device),
        "val_pix_loss": torch.tensor(0.0, device=device),
    }
    val_batches = 0

    for i, (img, mask) in enumerate(val_loader):
        img = img.to(device, non_blocking=True)
        if mask is not None:
            mask = mask.to(device, non_blocking=True)
        assert isinstance(val_loader.dataset, MEDiCDataset)

        with torch.autocast(device_type=device.type, dtype=dtype):
            T = None
            if teacher is not None:
                T = teacher(img, return_cls=True)

            pred_tok, pred_pix, mask_out, ids_restore = model(
                img, mask, teacher=teacher, epoch=epoch, model_ema=model_ema,
            )
            if mask is None:
                mask = mask_out

            use_mask_tokens = get_config_value(
                cfg, "model.student.use_mask_tokens", True
            )
            if not use_mask_tokens and T is not None:
                T = _extract_visible_teacher_features(T, mask, model)

        loss, l_dict = compute_loss(
            pred_tok=pred_tok,
            pred_pix=pred_pix,
            teacher_features=T,
            img=img,
            mask=mask,
            cfg=cfg,
        )

        val_batches += 1
        val_stats["val_loss"] += loss
        val_stats["val_head_loss"] += l_dict.get("L_head", 0.0)
        val_stats["val_cls_loss"] += l_dict.get("L_cls", 0.0)
        val_stats["val_pix_loss"] += l_dict.get("L_pix", 0.0)

    # All-reduce across GPUs
    if get_config_value(cfg, "distributed", False):
        for k in val_stats:
            dist.all_reduce(val_stats[k], op=dist.ReduceOp.SUM)

    world_size = (
        get_world_size()
        if get_config_value(cfg, "distributed", False)
        else 1
    )
    avg_val_batches = val_batches * world_size

    final_stats = {k: v.item() / avg_val_batches for k, v in val_stats.items()}

    if is_main_process():
        parts = [f"val_loss={final_stats['val_loss']:.4f}"]
        if final_stats.get('val_head_loss', 0) > 0:
            parts.append(f"head={final_stats['val_head_loss']:.4f}")
        if final_stats.get('val_cls_loss', 0) > 0:
            parts.append(f"cls={final_stats['val_cls_loss']:.4f}")
        if final_stats.get('val_pix_loss', 0) > 0:
            parts.append(f"pix={final_stats['val_pix_loss']:.4f}")
        print(f"  Validation — {', '.join(parts)}")

    return final_stats



def _extract_visible_teacher_features(
    T: torch.Tensor, mask: torch.Tensor, model: nn.Module,
) -> torch.Tensor:
    """Sub-select and reorder teacher features [B, N+1, D] to match
    the student's visible token ordering in sparse mode.

    Returns [B, num_visible+1, D] with CLS at position 0.
    """
    visible_mask = ~mask
    B, _, D = T.shape
    T_cls, T_patches = T[:, :1, :], T[:, 1:, :]

    # All samples must have the same number of visible patches (fixed-ratio masking)
    n_vis = visible_mask.sum(dim=1)
    assert torch.all(n_vis == n_vis[0]), "Variable visible counts unsupported"

    # Gather visible patches per sample
    vis_idx = visible_mask.nonzero(as_tuple=False)  # [total_vis, 2]
    idx_per_batch = []
    for b in range(B):
        idx_per_batch.append(vis_idx[vis_idx[:, 0] == b, 1])
    patch_indices = torch.stack(idx_per_batch, dim=0)  # [B, n_vis]
    T_visible = torch.gather(
        T_patches, 1, patch_indices.unsqueeze(-1).expand(-1, -1, D)
    )
    T_out = torch.cat([T_cls, T_visible], dim=1)

    # If student uses patch shuffling, align teacher to the shuffled order
    m = model.module if hasattr(model, "module") else model
    if hasattr(m, "student") and hasattr(m.student, "ids_shuffle"):
        ids_shuffle = m.student.ids_shuffle
        if ids_shuffle is not None:
            aligned = []
            for b in range(B):
                shuf = ids_shuffle if ids_shuffle.dim() == 1 else ids_shuffle[b]
                T_shuf = T_patches[b, shuf, :]
                if ids_shuffle.dim() == 1:
                    m_shuf = mask[b, shuf]
                else:
                    m_shuf = torch.gather(mask[b].float(), 0, shuf).bool()
                aligned.append(T_shuf[~m_shuf].unsqueeze(0))
            T_out = torch.cat([T_cls, torch.cat(aligned, dim=0)], dim=1)

    return T_out



def copy_source_files(output_dir: Path):
    """Snapshot key source files into the checkpoint directory."""
    if not is_main_process():
        return
    dst_root = output_dir / "src_snapshot"
    dst_root.mkdir(exist_ok=True)
    for fp in ["src/train.py", "src/utils/losses.py", "src/utils/evolved_masking_generator.py"]:
        src = Path(fp)
        if src.exists():
            dst = dst_root / fp
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    if Path("src/models").exists():
        shutil.copytree("src/models", dst_root / "models", dirs_exist_ok=True)
    print(f"Source snapshot saved to: {dst_root}")



def main():
    """Main entry point: setup, build, train, validate, checkpoint."""
    cfg = setup_environment()
    output_dir = Path(cfg["output_dir"])

    if is_main_process():
        print(f"{'='*80}\nCheckpoint Directory: {output_dir.absolute()}")
        slurm_id = os.environ.get("SLURM_JOB_ID")
        if slurm_id:
            print(f"SLURM Job: {slurm_id}")
        print(f"{'='*80}")
        copy_source_files(output_dir)

    use_cpu = cfg.get("cpu", False) or cfg["gpu"] == -1
    device = torch.device(
        "cpu" if use_cpu else f"cuda:{cfg['gpu']}"
    )
    dtype = (
        torch.bfloat16
        if get_config_value(cfg, "precision", "bf16") == "bf16"
        else torch.float16
    )

    # 1. Build data loaders
    train_loader, val_loader = build_loader(cfg)

    # 2. Build models and schedules
    model, model_without_ddp, teacher, model_ema, opt, lr_schedule, wd_schedule = (
        build_models_optimizer_and_schedules(cfg, device, len(train_loader))
    )

    # 3. Initialise W&B
    init_wandb(cfg, output_dir, lr_schedule, len(train_loader))

    viz_images = load_visualization_images(cfg, device)
    scaler = NativeScalerWithGradNormCount()

    # Training state
    global_step = 0
    best_loss = float("inf")
    top_k_checkpoints: List[Tuple[float, Path]] = []
    start_epoch = 0

    # Resume from checkpoint if specified
    if cfg.get("resume_path"):
        resume_path = get_config_value(cfg, "resume_path")
        vr = validate_checkpoint_completeness(resume_path)
        if not vr.get("file_exists", False):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        if not vr.get("loadable", True):
            raise ValueError(f"Cannot load checkpoint: {vr.get('error')}")

        ts = load_checkpoint(resume_path, model, opt, scaler, teacher, device, model_ema=model_ema)
        start_epoch = ts["epoch"] + 1
        global_step = ts["global_step"]
        best_loss = ts["best_loss"]
        top_k_checkpoints = ts["top_k_checkpoints"]
        if ts.get("lr_schedule") is not None:
            lr_schedule = ts["lr_schedule"]
        if ts.get("wd_schedule") is not None:
            wd_schedule = ts["wd_schedule"]
    else:
        print(f"Starting fresh training for {get_config_value(cfg, 'epochs')} epochs")

    # ---- Epoch loop ----
    start_time = time.time()
    total_epochs = get_config_value(cfg, "epochs")

    # Checkpoint at specific epochs (200-290 every 10, plus 295 and last)
    specific_save_epochs = (
        set(range(200, 291, 10)) | {295, total_epochs - 1}
    ) - {210, 230}

    for epoch in range(start_epoch, total_epochs):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        if is_main_process():
            wandb.log(
                {"epoch": epoch, "step": global_step}, step=global_step
            )

        # Train
        global_step, train_stats = train_one_epoch(
            model, teacher, train_loader, opt, lr_schedule,
            wd_schedule, scaler, epoch, global_step, dtype, device, cfg,
            model_ema=model_ema,
        )
        if is_main_process():
            wandb.log(
                {**train_stats, "epoch": epoch, "step": global_step},
                step=global_step,
            )

        # Validate
        val_stats = validate_one_epoch(
            model, teacher, val_loader, device, dtype, cfg, epoch,
            model_ema=model_ema,
        )

        if is_main_process():
            # Visualizations
            log_visualizations(
                model_without_ddp, viz_images, epoch, device,
                global_step, cfg,
            )

            wandb.log(
                {**val_stats, "epoch": epoch, "step": global_step},
                step=global_step,
            )

            # Checkpointing
            current_loss = val_stats.get("val_loss", float("inf"))
            best_loss, top_k_checkpoints = handle_checkpointing(
                current_loss, epoch, model, opt, cfg, output_dir,
                best_loss, top_k_checkpoints, specific_save_epochs,
                scaler, global_step, teacher, lr_schedule, wd_schedule,
                model_ema=model_ema,
            )

    print(f"Training time {datetime.timedelta(seconds=int(time.time() - start_time))}")
    if "_config_tracker" in cfg:
        cfg["_config_tracker"].print_report()


if __name__ == "__main__":
    main()
