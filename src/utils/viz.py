"""
Visualization utilities for MEDiC pretraining.

Provides:
- Learning rate schedule plotting
- Masked image visualization (original + masked overlay)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, List, Any, Optional


# ImageNet normalization constants
IMAGENET_DEFAULT_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_DEFAULT_STD = torch.tensor([0.229, 0.224, 0.225])
MASK_GRAY_VALUE = 0.5
MASK_ALPHA = 0.8


def plot_scheduler(schedule: List[float], niter_per_ep: int,
                   caption: str = '') -> Figure:
    """
    Plot learning rate schedule (per-iteration and per-epoch).

    Args:
        schedule: List of LR values per iteration
        niter_per_ep: Number of iterations per epoch
        caption: Figure title

    Returns:
        matplotlib Figure
    """
    total_steps = len(schedule)
    full_epochs = total_steps // niter_per_ep
    remaining_iters = total_steps % niter_per_ep
    total_epochs = full_epochs + 1 if remaining_iters > 0 else full_epochs
    epochs_range = np.arange(0, total_epochs)

    lr_per_epoch: List[float] = []
    if full_epochs > 0:
        lr_per_epoch = [schedule[i * niter_per_ep] for i in range(full_epochs)]
        if remaining_iters > 0:
            lr_per_epoch.append(schedule[-1])
    elif remaining_iters > 0:
        lr_per_epoch = [schedule[-1]]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9))

    # Per-iteration plot
    ax1.plot(schedule, label="LR per Iteration")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Learning Rate")
    ax1.set_title("Learning Rate Schedule (Iterations)")

    # Per-epoch plot
    if lr_per_epoch:
        ax2.plot(epochs_range, lr_per_epoch, label="LR per Epoch", color="orange")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning Rate")
    ax2.set_title("Learning Rate Schedule (Epochs)")

    # Annotations for iteration plot
    if schedule:
        max_x, max_y = int(np.argmax(schedule)), max(schedule)
        min_x, min_y = int(np.argmin(schedule)), min(schedule)

        points = [
            ("First Iter", 0, schedule[0], "red"),
            ("Last Iter", total_steps - 1, schedule[-1], "green"),
            ("Max LR Iter", max_x, max_y, "purple"),
            ("Min LR Iter", min_x, min_y, "blue"),
        ]

        for label, x, y, color in points:
            ax1.plot(x, y, 'o', color=color, label=f"{label} ({x}, {y:.6f})")
            ax1.axhline(y=y, color='grey', linestyle='--', linewidth=0.5)
            ax1.axvline(x=x, color='grey', linestyle='--', linewidth=0.5)

    # Annotations for epoch plot
    if lr_per_epoch:
        max_x_ep, max_y_ep = int(np.argmax(lr_per_epoch)), max(lr_per_epoch)
        min_x_ep, min_y_ep = int(np.argmin(lr_per_epoch)), min(lr_per_epoch)

        points_epoch = [
            ("First Epoch", 0, lr_per_epoch[0], "red"),
            ("Last Epoch", total_epochs - 1, lr_per_epoch[-1], "green"),
            ("Max LR Epoch", max_x_ep, max_y_ep, "purple"),
            ("Min LR Epoch", min_x_ep, min_y_ep, "blue"),
        ]

        for label, x, y, color in points_epoch:
            ax2.plot(x, y, 'o', color=color, label=f"{label} ({x}, {y:.6f})")
            ax2.axhline(y=y, color='grey', linestyle='--', linewidth=0.5)
            ax2.axvline(x=x, color='grey', linestyle='--', linewidth=0.5)

    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")
    fig.suptitle(caption, fontsize=16)
    plt.tight_layout()
    return fig


@torch.no_grad()
def get_reconstruction_fig(model, batch, mask, epoch, n_images=4) -> Figure:
    """
    Generates a figure showing original and masked images.

    Shows 2 columns: Original | Masked (with gray overlay on masked patches).
    When pixel decoder is available, also shows reconstructed patches.

    Args:
        model: MEDiCModel instance
        batch: (images, labels) tuple
        mask: Patch-level mask [B, N] (True = masked)
        epoch: Current epoch number
        n_images: Number of images to display

    Returns:
        matplotlib Figure
    """
    model.eval()

    if batch is None or not batch or len(batch) < 2:
        raise ValueError("Invalid batch format. Expected (images, labels) tuple.")

    images, _ = batch
    if images is None or images.numel() == 0:
        raise ValueError("Invalid images in batch.")

    images = images[:n_images].to(mask.device)
    mask = mask[:n_images]

    # Handle 3D masks (pixel-level) by converting to patch-level
    if mask.dim() == 3:
        patch_size = model.student.patch_embed.patch_size[0]
        batch_size, height, width = mask.shape
        num_patches_h = height // patch_size
        num_patches_w = width // patch_size
        mask = mask.float().view(batch_size, num_patches_h, patch_size, num_patches_w, patch_size)
        mask = mask.mean(dim=(2, 4))
        mask = mask.view(batch_size, num_patches_h * num_patches_w)
        mask = (mask > 0.5).float()

    # Denormalize images for display
    orig_imgs = (
        torch.einsum("nchw->nhwc", images).cpu() * IMAGENET_DEFAULT_STD
        + IMAGENET_DEFAULT_MEAN
    )

    # Create mask overlay
    patch_size = model.student.patch_embed.patch_size[0]
    mask_expanded = mask.unsqueeze(-1).float().repeat(1, 1, patch_size**2 * 3)
    unpatchified_mask = model.student.unpatchify(mask_expanded)
    mask_img = torch.einsum("nchw->nhwc", unpatchified_mask).cpu()

    # Ensure mask matches image size
    target_size = orig_imgs.shape[1:3]
    if mask_img.shape[1:3] != target_size:
        mask_img = torch.nn.functional.interpolate(
            mask_img.permute(0, 3, 1, 2),
            size=target_size,
            mode='nearest',
        ).permute(0, 2, 3, 1)

    # Overlay: visible patches shown normally, masked patches grayed out
    masked_imgs = (
        orig_imgs * (1 - mask_img * MASK_ALPHA)
        + (MASK_GRAY_VALUE * MASK_ALPHA) * mask_img
    )

    min_batch = min(orig_imgs.shape[0], mask_img.shape[0])

    fig, axs = plt.subplots(n_images, 2, figsize=(6, 3 * n_images))
    axs = np.atleast_2d(axs)
    fig.suptitle(f"Epoch {epoch} Masking Visualization", fontsize=14)

    for i in range(n_images):
        for j in range(2):
            if i < min_batch:
                if j == 0:
                    axs[i, j].imshow(torch.clip(orig_imgs[i], 0, 1).numpy())
                    axs[i, j].set_title("Original")
                else:
                    axs[i, j].imshow(torch.clip(masked_imgs[i], 0, 1).numpy())
                    axs[i, j].set_title("Masked")
            axs[i, j].axis("off")

    return fig


def fig_to_pil(fig) -> "Image.Image":
    """Convert a matplotlib figure to a PIL Image."""
    from io import BytesIO
    from PIL import Image as PILImage
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    return PILImage.open(buf)


@torch.no_grad()
def log_attention_plots_to_wandb(
    model_without_ddp: torch.nn.Module,
    viz_images: torch.Tensor,
    cfg: Dict[str, Any],
    epoch: int,
    global_step: int,
):
    """Generate and log attention visualizations to W&B."""
    import wandb
    from .attention_viz import (
        AttentionCapture,
        plot_attention_matrices_comparison,
        plot_head_evolution_by_layer,
        plot_layer_evolution,
    )
    from torchvision.transforms import ToPILImage

    viz_cfg = cfg.get("visualizations", {})
    if not viz_cfg.get("enable", False):
        return

    student_model = model_without_ddp.student
    attention_capture = AttentionCapture(student_model)

    mean = torch.tensor([0.485, 0.456, 0.406], device=viz_images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=viz_images.device).view(1, 3, 1, 1)

    all_attention_maps = {}
    pil_images = []

    for i in range(viz_images.shape[0]):
        img_tensor = viz_images[i].unsqueeze(0)
        img_name = f"image_{i}"

        img_denorm = (img_tensor * std) + mean
        img_pil = ToPILImage()(img_denorm.squeeze(0).cpu())
        pil_images.append(img_pil)

        try:
            with torch.autocast(
                device_type="cuda" if img_tensor.device.type == "cuda" else "cpu",
                enabled=True,
            ):
                _, attention_maps = attention_capture.forward(img_tensor)
            all_attention_maps[img_name] = attention_maps
        except NotImplementedError as e:
            if "fused attention" in str(e).lower():
                return  # Skip if fused attention blocks capture
            raise

    layers = viz_cfg.get("layers_to_analyze", [0, 5, 11])
    plots_to_run = viz_cfg.get("plots", {})

    if plots_to_run.get("raw_attention", False):
        fig = plot_attention_matrices_comparison(
            all_attention_maps,
            layers_to_analyze=layers,
            suptitle=f"Epoch {epoch}: Attention Matrix Comparison",
            mode="wandb",
        )
        if fig:
            wandb.log({"Attention/Matrices": wandb.Image(fig_to_pil(fig))}, step=global_step)
            plt.close(fig)

    for i, img_name in enumerate(all_attention_maps.keys()):
        viz_img_name = f"image_{i}"

        if plots_to_run.get("layer_evolution", False):
            fig = plot_layer_evolution(
                all_attention_maps[img_name],
                pil_images[i],
                layers_to_analyze=layers,
                suptitle=f"Epoch {epoch}: {img_name} Mean Attention Evolution",
                mode="wandb",
            )
            if fig:
                wandb.log(
                    {f"Layer_Evolution/{viz_img_name}/mean_attention_evolution": wandb.Image(fig_to_pil(fig))},
                    step=global_step,
                )
                plt.close(fig)

        if plots_to_run.get("head_comparison", False):
            heads = viz_cfg.get("attention_heads_to_show", 4)
            num_model_heads = student_model.blocks[0].attn.num_heads
            heads_to_show = np.linspace(
                0, num_model_heads - 1, min(heads, num_model_heads), dtype=int
            ).tolist()

            figs = plot_head_evolution_by_layer(
                all_attention_maps[img_name],
                pil_images[i],
                layers_to_analyze=layers,
                heads_to_show=heads_to_show,
                suptitle=f"Epoch {epoch}: {img_name} Head Evolution",
                mode="wandb",
            )
            if figs and isinstance(figs, dict):
                for key, fig in figs.items():
                    wandb.log(
                        {f"Head_Evolution/{viz_img_name}/{key}": wandb.Image(fig_to_pil(fig))},
                        step=global_step,
                    )
                    plt.close(fig)
