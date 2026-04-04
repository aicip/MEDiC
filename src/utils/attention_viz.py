import torch
import numpy as np
import cv2  # Required dependency - should always be available
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from typing import Dict, Any, cast, List, Optional, Literal
from torchvision.transforms import Compose
from pathlib import Path
import os

from src.data.transforms import build_transform

CONFIG = {
    # Image Selection (will be populated automatically)
    "selected_images": [0, 2, 3],  # Leave empty to show selection interface
    # Visualization Settings
    "layers_to_analyze": [0, 5, 11],  # Early, middle, late layers
    "attention_heads_to_show": 5,  # Number of attention heads to visualize
    "figure_dpi": 100,  # Plot resolution
    "max_figure_width": 15,  # Maximum figure width
    "max_figure_height": 12,  # Maximum figure height
    # Analysis Options
    "show_raw_attention": True,
    "show_layer_evolution": True,
    "show_head_comparison": True,
    "show_attention_rollout": False,
    "show_model_comparison": True,
    # Rollout Configuration
    "rollout_configs": [
        {"head_fusion": "mean", "discard_ratio": 0.0, "name": "Mean Fusion"},
        # {"head_fusion": "max", "discard_ratio": 0.0, "name": "Max Fusion"},
        # {"head_fusion": "mean", "discard_ratio": 0.1, "name": "Mean + 10% Discard"},
    ],
}


class MultiModelAttentionCapture:
    """
    This class captures attention maps from multiple models.
    """
    def __init__(self, models_dict):
        self.models = models_dict
        self.attention_captures = {}

        # Initialize attention capture for each model
        for model_name, model in models_dict.items():
            self.attention_captures[model_name] = AttentionCapture(model)

        print(f" Initialized attention capture for {len(models_dict)} models")

    def forward_all(self, x, mask=None):
        """Run forward pass on all models and capture attention maps."""
        all_outputs = {}
        all_attention_maps = {}

        for model_name, capture in self.attention_captures.items():
            print(f"🔄 Processing {model_name}...")
            output, attention_maps = capture.forward(x, mask)
            all_outputs[model_name] = output
            all_attention_maps[model_name] = attention_maps
            print(f" {model_name}: Captured {len(attention_maps)} layers")

        return all_outputs, all_attention_maps

    def remove_hooks(self):
        """Remove hooks from all models."""
        for capture in self.attention_captures.values():
            capture.remove_hooks()


class AttentionCapture:
    """
    This class captures attention maps from a single model.
    """
    def __init__(self, model):
        self.model = model
        self.attention_maps = {}

    def forward(self, x, mask=None):
        """Run forward pass and capture attention maps using the model's built-in method."""
        # --- ADAPTATION: Rewrote this method for the new model's API ---
        self.attention_maps.clear()

        # Manually perform the forward pass to capture attention at each block
        x_patches = self.model.patch_embed(x)

        cls_token = self.model.cls_token.expand(x_patches.shape[0], -1, -1)
        # We handle pos_embed separately for CLS and patch tokens
        if self.model.pos_embed is not None:
            cls_token = cls_token + self.model.pos_embed[:, :1, :]
            x_patches = x_patches + self.model.pos_embed[:, 1:, :]

        x_tokens = torch.cat([cls_token, x_patches], dim=1)
        x_tokens = self.model.pos_drop(x_tokens)

        rel_pos_bias = (
            self.model.rel_pos_bias() if self.model.rel_pos_bias is not None else None
        )

        with torch.no_grad():
            # Iterate through blocks to get attention from each one
            for i, block in enumerate(self.model.blocks):
                # Use the get_attn_graph method to capture attention AND get the output
                x_tokens, attn_abs, attn_softmax = block.get_attn_graph(x_tokens, shared_rel_pos_bias=rel_pos_bias)
                # Store the softmax attention weights (after positional bias and softmax)
                self.attention_maps[f"block_{i}"] = attn_softmax.detach().clone()

        # Final normalization to get the model output
        output = self.model.norm(x_tokens)
        return output, self.attention_maps.copy()

    def remove_hooks(self):
        """No hooks to remove in this implementation."""
        pass



# Utility function for handling matplotlib axes arrays
def ensure_axes_array(axes, n_rows, n_cols):
    """Ensure axes is always a 2D array for consistent indexing."""
    if n_rows == 1 and n_cols == 1:
        return np.array([[axes]])
    if n_rows == 1:
        return axes.reshape(1, -1)
    if n_cols == 1:
        return axes.reshape(-1, 1)
    return axes


def extract_attention_vec(attn, direction="cls2patch", head=None):
    """
    Return a (√N × √N) grid.
    direction : 'cls2patch'  (row=CLS,   col=patches)
                'patch2cls'  (row=patch, col=CLS)
    head      : None = mean across heads, else int
    """
    A = attn.mean(1).squeeze(0) if head is None else attn[0, head]
    if direction == "cls2patch":
        vec = A[0, 1:]  # CLS → patches
    elif direction == "patch2cls":
        vec = A[1:, 0]  # patches → CLS
    else:
        raise ValueError("direction must be 'cls2patch' or 'patch2cls'")

    side = int(np.sqrt(vec.shape[0]))
    return vec.reshape(side, side).cpu().numpy()


def get_dynamic_figsize(base_width, base_height, n_items, max_cols=4):
    """Calculate dynamic figure size based on number of items."""
    n_cols = min(max_cols, n_items)
    n_rows = (n_items + n_cols - 1) // n_cols

    width = min(CONFIG["max_figure_width"], n_cols * base_width)
    height = min(CONFIG["max_figure_height"], n_rows * base_height)

    return width, height


def plot_attention_matrix(
    attention, title, figsize=None,
    mode: Literal['show', 'save', 'wandb'] = 'show',
    output_path: Optional[str] = None
):
    """
    Plot attention matrix as heatmap.

    **What this shows**: Direct attention weights between all token pairs
    **Interpretation**:
    - Bright spots = strong attention connections
    - Diagonal patterns = self-attention
    - Off-diagonal = cross-token attention
    """
    if figsize is None:
        figsize = (8, 6)

    plt.figure(figsize=figsize, dpi=CONFIG["figure_dpi"])
    sns.heatmap(
        attention, cmap="viridis", cbar=True, square=True, cbar_kws={"shrink": 0.8}
    )
    plt.title(title, fontsize=12, fontweight="bold", pad=20)
    plt.xlabel("Key Tokens (What is being attended to)", fontsize=10)
    plt.ylabel("Query Tokens (What is attending)", fontsize=10)
    plt.tight_layout()

    if mode == 'save':
        if not output_path: raise ValueError("output_path must be provided for save mode.")
        plt.savefig(output_path)
        print(f"Saved attention matrix plot to {output_path}")
        plt.close()
        return None
    elif mode == 'show':
        plt.show()
        return None
    elif mode == 'wandb':
        return plt.gcf()


def plot_attention_grid(
    attention_grid, title, figsize=None,
    mode: Literal['show', 'save', 'wandb'] = 'show',
    output_path: Optional[str] = None
):
    """
    Plot attention as spatial grid.

    **What this shows**: Spatial attention pattern over image patches
    **Interpretation**:
    - Hot spots = regions receiving high attention
    - Cool areas = regions with low attention
    - Patterns reveal spatial biases
    """
    if figsize is None:
        figsize = (6, 6)

    plt.figure(figsize=figsize, dpi=CONFIG["figure_dpi"])
    im = plt.imshow(attention_grid, cmap="inferno", interpolation="bilinear")
    plt.title(title, fontsize=12, fontweight="bold", pad=20)
    plt.colorbar(im, shrink=0.8, label="Attention Weight")
    plt.axis("off")
    plt.tight_layout()

    if mode == 'save':
        if not output_path: raise ValueError("output_path must be provided for save mode.")
        plt.savefig(output_path)
        print(f"Saved attention grid plot to {output_path}")
        plt.close()
        return None
    elif mode == 'show':
        plt.show()
        return None
    elif mode == 'wandb':
        return plt.gcf()


def plot_attention_overlay(
    image, attention_grid, title, alpha=0.4, figsize=None,
    mode: Literal['show', 'save', 'wandb'] = 'show',
    output_path: Optional[str] = None
):
    """
    Overlay attention heatmap on original image.

    **What this shows**: Attention pattern overlaid on the actual image content
    **Interpretation**:
    - Red/yellow areas = high attention regions
    - Blue/purple areas = low attention regions
    - Shows which image features the model focuses on
    """
    if figsize is None:
        figsize = (12, 4)

    # Resize attention to match image size
    attention_resized = cv2.resize(attention_grid, (image.width, image.height))

    # Normalize attention
    attention_norm = (attention_resized - attention_resized.min()) / (
        attention_resized.max() - attention_resized.min() + 1e-8
    )
    attention_uint8 = np.uint8(255 * attention_norm)

    # Create heatmap
    heatmap = cv2.applyColorMap(attention_uint8, cv2.COLORMAP_JET)  # type: ignore
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Blend with original image
    image_array = np.array(image)
    blended = cv2.addWeighted(image_array, 1 - alpha, heatmap, alpha, 0)

    # Plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize, dpi=CONFIG["figure_dpi"])

    ax1.imshow(image)
    ax1.set_title("Original Image", fontsize=10, fontweight="bold")
    ax1.axis("off")

    im2 = ax2.imshow(attention_grid, cmap="inferno")
    ax2.set_title("Attention Map", fontsize=10, fontweight="bold")
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    ax3.imshow(blended)
    ax3.set_title(title, fontsize=10, fontweight="bold")
    ax3.axis("off")

    plt.tight_layout()

    if mode == 'save':
        if not output_path: raise ValueError("output_path must be provided for save mode.")
        plt.savefig(output_path)
        print(f"Saved attention overlay plot to {output_path}")
        plt.close()
        return None
    elif mode == 'show':
        plt.show()
        return None
    elif mode == 'wandb':
        return plt.gcf()


def plot_multi_attention_overlays(
    image,
    attention_grids,
    titles,
    suptitle="",
    max_cols=3,
    smooth=True,  # True = bilinear; False = nearest
    draw_grid=False,  # overlay patch grid if True
    alpha=0.45,
):
    """
    Overlay attention grids on top of the original PIL image.

    smooth     : Bilinear up‑sampling instead of nearest (no blocky look).
    draw_grid  : Optional white patch grid.
    """
    if not attention_grids:
        return

    n_cols = min(max_cols, len(attention_grids))
    n_rows = (len(attention_grids) + n_cols - 1) // n_cols
    w_fig, h_fig = get_dynamic_figsize(5, 4, len(attention_grids), max_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(w_fig, h_fig), dpi=CONFIG["figure_dpi"]
    )
    axes = ensure_axes_array(axes, n_rows, n_cols)

    h, w = image.height, image.width
    interp = cv2.INTER_LINEAR if smooth else cv2.INTER_NEAREST

    for i, (grid, title) in enumerate(zip(attention_grids, titles)):
        r, c = divmod(i, n_cols)
        ax = axes[r, c]

        grid_rs = cv2.resize(grid, (w, h), interpolation=interp)
        norm = (grid_rs - grid_rs.min()) / (np.ptp(grid_rs) + 1e-8)
        heat = cv2.cvtColor(
            cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET),
            cv2.COLOR_BGR2RGB,
        )
        blended = cv2.addWeighted(np.array(image), 1 - alpha, heat, alpha, 0)

        if draw_grid:
            side = grid.shape[0]
            step_x, step_y = w // side, h // side
            for k in range(1, side):
                cv2.line(blended, (0, k * step_y), (w, k * step_y), (255, 255, 255), 1)
                cv2.line(blended, (k * step_x, 0), (k * step_x, h), (255, 255, 255), 1)

        ax.imshow(blended)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")

    # hide unused cells
    for i in range(len(attention_grids), n_rows * n_cols):
        r, c = divmod(i, n_cols)
        axes[r, c].axis("off")

    if suptitle:
        plt.suptitle(suptitle, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_multi_attention_grids(attention_grids, titles, suptitle="", max_cols=4):
    """Plot multiple attention grids - used when overlay is not needed."""
    n_grids = len(attention_grids)
    if n_grids == 0:
        return

    n_cols = min(max_cols, n_grids)
    n_rows = (n_grids + n_cols - 1) // n_cols

    fig_width, fig_height = get_dynamic_figsize(4, 3, n_grids, max_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(fig_width, fig_height), dpi=CONFIG["figure_dpi"]
    )

    axes = ensure_axes_array(axes, n_rows, n_cols)

    for i, (grid, title) in enumerate(zip(attention_grids, titles)):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]
        im = ax.imshow(grid, cmap="inferno", interpolation="bilinear")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")
        plt.colorbar(im, ax=ax, shrink=0.8)

    # Hide unused subplots
    for i in range(n_grids, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row, col].axis("off")

    if suptitle:
        plt.suptitle(suptitle, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_layer_evolution(
    attention_maps: Dict[str, torch.Tensor],
    image: Image.Image,
    layers_to_analyze: List[int],
    suptitle: str = "Attention Layer Evolution",
    mode: Literal['show', 'save', 'wandb'] = 'show',
    output_path: Optional[str] = None
):
    """
    Plots the mean attention overlay for a selection of layers in a single row.

    Args:
        attention_maps (Dict): {layer_name: attention_tensor} for a single image.
        image (PIL.Image): The original image for overlay.
        layers_to_analyze (List[int]): List of layer indices to visualize.
        suptitle (str): The main title for the figure.
        mode (str): 'show' to display, 'save' to file, 'wandb' to return figure.
        output_path (str): Path to save the figure if mode is 'save'.

    Returns:
        Optional[plt.Figure]: The figure object if mode is 'wandb', otherwise None.
    """
    n_layers = len(layers_to_analyze)
    h_img, w_img = image.height, image.width
    interp, alpha = cv2.INTER_LINEAR, 0.45

    fig, axes = plt.subplots(1, n_layers, figsize=(n_layers * 4, 4), squeeze=False)

    for i, layer_idx in enumerate(layers_to_analyze):
        ax = axes[0, i]
        ax.set_title(f"Layer {layer_idx}", fontsize=10)
        attn = attention_maps.get(f"block_{layer_idx}")

        if attn is not None:
            # Get the mean attention across all heads for the CLS token
            grid = extract_attention_vec(attn, direction="cls2patch", head=None)
            grid_rs = cv2.resize(grid, (w_img, h_img), interpolation=interp)
            # Handle NaN/inf values before normalization
            grid_rs = np.nan_to_num(grid_rs, nan=0.0, posinf=1.0, neginf=0.0)
            norm = (grid_rs - grid_rs.min()) / (np.ptp(grid_rs) + 1e-8)
            # Ensure values are in valid range before casting
            norm = np.clip(norm, 0, 1)
            heat = cv2.cvtColor(cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            blend = cv2.addWeighted(np.asarray(image), 1 - alpha, heat, alpha, 0)
            ax.imshow(blend)
        else:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center")
        ax.axis('off')

    fig.suptitle(suptitle, fontsize=16, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.95))

    if mode == 'save':
        if not output_path: raise ValueError("output_path must be provided for save mode.")
        fig.savefig(output_path)
        print(f"Saved layer evolution plot to {output_path}")
        plt.close(fig)
        return None
    elif mode == 'show':
        plt.show()
        return None
    elif mode == 'wandb':
        return fig


def attention_rollout(attention_maps_list, head_fusion="mean", discard_ratio=0.0):
    """
    Compute attention rollout across all layers.

    **What this computes**: Aggregated attention flow from CLS token through all layers
    **Interpretation**: Shows the final "importance map" after all transformer processing
    """
    if not attention_maps_list:
        return None

    rollout = torch.eye(attention_maps_list[0].shape[-1], device=attention_maps_list[0].device)

    for attention in attention_maps_list:
        # Fuse attention heads
        if head_fusion == "mean":
            attention_fused = attention.mean(dim=1).squeeze(0)
        elif head_fusion == "max":
            attention_fused = attention.max(dim=1)[0].squeeze(0)
        elif head_fusion == "min":
            attention_fused = attention.min(dim=1)[0].squeeze(0)
        else:
            raise ValueError(f"Unsupported head fusion: {head_fusion}")

        # Discard low attention values if specified
        if discard_ratio > 0:
            flat_attention = attention_fused.flatten()
            threshold = torch.quantile(flat_attention, discard_ratio)
            attention_fused = torch.where(
                attention_fused >= threshold,
                attention_fused,
                torch.zeros_like(attention_fused),
            )

        # Add residual connection (identity matrix)
        eye = torch.eye(attention_fused.shape[-1], device=attention_fused.device)
        attention_with_residual = attention_fused + eye

        # Normalize rows
        attention_normalized = attention_with_residual / (
            attention_with_residual.sum(dim=-1, keepdim=True) + 1e-8
        )

        # Update rollout
        rollout = attention_normalized @ rollout

    return rollout


def plot_attention_matrices_comparison(
    all_attention_maps: Dict[str, Dict[str, torch.Tensor]],
    layers_to_analyze: List[int],
    suptitle: str = "Attention Matrix Comparison",
    mode: Literal['show', 'save', 'wandb'] = 'show',
    output_path: Optional[str] = None
):
    """
    Plots attention matrices in a grid where rows are layers and columns are images.

    Args:
        all_attention_maps (Dict): Nested dict: {image_name: {layer_name: attention_tensor}}.
        layers_to_analyze (List[int]): List of layer indices to visualize.
        suptitle (str): The main title for the entire figure.
        mode (str): 'show' to display, 'save' to file, 'wandb' to return figure.
        output_path (str): Path to save the figure if mode is 'save'.

    Returns:
        Optional[plt.Figure]: The figure object if mode is 'wandb', otherwise None.
    """
    image_names = list(all_attention_maps.keys())
    n_images = len(image_names)
    n_layers = len(layers_to_analyze)

    if n_images == 0 or n_layers == 0:
        return

    fig, axes = plt.subplots(n_layers, n_images, figsize=(n_images * 5, n_layers * 4), squeeze=False)

    for r, layer_idx in enumerate(layers_to_analyze):
        for c, img_name in enumerate(image_names):
            ax = axes[r, c]
            attention_maps = all_attention_maps[img_name]

            if f"block_{layer_idx}" in attention_maps:
                attention = attention_maps[f"block_{layer_idx}"]
                attention_avg = attention.mean(dim=1).squeeze(0).cpu().numpy()
                sns.heatmap(attention_avg, cmap="viridis", ax=ax, cbar=False)
                ax.set_title(f"L{layer_idx} - {img_name}", fontsize=10)
            else:
                ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes)

            # Add row labels to the first column
            if c == 0:
                ax.set_ylabel(f"Layer {layer_idx}", fontsize=10, fontweight='bold')

            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(suptitle, fontsize=16, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.96))

    if mode == 'save':
        if not output_path: raise ValueError("output_path must be provided for save mode.")
        fig.savefig(output_path)
        print(f"Saved attention matrix plot to {output_path}")
        plt.close(fig)
        return None
    elif mode == 'show':
        plt.show()
        return None
    elif mode == 'wandb':
        return fig


def plot_head_evolution_by_layer(
    attention_maps: Dict[str, torch.Tensor],
    image: Image.Image,
    layers_to_analyze: List[int],
    heads_to_show: List[int],
    suptitle: str = "Attention Head Evolution",
    mode: Literal['show', 'save', 'wandb'] = 'show',
    output_path: Optional[str] = None
):
    """
    Plots attention overlays in a grid where rows are layers and columns are heads.
    Generates two plots: one for CLS->patch and one for patch->CLS attention.

    Args:
        attention_maps (Dict): {layer_name: attention_tensor} for a single image.
        image (PIL.Image): The original image for overlay.
        layers_to_analyze (List[int]): List of layer indices to visualize.
        heads_to_show (List[int]): List of head indices to visualize.
        suptitle (str): The main title for the figure.
        mode (str): 'show' to display, 'save' to file, 'wandb' to return figure.
        output_path (str): Path to save the figure if mode is 'save'.

    Returns:
        Optional[Dict[str, plt.Figure]]: Dict of figures if mode is 'wandb', else None.
    """
    n_layers = len(layers_to_analyze)
    n_heads = len(heads_to_show)
    h_img, w_img = image.height, image.width
    interp, alpha = cv2.INTER_LINEAR, 0.45
    figs = {}

    for direction, dir_lbl in [("cls2patch", "CLS->patch"), ("patch2cls", "patch->CLS")]:
        fig, axes = plt.subplots(n_layers, n_heads, figsize=(n_heads * 3, n_layers * 3), squeeze=False)

        for r, layer_idx in enumerate(layers_to_analyze):
            for c, head_idx in enumerate(heads_to_show):
                ax = axes[r, c]
                attn = attention_maps.get(f"block_{layer_idx}")

                if attn is not None:
                    grid = extract_attention_vec(attn, direction, head=head_idx)
                    grid_rs = cv2.resize(grid, (w_img, h_img), interpolation=interp)
                    # Handle NaN/inf values before normalization
                    grid_rs = np.nan_to_num(grid_rs, nan=0.0, posinf=1.0, neginf=0.0)
                    norm = (grid_rs - grid_rs.min()) / (np.ptp(grid_rs) + 1e-8)
                    # Ensure values are in valid range before casting
                    norm = np.clip(norm, 0, 1)
                    heat = cv2.cvtColor(cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
                    blend = cv2.addWeighted(np.asarray(image), 1 - alpha, heat, alpha, 0)
                    ax.imshow(blend)
                else:
                    ax.text(0.5, 0.5, "No Data", ha="center", va="center")

                ax.axis('off')
                if r == 0: ax.set_title(f"Head {head_idx}", fontsize=10, fontweight='bold')

                if c == 0:
                    ax.text(-0.1, 0.5, f"Layer {layer_idx}",
                    transform=ax.transAxes,
                            horizontalalignment='right',
                            verticalalignment='center',
                            fontsize=10,
                            fontweight='bold')

        full_suptitle = f"{suptitle} ({dir_lbl})"
        fig.suptitle(full_suptitle, fontsize=16, fontweight="bold")
        plt.tight_layout(rect=(0, 0, 1, 0.95))
        figs[dir_lbl] = fig

    if mode == 'save':
        if not output_path: raise ValueError("output_path must be provided for save mode.")
        for key, fig_to_save in figs.items():
            path = Path(output_path)
            save_path = path.parent / f"{path.stem}_{key}{path.suffix}"
            fig_to_save.savefig(save_path)
            print(f"Saved head evolution plot to {save_path}")
            plt.close(fig_to_save)
        return None
    elif mode == 'show':
        plt.show()
        return None
    elif mode == 'wandb':
        return figs


def analyze_attention_heads(all_attention_maps, img_name, image):
    """
    Head comparison (middle layer).

    Two figures will be produced:

        • Figure A : CLS→patch   (rows=heads, cols=models)
        • Figure B : patch→CLS   (rows=heads, cols=models)
    """
    mid_layer    = (CONFIG["layers_to_analyze"][1]
                    if len(CONFIG["layers_to_analyze"]) > 1 else 5)
    model_names  = list(all_attention_maps.keys())

    # unified head subset
    ref_heads    = next(iter(all_attention_maps.values()))[f"block_{mid_layer}"].shape[1]
    want         = CONFIG["attention_heads_to_show"]
    heads_show   = (list(range(ref_heads)) if want >= ref_heads
                    else np.linspace(0, ref_heads-1, want, dtype=int).tolist())
    H = len(heads_show)
    h_img, w_img = image.height, image.width
    interp, α    = cv2.INTER_LINEAR, 0.45

    for direction, dir_label in [("cls2patch", "CLS→patch"),
                                 ("patch2cls", "patch→CLS")]:
        fig_w = min(CONFIG["max_figure_width"], len(model_names)*4)
        fig_h = min(CONFIG["max_figure_height"], H*2.6)
        fig, axes = plt.subplots(H, len(model_names),
                                 figsize=(fig_w, fig_h),
                                 dpi=CONFIG["figure_dpi"])
        axes = ensure_axes_array(axes, H, len(model_names))

        for col, m_name in enumerate(model_names):
            attn = all_attention_maps[m_name][f"block_{mid_layer}"]

            for row, h in enumerate(heads_show):
                ax   = axes[row, col]
                grid = extract_attention_vec(attn, direction, head=h)

                grid_rs = cv2.resize(grid, (w_img, h_img), interpolation=interp)
                norm    = (grid_rs-grid_rs.min())/(np.ptp(grid_rs)+1e-8)
                heat    = cv2.cvtColor(cv2.applyColorMap((norm*255).astype(np.uint8),
                                                         cv2.COLORMAP_JET),
                                       cv2.COLOR_BGR2RGB)
                blend   = cv2.addWeighted(np.asarray(image), 1-α, heat, α, 0)
                ax.imshow(blend); ax.axis("off")

                if row == 0:
                    ax.set_title(m_name, fontsize=10, fontweight="bold")
                if col == 0:
                    ax.text(-0.08, 0.5, f"Head {h}",
                            transform=ax.transAxes,
                            ha="right", va="center",
                            fontsize=9, fontweight="bold")

        plt.suptitle(f"Attention Heads • {dir_label} • L{mid_layer} • {img_name}",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()


def compare_models_attention(all_attention_maps, img_name, image):
    """
    Late‑layer attention comparison (one row, columns = models).
    Produces two figures: CLS→patch   and   patch→CLS.
    """
    last_layer  = CONFIG["layers_to_analyze"][-1] if CONFIG["layers_to_analyze"] else 11
    model_names = list(all_attention_maps.keys())
    h_img, w_img = image.height, image.width
    interp, α    = cv2.INTER_LINEAR, 0.45

    for direction, dir_label in [("cls2patch", "CLS→patch"),
                                 ("patch2cls", "patch→CLS")]:
        fig_w = min(CONFIG["max_figure_width"], len(model_names)*4)
        fig, axes = plt.subplots(1, len(model_names),
                                 figsize=(fig_w, 3.5),
                                 dpi=CONFIG["figure_dpi"])
        axes = ensure_axes_array(axes, 1, len(model_names))[0]

        for col, m_name in enumerate(model_names):
            grid = extract_attention_vec(
                all_attention_maps[m_name][f"block_{last_layer}"],
                direction
            )
            grid_rs = cv2.resize(grid, (w_img, h_img), interpolation=interp)
            norm    = (grid_rs-grid_rs.min())/(np.ptp(grid_rs)+1e-8)
            heat    = cv2.cvtColor(cv2.applyColorMap((norm*255).astype(np.uint8),
                                                     cv2.COLORMAP_JET),
                                   cv2.COLOR_BGR2RGB)
            blend   = cv2.addWeighted(np.asarray(image), 1-α, heat, α, 0)
            axes[col].imshow(blend); axes[col].axis("off")
            axes[col].set_title(m_name, fontsize=10, fontweight="bold")

        plt.suptitle(f"{dir_label} • Layer {last_layer} • {img_name}",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()


def analyze_attention_rollout(all_attention_maps, img_name, image):
    """
    Roll‑out comparison grid.

    • Columns  = models  (named at top)
    • Rows     = rollout configs  (labelled on the left)
    • Each cell = CLS global‑importance heat‑map overlay.
    """
    if not CONFIG["show_attention_rollout"]:
        return

    cfgs        = CONFIG["rollout_configs"]
    model_names = list(all_attention_maps.keys())
    n_rows, n_cols = len(cfgs), len(model_names)
    
    # Return early if no configs to process
    if n_rows == 0 or n_cols == 0:
        return
    h_img, w_img   = image.height, image.width
    interp, α      = cv2.INTER_LINEAR, 0.45

    # size ~5" per model wide, 3" per config tall
    fig_w = min(CONFIG["max_figure_width"],  n_cols * 5)
    fig_h = min(CONFIG["max_figure_height"], n_rows * 3)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(fig_w, fig_h),
                             dpi=CONFIG["figure_dpi"])
    axes = ensure_axes_array(axes, n_rows, n_cols)

    for c, model_name in enumerate(model_names):
        attn_seq = [all_attention_maps[model_name][f"block_{i}"]
                    for i in range(len(all_attention_maps[model_name]))]

        for r, cfg in enumerate(cfgs):
            ax  = axes[r, c]
            rol = attention_rollout(attn_seq,
                                    head_fusion  = cfg["head_fusion"],
                                    discard_ratio= cfg["discard_ratio"])
            if rol is None:
                ax.axis("off"); continue

            cls_vec = rol[0, 1:]                       # CLS importance
            side    = int(np.sqrt(cls_vec.shape[0]))
            grid    = cls_vec.reshape(side, side).cpu().numpy()

            # overlay
            grid_rs = cv2.resize(grid, (w_img, h_img), interpolation=interp)
            norm    = (grid_rs - grid_rs.min()) / (np.ptp(grid_rs)+1e-8)
            heat    = cv2.cvtColor(cv2.applyColorMap((norm*255).astype(np.uint8),
                                                     cv2.COLORMAP_JET),
                                   cv2.COLOR_BGR2RGB)
            blend   = cv2.addWeighted(np.asarray(image), 1-α, heat, α, 0)
            ax.imshow(blend); ax.axis("off")

            # --- labels -------------------------------------------------
            if r == 0:   # top row
                ax.set_title(model_name, fontsize=11, fontweight="bold")
            if c == 0:   # leftmost col – manual row label
                ax.text(-0.08, 0.5, cfg["name"],
                        transform=ax.transAxes,
                        ha="right", va="center",
                        fontsize=9, fontweight="bold")

    plt.suptitle(f"CLS Roll‑out Importance • {img_name}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    from PIL import Image
    import sys

    # Add project root to path to allow importing local modules
    sys.path.append('.')
    from src.models.medic_model import build_medic_model
    from src.data.transforms import build_transform
    from typing import cast
    from torchvision.transforms import Compose

    parser = argparse.ArgumentParser(description="Attention Visualization Debugger")
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help="Path to the model checkpoint (.pth file).")
    parser.add_argument('--image_dir', type=str, default='images',
                        help="Directory containing images.")
    parser.add_argument('--output_dir', type=str, default='notebooks/outputs',
                        help="Directory to save plots.")
    parser.add_argument('--mode', type=str, default='save', choices=['show', 'save'])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Model
    model_cfg = {
        'model': {
            'student': {
                'name': 'vit_base_patch16',
                'img_size': 224,
                'patch_size': 16,
                'embed_dim': 768,
                'depth': 12,
                'num_heads': 12,
                'mlp_ratio': 4,
                'qkv_bias': True,
                'norm_layer': 'LayerNorm',
                'init_values': 0.1,
                'use_rel_pos_bias': True,
                'use_abs_pos_emb': False,
                'rel_pos_bias_type': 'shared'
            },
            'teacher': {
                'name': 'ViT-B/16'
            },
            'decoder': {
                'name': 'mae_decoder',
        'decoder_embed_dim': 512,
                'decoder_depth': 8,
                'decoder_num_heads': 16,
                'mlp_ratio': 4.0,
                'norm_layer': 'LayerNorm'
            }
        },
        'losses': {}
    }
    model = build_medic_model(model_cfg)
    checkpoint = torch.load(args.checkpoint_path,
                            map_location='cpu',
                            weights_only=True)
    state_dict = checkpoint['model']
    unwrapped_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            unwrapped_state_dict[key[7:]] = value
        else:
            unwrapped_state_dict[key] = value
    model.load_state_dict(unwrapped_state_dict)
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # 2. Load Images based on CONFIG
    image_paths_all = sorted(list(Path(args.image_dir).glob("*.JPEG")))
    selected_indices = CONFIG.get("selected_images", [0])
    image_paths = [image_paths_all[i] for i in selected_indices if i < len(image_paths_all)]
    images_pil = [Image.open(p).convert("RGB") for p in image_paths]
    transform = cast(Compose, build_transform(is_train=False, cfg={'img_size': 224}))
    images_tensor = torch.stack([cast(torch.Tensor, transform(img)) for img in images_pil]).to(device)

    # 3. Capture Attention for all selected images
    all_attention_maps = {}
    attention_capture = AttentionCapture(model.student)
    for i, img_tensor in enumerate(images_tensor):
        img_name = Path(image_paths[i]).stem
        with torch.autocast(device_type='cuda' if img_tensor.device.type == 'cuda' else 'cpu', enabled=True):
            _, attention_maps = attention_capture.forward(img_tensor.unsqueeze(0))
        all_attention_maps[img_name] = attention_maps
    print(f"Captured attention for {len(all_attention_maps)} images.")

    # 4. Test Plotting Functions based on CONFIG
    viz_cfg = CONFIG
    layers = viz_cfg.get("layers_to_analyze", [0, 5, 11])
    plots_to_run = viz_cfg.get("plots", {})

    # --- Raw Attention Matrix Plot ---
    if plots_to_run.get("raw_attention", False):
        print("\n--- Testing Comparison Attention Matrix Plot ---")
        plot_attention_matrices_comparison(
        all_attention_maps,
            layers_to_analyze=layers,
            suptitle="Debug: Attention Matrix Comparison (Layers vs Images)",
            mode=args.mode,
            output_path=os.path.join(args.output_dir, 'debug_attention_matrix.png')
        )

    # --- Head Evolution / Comparison Plot ---
    if plots_to_run.get("head_comparison", False):
        print("\n--- Testing Head Evolution Plot ---")
        num_heads_to_show = viz_cfg.get("attention_heads_to_show", 4)
        num_model_heads = model.student.blocks[0].attn.num_heads
        num_heads_to_show = min(num_heads_to_show, num_model_heads)
        heads_to_show = np.linspace(0, num_model_heads - 1, num_heads_to_show, dtype=int).tolist()

        # Generate this plot for the first selected image
        first_image_name = list(all_attention_maps.keys())[0]
        plot_head_evolution_by_layer(
            all_attention_maps[first_image_name],
            images_pil[0],
            layers_to_analyze=layers,
            heads_to_show=heads_to_show,
            suptitle=f"Debug: Attention Head Evolution for {first_image_name}",
            mode=args.mode,
            output_path=os.path.join(args.output_dir, f'debug_head_evolution_{first_image_name}.png')
        )

    # --- Attention Rollout Plot ---
    if plots_to_run.get("attention_rollout", False):
        print("\n--- Testing Attention Rollout Plot ---")
        # Generate this plot for the first selected image
        first_image_name = list(all_attention_maps.keys())[0]
        attn_seq = [all_attention_maps[first_image_name][f"block_{i}"]
                    for i in range(len(model.student.blocks))]

        for rollout_cfg in viz_cfg.get("rollout_configs", []):
            rollout_map = attention_rollout(attn_seq,
                                            head_fusion=rollout_cfg['head_fusion'],
                                            discard_ratio=rollout_cfg['discard_ratio'])
            if rollout_map is not None:
                cls_vec = rollout_map[0, 1:]
                side = int(np.sqrt(cls_vec.shape[0]))
                grid = cls_vec.reshape(side, side).cpu().numpy()
                plot_attention_overlay(
                    images_pil[0],
                    grid,
                    title=f"Attention Rollout ({rollout_cfg['name']}) - {first_image_name}",
                    mode=args.mode,
                    output_path=os.path.join(args.output_dir,
                                             f"debug_rollout_{rollout_cfg['name'].replace(' ', '_')}.png")
                )

    print("\n--- Visualization script finished ---")
