"""
MEDiC Evolved Part Masking Demo

Interactive visualization of the Evolved Part Masking algorithm from MEDiC.
Uses CLIP ViT-B/16 attention maps to discover semantic parts via EM clustering,
then generates masks that progressively transition from spatial to semantic.

No trained checkpoint needed -- only the frozen CLIP teacher.
"""

import math
import random as py_random
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image

# ── CLIP Model ────────────────────────────────────────────────────────

CLIP_MODEL = None
CLIP_PREPROCESS = None


def load_clip():
    global CLIP_MODEL, CLIP_PREPROCESS
    if CLIP_MODEL is not None:
        return CLIP_MODEL, CLIP_PREPROCESS
    import open_clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    model = model.to(device).eval()
    CLIP_MODEL = model
    CLIP_PREPROCESS = preprocess
    return model, preprocess


def get_clip_attention(image_pil):
    """Extract attention maps from CLIP's last transformer layer.
    Returns: [N, N] attention map (CLS removed), N=196.
    Uses open_clip's ViT-B-16 (same weights as openai/CLIP ViT-B/16).
    """
    model, preprocess = load_clip()
    device = next(model.parameters()).device
    img_tensor = preprocess(image_pil).unsqueeze(0).to(device)

    visual = model.visual

    with torch.no_grad():
        # open_clip ViT: patch embed -> cls token -> pos embed -> transformer
        x = visual.conv1(img_tensor)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

        # CLS token + positional embedding
        cls_embed = visual.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=device)
        x = torch.cat([cls_embed, x], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND

        # Forward through transformer, extract attention from last layer
        num_blocks = len(visual.transformer.resblocks)
        for i, resblock in enumerate(visual.transformer.resblocks):
            if i < num_blocks - 1:
                x = resblock(x)
            else:
                x_norm = resblock.ln_1(x)
                L, N, D = x_norm.shape
                nh = resblock.attn.num_heads
                hd = D // nh
                w_q, w_k, w_v = resblock.attn.in_proj_weight.chunk(3)
                b_q, b_k, b_v = (resblock.attn.in_proj_bias.chunk(3)
                                  if resblock.attn.in_proj_bias is not None
                                  else (None, None, None))
                q = F.linear(x_norm, w_q, b_q).contiguous().view(L, N * nh, hd).transpose(0, 1)
                k = F.linear(x_norm, w_k, b_k).contiguous().view(L, N * nh, hd).transpose(0, 1)
                attn = F.softmax(torch.bmm(q, k.transpose(-2, -1)) / math.sqrt(hd), dim=-1)
                attn = attn.view(N, nh, L, L).mean(dim=1)

    return attn[0, 1:, 1:].cpu()


# ── Clustering ────────────────────────────────────────────────────────

def em_clustering(x, stage_num=15, k=3):
    """EM clustering (EPM paper). x: [1, C, N]. Returns [N] assignments."""
    def l2norm(inp, dim):
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))
    mu = l2norm(torch.randn(1, x.size(1), k), dim=1)
    with torch.no_grad():
        for _ in range(stage_num):
            z = F.softmax(40.0 * torch.bmm(x.permute(0, 2, 1), mu), dim=2)
            mu = l2norm(torch.bmm(x, z / (1e-6 + z.sum(dim=1, keepdim=True))), dim=1)
    z = F.softmax(40.0 * torch.bmm(x.permute(0, 2, 1), mu), dim=2)
    return z[0].argmax(dim=-1)


def hierarchical_clustering(x, k=10):
    """Agglomerative clustering. x: [1, C, N]. Returns [N] assignments."""
    features = F.normalize(x[0].T, dim=-1)  # [N, C]
    N = features.shape[0]
    assignments = torch.arange(N)
    centers = features.clone()

    while len(torch.unique(assignments)) > k:
        active = torch.unique(assignments)
        if len(active) <= k:
            break
        # Find most similar pair
        active_centers = torch.stack([centers[a] for a in active])
        sim = torch.mm(active_centers, active_centers.T)
        sim.fill_diagonal_(-1)
        flat_idx = sim.argmax().item()
        i, j = flat_idx // len(active), flat_idx % len(active)
        ci, cj = active[i].item(), active[j].item()
        assignments[assignments == cj] = ci
        mask_i = (assignments == ci)
        if mask_i.sum() > 0:
            centers[ci] = F.normalize(features[mask_i].mean(0, keepdim=True), dim=-1).squeeze()

    unique = torch.unique(assignments)
    remap = {old.item(): new for new, old in enumerate(unique)}
    return torch.tensor([remap[a.item()] for a in assignments])


# ── Spatial Noise (matches EPM code) ──────────────────────────────────

def generate_spatial_noise(N, grid_size, noise_type="grid"):
    if noise_type == "grid":
        noise_grid = torch.rand(2, 2)
        noise = torch.zeros(N)
        for i in range(N):
            noise[i] = noise_grid[(i // grid_size) % 2, i % 2]
        return noise
    return torch.rand(N)


# ── Evolution (matches EPM code exactly) ──────────────────────────────

def compute_alpha(epoch, total_epochs, gamma):
    """alpha = ((epoch+1) / total_epochs)^gamma"""
    return math.pow((epoch + 1) / total_epochs, gamma)


def compute_dynamic_k(epoch, total_epochs, cluster_low, cluster_high):
    """K decreases linearly from high to low during training.
    Exact formula from EPM: cluster_idx = (high-low)*(total-epoch)/total + low
    Returns [int(cluster_idx), int(cluster_idx+2)] range.
    """
    if epoch is None:
        return int((cluster_high + cluster_low) / 2)
    cluster_idx = (cluster_high - cluster_low) * (total_epochs - epoch) / total_epochs + cluster_low
    return int(cluster_idx), int(cluster_idx + 2)


# ── Auto-update callbacks ─────────────────────────────────────────────

def update_derived(epoch, total_epochs, gamma, cluster_low, cluster_high):
    """Auto-compute alpha and K when epoch/gamma/cluster range changes."""
    alpha = compute_alpha(int(epoch), int(total_epochs), gamma)
    k_low, k_high = compute_dynamic_k(int(epoch), int(total_epochs), int(cluster_low), int(cluster_high))
    return f"{alpha:.3f}", f"{k_low}-{k_high}"


# ── Visualization ─────────────────────────────────────────────────────

CMAP = ListedColormap(plt.cm.tab20.colors[:20])


def visualize(
    image_pil, mask_ratio, clustering_method, cluster_low, cluster_high,
    em_iterations, spatial_noise_type, gamma, epoch, total_epochs,
    seed,
):
    if image_pil is None:
        return None, None, "Upload an image first."

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    py_random.seed(int(seed))

    image_pil = image_pil.convert("RGB").resize((224, 224))
    img_np = np.array(image_pil).astype(float) / 255.0
    grid_size = 14
    N = 196

    attention = get_clip_attention(image_pil)

    alpha = compute_alpha(int(epoch), int(total_epochs), gamma)
    k_low, k_high = compute_dynamic_k(int(epoch), int(total_epochs), int(cluster_low), int(cluster_high))
    # Random K in range [k_low, k_high] (matches EPM code: random.randint)
    cluster_k = py_random.randint(k_low, k_high)
    parts_num = k_high + 1  # matches EPM: cluster_num[1] + 1

    # Clustering
    attn_features = attention.permute(1, 0).unsqueeze(0)
    if "EM" in clustering_method:
        clusters = em_clustering(attn_features, stage_num=int(em_iterations), k=cluster_k)
        method_short = "EM"
    else:
        clusters = hierarchical_clustering(attn_features, k=cluster_k)
        method_short = "HC"

    unique_parts = sorted(clusters.unique().tolist())
    n_parts = len(unique_parts)

    # Spatial noise: random 2x2 grid (matches EPM exactly)
    spatial_noise = generate_spatial_noise(N, grid_size, spatial_noise_type.lower())

    # Semantic noise: per-part random, gathered by assignment (matches EPM)
    parts_noise = torch.rand(parts_num)
    semantic_noise = parts_noise[clusters]

    noise = (1 - alpha) * spatial_noise + alpha * semantic_noise

    len_keep = int(N * (1 - mask_ratio))
    ids = torch.argsort(noise)
    mask = torch.ones(N, dtype=torch.bool)
    mask[:len_keep] = False
    mask = mask[torch.argsort(ids)]

    pct = mask.sum().item() / N * 100

    # ── Main figure: 4 panels ─────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(img_np)
    axes[0].set_title("Original", fontsize=14)
    axes[0].axis("off")

    attn_map = attention.mean(dim=0).reshape(grid_size, grid_size).numpy()
    axes[1].imshow(img_np)
    axes[1].imshow(np.kron(attn_map, np.ones((16, 16))), alpha=0.6, cmap="hot")
    axes[1].set_title("CLIP Attention", fontsize=14)
    axes[1].axis("off")

    cluster_map = clusters.reshape(grid_size, grid_size).numpy()
    axes[2].imshow(img_np)
    axes[2].imshow(np.kron(cluster_map, np.ones((16, 16))), alpha=0.5,
                    cmap=CMAP, vmin=0, vmax=max(cluster_k - 1, 1))
    axes[2].set_title(f"Semantic Parts ({method_short}, K={cluster_k}, found {n_parts})", fontsize=13)
    axes[2].axis("off")

    mask_2d = mask.reshape(grid_size, grid_size).float().numpy()
    mask_up = np.kron(mask_2d, np.ones((16, 16)))
    masked_img = img_np.copy()
    masked_img[mask_up > 0.5] = masked_img[mask_up > 0.5] * 0.3 + 0.35
    axes[3].imshow(masked_img)
    axes[3].set_title(f"Evolved Mask ({pct:.0f}% masked, alpha={alpha:.2f})", fontsize=13)
    axes[3].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # ── Cluster figure: only non-empty parts ──────────────────────
    nonempty = [(pid, (clusters == pid).sum().item()) for pid in unique_parts if (clusters == pid).sum() > 0]
    n_show = min(len(nonempty), 8)

    if n_show > 0:
        cluster_fig, cluster_axes = plt.subplots(1, n_show, figsize=(n_show * 3.5, 3.5))
        if n_show == 1:
            cluster_axes = [cluster_axes]
        for j, (pid, count) in enumerate(nonempty[:n_show]):
            c_mask = (clusters == pid).reshape(grid_size, grid_size).float().numpy()
            c_up = np.kron(c_mask, np.ones((16, 16)))
            highlighted = img_np.copy()
            highlighted[c_up < 0.5] = highlighted[c_up < 0.5] * 0.15
            cluster_axes[j].imshow(highlighted)
            cluster_axes[j].set_title(f"Part {j+1} ({count} patches)", fontsize=11)
            cluster_axes[j].axis("off")
        cluster_fig.suptitle(f"Semantic Parts ({method_short}, {n_parts} found)", fontsize=13)
        cluster_fig.tight_layout(rect=[0, 0, 1, 0.92])
    else:
        cluster_fig = plt.figure(figsize=(4, 2))
        plt.text(0.5, 0.5, "No clusters found", ha="center", va="center")
        plt.axis("off")

    info = (f"alpha={alpha:.3f} (epoch {int(epoch)}/{int(total_epochs)}, gamma={gamma}) | "
            f"{method_short} K={cluster_k} (range [{k_low},{k_high}] from [{int(cluster_low)},{int(cluster_high)}]), "
            f"found {n_parts} parts | "
            f"Mask: {pct:.0f}% | Spatial: {spatial_noise_type}")

    return fig, cluster_fig, info


# ── Gradio Interface ──────────────────────────────────────────────────

def main():
    with gr.Blocks(title="MEDiC Evolved Part Masking", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # MEDiC: Evolved Part Masking Visualization

        Interactive demo of **Evolved Part Masking** from [MEDiC](https://arxiv.org/abs/2603.29009).
        Uses a frozen CLIP ViT-B/16 for attention extraction + EM/HC clustering for
        semantic part discovery. **Alpha** and **K** auto-adjust as you move the epoch slider,
        simulating how masking evolves during training.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Input Image")

                gr.Markdown("### Masking")
                mask_ratio = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Mask Ratio")

                gr.Markdown("### Clustering")
                clustering_method = gr.Radio(
                    ["EM (Expectation-Maximization)", "Hierarchical Clustering (HC)"],
                    value="EM (Expectation-Maximization)", label="Algorithm")
                with gr.Row():
                    cluster_low = gr.Slider(3, 30, value=10, step=1, label="K min")
                    cluster_high = gr.Slider(10, 50, value=40, step=1, label="K max")
                em_iterations = gr.Slider(5, 30, value=15, step=1, label="EM Iterations")

                gr.Markdown("### Spatial Noise")
                spatial_noise_type = gr.Radio(["Grid", "Random"], value="Grid", label="Type")

                gr.Markdown("### Training Schedule")
                gamma = gr.Slider(0.1, 3.0, value=0.5, step=0.1,
                                   label="Gamma (0.5=sqrt, 1.0=linear, 2.0=quadratic)")
                epoch = gr.Slider(0, 300, value=150, step=1, label="Current Epoch")
                total_epochs = gr.Slider(100, 500, value=300, step=10, label="Total Epochs")

                gr.Markdown("### Auto-computed (from epoch, gamma, K range)")
                with gr.Row():
                    alpha_display = gr.Textbox(value="0.707", label="Alpha", interactive=False)
                    k_display = gr.Textbox(value="20-22", label="K range", interactive=False)

                seed = gr.Slider(0, 100, value=42, step=1, label="Random Seed")
                run_btn = gr.Button("Generate", variant="primary", size="lg")

            with gr.Column(scale=3):
                info_text = gr.Textbox(label="Configuration", interactive=False)
                main_plot = gr.Plot(label="Masking Visualization")
                cluster_plot = gr.Plot(label="Individual Semantic Parts")

        # Auto-update alpha and K when epoch/gamma/cluster range changes
        for trigger in [epoch, gamma, total_epochs, cluster_low, cluster_high]:
            trigger.change(
                fn=update_derived,
                inputs=[epoch, total_epochs, gamma, cluster_low, cluster_high],
                outputs=[alpha_display, k_display],
            )

        run_btn.click(
            fn=visualize,
            inputs=[image_input, mask_ratio, clustering_method, cluster_low, cluster_high,
                    em_iterations, spatial_noise_type, gamma, epoch, total_epochs,
                    seed],
            outputs=[main_plot, cluster_plot, info_text],
            api_name="generate",
        )

        gr.Markdown("### Examples")
        gr.Examples(
            examples=[
                ["examples/dogs.jpg", 0.5, "EM (Expectation-Maximization)", 10, 40, 15, "Grid", 0.5, 150, 300, 42],
                ["examples/bird.jpg", 0.4, "EM (Expectation-Maximization)", 10, 40, 15, "Grid", 0.5, 50, 300, 7],
                ["examples/cat.jpg", 0.6, "Hierarchical Clustering (HC)", 5, 20, 15, "Grid", 0.5, 250, 300, 13],
                ["examples/baseball.jpg", 0.5, "EM (Expectation-Maximization)", 10, 40, 15, "Random", 1.0, 299, 300, 42],
            ],
            inputs=[image_input, mask_ratio, clustering_method, cluster_low, cluster_high,
                    em_iterations, spatial_noise_type, gamma, epoch, total_epochs, seed],
            outputs=[main_plot, cluster_plot, info_text],
            fn=visualize,
            cache_examples=False,
        )

        gr.Markdown("""
        ### How it works

        **Alpha** controls the spatial-to-semantic transition: `alpha = ((epoch+1) / total_epochs)^gamma`
        - Early training (alpha near 0): grid-based spatial masking
        - Late training (alpha near 1): semantic part-based masking

        **K** (cluster count) decreases during training from K_max to K_min, making parts coarser over time.

        **Gamma** controls the evolution speed:
        `0.5` = fast start (sqrt), `1.0` = linear, `2.0` = slow start (quadratic)

        Ref: "Evolved Part Masking for Self-Supervised Learning" (CVPR 2023) integrated into MEDiC.
        """)

    demo.launch(show_error=True)


if __name__ == "__main__":
    main()
