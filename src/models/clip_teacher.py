import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import math
from einops import rearrange
from typing import Union, List, Optional, Tuple


class ClipTeacher(nn.Module):
    """
    Teacher model wrapper for the official OpenAI CLIP ViT.

    This module loads a specified CLIP model, freezes its weights, and provides
    a forward pass that extracts the final hidden states (patch and CLS tokens).
    It also handles positional embedding interpolation for different input sizes.

    Attention extraction follows the MILAN approach:
    - Extracts attention from the LAST transformer block only
    - Averages attention across heads instead of summing
    - Provides properly normalized attention weights for importance sampling
    """

    def __init__(self, model_name: str = "ViT-B/16", **kwargs):
        """
        Initializes the ClipTeacher.

        Args:
            model_name (str, optional): The name of the CLIP model to load.
                Defaults to "ViT-B/16".
        """
        super().__init__()
        self.net, _ = clip.load(model_name, device="cpu", jit=False)
        self.net = self.net.visual

        self.patch_size = self.net.conv1.kernel_size[0]

        for param in self.net.parameters():
            param.requires_grad = False
        self.net.eval()
        self.eval()

    def forward(
        self, x: torch.Tensor, return_cls: bool = True, return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Performs a forward pass through the frozen CLIP visual encoder.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
            return_cls (bool, optional): Whether to include the CLS token in the
                                         output. Defaults to True.
            return_attention (bool, optional): Whether to return attention maps.
                                             Defaults to False.

        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]: 
                - If return_attention=False: Raw output tokens of shape (B, 197, D) or (B, 196, D).
                - If return_attention=True: Tuple of (features, attention_maps) where
                  attention_maps has shape (B, N, N) averaged across heads from the last layer.
        """
        with torch.no_grad():
            w, h = x.shape[2:]
            x = self.net.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

            class_embedding = self.net.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
            )
            x = torch.cat([class_embedding, x], dim=1)
            x = x + self.interpolate_pos_encoding(x, w, h).to(x.dtype)
            x = self.net.ln_pre(x)

            x = x.permute(1, 0, 2)  # NLD -> LND

            if return_attention:
                # Extract features with attention maps using MILAN approach
                x, attention_maps = self._forward_with_attention(x)
            else:
                # Regular forward without attention
                x = self.net.transformer(x)

            x = x.permute(1, 0, 2)  # LND -> NLD

            if return_attention:
                if return_cls:
                    return x, attention_maps
                else:
                    return x[:, 1:, :], attention_maps
            else:
                if return_cls:
                    return x
                else:
                    return x[:, 1:, :]


    def _forward_with_attention(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through CLIP transformer while collecting attention maps.
        
        MILAN-inspired approach:
        - Extract attention from LAST layer only (not all layers)
        - Average across heads (not sum)
        - Use softmax attention weights directly
        
        Args:
            x (torch.Tensor): Input tensor after patch embedding and positional encoding.
                             Shape: (L, N, D) where L=seq_len, N=batch_size, D=embed_dim
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Features tensor (L, N, D)
                - Attention maps (N, L-1, L-1) - excludes CLS token, averaged across heads
        """
        attention_maps = None
        
        # Process through all blocks, but only extract attention from the last one
        num_blocks = len(self.net.transformer.resblocks)
        
        for i, resblock in enumerate(self.net.transformer.resblocks):
            if i < num_blocks - 1:
                # Regular forward pass for all blocks except the last
                x = resblock(x)
            else:
                # For the LAST block, extract attention (MILAN approach)
                # Layer norm before attention
                x_norm = resblock.ln_1(x)
                
                # Compute QKV
                # CLIP uses in_proj_weight and in_proj_bias for combined QKV projection
                L, N, D = x_norm.shape
                num_heads = resblock.attn.num_heads
                head_dim = D // num_heads
                
                # Split the weight and bias into Q, K, V components
                w_q, w_k, w_v = resblock.attn.in_proj_weight.chunk(3)
                if resblock.attn.in_proj_bias is not None:
                    b_q, b_k, b_v = resblock.attn.in_proj_bias.chunk(3)
                else:
                    b_q = b_k = b_v = None
                
                # Project to Q, K, V separately
                q = F.linear(x_norm, w_q, b_q)
                k = F.linear(x_norm, w_k, b_k)
                v = F.linear(x_norm, w_v, b_v)
                
                # Reshape for multi-head attention
                # Critical: Use contiguous view and correct transpose
                q = q.contiguous().view(L, N * num_heads, head_dim).transpose(0, 1)
                k = k.contiguous().view(L, N * num_heads, head_dim).transpose(0, 1)
                v = v.contiguous().view(L, N * num_heads, head_dim).transpose(0, 1)
                # Now q, k, v are (N*num_heads, L, head_dim)
                
                # Compute attention scores
                scale = 1.0 / math.sqrt(head_dim)
                scores = torch.bmm(q, k.transpose(-2, -1)) * scale  # (N*num_heads, L, L)
                attn = F.softmax(scores, dim=-1)
                
                # Reshape to separate batch and heads
                attn = attn.view(N, num_heads, L, L)  # (N, num_heads, L, L)
                
                # MILAN approach: Average attention across heads (not sum)
                attn_avg = attn.mean(dim=1)  # (N, L, L)
                attention_maps = attn_avg
                
                # Complete the forward pass through the block
                # Apply attention to values
                out = torch.bmm(attn.view(N * num_heads, L, L), v)  # (N*num_heads, L, head_dim)
                out = out.transpose(0, 1).contiguous().view(L, N, D)  # (L, N, D)
                out = F.linear(out, resblock.attn.out_proj.weight, resblock.attn.out_proj.bias)
                
                # Residual connection
                x = x + out
                
                # MLP block
                x = x + resblock.mlp(resblock.ln_2(x))
        
        # Remove CLS token from attention maps: (N, L, L) -> (N, L-1, L-1)
        # CLS token is at position 0
        attention_maps = attention_maps[:, 1:, 1:]
        
        return x, attention_maps

    def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
        """
        Interpolates positional embeddings for different input sizes.
        Adapted from the DINO repository.
        """
        num_patches_x = x.shape[1] - 1
        num_patches_embed = self.net.positional_embedding.shape[0] - 1
        if num_patches_x == num_patches_embed and w == h:
            return self.net.positional_embedding

        class_pos_embed = self.net.positional_embedding[:1]
        patch_pos_embed = self.net.positional_embedding[1:]

        w0 = w // self.patch_size
        h0 = h // self.patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1

        patch_pos_embed = nn.functional.interpolate(
            rearrange(
                patch_pos_embed,
                "(h w) d -> 1 d h w",
                h=int(math.sqrt(num_patches_embed)),
                w=int(math.sqrt(num_patches_embed)),
            ),
            scale_factor=(
                h0 / math.sqrt(num_patches_embed),
                w0 / math.sqrt(num_patches_embed),
            ),
            mode="bicubic",
        )
        patch_pos_embed = rearrange(patch_pos_embed, "1 d h w -> (h w) d")
        return torch.cat((class_pos_embed, patch_pos_embed), dim=0)

    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract attention maps from the CLIP model.

        Args:
            x (torch.Tensor): Input images of shape (B, 3, H, W)

        Returns:
            torch.Tensor: Attention maps of shape (B, N, N) where N is number of patches
        """
        # Get features and attention maps
        _, attention_maps = self.forward(x, return_cls=False, return_attention=True)
        return attention_maps

