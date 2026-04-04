# --------------------------------------------------------
# MEDiC: Masked Enhanced Distillation with Contrastive Learning
# Object Detection Backbone Adapter
# Based on semantic segmentation adapter and CAE detection backbone
# --------------------------------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

try:
    from mmdet.models.builder import BACKBONES
    from mmcv.runner import load_checkpoint as mmcv_load_checkpoint
    from mmcv.utils import get_logger
    MMDET_AVAILABLE = True
except ImportError:
    print("Warning: mmdetection not installed. Install with: pip install mmdet")
    BACKBONES = None
    mmcv_load_checkpoint = None
    get_logger = None
    MMDET_AVAILABLE = False


class MEDiC(nn.Module):
    """MEDiC backbone adapter for object detection.

    This adapter wraps the pretrained MEDiC VisionTransformerMIM model
    and adds FPN layers to produce multi-scale features for detection.

    Args:
        img_size (int): Input image size for position embedding. Default: 224
        patch_size (int): Patch size. Default: 16
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Embedding dimension. Default: 768
        depth (int): Number of transformer blocks. Default: 12
        num_heads (int): Number of attention heads. Default: 12
        mlp_ratio (float): MLP hidden dim ratio. Default: 4.0
        qkv_bias (bool): Enable bias for qkv. Default: True
        drop_rate (float): Dropout rate. Default: 0.0
        attn_drop_rate (float): Attention dropout rate. Default: 0.0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        init_values (float): Layer scale init value. Default: 0.1
        use_abs_pos_emb (bool): Use absolute position embedding. Default: False
        use_sincos_pos_emb (bool): Use fixed sincos position embedding. Default: True
        use_checkpoint (bool): Use gradient checkpointing. Default: False
        out_indices (list): Indices of blocks to output features. Default: [3, 5, 7, 11]
        with_fpn (bool): Whether to create FPN layers. Default: True
        frozen_stages (int): Stages to freeze. -1 means not freezing any. Default: -1
        pretrained (str): Path to pretrained checkpoint. Default: None
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.1,
                 init_values=0.1,
                 use_abs_pos_emb=False,
                 use_sincos_pos_emb=True,  # Default for detection
                 use_checkpoint=False,
                 # Detection-specific parameters
                 out_indices=[3, 5, 7, 11],
                 with_fpn=True,
                 frozen_stages=-1,
                 # Pretrained checkpoint path
                 pretrained=None):
        super().__init__()

        # Import VisionTransformerMIM from MEDiC codebase
        import sys
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from src.models.vision_transformer import VisionTransformerMIM

        # IMPORTANT: For detection (variable input sizes), we cannot use relative position bias
        # because it requires fixed sequence length. Instead, we use sinusoidal embeddings.
        #
        # Priority order for position embeddings:
        # 1. If use_abs_pos_emb=True, use learned absolute embeddings (will be interpolated)
        # 2. Else if use_sincos_pos_emb=True, use fixed sinusoidal embeddings
        # 3. Else no position embeddings
        #
        # NOTE: If checkpoint has learned pos_embed, we MUST use abs_pos_emb to load them!
        # The previous logic incorrectly gave sincos priority, which prevented loading
        # pretrained position embeddings.
        actual_use_abs = use_abs_pos_emb  # Learned embeddings take priority if enabled
        actual_use_sincos = use_sincos_pos_emb and not use_abs_pos_emb  # Fallback to sincos

        # Store config for later
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_features = self.embed_dim = embed_dim
        self.out_indices = out_indices
        self.use_checkpoint = use_checkpoint
        self.with_fpn = with_fpn
        self.frozen_stages = frozen_stages
        self.depth = depth

        # Create the backbone model
        self.backbone = VisionTransformerMIM(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            init_values=init_values,
            use_abs_pos_emb=actual_use_abs,
            use_sincos_pos_emb=actual_use_sincos,
            use_rel_pos_bias=False,  # Cannot use rel_pos_bias for detection (variable sizes)
            use_shared_rel_pos_bias=False,
            use_mask_tokens=False,  # No masking for detection
            num_classes=0,  # No classification head
        )

        # CRITICAL: Disable strict image size checking for variable-size detection inputs
        # VisionTransformerMIM uses timm's PatchEmbed which enforces img_size by default
        # Detection uses variable sizes (800+) while pretraining uses 224
        self.backbone.patch_embed.strict_img_size = False

        # No normalization before FPN — CAE uses nn.Identity() here.
        # LayerNorm would erase magnitude differences between blocks
        # that FPN relies on for multi-scale feature discrimination.
        self.norm = nn.Identity()

        # Feature Pyramid Network (FPN) layers
        # Creates multi-scale features from single-scale ViT output
        # For patch_size=16: strides [4, 8, 16, 32]
        if with_fpn:
            if patch_size == 16:
                # FPN level 1: stride 4 (upsample 4x from stride 16)
                self.fpn1 = nn.Sequential(
                    nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                    nn.SyncBatchNorm(embed_dim),
                    nn.GELU(),
                    nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                )
                # FPN level 2: stride 8 (upsample 2x from stride 16)
                self.fpn2 = nn.Sequential(
                    nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                )
                # FPN level 3: stride 16 (no change)
                self.fpn3 = nn.Identity()
                # FPN level 4: stride 32 (downsample 2x from stride 16)
                self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
            elif patch_size == 8:
                # For patch_size=8: strides [4, 8, 16, 32]
                self.fpn1 = nn.Sequential(
                    nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                )
                self.fpn2 = nn.Identity()
                self.fpn3 = nn.Sequential(
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
                self.fpn4 = nn.Sequential(
                    nn.MaxPool2d(kernel_size=4, stride=4),
                )
            else:
                raise ValueError(f"Unsupported patch_size={patch_size}. Use 8 or 16.")

        # Initialize weights if pretrained path is provided
        self.pretrained = pretrained
        if pretrained is not None:
            self.init_weights(pretrained)

    def _freeze_stages(self):
        """Freeze stages to prevent them from training.

        frozen_stages behavior:
        - -1: Don't freeze anything
        - 0: Freeze patch embedding and position embedding
        - 1-depth: Freeze patch embed + first N transformer blocks
        """
        if self.frozen_stages >= 0:
            # Freeze patch embedding
            self.backbone.patch_embed.eval()
            for param in self.backbone.patch_embed.parameters():
                param.requires_grad = False

            # Freeze position embedding and CLS token
            if self.backbone.pos_embed is not None:
                self.backbone.pos_embed.requires_grad = False
            if self.backbone.cls_token is not None:
                self.backbone.cls_token.requires_grad = False

        # Freeze transformer blocks up to frozen_stages
        for i in range(self.frozen_stages):
            block = self.backbone.blocks[i]
            block.eval()
            for param in block.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained MEDiC checkpoint.
        """
        if pretrained is None:
            pretrained = self.pretrained

        if isinstance(pretrained, str):
            logger = get_logger('mmdet') if get_logger is not None else None

            # Load checkpoint with strict=False to allow missing FPN weights
            # PyTorch 2.6 fix: set weights_only=False for numpy compatibility
            checkpoint = torch.load(pretrained, map_location='cpu', weights_only=False)

            # Extract student weights from MEDiC checkpoint
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # Filter to student weights only (remove teacher, decoder, etc.)
            student_state_dict = {}
            for k, v in state_dict.items():
                # Handle different checkpoint formats
                if k.startswith('module.student.'):
                    new_key = k.replace('module.student.', 'backbone.')
                elif k.startswith('student.'):
                    new_key = k.replace('student.', 'backbone.')
                elif k.startswith('module.'):
                    new_key = k.replace('module.', 'backbone.')
                else:
                    new_key = 'backbone.' + k

                # Skip decoder, teacher, and head weights
                if any(skip in new_key for skip in ['decoder', 'teacher', 'head', 'fc_norm', 'distill']):
                    continue

                # Handle position embedding interpolation for different image sizes
                if 'pos_embed' in new_key and v is not None:
                    # Get current and checkpoint position embedding shapes
                    current_pos_embed = self.backbone.pos_embed
                    if current_pos_embed is not None:
                        checkpoint_pos_embed = v
                        embedding_size = checkpoint_pos_embed.shape[2]  # Should be 768

                        # Calculate grid sizes
                        num_extra_tokens = 1  # CLS token
                        orig_size = int((checkpoint_pos_embed.shape[1] - num_extra_tokens) ** 0.5)
                        # For detection, we don't know the final size at init, so skip interpolation here
                        # Position embedding interpolation will happen in forward_features if needed
                        pass

                student_state_dict[new_key] = v

            # Load with strict=False (FPN weights will be randomly initialized)
            msg = self.load_state_dict(student_state_dict, strict=False)

            if logger is not None:
                logger.info(f"Loaded pretrained MEDiC from: {pretrained}")
                logger.info(f"Missing keys: {len(msg.missing_keys)} (expected: FPN layers)")
                logger.info(f"Unexpected keys: {len(msg.unexpected_keys)}")
            else:
                print(f"Loaded pretrained MEDiC from: {pretrained}")
                print(f"Missing keys: {len(msg.missing_keys)} (expected FPN layers)")
                print(f"Unexpected keys: {len(msg.unexpected_keys)}")

            # Freeze stages if specified
            self._freeze_stages()

        elif pretrained is None:
            # Random initialization (default)
            pass
        else:
            raise TypeError('pretrained must be a str or None')

    def get_num_layers(self):
        """Get number of transformer blocks."""
        return len(self.backbone.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        """Specify parameters that should not have weight decay."""
        return {'backbone.pos_embed', 'backbone.cls_token', 'backbone.mask_token'}

    def interpolate_pos_encoding(self, x, w, h):
        """Interpolate position encoding for different input sizes.

        Args:
            x: Input tensor after patch embedding, shape (B, N, D)
            w: Width in patches
            h: Height in patches

        Returns:
            Interpolated position embedding
        """
        npatch = x.shape[1] - 1  # -1 for CLS token
        N = self.backbone.pos_embed.shape[1] - 1  # Original number of patches

        if npatch == N:
            # No interpolation needed
            return self.backbone.pos_embed

        # Separate CLS token position embedding
        class_pos_embed = self.backbone.pos_embed[:, 0:1, :]  # (1, 1, D)
        patch_pos_embed = self.backbone.pos_embed[:, 1:, :]   # (1, N, D)

        dim = self.backbone.pos_embed.shape[-1]

        # Original grid size
        orig_size = int(N ** 0.5)

        # Reshape to 2D grid and interpolate
        patch_pos_embed = patch_pos_embed.reshape(1, orig_size, orig_size, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)  # (1, D, H, W)

        # Bicubic interpolation
        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(h, w),
            mode='bicubic',
            align_corners=False
        )

        # Reshape back
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1)  # (1, H, W, D)
        patch_pos_embed = patch_pos_embed.reshape(1, -1, dim)   # (1, H*W, D)

        # Concatenate with CLS token
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward_features(self, x):
        """Extract multi-scale features from input images.

        Args:
            x: Input images, shape (B, C, H, W)

        Returns:
            tuple: Multi-scale features at different strides
        """
        B, C, H, W = x.shape

        # Patch embedding
        x = self.backbone.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        # Calculate spatial dimensions after patching
        Hp, Wp = H // self.patch_size, W // self.patch_size

        # Add CLS token
        cls_tokens = self.backbone.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add position embeddings with interpolation for different sizes
        if self.backbone.pos_embed is not None:
            # Interpolate position embedding if needed
            pos_embed = self.interpolate_pos_encoding(x, Wp, Hp)
            x = x + pos_embed
        x = self.backbone.pos_drop(x)

        # Get relative position bias if using it (should be None for detection)
        rel_pos_bias = None
        if self.backbone.rel_pos_bias is not None:
            rel_pos_bias = self.backbone.rel_pos_bias()

        # Forward through transformer blocks and collect features
        features = []
        for i, blk in enumerate(self.backbone.blocks):
            if self.use_checkpoint:
                # Gradient checkpointing to save memory
                # CRITICAL: use_reentrant=False is required for DDP compatibility
                # with LayerScale parameters (avoids "mark variable ready only once" error)
                x = checkpoint.checkpoint(blk, x, rel_pos_bias, use_reentrant=False)
            else:
                x = blk(x, rel_pos_bias)

            # Extract features at specified indices
            if i in self.out_indices:
                # Remove CLS token and reshape to spatial format
                # x[:, 0] is CLS token, x[:, 1:] are patch tokens
                xp = self.norm(x[:, 1:, :]).permute(0, 2, 1).reshape(B, -1, Hp, Wp)
                features.append(xp.contiguous())

        # Apply FPN to create multi-scale pyramid
        if self.with_fpn:
            ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
            for i in range(len(features)):
                features[i] = ops[i](features[i])

        return tuple(features)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input images, shape (B, C, H, W)

        Returns:
            tuple: Multi-scale features for detection neck/head
        """
        return self.forward_features(x)

    def train(self, mode=True):
        """Override train to handle frozen stages."""
        super(MEDiC, self).train(mode)
        self._freeze_stages()


# Register with mmdet if available
if MMDET_AVAILABLE:
    MEDiC = BACKBONES.register_module()(MEDiC)
