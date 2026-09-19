"""
Base Transformer modules for both VAE and AE architectures.
Supports invvar (inverse variance) information for astronomical image processing.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Union
from einops import rearrange


def get_2d_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False):
    """
    Create 2D sin/cos positional embeddings.
    
    Args:
        embed_dim (int): Embedding dimension.
        grid_size (int): The grid height and width.
        add_cls_token (bool): Whether or not to add a classification (CLS) token.
    
    Returns:
        pos_embed: Position embeddings (grid_size*grid_size, embed_dim) or (1+grid_size*grid_size, embed_dim)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if add_cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")
    
    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    
    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")
    
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)
    
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product
    
    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)
    
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class PatchEmbedding(nn.Module):
    """
    Convert image to patch embeddings with support for invvar weighting.
    """
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768, 
                 with_invvar=False):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.with_invvar = with_invvar
        
        # Projection layer
        if with_invvar:
            # When using invvar, we concatenate image and invvar
            self.projection = nn.Conv2d(in_channels * 2, embed_dim, 
                                       kernel_size=patch_size, stride=patch_size)
        else:
            self.projection = nn.Conv2d(in_channels, embed_dim, 
                                       kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x, invvar=None):
        """
        Args:
            x: (B, C, H, W) image tensor
                如果with_invvar=True且invvar=None，则x应该已经包含拼接好的image+invvar/sigma
                如果with_invvar=True且invvar不为None，则在这里拼接
            invvar: (B, C, H, W) inverse variance tensor (optional)
        Returns:
            embeddings: (B, num_patches, embed_dim)
        """
        B, C, H, W = x.shape
        
        # 如果提供了单独的invvar参数，则拼接
        if self.with_invvar and invvar is not None:
            # Concatenate image and invvar along channel dimension
            x = torch.cat([x, invvar], dim=1)
        # 否则，如果with_invvar=True，假设x已经包含了拼接好的数据
        # projection层会根据初始化时的in_channels*2来处理
        
        # Project to embeddings
        x = self.projection(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        return x


class TransformerLayer(nn.Module):
    """
    Base transformer layer with self-attention and MLP.
    支持可选的cross-attention来处理invvar条件信息。
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.0, 
                 attention_dropout=0.0, use_cross_attention=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.use_cross_attention = use_cross_attention
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=attention_dropout, batch_first=True
        )
        
        # Optional cross-attention for invvar conditioning
        if use_cross_attention:
            self.norm_cross = nn.LayerNorm(hidden_size, eps=1e-6)
            self.cross_attention = nn.MultiheadAttention(
                hidden_size, num_heads, dropout=attention_dropout, batch_first=True
            )
        
        # MLP
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_size),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, context=None):
        """
        Args:
            x: (B, N, hidden_size) - 主要输入（图像tokens）
            context: (B, M, hidden_size) - 可选的条件信息（invvar tokens）
        Returns:
            x: (B, N, hidden_size)
        """
        # Self-attention with residual connection
        x_norm = self.norm1(x)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)
        
        # Cross-attention with context (if provided and enabled)
        if self.use_cross_attention and context is not None:
            x_norm = self.norm_cross(x)
            cross_attn_out, _ = self.cross_attention(x_norm, context, context)
            x = x + self.dropout(cross_attn_out)
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x


class BaseTransformerEncoder(nn.Module):
    """
    Base transformer encoder that can be used for both VAE and AE.
    支持通过cross-attention使用invvar作为条件信息。
    """
    def __init__(
        self,
        resolution=224,
        in_channels=3,
        patch_size=16,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.0,
        attention_dropout=0.0,
        with_invvar=False,
        use_invvar_cross_attention=False,  # 新增：是否使用cross-attention处理invvar
        **kwargs
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.with_invvar = with_invvar
        self.use_invvar_cross_attention = use_invvar_cross_attention
        
        # Calculate number of patches
        self.num_patches = (resolution // patch_size) ** 2
        
        # Patch embedding for image
        self.patch_embed = PatchEmbedding(
            image_size=resolution,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=hidden_size,
            with_invvar=False  # 图像单独处理
        )
        
        # 如果使用cross-attention，为invvar创建单独的patch embedding
        if with_invvar and use_invvar_cross_attention:
            self.invvar_patch_embed = PatchEmbedding(
                image_size=resolution,
                patch_size=patch_size,
                in_channels=in_channels,  # invvar通道数与图像相同
                embed_dim=hidden_size,
                with_invvar=False
            )
            # invvar的位置编码（与图像共享）
            self.invvar_pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches, hidden_size), requires_grad=False
            )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        
        # Position embeddings (fixed sin-cos)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, hidden_size), requires_grad=False
        )
        
        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout,
                use_cross_attention=use_invvar_cross_attention  # 启用cross-attention
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        
        # Initialize weights
        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize position embeddings
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], 
            int(self.num_patches ** 0.5), 
            add_cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Initialize invvar position embeddings if using cross-attention
        if self.with_invvar and self.use_invvar_cross_attention:
            invvar_pos_embed = get_2d_sincos_pos_embed(
                self.invvar_pos_embed.shape[-1],
                int(self.num_patches ** 0.5),
                add_cls_token=False  # invvar不需要cls token
            )
            self.invvar_pos_embed.data.copy_(torch.from_numpy(invvar_pos_embed).float().unsqueeze(0))
        
        # Initialize cls token
        torch.nn.init.normal_(self.cls_token, std=0.02)
        
        # Initialize other layers
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight.view([m.weight.shape[0], -1]))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward_features(self, x, invvar=None):
        """
        Forward pass to get features (without final projection).
        
        Args:
            x: (B, C, H, W) image tensor 或 (B, 2*C, H, W) 拼接了invvar的tensor
            invvar: (B, C, H, W) inverse variance tensor (optional)
                   如果提供了invvar且use_invvar_cross_attention=True，会使用cross-attention
                   如果没提供invvar但x包含拼接的invvar，会自动分离
        Returns:
            features: (B, num_patches+1, hidden_size) with CLS token
        """
        B = x.shape[0]
        
        # 处理输入：分离图像和invvar
        if self.with_invvar and self.use_invvar_cross_attention:
            if invvar is None:
                # 如果没有单独提供invvar，从x中分离
                c = x.shape[1]
                if c == 2 * self.in_channels:
                    img = x[:, :self.in_channels, :, :]
                    invvar = x[:, self.in_channels:, :, :]
                else:
                    img = x
                    invvar = None
            else:
                img = x
            
            # 对图像进行patch embedding
            x_patches = self.patch_embed(img, None)  # (B, num_patches, hidden_size)
            
            # 对invvar进行patch embedding（如果提供了）
            if invvar is not None:
                invvar_patches = self.invvar_patch_embed(invvar, None)  # (B, num_patches, hidden_size)
                # 添加位置编码到invvar
                invvar_context = invvar_patches + self.invvar_pos_embed
            else:
                invvar_context = None
        else:
            # 不使用cross-attention，使用原来的方式（拼接）
            x_patches = self.patch_embed(x, invvar)  # (B, num_patches, hidden_size)
            invvar_context = None
        
        # Add cls token to image patches
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x_patches], dim=1)  # (B, num_patches+1, hidden_size)
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Apply transformer layers with cross-attention
        for layer in self.layers:
            x = layer(x, context=invvar_context)
        
        # Final layer norm
        x = self.norm(x)
        
        return x


class BaseTransformerDecoder(nn.Module):
    """
    Base transformer decoder that can be used for both VAE and AE.
    """
    def __init__(
        self,
        resolution=224,
        out_ch=3,
        patch_size=16,
        hidden_size=512,
        num_layers=8,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
        attention_dropout=0.0,
        with_invvar=False,
        predict_invvar=False,
        positive_output=False,  # 是否对输出应用恒正约束
        output_activation='softplus',  # 恒正约束的激活函数：'softplus', 'relu', 'exp'
        **kwargs
    ):
        super().__init__()
        self.resolution = resolution
        self.out_ch = out_ch
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.with_invvar = with_invvar
        self.predict_invvar = predict_invvar
        self.positive_output = positive_output
        self.output_activation = output_activation
        
        # Calculate number of patches
        self.num_patches = (resolution // patch_size) ** 2
        self.latent_size = resolution // patch_size
        
        # Learnable CLS token for decoder
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        
        # Position embeddings (fixed sin-cos)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, hidden_size), requires_grad=False
        )
        
        # Transformer decoder layers
        self.layers = nn.ModuleList([
            TransformerLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        
        # Prediction head to reconstruct patches
        self.pred_head = nn.Linear(
            hidden_size, 
            patch_size ** 2 * out_ch, 
            bias=True
        )
        
        # Optional invvar prediction head
        if predict_invvar:
            self.invvar_head = nn.Linear(
                hidden_size,
                patch_size ** 2 * out_ch,
                bias=True
            )
        
        # Initialize weights
        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize position embeddings
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], 
            int(self.num_patches ** 0.5), 
            add_cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Initialize cls token
        torch.nn.init.normal_(self.cls_token, std=0.02)
        
        # Initialize other layers
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def unpatchify(self, x):
        """
        Args:
            x: (B, num_patches, patch_size**2 * out_ch)
        Returns:
            imgs: (B, out_ch, H, W)
        """
        p = self.patch_size
        h = w = int(self.num_patches ** 0.5)
        
        x = x.reshape(x.shape[0], h, w, p, p, self.out_ch)
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], self.out_ch, h * p, w * p)
        return imgs
    
    def forward_decode(self, x):
        """
        Decode features to image.
        
        Args:
            x: (B, num_patches, hidden_size) features without CLS token
        Returns:
            x_rec: (B, out_ch, resolution, resolution) reconstructed image
            invvar_rec: (B, out_ch, resolution, resolution) predicted invvar (if predict_invvar=True)
        """
        B = x.shape[0]
        
        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, hidden_size)
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Final layer norm
        x = self.norm(x)
        
        # Remove cls token
        x = x[:, 1:, :]  # (B, num_patches, hidden_size)
        
        # Predict pixel values
        x_pred = self.pred_head(x)  # (B, num_patches, patch_size**2 * out_ch)
        
        # Unpatchify to get image
        x_rec = self.unpatchify(x_pred)  # (B, out_ch, resolution, resolution)
        
        # Apply positive constraint if enabled
        if self.positive_output:
            if self.output_activation == 'softplus':
                x_rec = torch.nn.functional.softplus(x_rec)
            elif self.output_activation == 'relu':
                x_rec = torch.nn.functional.relu(x_rec)
            elif self.output_activation == 'exp':
                x_rec = torch.exp(x_rec)
            else:
                raise ValueError(f"Unknown output activation: {self.output_activation}")
        
        # Optionally predict invvar
        if self.predict_invvar:
            invvar_pred = self.invvar_head(x)
            invvar_rec = self.unpatchify(invvar_pred)
            # Apply softplus to ensure positive invvar
            invvar_rec = torch.nn.functional.softplus(invvar_rec)
            return x_rec, invvar_rec
        
        return x_rec
    
    def get_last_layer(self):
        """
        Return the last layer for discriminator training.
        """
        return self.pred_head.weight
