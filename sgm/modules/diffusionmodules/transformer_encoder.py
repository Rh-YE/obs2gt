"""
Transformer-based Encoder for VAE, inspired by RAE-main architecture.
Supports invvar (inverse variance) information for astronomical image processing.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from einops import rearrange

from .transformer_base import BaseTransformerEncoder


class TransformerEncoder(BaseTransformerEncoder):
    """
    Transformer-based encoder for VAE, inspired by ViT and MAE architectures.
    Supports invvar information for astronomical image processing.
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
        z_channels=64,
        double_z=True,
        with_invvar=False,
        **kwargs
    ):
        super().__init__(
            resolution=resolution,
            in_channels=in_channels,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            with_invvar=with_invvar,
            **kwargs
        )
        
        self.z_channels = z_channels
        self.double_z = double_z
        
        # Output projection to latent space
        out_channels = z_channels * 2 if double_z else z_channels
        self.head = nn.Linear(hidden_size, out_channels)
    
    def forward(self, x, invvar=None):
        """
        Args:
            x: (B, C, H, W) image tensor
            invvar: (B, C, H, W) inverse variance tensor (optional)
        Returns:
            z: (B, z_channels, H', W') latent representation
               where H' = W' = resolution // patch_size
        """
        # Get features from base encoder
        features = self.forward_features(x, invvar)  # (B, num_patches+1, hidden_size)
        
        # Remove cls token and project to latent space
        x = features[:, 1:, :]  # (B, num_patches, hidden_size)
        z = self.head(x)  # (B, num_patches, z_channels*2 or z_channels)
        
        # Reshape to spatial format
        h = w = int(self.num_patches ** 0.5)
        z = rearrange(z, 'b (h w) c -> b c h w', h=h, w=w)
        
        return z

