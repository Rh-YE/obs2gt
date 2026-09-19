"""
Transformer-based Decoder for VAE, using base transformer modules.
Supports invvar (inverse variance) information for astronomical image processing.
"""

import torch
import torch.nn as nn
from typing import Optional
from einops import rearrange

from .transformer_base import BaseTransformerDecoder


class TransformerVAEDecoder(BaseTransformerDecoder):
    """
    Transformer-based decoder for VAE, inspired by ViT-MAE decoder architecture.
    Supports invvar (inverse variance) information for astronomical image processing.
    """
    def __init__(
        self,
        resolution=224,
        out_ch=3,
        patch_size=16,
        z_channels=64,
        hidden_size=512,
        num_layers=8,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
        attention_dropout=0.0,
        with_invvar=False,
        predict_invvar=False,
        **kwargs
    ):
        super().__init__(
            resolution=resolution,
            out_ch=out_ch,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            with_invvar=with_invvar,
            predict_invvar=predict_invvar,
            **kwargs
        )
        
        self.z_channels = z_channels
        
        # Embedding layer to project latent to decoder hidden size
        self.decoder_embed = nn.Linear(z_channels, hidden_size, bias=True)
    
    def forward(self, z):
        """
        Args:
            z: (B, z_channels, H, W) latent representation
        Returns:
            x: (B, out_ch, resolution, resolution) reconstructed image
            invvar: (B, out_ch, resolution, resolution) predicted invvar (if predict_invvar=True)
        """
        B, C, H, W = z.shape
        
        # Interpolate latent if needed
        z = self.interpolate_latent(z)
        
        # Reshape latent to sequence format
        z = rearrange(z, 'b c h w -> b (h w) c')
        
        # Embed latent to decoder hidden size
        x = self.decoder_embed(z)  # (B, num_patches, hidden_size)
        
        # Decode to image using base decoder
        return self.forward_decode(x)


# Alias for backward compatibility
class TransformerDecoder(TransformerVAEDecoder):
    """Alias for TransformerVAEDecoder for backward compatibility."""
    pass
