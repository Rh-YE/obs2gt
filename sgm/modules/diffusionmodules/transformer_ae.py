"""
Transformer-based Autoencoder (AE) modules, inspired by RAE-main architecture.
Supports invvar (inverse variance) information for astronomical image processing.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from einops import rearrange

from .transformer_base import BaseTransformerEncoder, BaseTransformerDecoder


class TransformerAEEncoder(BaseTransformerEncoder):
    """
    Transformer-based encoder for AE (not VAE).
    Outputs feature representations directly without probabilistic modeling.
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
        latent_dim=512,
        with_invvar=False,
        output_format='spatial',  # 'spatial' or 'sequence'
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
        
        self.latent_dim = latent_dim
        self.output_format = output_format
        
        # Output projection to latent space
        self.head = nn.Linear(hidden_size, latent_dim)
    
    def forward(self, x, invvar=None):
        """
        Args:
            x: (B, C, H, W) image tensor
            invvar: (B, C, H, W) inverse variance tensor (optional)
        Returns:
            z: latent representation
                - if output_format='spatial': (B, latent_dim, H', W') 
                - if output_format='sequence': (B, num_patches, latent_dim)
        """
        # Get features from base encoder
        features = self.forward_features(x, invvar)  # (B, num_patches+1, hidden_size)
        
        # Remove cls token and project to latent space
        patch_features = features[:, 1:, :]  # (B, num_patches, hidden_size)
        z = self.head(patch_features)  # (B, num_patches, latent_dim)
        
        if self.output_format == 'spatial':
            # Reshape to spatial format
            h = w = int(self.num_patches ** 0.5)
            z = rearrange(z, 'b (h w) c -> b c h w', h=h, w=w)
        
        return z


class TransformerAEDecoder(BaseTransformerDecoder):
    """
    Transformer-based decoder for AE.
    Takes latent representations and reconstructs images.
    """
    def __init__(
        self,
        resolution=224,
        out_ch=3,
        patch_size=16,
        latent_dim=512,
        hidden_size=512,
        num_layers=8,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
        attention_dropout=0.0,
        with_invvar=False,
        predict_invvar=False,
        input_format='spatial',  # 'spatial' or 'sequence'
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
        
        self.latent_dim = latent_dim
        self.input_format = input_format
        
        # Add z_shape attribute for compatibility with autoencoder.py
        # Format: (batch_size, channels, height, width) -> we use (channels, height, width)
        latent_spatial_size = resolution // patch_size
        self.z_shape = (1, latent_dim, latent_spatial_size, latent_spatial_size)
        
        # Embedding layer to project latent to decoder hidden size
        self.decoder_embed = nn.Linear(latent_dim, hidden_size, bias=True)
    
    def interpolate_latent(self, z):
        """
        Interpolate latent if its spatial size doesn't match expected size.
        
        Args:
            z: (B, C, H, W) latent tensor
        Returns:
            z: (B, C, latent_size, latent_size) interpolated latent
        """
        B, C, H, W = z.shape
        if H != self.latent_size or W != self.latent_size:
            z = torch.nn.functional.interpolate(
                z, 
                size=(self.latent_size, self.latent_size), 
                mode='bilinear', 
                align_corners=False
            )
        return z
    
    def forward(self, z):
        """
        Args:
            z: latent representation
                - if input_format='spatial': (B, latent_dim, H', W')
                - if input_format='sequence': (B, num_patches, latent_dim)
        Returns:
            x: (B, out_ch, resolution, resolution) reconstructed image
            invvar: (B, out_ch, resolution, resolution) predicted invvar (if predict_invvar=True)
        """
        if self.input_format == 'spatial':
            B, C, H, W = z.shape
            
            # Interpolate latent if needed
            z = self.interpolate_latent(z)
            
            # Reshape latent to sequence format
            z = rearrange(z, 'b c h w -> b (h w) c')
        else:
            # Already in sequence format
            B = z.shape[0]
        
        # Embed latent to decoder hidden size
        x = self.decoder_embed(z)  # (B, num_patches, hidden_size)
        
        # Decode to image
        result = self.forward_decode(x)
        
        return result


class TransformerAutoencoder(nn.Module):
    """
    Complete Transformer-based Autoencoder combining encoder and decoder.
    Similar to RAE architecture but adapted for astronomical images with invvar support.
    """
    def __init__(
        self,
        # Image parameters
        resolution=224,
        in_channels=3,
        out_channels=None,
        patch_size=16,
        
        # Architecture parameters
        encoder_hidden_size=768,
        encoder_num_layers=12,
        encoder_num_heads=12,
        decoder_hidden_size=512,
        decoder_num_layers=8,
        decoder_num_heads=8,
        latent_dim=512,
        
        # Training parameters
        mlp_ratio=4.0,
        dropout=0.0,
        attention_dropout=0.0,
        
        # Special features
        with_invvar=False,
        predict_invvar=False,
        
        # Data flow format
        latent_format='spatial',  # 'spatial' or 'sequence'
        
        **kwargs
    ):
        super().__init__()
        
        if out_channels is None:
            out_channels = in_channels
        
        self.resolution = resolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.with_invvar = with_invvar
        self.predict_invvar = predict_invvar
        self.latent_format = latent_format
        
        # Encoder
        self.encoder = TransformerAEEncoder(
            resolution=resolution,
            in_channels=in_channels,
            patch_size=patch_size,
            hidden_size=encoder_hidden_size,
            num_layers=encoder_num_layers,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            latent_dim=latent_dim,
            with_invvar=with_invvar,
            output_format=latent_format,
            **kwargs
        )
        
        # Decoder
        self.decoder = TransformerAEDecoder(
            resolution=resolution,
            out_ch=out_channels,
            patch_size=patch_size,
            latent_dim=latent_dim,
            hidden_size=decoder_hidden_size,
            num_layers=decoder_num_layers,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            with_invvar=with_invvar,
            predict_invvar=predict_invvar,
            input_format=latent_format,
            **kwargs
        )
    
    def encode(self, x, invvar=None):
        """
        Encode image to latent representation.
        
        Args:
            x: (B, C, H, W) image tensor
            invvar: (B, C, H, W) inverse variance tensor (optional)
        Returns:
            z: latent representation
        """
        return self.encoder(x, invvar)
    
    def decode(self, z):
        """
        Decode latent representation to image.
        
        Args:
            z: latent representation
        Returns:
            x_rec: reconstructed image
            invvar_rec: predicted invvar (if predict_invvar=True)
        """
        return self.decoder(z)
    
    def forward(self, x, invvar=None):
        """
        Full forward pass: encode then decode.
        
        Args:
            x: (B, C, H, W) image tensor
            invvar: (B, C, H, W) inverse variance tensor (optional)
        Returns:
            x_rec: reconstructed image
            z: latent representation
            invvar_rec: predicted invvar (if predict_invvar=True, else None)
        """
        # Encode
        z = self.encode(x, invvar)
        
        # Decode
        decode_result = self.decode(z)
        
        if self.predict_invvar:
            x_rec, invvar_rec = decode_result
            return x_rec, z, invvar_rec
        else:
            x_rec = decode_result
            return x_rec, z, None
    
    def get_last_layer(self):
        """
        Return the last layer for discriminator training.
        """
        return self.decoder.get_last_layer()


# Compatibility aliases for easy switching between VAE and AE
class TransformerEncoder(TransformerAEEncoder):
    """Alias for TransformerAEEncoder for backward compatibility."""
    pass


class TransformerDecoder(TransformerAEDecoder):
    """Alias for TransformerAEDecoder for backward compatibility."""
    pass
