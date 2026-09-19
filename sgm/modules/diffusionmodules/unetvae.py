"""
U-Net encoder/decoder for VAE backbone comparison.

Classic U-Net with skip connections:
  Encoder: conv blocks + MaxPool downsampling, saves skip features per level
  Decoder: transposed-conv upsampling + skip concat + conv blocks
           -> outputs 2*z_channels from encoder, out_ch from decoder

Skip connections are stored as instance state during encode() and consumed
during decode().  The VAE forward() calls encode() then decode() in sequence,
so this is safe in the standard AutoencodingEngine training loop.

Interface matches model.py Encoder/Decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _double_conv(in_ch, out_ch, dropout=0.0):
    layers = [
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.GroupNorm(min(32, out_ch), out_ch),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.GroupNorm(min(32, out_ch), out_ch),
        nn.SiLU(inplace=True),
    ]
    if dropout > 0:
        layers.append(nn.Dropout2d(dropout))
    return nn.Sequential(*layers)


class UNetEncoder(nn.Module):
    """
    U-Net encoder for VAE.  Stores skip activations as self.skips for the
    paired UNetDecoder to consume.

    Args (YAML encoder_config.params compatible):
      in_channels   : 3 (implicit invvar)
      z_channels    : latent channels (output is 2*z_channels)
      ch            : base channel width
      ch_mult       : per-level channel multipliers (defines depth)
      num_res_blocks: ignored (U-Net uses double-conv per level, not ResBlocks)
      double_z      : True for DiagonalGaussianRegularizer
      softplus_out  : ignored
    """
    def __init__(
        self,
        in_channels=3,
        z_channels=80,
        resolution=64,
        ch=64,
        ch_mult=(1, 2, 4),
        num_res_blocks=2,   # accepted, unused
        double_z=True,
        softplus_out=False,
        dropout=0.0,
        **ignore_kwargs,
    ):
        super().__init__()
        self.double_z = double_z
        channels = [ch * m for m in ch_mult]

        # Encoder blocks (one per level, each followed by maxpool)
        self.enc_blocks = nn.ModuleList()
        in_ch = in_channels
        for out_ch in channels:
            self.enc_blocks.append(_double_conv(in_ch, out_ch, dropout))
            in_ch = out_ch

        # Bottleneck (deepest level, no skip)
        bot_ch = channels[-1] * 2
        self.bottleneck = _double_conv(channels[-1], bot_ch, dropout)

        # Head: bottleneck -> 2*z_channels
        out_z = 2 * z_channels if double_z else z_channels
        self.head = nn.Conv2d(bot_ch, out_z, 1)

        self.pool = nn.MaxPool2d(2)
        self.skips = []   # populated during forward

    def forward(self, x):
        self.skips = []
        h = x
        for block in self.enc_blocks:
            h = block(h)
            self.skips.append(h)    # save before pooling
            h = self.pool(h)
        h = self.bottleneck(h)
        return self.head(h)


class UNetDecoder(nn.Module):
    """
    U-Net decoder for VAE.  Consumes skip activations stored by the paired
    UNetEncoder instance.

    The encoder instance is passed at construction time so decoder can access
    .skips at forward time.
    """
    def __init__(
        self,
        encoder: UNetEncoder,
        out_ch=3,
        softplus_out=False,
        dropout=0.0,
        z_channels=80,
        **ignore_kwargs,
    ):
        super().__init__()
        self.encoder_ref = encoder
        self.softplus_out = softplus_out

        # Rebuild channel list from encoder blocks (first conv weight shape[0] = out_ch)
        enc_channels = [block[0].weight.shape[0] for block in encoder.enc_blocks]
        bot_ch = enc_channels[-1] * 2   # bottleneck channels

        # Project z_channels -> bottleneck before upsampling
        self.stem = nn.Conv2d(z_channels, bot_ch, 1)

        # Decoder stages (reversed)
        self.up_convs = nn.ModuleList()   # transposed conv for upsample+project
        self.dec_blocks = nn.ModuleList()

        in_ch = bot_ch
        for skip_ch in reversed(enc_channels):
            # upsample: in_ch -> skip_ch (to match skip shape for concat)
            self.up_convs.append(
                nn.ConvTranspose2d(in_ch, skip_ch, 2, stride=2)
            )
            # after concat: skip_ch + skip_ch -> skip_ch
            self.dec_blocks.append(_double_conv(skip_ch * 2, skip_ch, dropout))
            in_ch = skip_ch

        self.head = nn.Conv2d(in_ch, out_ch, 1)

    def get_last_layer(self, **kwargs):
        return self.head.weight

    def forward(self, z, **kwargs):
        skips = self.encoder_ref.skips   # list of skip tensors, shallowest last

        h = self.stem(z)   # z_channels -> bot_ch

        for up_conv, dec_block, skip in zip(
            self.up_convs, self.dec_blocks, reversed(skips)
        ):
            h = up_conv(h)
            # Handle potential size mismatch from odd resolutions
            if h.shape != skip.shape:
                h = F.interpolate(h, size=skip.shape[2:], mode='bilinear',
                                  align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = dec_block(h)

        h = self.head(h)
        if self.softplus_out:
            h = F.softplus(h)
        return h


def build_unet_vae(
    in_channels=3,
    z_channels=80,
    out_ch=3,
    ch=64,
    ch_mult=(1, 2, 4),
    num_res_blocks=2,
    double_z=True,
    softplus_out=False,
    dropout=0.0,
    resolution=64,
    **ignore_kwargs,
):
    """Factory: returns (encoder, decoder) with decoder holding encoder ref."""
    enc = UNetEncoder(
        in_channels=in_channels, z_channels=z_channels, resolution=resolution,
        ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
        double_z=double_z, softplus_out=softplus_out, dropout=dropout,
    )
    dec = UNetDecoder(
        encoder=enc, out_ch=out_ch, softplus_out=softplus_out, dropout=dropout,
    )
    return enc, dec


# ──────────────────────────────────────────────────────────────────────────────
# Standalone wrappers that match AutoencodingEngine's instantiate_from_config
# interface (each takes the same flat kwargs as ConvVAE Encoder/Decoder).
#
# The UNetDecoder needs a reference to the encoder to read skip activations.
# We solve this via a shared SkipStore singleton keyed by (model_id).
# AutoencodingEngine creates encoder first, then decoder, both from config.
# We use a module-level registry so decoder can find encoder at runtime.
# ──────────────────────────────────────────────────────────────────────────────

_encoder_registry: dict = {}   # instance_id -> UNetEncoder


class UNetEncoderStandalone(UNetEncoder):
    """
    Standalone encoder that registers itself in _encoder_registry.
    decoder_key must match the one passed to UNetDecoderStandalone.
    """
    def __init__(self, decoder_key="default", **kwargs):
        super().__init__(**kwargs)
        self._decoder_key = decoder_key
        _encoder_registry[decoder_key] = self


class UNetDecoderStandalone(nn.Module):
    """
    Standalone decoder that retrieves the encoder from _encoder_registry
    at first forward call (lazy init).
    """
    def __init__(
        self,
        decoder_key="default",
        out_ch=3,
        softplus_out=False,
        dropout=0.0,
        # Accept all encoder params for config parity (unused here)
        in_channels=3, z_channels=80, ch=64, ch_mult=(1,2,4),
        num_res_blocks=2, double_z=True, resolution=64,
        **ignore_kwargs,
    ):
        super().__init__()
        self._decoder_key = decoder_key
        self._out_ch = out_ch
        self._softplus_out = softplus_out
        self._dropout = dropout
        self._z_channels = z_channels
        self._built = False
        # Build immediately if encoder is already registered (normal usage)
        if decoder_key in _encoder_registry:
            self._build()

    def _build(self):
        enc = _encoder_registry.get(self._decoder_key)
        if enc is None:
            raise RuntimeError(
                f"UNetDecoderStandalone: no encoder registered under key "
                f"'{self._decoder_key}'. Ensure UNetEncoderStandalone with "
                f"the same decoder_key is instantiated first."
            )
        inner = UNetDecoder(
            encoder=enc,
            out_ch=self._out_ch,
            softplus_out=self._softplus_out,
            dropout=self._dropout,
            z_channels=self._z_channels,
        )
        # Register as proper nn.Module child so parameters() / state_dict() work
        self.add_module('inner', inner)
        self._built = True

    def get_last_layer(self, **kwargs):
        if not self._built:
            self._build()
        return self.inner.get_last_layer(**kwargs)

    def forward(self, z, **kwargs):
        if not self._built:
            self._build()
        return self.inner(z, **kwargs)

