"""
NAFNet-style encoder/decoder for VAE backbone comparison.
Reference: "Simple Baselines for Image Restoration" (ECCV 2022).

Encoder: NAF blocks (SimpleGate + channel attention) with strided conv downsampling
         -> outputs 2*z_channels feature map (mean+logvar for DiagonalGaussianRegularizer)
Decoder: symmetric NAF blocks with bilinear upsampling
         -> outputs out_ch feature map

Interface matches sgm/modules/diffusionmodules/model.py Encoder/Decoder:
  Encoder.forward(x)          -> [B, 2*z_channels, H/s, W/s]
  Decoder.forward(z, **kw)    -> [B, out_ch, H, W]
  Decoder.get_last_layer()    -> weight tensor for adaptive loss weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Per-channel LayerNorm for 2-D feature maps (NCHW)."""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # x: [B, C, H, W] -> norm over C dim
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class SimpleGate(nn.Module):
    """Split channel-wise and gate: y = x1 * x2."""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """
    NAFNet basic block.
    Input/output channels: c  (no expansion outside)
    Internal: expand to 2*dw_expand*c for SimpleGate DWConv, 2*ffn_expand*c for FFN.
    """
    def __init__(self, c, dw_expand=2, ffn_expand=2, drop=0.0):
        super().__init__()
        dw_ch = c * dw_expand
        ffn_ch = c * ffn_expand

        # Depthwise branch
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw_ch * 2, 1)        # pointwise in
        self.conv2 = nn.Conv2d(dw_ch * 2, dw_ch * 2, 3, padding=1, groups=dw_ch * 2)  # depthwise
        self.gate  = SimpleGate()                        # dw_ch*2 -> dw_ch
        # Simplified channel attention (mean-pool)
        self.sca    = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_ch, dw_ch, 1),
        )
        self.conv3  = nn.Conv2d(dw_ch, c, 1)            # pointwise out

        # FFN branch
        self.norm2 = LayerNorm2d(c)
        self.conv4 = nn.Conv2d(c, ffn_ch * 2, 1)   # c -> ffn_ch*2
        self.gate2 = SimpleGate()                    # ffn_ch*2 -> ffn_ch
        self.conv5 = nn.Conv2d(ffn_ch, c, 1)        # ffn_ch -> c

        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

        # learnable residual scalings (near-zero init -> near-identity at start)
        self.beta  = nn.Parameter(torch.ones(1, c, 1, 1) * 1e-3)
        self.gamma = nn.Parameter(torch.ones(1, c, 1, 1) * 1e-3)

    def forward(self, x):
        inp = x
        # --- depthwise attention branch ---
        h = self.norm1(inp)
        h = self.conv1(h)
        h = self.conv2(h)
        h = self.gate(h)       # SimpleGate: dw_ch*2 -> dw_ch
        h = h * self.sca(h)
        h = self.conv3(h)
        h = self.drop(h)
        y = inp + h * self.beta

        # --- FFN branch ---
        h = self.norm2(y)
        h = self.conv4(h)      # c -> ffn_ch*2
        h = self.gate2(h)      # SimpleGate: ffn_ch*2 -> ffn_ch; conv5 maps ffn_ch -> c
        h = self.conv5(h)
        h = self.drop(h)
        return y + h * self.gamma


class NAFEncoder(nn.Module):
    """
    NAFNet encoder for VAE.

    Architecture:
      stem conv (in_channels -> enc_blks[0] channels)
      N encoder stages: each stage has num_res_blocks NAFBlocks, then strided conv
      bottleneck: mid_blks NAFBlocks
      head conv -> 2*z_channels (mean + logvar)

    Args (compatible with YAML encoder_config.params):
      in_channels   : input image channels (3 for implicit invvar)
      z_channels    : latent channels (encoder outputs 2*z_channels)
      resolution    : spatial input size (must be divisible by 2^n_stages)
      ch            : base channel width
      ch_mult       : tuple of channel multipliers per down-stage
      num_res_blocks: NAFBlocks per stage
      double_z      : always True for DiagonalGaussianRegularizer
      softplus_out  : ignored (encoder has no activation on output)
      dropout       : dropout rate in NAFBlocks
    """
    def __init__(
        self,
        in_channels=3,
        z_channels=80,
        resolution=64,
        ch=64,
        ch_mult=(1, 2, 4),
        num_res_blocks=2,
        double_z=True,
        dropout=0.0,
        softplus_out=False,   # unused, accepted for config compat
        mid_blks=4,
        **ignore_kwargs,
    ):
        super().__init__()
        self.double_z = double_z
        n_stages = len(ch_mult)
        channels = [ch * m for m in ch_mult]

        # stem
        self.stem = nn.Conv2d(in_channels, channels[0], 3, padding=1)

        # encoder stages: project in_ch -> out_ch first, then NAFBlocks at out_ch
        self.downs = nn.ModuleList()
        self.down_projs = nn.ModuleList()   # 1x1 proj in_ch -> out_ch before blocks
        self.down_convs = nn.ModuleList()   # strided-2 conv for spatial downsampling
        in_ch = channels[0]
        for i, out_ch in enumerate(channels):
            self.down_projs.append(nn.Conv2d(in_ch, out_ch, 1))
            stage = nn.Sequential(*[NAFBlock(out_ch, drop=dropout)
                                    for _ in range(num_res_blocks)])
            self.downs.append(stage)
            self.down_convs.append(nn.Conv2d(out_ch, out_ch, 2, stride=2))
            in_ch = out_ch

        # bottleneck
        self.mid = nn.Sequential(*[NAFBlock(in_ch, drop=dropout) for _ in range(mid_blks)])

        # head: -> 2*z_channels
        out_z = (2 * z_channels) if double_z else z_channels
        self.head = nn.Conv2d(in_ch, out_z, 3, padding=1)

    def forward(self, x):
        h = self.stem(x)
        for proj, stage, down in zip(self.down_projs, self.downs, self.down_convs):
            h = proj(h)
            h = stage(h)
            h = down(h)
        h = self.mid(h)
        return self.head(h)


class NAFDecoder(nn.Module):
    """
    NAFNet decoder for VAE (symmetric to NAFEncoder).

    Architecture:
      stem conv (z_channels -> channels[-1])
      bottleneck: mid_blks NAFBlocks
      N decoder stages: bilinear upsample then num_res_blocks+1 NAFBlocks
      head -> out_ch with optional softplus

    Args (compatible with YAML decoder_config.params, same as encoder_config.params):
      z_channels    : latent channels (decoder input)
      out_ch        : output image channels
      ch            : base channel width
      ch_mult       : tuple of channel multipliers (reversed for upsampling)
      num_res_blocks: NAFBlocks per stage
      softplus_out  : apply softplus to output (match existing decoder behavior)
    """
    def __init__(
        self,
        in_channels=3,      # accepted but unused (for config compat)
        z_channels=80,
        out_ch=3,
        ch=64,
        ch_mult=(1, 2, 4),
        num_res_blocks=2,
        softplus_out=False,
        dropout=0.0,
        mid_blks=4,
        double_z=True,      # accepted, unused
        resolution=64,      # accepted, unused
        **ignore_kwargs,
    ):
        super().__init__()
        self.softplus_out = softplus_out
        channels = [ch * m for m in ch_mult]   # e.g. [64, 128, 256]
        # decoder processes from deepest to shallowest
        rev_channels = list(reversed(channels))  # [256, 128, 64]

        # stem: z_channels -> deepest channel
        self.stem = nn.Conv2d(z_channels, rev_channels[0], 3, padding=1)

        # bottleneck
        self.mid = nn.Sequential(*[NAFBlock(rev_channels[0], drop=dropout)
                                   for _ in range(mid_blks)])

        # decoder stages
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        in_ch = rev_channels[0]
        for i, out_ch_stage in enumerate(rev_channels):
            # upsample projection
            self.up_convs.append(nn.Conv2d(in_ch, out_ch_stage, 1))
            stage = nn.Sequential(*[NAFBlock(out_ch_stage, drop=dropout)
                                    for _ in range(num_res_blocks + 1)])
            self.ups.append(stage)
            in_ch = out_ch_stage

        # head
        self.head = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def get_last_layer(self, **kwargs):
        return self.head.weight

    def forward(self, z, **kwargs):
        h = self.stem(z)
        h = self.mid(h)
        for up_conv, stage in zip(self.up_convs, self.ups):
            h = F.interpolate(h, scale_factor=2, mode='bilinear', align_corners=False)
            h = up_conv(h)
            h = stage(h)
        h = self.head(h)
        if self.softplus_out:
            h = F.softplus(h)
        return h
