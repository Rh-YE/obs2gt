"""
Restormer-style encoder/decoder for VAE backbone comparison.
Reference: "Restormer: Efficient Transformer for High-Resolution Image Restoration"
           (CVPR 2022, Zamir et al.)

Key design:
  - MDTA (Multi-Dconv Head Transposed Attention): attention on channel dimension
    using transposed Q,K to keep O(HW*C) not O((HW)^2)
  - GDFN (Gated-Dconv Feed-Forward): depth-wise conv + gating

Encoder: hierarchical MDTA blocks + strided conv downsampling -> 2*z_channels
Decoder: symmetric MDTA blocks + bilinear upsample -> out_ch

Interface matches model.py Encoder/Decoder (same forward signatures).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LayerNorm2d(nn.Module):
    """Per-channel LayerNorm for NCHW tensors."""
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class MDTA(nn.Module):
    """
    Multi-Dconv Head Transposed Attention.
    Computes attention in the channel dimension:
      Q, K, V each shaped [B, heads, C/heads, HW] after depth-wise conv projection.
      Attention matrix is [heads, C/heads, C/heads] -> O(C^2) not O((HW)^2).
    """
    def __init__(self, channels, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5

        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.qkv_dw = nn.Conv2d(channels * 3, channels * 3, 3, padding=1,
                                  groups=channels * 3, bias=False)
        self.proj_out = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm = LayerNorm2d(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x
        x = self.norm(x)

        qkv = self.qkv_dw(self.qkv(x))          # [B, 3C, H, W]
        q, k, v = qkv.chunk(3, dim=1)             # each [B, C, H, W]

        # reshape to [B, heads, C//heads, HW]
        h = self.num_heads
        d = C // h
        q = q.reshape(B, h, d, H * W)
        k = k.reshape(B, h, d, H * W)
        v = v.reshape(B, h, d, H * W)

        # transposed attention: [B, h, d, d]
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale   # [B, h, d, d]
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v)              # [B, h, d, HW]
        out = out.reshape(B, C, H, W)
        return shortcut + self.proj_out(out)


class GDFN(nn.Module):
    """
    Gated-Dconv Feed-Forward Network.
    Two parallel depth-wise conv paths with gating.
    """
    def __init__(self, channels, expand=2.66):
        super().__init__()
        hidden = int(channels * expand)
        self.norm = LayerNorm2d(channels)
        self.proj_in  = nn.Conv2d(channels, hidden * 2, 1, bias=False)
        self.dw_conv  = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1,
                                   groups=hidden * 2, bias=False)
        self.proj_out = nn.Conv2d(hidden, channels, 1, bias=False)

    def forward(self, x):
        shortcut = x
        h = self.dw_conv(self.proj_in(self.norm(x)))   # [B, 2*hidden, H, W]
        h1, h2 = h.chunk(2, dim=1)
        h = h1 * F.gelu(h2)                            # gating
        return shortcut + self.proj_out(h)


class RestormerBlock(nn.Module):
    def __init__(self, channels, num_heads, ffn_expand=2.66):
        super().__init__()
        self.attn = MDTA(channels, num_heads)
        self.ffn  = GDFN(channels, ffn_expand)

    def forward(self, x):
        x = self.attn(x)
        x = self.ffn(x)
        return x


class RestormerEncoder(nn.Module):
    """
    Restormer-based VAE encoder.

    Args (YAML encoder_config.params compatible):
      in_channels   : 3 (implicit invvar mode)
      z_channels    : latent depth (encoder output is 2*z_channels)
      ch            : base channel count
      ch_mult       : per-stage channel multipliers
      num_res_blocks: Restormer blocks per stage
      num_heads     : attention heads per stage (scalar -> same for all stages)
      double_z      : True (required by DiagonalGaussianRegularizer)
      softplus_out  : ignored
    """
    def __init__(
        self,
        in_channels=3,
        z_channels=80,
        resolution=64,
        ch=48,
        ch_mult=(1, 2, 4),
        num_res_blocks=2,
        num_heads=1,
        double_z=True,
        softplus_out=False,
        ffn_expand=2.66,
        **ignore_kwargs,
    ):
        super().__init__()
        channels = [ch * m for m in ch_mult]
        # heads per stage: scale with channel count, min 1
        heads = [max(1, c // ch) * num_heads for c in channels]

        self.stem = nn.Conv2d(in_channels, channels[0], 3, padding=1)

        self.stages = nn.ModuleList()
        self.down_projs = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        in_ch = channels[0]
        for out_ch, h in zip(channels, heads):
            self.down_projs.append(nn.Conv2d(in_ch, out_ch, 1))
            self.stages.append(nn.Sequential(
                *[RestormerBlock(out_ch, h, ffn_expand) for _ in range(num_res_blocks)]
            ))
            self.down_convs.append(nn.Conv2d(out_ch, out_ch, 2, stride=2))
            in_ch = out_ch

        # bottleneck (same depth as last stage)
        self.mid = nn.Sequential(
            *[RestormerBlock(in_ch, heads[-1], ffn_expand) for _ in range(num_res_blocks)]
        )

        out_z = 2 * z_channels if double_z else z_channels
        self.head = nn.Conv2d(in_ch, out_z, 3, padding=1)

    def forward(self, x):
        h = self.stem(x)
        for proj, stage, down in zip(self.down_projs, self.stages, self.down_convs):
            h = proj(h)
            h = stage(h)
            h = down(h)
        h = self.mid(h)
        return self.head(h)


class RestormerDecoder(nn.Module):
    """
    Restormer-based VAE decoder (symmetric to RestormerEncoder).
    """
    def __init__(
        self,
        in_channels=3,       # accepted, unused
        z_channels=80,
        out_ch=3,
        ch=48,
        ch_mult=(1, 2, 4),
        num_res_blocks=2,
        num_heads=1,
        softplus_out=False,
        ffn_expand=2.66,
        double_z=True,       # accepted, unused
        resolution=64,       # accepted, unused
        **ignore_kwargs,
    ):
        super().__init__()
        self.softplus_out = softplus_out
        channels = [ch * m for m in ch_mult]
        rev_channels = list(reversed(channels))
        heads = [max(1, c // ch) * num_heads for c in reversed(channels)]

        self.stem = nn.Conv2d(z_channels, rev_channels[0], 3, padding=1)

        self.mid = nn.Sequential(
            *[RestormerBlock(rev_channels[0], heads[0], ffn_expand)
              for _ in range(num_res_blocks)]
        )

        self.stages = nn.ModuleList()
        self.up_projs = nn.ModuleList()
        in_ch = rev_channels[0]
        for out_ch_stage, h in zip(rev_channels, heads):
            self.up_projs.append(nn.Conv2d(in_ch, out_ch_stage, 1))
            self.stages.append(nn.Sequential(
                *[RestormerBlock(out_ch_stage, h, ffn_expand)
                  for _ in range(num_res_blocks + 1)]
            ))
            in_ch = out_ch_stage

        self.head = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def get_last_layer(self, **kwargs):
        return self.head.weight

    def forward(self, z, **kwargs):
        h = self.stem(z)
        h = self.mid(h)
        for proj, stage in zip(self.up_projs, self.stages):
            h = F.interpolate(h, scale_factor=2, mode='bilinear', align_corners=False)
            h = proj(h)
            h = stage(h)
        h = self.head(h)
        if self.softplus_out:
            h = F.softplus(h)
        return h
