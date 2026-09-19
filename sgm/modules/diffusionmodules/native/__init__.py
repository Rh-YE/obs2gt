"""Native (image->image, no VAE bottleneck) SOTA restoration backbones.

Each arch file is the *original* model definition pulled verbatim from the
official repository (see header of each file). The thin wrappers below only
adapt the entry signature to a uniform `forward(x: [B,C,H,W]) -> [B,C,H,W]`
interface and expose `in_channels` / `out_channels`; they do NOT modify the
internal architecture.

Backbones:
  Restormer  - swz30/Restormer            (CVPR 2022, MDTA + GDFN transformer)
  NAFNet     - megvii-research/NAFNet      (ECCV 2022, SimpleGate CNN)
  SwinIR     - JingyunLiang/SwinIR         (ICCV 2021 W, Swin transformer)
  Uformer    - ZhendongWang6/Uformer       (CVPR 2022, LeWin U-transformer)
"""
from .wrappers import (
    NativeRestormer,
    NativeNAFNet,
    NativeSwinIR,
    NativeUformer,
    NativeMAE,
)

__all__ = [
    "NativeRestormer",
    "NativeNAFNet",
    "NativeSwinIR",
    "NativeUformer",
    "NativeMAE",
]
