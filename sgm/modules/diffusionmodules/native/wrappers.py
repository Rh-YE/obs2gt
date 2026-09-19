"""Thin, non-invasive wrappers around the native restoration backbones.

The wrappers only:
  * translate a uniform constructor (`in_channels`, `out_channels`, plus a few
    capacity knobs that are the model's *own* standard hyper-parameters) into
    each backbone's native signature, and
  * provide a uniform `forward(x) -> x` that returns a same-resolution image.

No internal block is modified, so these remain the *native* models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .restormer_arch import Restormer
from .nafnet_arch import NAFNet
from .swinir_arch import SwinIR
from .uformer_arch import Uformer
from .mae_arch import MaskedAutoencoderViT


def _pad_to_multiple(x, m):
    """Reflect-pad H,W up to a multiple of m; return padded x and (H,W) to crop back."""
    _, _, h, w = x.shape
    ph = (m - h % m) % m
    pw = (m - w % m) % m
    if ph or pw:
        # reflect needs pad < dim; fall back to replicate for large pads.
        mode = "reflect" if (ph < h and pw < w) else "replicate"
        x = F.pad(x, (0, pw, 0, ph), mode=mode)
    return x, h, w


class NativeRestormer(nn.Module):
    """Restormer (swz30/Restormer). 4-level U; needs H,W divisible by 8.

    通道数说明 (2026-07 修复): 原生 Restormer 在 dual_pixel_task=False 时最后
    一层是 `self.output(feat) + inp_img` (restormer_arch.py:281) 的全局残差,
    要求 in_channels==out_channels, 但构造函数并未对此做 assert——当
    in_channels!=out_channels 时 (例如 sigma 拼进输入通道的消融实验),
    这个加法会静默触发 PyTorch 的广播规则而不是报错, 产出通道数错误的
    "看似正常"输出 (已实测复现: in=2,out=1 时输出变成 2 通道)。
    dual_pixel_task=True 分支改用 `self.output(feat) + skip_conv(patch_embed特征)`
    (restormer_arch.py:239-240,277), skip_conv 的输入是 patch_embed 之后的
    `dim` 通道特征、与原始图像通道数无关, 因此天然支持任意 in_channels!=
    out_channels。这里在 in_channels!=out_channels 时自动启用该分支, 使
    "σ 拼进输入通道" 的消融实验行为正确 (输出通道数=out_channels)。"""

    def __init__(self, in_channels=1, out_channels=1, dim=48,
                 num_blocks=(4, 6, 6, 8), num_refinement_blocks=4,
                 heads=(1, 2, 4, 8), ffn_expansion_factor=2.66,
                 bias=False, LayerNorm_type="WithBias"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.net = Restormer(
            inp_channels=in_channels, out_channels=out_channels, dim=dim,
            num_blocks=list(num_blocks), num_refinement_blocks=num_refinement_blocks,
            heads=list(heads), ffn_expansion_factor=ffn_expansion_factor,
            bias=bias, LayerNorm_type=LayerNorm_type,
            dual_pixel_task=(in_channels != out_channels))

    def forward(self, x):
        xp, h, w = _pad_to_multiple(x, 8)
        y = self.net(xp)
        return y[..., :h, :w]


class NativeNAFNet(nn.Module):
    """NAFNet (megvii-research/NAFNet). Depth set by enc/dec block counts."""

    def __init__(self, in_channels=1, out_channels=1, width=32,
                 middle_blk_num=1, enc_blk_nums=(1, 1, 1, 2),
                 dec_blk_nums=(1, 1, 1, 1)):
        super().__init__()
        assert in_channels == out_channels, "NAFNet is image->image with equal channels"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self._mult = 2 ** len(enc_blk_nums)
        self.net = NAFNet(
            img_channel=in_channels, width=width, middle_blk_num=middle_blk_num,
            enc_blk_nums=list(enc_blk_nums), dec_blk_nums=list(dec_blk_nums))

    def forward(self, x):
        # NAFNet has its own check_image_size padding, but guard anyway.
        xp, h, w = _pad_to_multiple(x, self._mult)
        y = self.net(xp)
        return y[..., :h, :w]


class NativeSwinIR(nn.Module):
    """SwinIR (JingyunLiang/SwinIR) in restoration (no upscale) mode.

    upscale=1, upsampler='' => classic image restoration head. Needs H,W
    divisible by window_size (handled by reflect padding)."""

    def __init__(self, in_channels=1, out_channels=1, img_size=64,
                 embed_dim=60, depths=(6, 6, 6, 6), num_heads=(6, 6, 6, 6),
                 window_size=8, mlp_ratio=2.0, resi_connection="1conv"):
        super().__init__()
        assert in_channels == out_channels, "SwinIR restoration head keeps channel count"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.window_size = window_size
        self.net = SwinIR(
            img_size=img_size, patch_size=1, in_chans=in_channels,
            embed_dim=embed_dim, depths=list(depths), num_heads=list(num_heads),
            window_size=window_size, mlp_ratio=mlp_ratio, upscale=1,
            img_range=1.0, upsampler="", resi_connection=resi_connection)

    def forward(self, x):
        xp, h, w = _pad_to_multiple(x, self.window_size)
        y = self.net(xp)
        return y[..., :h, :w]


class NativeUformer(nn.Module):
    """Uformer (ZhendongWang6/Uformer). 9 depths => 4-level U; needs H,W div by
    win_size * 2**(num_enc_layers). For win_size=8, 4 levels => div by 128;
    we reflect-pad up and crop back."""

    def __init__(self, in_channels=1, out_channels=1, img_size=64,
                 embed_dim=16, depths=(2, 2, 2, 2, 2, 2, 2, 2, 2),
                 num_heads=(1, 2, 4, 8, 16, 16, 8, 4, 2), win_size=4,
                 mlp_ratio=2.0, token_projection="linear", token_mlp="leff",
                 residual=True):
        super().__init__()
        assert in_channels == out_channels, "Uformer is image->image with equal channels"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.win_size = win_size
        num_enc = len(depths) // 2
        self._mult = win_size * (2 ** num_enc)
        self.residual = residual
        self.net = Uformer(
            img_size=img_size, in_chans=in_channels, dd_in=in_channels,
            embed_dim=embed_dim, depths=list(depths), num_heads=list(num_heads),
            win_size=win_size, mlp_ratio=mlp_ratio, token_projection=token_projection,
            token_mlp=token_mlp, shift_flag=True)

    def forward(self, x):
        xp, h, w = _pad_to_multiple(x, self._mult)
        y = self.net(xp)
        if self.residual:
            y = y + xp
        return y[..., :h, :w]


class _MAEChannelAgnostic(MaskedAutoencoderViT):
    """Subclass that (a) parameterizes patchify/unpatchify channel count (the
    official file hardcodes 3), and (b) lets an external caller force the
    masking noise so the training engine can know exactly which patches were
    masked (needed to restrict the loss to masked pixels). The transformer
    itself is untouched."""

    def __init__(self, *args, in_chans=1, **kwargs):
        super().__init__(*args, in_chans=in_chans, **kwargs)
        self.in_chans = in_chans
        self._forced_noise = None  # [N, L]; consumed once by random_masking

    def patchify(self, imgs):
        p = self.patch_embed.patch_size[0]
        c = self.in_chans
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * c))
        return x

    def unpatchify(self, x):
        p = self.patch_embed.patch_size[0]
        c = self.in_chans
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        if self._forced_noise is None:
            return super().random_masking(x, mask_ratio)
        N, L, D = x.shape
        noise = self._forced_noise.to(x.device)
        self._forced_noise = None
        len_keep = int(L * (1 - mask_ratio))
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore


class NativeMAE(nn.Module):
    """Masked Autoencoder ViT (facebookresearch/mae) for masked-reconstruction.

    Training: one random-mask pass; returns the full unpatchified prediction and
    remembers which pixels were masked (`last_loss_mask`, 1 = masked = supervise
    here, per the native MAE objective of reconstructing removed patches).

    Eval: complementary-partition multi-pass -- patch ids are split into
    ceil(1/(1-mask_ratio)) groups; in pass k group k is *visible*, the rest are
    masked, and predictions are averaged over the passes in which a patch was
    masked. Every patch therefore receives a supervised-style prediction and the
    output is a full denoised image (input is never copied through)."""

    def __init__(self, in_channels=1, out_channels=1, img_size=64, patch_size=4,
                 embed_dim=192, depth=8, num_heads=6,
                 decoder_embed_dim=128, decoder_depth=4, decoder_num_heads=4,
                 mlp_ratio=4.0, mask_ratio=0.75):
        super().__init__()
        assert in_channels == out_channels, "MAE reconstructs its own input space"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.net = _MAEChannelAgnostic(
            img_size=img_size, patch_size=patch_size, in_chans=in_channels,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            decoder_embed_dim=decoder_embed_dim, decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads, mlp_ratio=mlp_ratio,
            norm_pix_loss=False)
        self.last_loss_mask = None  # [B,1,H,W]; 1 = masked patch (supervised)

    def _token_mask_to_pixels(self, mask, h, w):
        # mask: [N, L] with 1 = removed/masked; row-major patch grid
        p = self.patch_size
        gh, gw = h // p, w // p
        m = mask.reshape(-1, 1, gh, gw)
        return m.repeat_interleave(p, dim=2).repeat_interleave(p, dim=3)

    def forward(self, x):
        b, c, h, w = x.shape
        if self.training:
            latent, mask, ids_restore = self.net.forward_encoder(x, self.mask_ratio)
            pred = self.net.forward_decoder(latent, ids_restore)
            rec = self.net.unpatchify(pred)
            self.last_loss_mask = self._token_mask_to_pixels(mask, h, w)
            return rec
        # eval: complementary partition for full coverage
        L = (h // self.patch_size) * (w // self.patch_size)
        n_groups = max(2, int(round(1.0 / max(1e-6, 1.0 - self.mask_ratio))))
        perm = torch.stack([torch.randperm(L, device=x.device) for _ in range(b)])
        acc = torch.zeros_like(x)
        cnt = torch.zeros_like(x)
        for k in range(n_groups):
            group = (perm % n_groups) == k  # [B, L] True = visible this pass
            # visible patches must sort first => give them small noise
            noise = torch.where(group, torch.zeros_like(perm, dtype=torch.float),
                                torch.ones_like(perm, dtype=torch.float))
            noise = noise + 1e-3 * torch.rand(b, L, device=x.device)  # tie-break
            self.net._forced_noise = noise
            # mask_ratio for this pass = fraction masked = 1 - |group|/L
            ratio = 1.0 - group.float().mean().item()
            latent, mask, ids_restore = self.net.forward_encoder(x, ratio)
            pred = self.net.forward_decoder(latent, ids_restore)
            rec_k = self.net.unpatchify(pred)
            pix = self._token_mask_to_pixels(mask, h, w)  # 1 = masked here
            acc = acc + rec_k * pix
            cnt = cnt + pix
        self.last_loss_mask = None
        return acc / cnt.clamp(min=1.0)
