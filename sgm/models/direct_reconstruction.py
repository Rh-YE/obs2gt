"""Direct (no-VAE-bottleneck) reconstruction engines for the paradigm study.

These engines reuse *all* of AutoencodingEngine's Lightning machinery
(manual optimization, EMA, validation histograms, ImageLogger hooks, GAN
optimizer_idx, and the DiscVAELoss chi2/mse/mae + discriminator) but replace
the encoder->regularizer->decoder VAE pipeline with a single *native* image->
image restoration backbone. The only thing that changes between the loss
ablations (chi2 / mse / mae) is `loss_config.params.loss_type`, so the network
input is byte-identical across the three -- a clean causal comparison.

Paradigms (all share DiscVAELoss; sigma enters only the loss weighting):
  DirectReconstructionEngine  -- supervised backbone, x -> rec (Restormer/NAFNet/
                                 SwinIR/Uformer). GAN = same engine, disc_weight>0.
  MaskedReconstructionEngine  -- MAE-style: random patch masking on the input,
                                 loss is evaluated on the full image (the network
                                 must inpaint masked regions from context).
  BlindSpotEngine             -- Noise2Void/Self-style blind-spot self-supervision:
                                 selected pixels are replaced by a neighbour, and
                                 the loss is computed ONLY on those blind pixels,
                                 so the net cannot learn identity. No clean GT used.

GT (HDU 'GT', noiseless SKIRT truth) is never fed during training; it is only
written to the test FITS by rec.py and consumed by analyze_results.py to report
MAE-to-GT. This keeps the self-supervised paradigms honest.
"""
from typing import Dict

import torch

from .autoencoder import AutoencodingEngine
from ..util import instantiate_from_config


class DirectReconstructionEngine(AutoencodingEngine):
    """Single native backbone, no latent bottleneck.

    Config contract: put the native backbone in `encoder_config`; `decoder_config`
    and `regularizer_config` are ignored (kept Identity in YAML for compatibility
    with the parent __init__, but never used on the forward path)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The native restoration net lives in self.encoder; alias for clarity.
        self.model = self.encoder

    # --- forward path: x -> rec, no z, no regularization ---
    def encode(self, x, return_reg_log: bool = False, unregularized: bool = False):
        z = self.model(x)
        if return_reg_log:
            return z, dict()
        return z

    def decode(self, z, **kwargs):
        return z

    def forward(self, x, sigma=None, **additional_decode_kwargs):
        rec = self.model(x)
        return rec, rec, dict()  # (z placeholder, xrec, empty reg_log)

    def get_last_layer(self):
        # DiscVAELoss only calls this when adaptive disc weighting is on; return a
        # representative final-conv weight so adaptive d_weight works if enabled.
        for m in reversed(list(self.model.modules())):
            if isinstance(m, torch.nn.Conv2d):
                return m.weight
        # fallback: any parameter
        return next(self.model.parameters())

    def get_autoencoder_params(self) -> list:
        params = []
        if hasattr(self.loss, "get_trainable_autoencoder_parameters"):
            params += list(self.loss.get_trainable_autoencoder_parameters())
        params += list(self.model.parameters())
        return params

    @torch.no_grad()
    def log_images(self, batch, additional_log_kwargs=None, **kwargs):
        # Parent's version samples from the VAE prior via decoder.z_shape, which
        # does not exist for a direct image->image net. Keep the same panel
        # layout (input | gt | input-rec | rec | gt-rec) and chi2 map.
        log = dict()
        input_img = self.get_input(batch)
        x_gt = batch.get("gt", input_img)
        # cat_invvar=True 时网络的第一层卷积期望 [图像, sigma] 拼接的多通道
        # 输入 (与 inner_training_step/_validation_step 的构造一致, 见
        # autoencoder.py:288-294,503起) ——这里之前遗漏了这个分支, 单通道
        # input_img 直接喂进去会在 σ 进输入通道的消融实验里让第一层卷积报
        # 通道数不匹配 (已实测复现)。
        forward_input = input_img
        if self.with_invvar and self.cat_invvar:
            forward_input = torch.cat([input_img, self.get_error(batch)], dim=1)
        _, rec, _ = self(forward_input)
        log["raw_input-rec"] = torch.cat(
            [input_img, x_gt, input_img - rec, rec, x_gt - rec], dim=-1)
        if self.with_invvar:
            error = self.get_error(batch)
            chi2 = ((x_gt - rec) / (error + 1e-8)) ** 2
            B, C, H, W = chi2.shape
            reduced = chi2.sum(dim=(2, 3)) / (H * W)
            disp = chi2.clone()
            disp[:, :, :5, :5] = 0
            for b in range(B):
                for c in range(C):
                    disp[b, c, :3, :3] = reduced[b, c]
            log["chi2"] = disp
        import torchvision
        log["samples"] = torchvision.utils.make_grid(rec[:36], nrow=6, padding=0)
        return log


class MaskedReconstructionEngine(DirectReconstructionEngine):
    """MAE-style masked reconstruction (native masking, not a patched arch).

    A fraction `mask_ratio` of non-overlapping `patch_size` patches in the input
    are zeroed before the backbone sees them; the reconstruction loss is computed
    on the whole image, forcing the net to infer masked content from context.
    Implemented at the engine level so any native backbone can be used as the
    masked-reconstruction network."""

    def __init__(self, *args, mask_ratio: float = 0.5, patch_size: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size

    def _random_patch_mask(self, x):
        b, c, h, w = x.shape
        ps = self.patch_size
        gh, gw = h // ps, w // ps
        n = gh * gw
        keep = torch.rand(b, n, device=x.device) >= self.mask_ratio  # True = visible
        keep = keep.view(b, 1, gh, gw).float()
        mask = keep.repeat_interleave(ps, dim=2).repeat_interleave(ps, dim=3)
        # pad to full size if h,w not divisible by ps
        if mask.shape[2] != h or mask.shape[3] != w:
            mask = torch.nn.functional.pad(mask, (0, w - mask.shape[3], 0, h - mask.shape[2]), value=1.0)
        return mask  # [b,1,h,w], 1=visible 0=masked

    def inner_training_step(self, batch, batch_idx, optimizer_idx: int = 0):
        # Mask the input image in-place (only the image channels, not invvar).
        if self.training:
            img = batch[self.input_key]
            mask = self._random_patch_mask(img)
            batch = dict(batch)
            batch[self.input_key] = img * mask
        return super().inner_training_step(batch, batch_idx, optimizer_idx)


class R2REngine(DirectReconstructionEngine):
    """Recorrupted-to-Recorrupted (Pang et al., CVPR 2021) self-supervision.

    Plain obs->obs regression is trivial for a no-bottleneck image->image net
    (identity is the global optimum -- NAFNet et al. even have a global input
    residual), so we de-correlate input and target with the known per-pixel
    sigma map:  y_in = y + a*sigma*z,  y_tgt = y - sigma*z/a  (z~N(0,1)).
    E[loss(f(y_in), y_tgt)] equals the supervised loss up to a constant, no
    clean GT is ever used, and identity is no longer optimal. The chi2
    weighting uses the corrupted target's effective sigma sqrt(1+1/a^2)*sigma.

    推理修正 (2026-07, 对应审计条目 A6): 训练时网络看到的输入方差是
    (1+a^2)*sigma^2 (即 y_in = y + a*sigma*z), 若测试/验证时直接对原始观测 y
    做单次前向, 输入的噪声水平与训练时不匹配 (CVPR2021 原文的做法是: 测试时同样
    对 y 做 M 次独立再腐蚀 y_pert_m = y + a*sigma*z_m, 推理后对 M 次预测取平均,
    该平均是训练分布下 E[f(y_pert)] 的蒙特卡洛估计, 与训练输入分布严格对齐)。
    这里按 BlindSpotEngine.forward 的"eval 走特殊多趟推理"结构实现: eval_r2r_avg
    关闭时保留旧行为 (单次 plain forward, 仅用于对照量化旧实验的系统偏差)。"""

    def __init__(self, *args, r2r_alpha: float = 1.0, eval_r2r_avg: bool = True,
                 eval_r2r_m: int = 50, **kwargs):
        super().__init__(*args, **kwargs)
        self.r2r_alpha = r2r_alpha
        self.eval_r2r_avg = eval_r2r_avg
        self.eval_r2r_m = eval_r2r_m

    def inner_training_step(self, batch, batch_idx, optimizer_idx: int = 0):
        # Validation goes through the parent (plain obs->obs reconstruction loss).
        if not self.training:
            return super().inner_training_step(batch, batch_idx, optimizer_idx)

        img = self.get_input(batch)
        sig = self.get_error(batch)
        assert sig is not None, "R2R needs the per-pixel sigma map"
        a = self.r2r_alpha
        z = torch.randn_like(img)
        valid = sig <= 1.0  # sigma>1 are sentinel/masked pixels; don't recorrupt
        n = torch.where(valid, sig, torch.zeros_like(sig)) * z
        y_in = img + a * n
        y_tgt = img - n / a
        # target noise variance: (1 + 1/a^2) * sigma^2
        sig_eff = torch.where(valid, sig * (1.0 + 1.0 / a ** 2) ** 0.5, sig)

        _, xrec, regularization_log = self(y_in)

        if hasattr(self.loss, "forward_keys"):
            extra_info = {
                "z": xrec,
                "optimizer_idx": optimizer_idx,
                "global_step": self.global_step,
                "last_layer": self.get_last_layer(),
                "split": "train",
                "regularization_log": regularization_log,
                "autoencoder": self,
            }
            extra_info = {k: extra_info[k] for k in self.loss.forward_keys}
        else:
            extra_info = dict()

        out_loss = self.loss(y_tgt, xrec, sig_eff, **extra_info)
        if isinstance(out_loss, tuple):
            loss, log_dict = out_loss
        else:
            loss, log_dict = out_loss, {"train/loss/rec": out_loss.detach()}

        if optimizer_idx == 0:
            self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True,
                          on_epoch=True, sync_dist=True, batch_size=img.shape[0])
            self.log("loss", loss.mean().detach(), prog_bar=True, logger=False,
                     on_epoch=False, on_step=True, sync_dist=True, batch_size=img.shape[0])
        else:
            self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True,
                          on_epoch=True, batch_size=img.shape[0])
        return loss

    def forward(self, x, sigma=None, **kw):
        """训练时 (self.training=True) 走标准单次前向 (inner_training_step 已经
        自己构造好 y_in 并直接调 self(y_in), 不经过这里的 M 次平均分支)。

        验证/推理 (self.training=False) 时, 若 sigma 可用且 eval_r2r_avg=True,
        对同一输入做 M 次独立再腐蚀 (与训练时相同的 y_in = x + a*sigma*z_m 构造,
        使用不同的 z_m), 逐次前向后取平均预测, 这是训练分布 E[f(y_in)] 的蒙特
        卡洛估计 (CVPR2021 官方: alpha=0.5, M=50)。若 sigma 不可用 (未开
        sigma_film) 或 eval_r2r_avg=False, 退化为旧行为的单次 plain forward,
        仅用于对照量化"用错误的输入分布做推理"引入的系统偏差。"""
        if self.training or not self.eval_r2r_avg or sigma is None:
            rec = self.model(x)
            return rec, rec, dict()
        a = self.r2r_alpha
        valid = sigma <= 1.0
        sig = torch.where(valid, sigma, torch.zeros_like(sigma))
        acc = torch.zeros_like(x)
        for _ in range(self.eval_r2r_m):
            z = torch.randn_like(x)
            y_pert = x + a * sig * z
            acc = acc + self.model(y_pert)
        out = acc / self.eval_r2r_m
        return out, out, dict()


class _FixedTargetEngine(DirectReconstructionEngine):
    """训练目标固定为 batch 中某个字段 (与输入图像不同) 的自监督/监督基线。

    父类 AutoencodingEngine.inner_training_step 硬编码"输入=目标=raw_img"
    (autoencoder.py:280,285,302), 无法表达"网络输入是 OBS, 训练目标是另一个
    与噪声实现无关/独立的字段"这类范式, 所以完整覆盖该方法, 只把目标换成
    self.target_key 指向的 batch 字段, 其余 (mask/权重/日志) 与父类一致。

    子类只需要设置 self.target_key:
      N2NEngine: target_key='obs_b'  (Noise2Noise, 目标=同一场景独立第二次
                 噪声实现, 训练时不接触 GT, 理论上是最干净的自监督)
      N2CEngine: target_key='gt'     (Noise2Clean, 监督上界, 训练时使用 GT
                 —— 只在仿真里合法, 真实观测没有这个数据)
    """
    target_key: str = "images"

    def inner_training_step(self, batch, batch_idx, optimizer_idx: int = 0):
        img = self.get_input(batch)  # 网络输入, 恒为 OBS
        target = batch[self.target_key]

        error_for_loss = None
        sigma_cond = None
        x = img
        if self.with_invvar:
            error = self.get_error(batch)
            if self.cat_invvar:
                x = torch.cat([img, error], dim=1)
            elif self.sigma_film:
                sigma_cond = error
        additional_decode_kwargs = {
            key: batch[key] for key in self.additional_decode_keys.intersection(batch)
        }
        z, xrec, regularization_log = self(x, sigma=sigma_cond, **additional_decode_kwargs)

        if self.with_invvar and self.cat_invvar:
            n_img_channels = img.shape[1]
            error_for_loss = x[:, n_img_channels:, :, :]

        if hasattr(self.loss, "forward_keys"):
            extra_info = {
                "z": z, "optimizer_idx": optimizer_idx, "global_step": self.global_step,
                "last_layer": self.get_last_layer(), "split": "train" if self.training else "val",
                "regularization_log": regularization_log, "autoencoder": self,
            }
            extra_info = {k: extra_info[k] for k in self.loss.forward_keys}
        else:
            extra_info = dict()

        if self.with_invvar:
            sig = error_for_loss if error_for_loss is not None else self.get_error(batch)
            out_loss = self.loss(target, xrec, sig, **extra_info)
        else:
            out_loss = self.loss(target, xrec, **extra_info)
        loss, log_dict = out_loss if isinstance(out_loss, tuple) else (
            out_loss, {"train/loss/rec": out_loss.detach()})

        prefix = "train" if self.training else "val"
        if optimizer_idx == 0:
            self.log_dict(log_dict, prog_bar=False, logger=True, on_step=self.training,
                          on_epoch=True, sync_dist=True, batch_size=img.shape[0])
            if self.training:
                self.log("loss", loss.mean().detach(), prog_bar=True, logger=False,
                         on_epoch=False, on_step=True, sync_dist=True, batch_size=img.shape[0])
        else:
            self.log_dict(log_dict, prog_bar=False, logger=True, on_step=self.training,
                          on_epoch=True, batch_size=img.shape[0])
        return loss


class N2NEngine(_FixedTargetEngine):
    """Noise2Noise: 训练目标 = 同一场景独立第二次噪声实现 (OBS_B)。

    两次曝光的噪声相互独立、期望相同, 网络在平方损失下的总体最优解仍是
    E[真值|观测] (Lehtinen et al. 2018), 训练全程不接触 GT。数据集必须提供
    'obs_b' 字段 (MultiHDUDataset 已支持)。这是本实验里除 N2C 外理论上最
    干净的范式, 也是 σ 进输入通道消融 (cat_invvar) 的载体——见 gen_
    benchmark_configs.py 中 restormer 骨干的相关配置。"""
    target_key = "obs_b"


class N2CEngine(_FixedTargetEngine):
    """Noise2Clean: 训练目标 = 无噪真值 (GT), 监督上界。

    只在仿真数据里合法 (真实观测没有 GT); 作用是把"损失函数本身的容量分配
    差异"与"自监督目标含噪导致的额外代价"分离——若某损失在 N2C 上已经落后,
    说明差距来自损失函数本身, 与自监督范式无关。"""
    target_key = "gt"


class MAEEngine(DirectReconstructionEngine):
    """Native MAE (Masked Autoencoder ViT) paradigm.

    The backbone must be NativeMAE. During training the engine pre-samples the
    masking noise, forces it onto the MAE (so engine and model agree on which
    patches are masked), and inflates sigma on *visible* pixels so the loss is
    evaluated only on masked patches -- exactly the native MAE objective, but
    with the reconstruction term swappable between chi2/mse/mae/huber.
    At eval the wrapper runs a complementary-mask multi-pass to emit a full
    denoised image (rec.py needs a complete prediction)."""

    def inner_training_step(self, batch, batch_idx, optimizer_idx: int = 0):
        if self.training:
            img = batch[self.input_key]
            b, c, h, w = img.shape
            p = self.model.patch_size
            L = (h // p) * (w // p)
            noise = torch.rand(b, L, device=img.device)
            # replicate MAE random_masking bookkeeping to get the pixel mask
            len_keep = int(L * (1 - self.model.mask_ratio))
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            mask = torch.ones(b, L, device=img.device)
            mask[:, :len_keep] = 0
            mask = torch.gather(mask, 1, ids_restore)  # 1 = masked (supervised)
            pix = self.model._token_mask_to_pixels(mask, h, w)  # [B,1,H,W]
            self.model.net._forced_noise = noise
            batch = dict(batch)
            if "error" in batch:
                sel = pix.bool().expand_as(batch["error"])
                big = torch.full_like(batch["error"], 1e6)
                batch["error"] = torch.where(sel, batch["error"], big)
        return super().inner_training_step(batch, batch_idx, optimizer_idx)


class BlindSpotEngine(DirectReconstructionEngine):
    """Noise2Void/Self-style blind-spot self-supervision.

    A random subset (`blind_ratio`) of pixels in the input is replaced by a random
    neighbour pixel; the loss is then evaluated ONLY on those replaced pixels (via
    a per-pixel loss mask). The network must predict each blind pixel from its
    surroundings, so it cannot collapse to identity and learns to denoise from a
    single noisy frame -- no clean GT, no noisy pair."""

    def __init__(self, *args, blind_ratio: float = 0.02, neighbour_radius: int = 2,
                 eval_blind: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.blind_ratio = blind_ratio
        self.neighbour_radius = neighbour_radius
        self.eval_blind = eval_blind

    def _blind_spot(self, img, sel=None):
        b, c, h, w = img.shape
        if sel is None:
            sel = torch.rand(b, 1, h, w, device=img.device) < self.blind_ratio  # blind pixels
        # replacement = random neighbour within +/- radius
        r = self.neighbour_radius
        dy = torch.randint(-r, r + 1, (b, 1, h, w), device=img.device)
        dx = torch.randint(-r, r + 1, (b, 1, h, w), device=img.device)
        ys = (torch.arange(h, device=img.device).view(1, 1, h, 1) + dy).clamp(0, h - 1)
        xs = (torch.arange(w, device=img.device).view(1, 1, 1, w) + dx).clamp(0, w - 1)
        bidx = torch.arange(b, device=img.device).view(b, 1, 1, 1).expand(b, 1, h, w)
        neigh = img[bidx, :, ys, xs] if c == 1 else torch.stack(
            [img[:, ci][bidx[:, 0], ys[:, 0], xs[:, 0]] for ci in range(c)], dim=1)
        neigh = neigh.view(b, c, h, w) if c == 1 else neigh
        out = torch.where(sel.expand(-1, c, -1, -1), neigh, img)
        return out, sel  # sel [b,1,h,w] = where loss is evaluated

    def inner_training_step(self, batch, batch_idx, optimizer_idx: int = 0):
        if self.training:
            img = batch[self.input_key]
            corrupted, sel = self._blind_spot(img)
            batch = dict(batch)
            batch[self.input_key] = corrupted
            # restrict the loss to blind pixels by inflating sigma elsewhere:
            # the loss masks pixels with sigma>clip_max; set non-blind sigma huge.
            if "error" in batch:
                big = torch.full_like(batch["error"], 1e6)
                batch["error"] = torch.where(sel.expand_as(batch["error"]), batch["error"], big)
        return super().inner_training_step(batch, batch_idx, optimizer_idx)

    def forward(self, x, sigma=None, **kw):
        """Eval uses blind-spot inference: the network was only ever supervised
        on blind (neighbour-replaced) pixels, and a plain full-image forward
        collapses to identity for backbones with a global input residual.
        Partition pixels into K=ceil(1/blind_ratio) groups; pass k replaces
        group k with neighbours and we keep the predictions for that group only,
        so every output pixel comes from the train-time input distribution."""
        if self.training or not self.eval_blind:
            rec = self.model(x)
            return rec, rec, dict()
        b, c, h, w = x.shape
        K = max(2, int(round(1.0 / max(self.blind_ratio, 1e-6))))
        gid = torch.randint(0, K, (b, 1, h, w), device=x.device)
        out = torch.zeros_like(x)
        for k in range(K):
            sel = gid == k
            corrupted, _ = self._blind_spot(x, sel=sel)
            pred = self.model(corrupted)
            out = torch.where(sel.expand_as(x), pred, out)
        return out, out, dict()
