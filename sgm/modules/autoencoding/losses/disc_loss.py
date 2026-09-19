from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision
from einops import rearrange
from matplotlib import colormaps
from matplotlib import pyplot as plt
from torchmetrics.functional import structural_similarity_index_measure as ssim
from ....util import default, instantiate_from_config
from ..lpips.loss.lpips import LPIPS
from ..lpips.model.model import weights_init
from ..lpips.vqperceptual import hinge_d_loss, vanilla_d_loss

import torch
import torch.fft as fft
import torch.nn.functional as F

from astropy.io import fits
import numpy as np
import os
import math
from tqdm import tqdm
import wandb

def write_grid(images, fn):
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    batch_size = images.shape[0]
    channels, height, width = images[0].shape
    grid_size = math.ceil(math.sqrt(batch_size))
    rows, cols = grid_size, grid_size
    
    grid_img = np.zeros((channels, rows * height, cols * width))
    
    for idx in range(batch_size):
        if idx >= rows * cols:
            break
        img = images[idx]
        row_idx = idx // cols
        col_idx = idx % cols
        
        grid_img[:, row_idx*height:(row_idx+1)*height, col_idx*width:(col_idx+1)*width] = img
    
    fits.writeto(fn, grid_img, overwrite=True)
    
def extract_step_data(i, images=None, invvars=None, rec=None, mask=None):
    os.makedirs("./step_data", exist_ok=True)
    if images is not None:
        write_grid(images, f"./step_data/step_{i}_grid.fits")
    if invvars is not None:
        write_grid(invvars, f"./step_data/step_{i}_invvar_grid.fits")
    if rec is not None:
        write_grid(rec, f"./step_data/step_{i}_rec_grid.fits")
    if mask is not None:
        write_grid(mask, f"./step_data/step_{i}_mask_grid.fits")
    
class DiscVAELoss(nn.Module):
    def __init__(
        self,
        logvar_init: float = 0.0,
        dims: int = 2,
        learn_logvar: bool = False,
        regularization_weights: Optional[Dict[str, float]] = None,
        additional_log_keys: Optional[List[str]] = None,
        loss_type: str = "chi2",
        use_mask: bool = False,  # 是否使用掩码
        mask_rules: Optional[Dict[str, Union[float, List[float]]]] = None,  # 新增掩码规则参数
        # snr_weight: bool = False,
        perceptual_weight: float = 0.0,
        use_random_perceptual: bool = False,  # 是否使用随机权重的感知损失
        cat_invvar: bool = False,
        normalize: bool = False,
        logvar: bool = False,
        cat_psf: bool = False,
        ssim_weight: float = 0.0,
        multiscale_chi2_weight: float = 0.0,  # 新增多尺度chi2损失权重
        multiscale_levels: int = 3,  # 新增多尺度级别数
        huber_delta: float = 0.1,  # Huber损失的delta阈值（残差尺度，约1-2倍典型sigma）
        discriminator_config: Optional[Dict] = None,
        disc_start: int = 20001,
        disc_num_layers: int = 3,
        disc_in_channels: int = 4,
        disc_factor: float = 1.0,
        disc_weight: float = 0,
        disc_loss: str = "hinge",
    ):
        super().__init__()
        self.perceptual_loss = LPIPS(use_random_weights=use_random_perceptual, input_channels=disc_in_channels).eval() if perceptual_weight > 0 else None
        self.perceptual_weight = perceptual_weight
        self.dims = dims
        self.loss_type = loss_type
        self.use_mask = use_mask  # 是否使用掩码
        self.mask_rules = default(mask_rules, {})  # 默认掩码规则
        self.cat_invvar = cat_invvar
        self.normalize = normalize
        self.cat_psf = cat_psf
        self.ssim_weight = ssim_weight
        self.multiscale_chi2_weight = multiscale_chi2_weight  # 新增多尺度chi2损失权重
        self.multiscale_levels = multiscale_levels  # 新增多尺度级别数
        self.huber_delta = huber_delta
        # if self.dims > 2:
        #     print(
        #         f"running with dims={dims}. This means that for perceptual loss "
        #         f"calculation, the LPIPS loss will be applied to each frame "
        #         f"independently."
        #     )
        self.logvar = nn.Parameter(
            torch.full((), logvar_init), requires_grad=learn_logvar
        )
        self.learn_logvar = learn_logvar
        self.regularization_weights = default(regularization_weights, {})
        self.forward_keys = [
            "optimizer_idx",
            "global_step",
            "last_layer",
            "split",
            "regularization_log",
        ]

        self.additional_log_keys = set(default(additional_log_keys, []))
        self.additional_log_keys.update(set(self.regularization_weights.keys()))

        # 只有当判别器权重大于0时才初始化判别器
        self.discriminator_weight = disc_weight
        if disc_weight > 0:
            discriminator_config = default(
                discriminator_config,
                {
                    "target": "sgm.modules.autoencoding.lpips.model.model.NLayerDiscriminator",
                    "params": {
                        "input_nc": disc_in_channels,
                        "n_layers": disc_num_layers,
                        "use_actnorm": False,
                    },
                },
            )
            self.discriminator = instantiate_from_config(discriminator_config).apply(weights_init)
            self.discriminator_iter_start = disc_start
            self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
            self.disc_factor = disc_factor
        else:
            self.discriminator = None
            self.discriminator_iter_start = disc_start
            self.disc_loss = None
            self.disc_factor = disc_factor

    def calculate_adaptive_weight(
        self, nll_loss: torch.Tensor, g_loss: torch.Tensor, last_layer: torch.Tensor
    ) -> torch.Tensor:
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight
    
    def get_trainable_parameters(self) -> Iterator[nn.Parameter]:
        if self.discriminator is not None:
            return self.discriminator.parameters()
        else:
            return iter([])  # 返回空的迭代器
    
    def get_trainable_autoencoder_parameters(self) -> Iterator[nn.Parameter]:
        if self.learn_logvar:
            yield self.logvar
        yield from ()
        
    def generate_mask(
        self, 
        inputs: torch.Tensor, 
        error: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        生成mask，True表示有效区域
        
        Args:
            inputs: 输入图像
            error: sigma误差图（已经在数据加载时转换并裁剪到[1e-5, 1]）
        """
        mask = torch.ones_like(inputs, dtype=torch.bool)
        
        # 图像值范围检查
        if "images_min" in self.mask_rules:
            mask = mask & (inputs >= self.mask_rules["images_min"])
            
        if "images_max" in self.mask_rules:
            mask = mask & (inputs <= self.mask_rules["images_max"])
        
        if "images_equal" in self.mask_rules:
            mask = mask & (inputs == 0)
        
        # sigma误差图范围检查
        if error is not None:
            if error.shape != mask.shape:
                raise ValueError(f"error shape {error.shape} does not match mask shape {mask.shape}")
            
            # 检查sigma是否在有效范围内[1e-5, 1]
            # 注意：数据加载时已经做了裁剪，这里是双重保险
            if "error_min" in self.mask_rules:
                mask = mask & (error >= self.mask_rules["error_min"])
            else:
                # 默认：排除过小的sigma值（对应极高的invvar）
                mask = mask & (error >= 1e-5)
                
            if "error_max" in self.mask_rules:
                mask = mask & (error <= self.mask_rules["error_max"])
            else:
                # 默认：排除过大的sigma值（对应极低的invvar）
                mask = mask & (error <= 1.0)
        
        return mask
        
    def forward(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        error: Optional[torch.Tensor] = None,
        *,  # added because I changed the order here
        regularization_log: Dict[str, torch.Tensor],
        optimizer_idx: int,
        global_step: int,
        last_layer: torch.Tensor,
        split: str = "train",
        weights: Union[None, float, torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:

        # 初始化log字典
        log = dict()

        if self.dims > 2:
            inputs, reconstructions = map(
                lambda x: rearrange(x, "b c t h w -> (b t) c h w"),
                (inputs, reconstructions),
            )
        # 注意：cat_invvar的处理已经在autoencoder.py中完成
        # 这里的inputs已经是纯图像部分，不包含error（sigma）
        
        self.mask = self.generate_mask(inputs, error) if self.use_mask else torch.ones_like(inputs, dtype=torch.bool)
        
        if self.cat_psf:
            index = inputs.shape[1]//2
            psf = inputs[:, index:, :, :].contiguous()
            output = reconstructions[:, :index, :, :].contiguous()
            
            rec_loss = inputs[:, :index, :, :].contiguous() - output
        else:
            rec_loss = inputs.contiguous() - reconstructions.contiguous()
        
        if error is not None:
            chi2_loss, weighted_chi2_loss, median_chi2_loss = self.astro_nll_loss(inputs,rec_loss, weights, error, normalize=self.normalize, median=True)
            
            # 添加epsilon防止除零
            epsilon = 1e-8
            
            # 计算每个样本每个通道的平均chi2
            if self.mask is not None:
                masked_rec_loss = rec_loss * self.mask + torch.logical_not(self.mask) * rec_loss.detach() * 0.0
                # 对于masked_error，在无效区域使用一个大的sigma值（相当于权重为0）
                masked_error = torch.where(self.mask, error, torch.ones_like(error))
            else:
                masked_rec_loss = rec_loss
                masked_error = error
            
            # 计算每个像素的chi2值：chi2 = (residual/sigma)^2
            # 添加epsilon确保分母不为0
            pixel_chi2 = (torch.abs(masked_rec_loss) / (masked_error + epsilon))**2  # [B, C, H, W]
            
            # 计算每个样本每个通道的有效像素数
            valid_pixels = self.mask.sum(dim=(2,3)) if self.mask is not None else torch.ones_like(pixel_chi2[:,:,0,0]) * pixel_chi2.shape[2] * pixel_chi2.shape[3]  # [B, C]
            
            # 计算每个通道的平均chi2
            channel_chi2 = pixel_chi2.sum(dim=(2,3)) / valid_pixels.clamp(min=1.0)  # [B, C]
            
            # 将数据转移到CPU并转换为numpy数组
            channel_chi2_np = channel_chi2.detach().cpu().numpy()
            
            # 为每个通道创建直方图数据
            for c in range(channel_chi2_np.shape[1]):
                channel_data = channel_chi2_np[:, c]
                # 过滤掉无效值（inf和nan）
                valid_mask = np.isfinite(channel_data)
                if valid_mask.any():  # 只有在有有效数据时才创建直方图
                    valid_data = channel_data[valid_mask]
                    # 记录统计量
                    log[f"{split}/chi2_stats/channel_{c}/mean"] = float(np.mean(valid_data))
                    log[f"{split}/chi2_stats/channel_{c}/median"] = float(np.median(valid_data))
                    log[f"{split}/chi2_stats/channel_{c}/std"] = float(np.std(valid_data))
                    # 记录直方图数据
                    # log[f"{split}/chi2_stats/channel_{c}/histogram"] = torch.tensor(valid_data)
            
        # GLS 归一化 chi2: per-sample Σ(w·r²)/Σ(w), w=1/(2σ²)。逆方差相对权重不变,
        # 但每样本损失尺度自归一 → 消除 batch 间 σ 分布波动导致的 Adam 二阶矩漂移
        # (诊断发现 naive mean(r²/2σ²) 在深度网络下欠收敛: chi2模型在自身泛函上输给mse模型)
        if error is not None:
            epsilon_gls = 1e-8
            w_gls = torch.where(self.mask, 1.0 / (2 * error ** 2 + epsilon_gls),
                                torch.zeros_like(error))
            gls_num = (w_gls * rec_loss ** 2).sum(dim=(1, 2, 3))
            gls_den = w_gls.sum(dim=(1, 2, 3)).clamp(min=1e-12)
            chi2_gls_loss = gls_num / gls_den
            weighted_chi2_gls_loss = chi2_gls_loss.mean()
        else:
            chi2_gls_loss = torch.tensor(0.0, device=inputs.device)
            weighted_chi2_gls_loss = torch.tensor(0.0, device=inputs.device)

        ssim_loss, weighted_ssim_loss = self.ssim_loss(inputs, reconstructions, weights)
        mse_loss, weighted_mse_loss = self.mse_loss(rec_loss**2, weights)
        mae_loss, weighted_mae_loss = self.mse_loss(torch.abs(rec_loss), weights)
        # Huber: 0.5*r^2 (|r|<=delta), delta*(|r|-0.5*delta) (|r|>delta)；与mse/mae走同一mask管线
        abs_r = torch.abs(rec_loss)
        pixel_huber = torch.where(
            abs_r <= self.huber_delta,
            0.5 * rec_loss**2,
            self.huber_delta * (abs_r - 0.5 * self.huber_delta),
        )
        huber_loss, weighted_huber_loss = self.mse_loss(pixel_huber, weights)
        cross_entropy_loss, weighted_cross_entropy_loss = self.cross_entropy_loss(rec_loss, error, weights)
        flux_consistency_loss, weighted_flux_consistency_loss = self.flux_consistency_loss(inputs, reconstructions, weights)
        poison_nll_loss, weighted_poison_nll_loss = self.poison_nll(inputs, reconstructions, weights, error)
        # 添加多尺度chi2损失计算
        if self.multiscale_chi2_weight > 0 and error is not None:
            multiscale_chi2_loss, weighted_multiscale_chi2_loss = self.multiscale_chi2_loss(inputs, reconstructions, error, weights)
        else:
            multiscale_chi2_loss, weighted_multiscale_chi2_loss = torch.tensor(0.0, device=inputs.device), torch.tensor(0.0, device=inputs.device)
        
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(
                inputs.contiguous(), reconstructions.contiguous()
            )
            if isinstance(p_loss, torch.Tensor) and p_loss.dim() > 0:
                p_loss = p_loss.mean()
        else:
            p_loss = torch.tensor(0.0, device=inputs.device)
        
        if optimizer_idx == 0:
            # loss = loss1 + loss2 / (loss2 / loss1).detach() + loss3 / (loss3 / loss1).detach()
            if self.loss_type == "chi2":
                loss = weighted_chi2_loss
            elif self.loss_type == "poison_nll":
                loss = weighted_poison_nll_loss
            elif self.loss_type == "mse":
                loss = weighted_mse_loss
            elif self.loss_type == "cross_entropy":
                loss = weighted_cross_entropy_loss
            elif self.loss_type == "flux_consistency":
                loss = weighted_flux_consistency_loss
            elif self.loss_type == "mae":
                loss = weighted_mae_loss
            elif self.loss_type == "huber":
                loss = weighted_huber_loss
            elif self.loss_type == "chi2_gls":
                loss = weighted_chi2_gls_loss
            elif self.loss_type == "chi2_log":
                loss = torch.abs(torch.log(weighted_chi2_loss-1))

            # generator update - 只有当判别器存在时才计算判别器损失
            if self.discriminator is not None and self.discriminator_weight > 0 and (global_step >= self.discriminator_iter_start or not self.training):
                _dw = next(self.discriminator.parameters()).dtype
                logits_fake = self.discriminator(reconstructions.contiguous().to(_dw))
                g_loss = -torch.mean(logits_fake)
                if self.training:
                    d_weight = self.calculate_adaptive_weight(
                        loss, g_loss, last_layer=last_layer
                    )
                else:
                    d_weight = torch.tensor(1.0)
            else:
                d_weight = torch.tensor(0.0)
                g_loss = torch.tensor(0.0, requires_grad=True)

            if self.perceptual_weight > 0:
                loss = loss + self.perceptual_weight * p_loss
            if self.ssim_weight > 0:
                loss = loss + self.ssim_weight * ssim_loss
            if self.multiscale_chi2_weight > 0:
                loss = loss + self.multiscale_chi2_weight * weighted_multiscale_chi2_loss

            loss = loss + d_weight * self.disc_factor * g_loss
            # log = dict()
            for k in regularization_log:
                if k in self.regularization_weights:
                    loss = loss + self.regularization_weights[k] * regularization_log[k]
                if k in self.additional_log_keys:
                    log[f"{split}/{k}"] = regularization_log[k].detach().float().mean()

            log.update(
                {
                    f"{split}/loss/median_chi2": median_chi2_loss.detach().mean(),
                    f"{split}/loss/total": loss.clone().detach().mean(),
                    f"{split}/loss/mse": mse_loss.detach().mean(),
                    f"{split}/loss/mae": mae_loss.detach().mean(),
                    f"{split}/loss/huber": huber_loss.detach().mean(),
                    f"{split}/loss/chi2_gls": chi2_gls_loss.detach().mean(),
                    f"{split}/loss/chi2": chi2_loss.detach().mean(),
                    f"{split}/loss/KL": cross_entropy_loss.detach().mean(),
                    f"{split}/loss/flux": flux_consistency_loss.detach().mean(),
                    f"{split}/loss/poison_nll": poison_nll_loss.detach().mean(),
                    f"{split}/step": global_step,
                    # f"{split}/scalars/logvar": self.logvar.detach(),
                }
            )
            if self.ssim_weight > 0:
                log.update(
                    {
                        f"{split}/loss/ssim": ssim_loss.detach().mean(),
                    }
                )
            if self.perceptual_weight > 0:
                log.update(
                    {
                        f"{split}/loss/p_loss": p_loss.detach().mean(),
                    }
                )
            if self.multiscale_chi2_weight > 0:
                log.update(
                    {
                        f"{split}/loss/multiscale_chi2": multiscale_chi2_loss.detach().mean(),
                    }
                )
            return loss, log
        elif optimizer_idx == 1 and self.discriminator is not None and self.discriminator_weight > 0:
            # second pass for discriminator update - 只有当判别器存在时才进行
            _dw = next(self.discriminator.parameters()).dtype
            logits_real = self.discriminator(inputs.contiguous().detach().to(_dw))
            logits_fake = self.discriminator(reconstructions.contiguous().detach().to(_dw))

            if global_step >= self.discriminator_iter_start or not self.training:
                d_loss = self.disc_factor * self.disc_loss(logits_real, logits_fake)
            else:
                d_loss = torch.tensor(0.0, requires_grad=True)

            log = {
                f"{split}/loss/disc": d_loss.clone().detach().mean(),
                f"{split}/logits/real": logits_real.detach().mean(),
                f"{split}/logits/fake": logits_fake.detach().mean(),
            }
            return d_loss, log
        else:
            # 如果没有判别器或判别器权重为0，返回零损失
            if optimizer_idx == 1:
                d_loss = torch.tensor(0.0, requires_grad=True)
                log = {
                    f"{split}/loss/disc": d_loss.clone().detach().mean(),
                    f"{split}/logits/real": torch.tensor(0.0),
                    f"{split}/logits/fake": torch.tensor(0.0),
                }
                return d_loss, log
            else:
                raise NotImplementedError(f"Unknown optimizer_idx {optimizer_idx}")

    def ssim_loss(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        weights: Optional[Union[float, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # residual = inputs - reconstructions

        # # 生成1000个与residual形状相同的高斯噪声
        # ssim_values = []
        # for _ in range(100):
        #     gaussian_noise = torch.randn_like(residual)
        #     ssim_val = ssim(residual, gaussian_noise)
        #     ssim_values.append(ssim_val)
        # ssim_values = torch.stack(ssim_values, dim=0)  # [1000, ...]
        # ssim_mean = ssim_values.mean(dim=0)  # 对1000个ssim取均值

        # # 损失项（1-ssim均值，越小越好）
        # ssim_loss = 1.0 - ssim_mean
        # return ssim_loss, ssim_loss
        ssim_value = ssim(inputs*self.mask, reconstructions*self.mask)
        if isinstance(ssim_value, tuple):
            ssim_value = ssim_value[0]  # 如果返回元组，取第一个值
        ssim_loss = 1.0 - ssim_value
        return ssim_loss, ssim_loss
    
    def mask_loss(self, loss: torch.Tensor, normalize: bool = False, inputs: Optional[torch.Tensor] = None) -> torch.Tensor:
        if normalize:
            assert inputs is not None
        _, C, _, _ = loss.shape
        # if torch.sum(self.mask == 0) > 1000:
        #     print(torch.sum(self.mask == 0))
        # 使用detach断开非掩码区域的梯度流
        masked_loss = loss * self.mask + torch.logical_not(self.mask) * loss.detach() * 0.0
        
        # 计算每个样本每个通道的有效像素数
        valid_pixels = self.mask.sum(dim=(2,3))  # [B, C]
        
        # 标记有效像素数为0的通道
        valid_channels = (valid_pixels > 0)  # [B, C]
        
        # 计算batch中每个样本的有效通道数
        valid_channels_count = valid_channels.sum(dim=1)  # [B]
        
        # 处理没有任何有效通道的样本
        all_invalid_samples = (valid_channels_count == 0)
        
        # 确保有效通道数至少为1，用于后续除法
        valid_channels_count = valid_channels_count.clamp(min=1.0)  # [B]
        
        # 对有效通道计算损失平均值，无效通道使用0
        loss_sum = masked_loss.sum(dim=(2,3))  # [B, C]
        loss_mean = torch.where(
            valid_channels,
            loss_sum / valid_pixels.clamp(min=1.0),
            torch.zeros_like(loss_sum)
        )  # [B, C]
        
        # 如果需要归一化，则除以每个通道的最大值
        if normalize:
            # 计算每个通道的最大值
            # 修复torch.max()参数错误，不能使用元组作为dim参数
            # 先在dim=2上取最大值，再在dim=3上取最大值
            channel_max = torch.max(torch.max(torch.abs(inputs), dim=2)[0], dim=2)[0]  # [B, C]
            # 防止除以零
            channel_max = channel_max.clamp(min=1e-8)
            # 归一化损失
            loss_mean = loss_mean / channel_max
        
        # 只对有效通道求平均
        sample_loss = loss_mean.sum(dim=1) / valid_channels_count  # [B]
        
        # 对完全无效的样本，返回一个detach的零张量，确保不会有无效梯度
        sample_loss = torch.where(
            all_invalid_samples,
            torch.zeros_like(sample_loss, requires_grad=False),
            sample_loss
        )
        
        return sample_loss

    def weighted_loss(
        self,
        loss: torch.Tensor,
        weights: Optional[Union[float, torch.Tensor]] = None,
        median: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        input loss shape: [B, C, H, W], [B, C], [B], only mean the input loss
        """
        weighted_loss = loss
        if weights is not None:
            weighted_loss = weights * loss
        weighted_loss = torch.mean(weighted_loss)
        if median:
            median_loss = torch.median(loss)
            return loss, weighted_loss, median_loss
        else:
            return loss, weighted_loss
    
    def flux_consistency_loss(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        weights: Optional[Union[float, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # 使用detach确保非掩码区域不贡献梯度
        masked_inputs = inputs * self.mask + torch.logical_not(self.mask) * inputs.detach() * 0.0
        masked_reconstructions = reconstructions * self.mask + torch.logical_not(self.mask) * reconstructions.detach() * 0.0
        
        # 添加小常数防止log(0)
        epsilon = 1e-5
        input_total_flux = torch.abs(masked_inputs).sum(dim=(2,3)) + epsilon
        rec_total_flux = torch.abs(masked_reconstructions).sum(dim=(2,3)) + epsilon
        
        input_total_flux = 22.5-2.5*torch.log10(input_total_flux)
        rec_total_flux = 22.5-2.5*torch.log10(rec_total_flux)
        
        loss = torch.abs(input_total_flux - rec_total_flux)
        
        return self.weighted_loss(loss, weights)
    
    def mse_loss(
        self,
        rec_loss: torch.Tensor,
        weights: Optional[Union[float, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rec_loss = self.mask_loss(rec_loss)
        return self.weighted_loss(rec_loss, weights)
    
    def astro_nll_loss(
        self,
        inputs: torch.Tensor,
        rec_loss: torch.Tensor,
        weights: Optional[Union[float, torch.Tensor]] = None,
        error: Optional[torch.Tensor] = None,
        normalize: bool = False,
        logvar:bool=False,
        median:bool=False,
        target_chi2=1.0,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        
        # 添加epsilon防止除零
        epsilon = 1e-8
        
        # 使用detach断开非掩码区域的梯度流
        if self.mask is not None:
            masked_rec_loss = rec_loss * self.mask + torch.logical_not(self.mask) * rec_loss.detach() * 0.0
            # 对于masked_error，在无效区域使用一个大的sigma值（相当于权重为0）
            masked_error = torch.where(self.mask, error, torch.ones_like(error))
        else:
            masked_rec_loss = rec_loss
            masked_error = error
        
        # chi2损失：chi2 = (residual/sigma)^2
        # 添加epsilon确保分母不为0
        # nll_loss = (torch.abs(masked_rec_loss) / (masked_error + epsilon))**2
        nll_loss = (masked_rec_loss**2) / (2 * masked_error**2 + epsilon)
        # if self.logvar:
        #     # 如果需要log variance项：log(sigma^2) = 2*log(sigma)
        #     nll_loss = nll_loss + 2.0 * torch.log(masked_error + epsilon)
        # np.save('nll_loss.npy', nll_loss.detach().cpu().numpy())
        
        nor_nll_loss = self.mask_loss(nll_loss, normalize=normalize, inputs=inputs)
        nll_loss = self.mask_loss(nll_loss)
        # if not self.logvar:
        #     nll_loss = nll_loss - target_chi2*torch.ones_like(nll_loss)
        return self.weighted_loss(nor_nll_loss, weights, median=median)
    def poison_nll(
        self,
        inputs: torch.Tensor,
        gen: torch.Tensor,
        weights: Optional[Union[float, torch.Tensor]] = None,
        error: Optional[torch.Tensor] = None,
        normalize: bool = False,
        logvar: bool = False,
        median: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        epsilon = 1e-8
        # Ensure all operations are done in PyTorch, not numpy
        if self.mask is not None:
            masked_error = torch.where(self.mask, error, torch.ones_like(error))
            masked_inputs = torch.where(self.mask, inputs, torch.ones_like(inputs))
            masked_gen = torch.where(self.mask, gen, torch.ones_like(gen))
        else:
            masked_error = error
            masked_inputs = inputs
            masked_gen = gen
        # Use torch.log instead of numpy log
        nll_loss = masked_gen - masked_inputs * torch.log(masked_gen + epsilon)
        nor_nll_loss = self.mask_loss(nll_loss, normalize=normalize, inputs=inputs)
        nll_loss = self.mask_loss(nll_loss)
        # if not self.logvar:
        #     nll_loss = nll_loss - target_chi2 * torch.ones_like(nll_loss)
        return self.weighted_loss(nor_nll_loss, weights, median=median)
    # def snr_chi2_nll_loss(
    #     self,
    #     rec_loss: torch.Tensor,
    #     weights: Optional[Union[float, torch.Tensor]] = None,
    #     invvar: torch.Tensor = None,
    #     inputs: torch.Tensor = None,
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
        
    #     # 使用detach断开非掩码区域的梯度流
    #     masked_rec_loss = rec_loss * self.mask + torch.logical_not(self.mask) * rec_loss.detach() * 0.0
    #     masked_invvar = invvar * self.mask + torch.logical_not(self.mask) * invvar.detach() * 0.0
        
    #     nll_loss = torch.abs(masked_rec_loss)**2 * masked_invvar
    #     if inputs is not None:
    #         # 计算信噪比SNR = inputs * sqrt(invvar)
    #         # 高信噪比区域的预测错误应该有更大的惩罚
    #         snr = inputs * torch.sqrt(masked_invvar)
    #         # 使用信噪比作为权重因子，增强高信噪比区域的损失贡献
    #         nll_loss = nll_loss * snr
    #     # 直接使用已修改的mask_loss函数
    #     nll_loss = self.mask_loss(nll_loss)
    #     return self.weighted_loss(nll_loss, weights)
    
    def cross_entropy_loss(
        self,
        rec_loss: torch.Tensor,
        error: torch.Tensor,
        weights: Optional[Union[float, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 使用detach断开非掩码区域的梯度流
        masked_rec_loss = rec_loss * self.mask + torch.logical_not(self.mask) * rec_loss.detach() * 0.0
        masked_error = error * self.mask + torch.logical_not(self.mask) * error.detach() * 0.0
        
        # 添加一个小常数防止除零
        epsilon = 1e-5
        # residual/sigma
        residual = torch.abs(masked_rec_loss) / (masked_error + epsilon)
        cross_entropy = 0.5 * residual**2 + 0.5 * torch.log(torch.tensor(2 * np.pi, device=residual.device))
        
        # 直接使用已修改的mask_loss函数
        cross_entropy = self.mask_loss(cross_entropy)
        
        return self.weighted_loss(cross_entropy, weights)
    
    # def poisson_nll_loss(
    #     self,
    #     y_true: torch.Tensor,
    #     y_pred: torch.Tensor,
    #     invvar: torch.Tensor,
    #     rec_loss: torch.Tensor,
    #     snr_threshold: float = 5.0,
    #     weight_factor: float = 1,
    #     mask: torch.Tensor = None,
    # ) -> torch.Tensor:
    #     """
    #     计算加权的泊松和高斯负对数似然损失

    #     参数:
    #     y_true -- 观测值（真实的光子数）
    #     y_pred -- 预测值（预测的光子数）
    #     invvar -- 观测误差的倒数
    #     rec_loss -- 重建损失
    #     snr_threshold -- SNR的阈值，默认值为5
    #     weight_factor -- 权重因子，用于加权两个损失
    #     mask -- 掩码，指示哪些像素需要计算损失

    #     返回:
    #     加权损失值
    #     """
    #     # 确保预测值大于0以避免log(0)的问题
    #     y_pred = torch.clamp(y_pred, min=1e-5)
        
    #     # 计算泊松负对数似然损失
    #     poisson_loss = y_pred - y_true * torch.log(y_pred) + torch.lgamma(y_true + 1)
        
    #     # 计算信噪比
    #     snr = y_true * torch.sqrt(invvar)
        
    #     # 计算高斯负对数似然损失
    #     nll_loss = rec_loss * (invvar + 1e-9)
        
    #     # 根据SNR选择基础损失
    #     use_poisson = snr <= snr_threshold
    #     base_loss = torch.where(use_poisson, poisson_loss, nll_loss)
        
    #     # 加权损失计算
    #     weighted_loss = weight_factor * base_loss + (1 - weight_factor) * nll_loss
    #     base_loss = base_loss.sum(axis=(1, 2, 3))
    #     weighted_loss = weighted_loss.sum(axis=(1, 2, 3))
        
    #     # === 关键修改部分 ===
    #     # 创建与模型所有参数相关的微小扰动
    #     param_dummy_loss = torch.tensor(0.0, device=base_loss.device)
    #     for p in self.parameters():
    #         param_dummy_loss = param_dummy_loss + p.mean() * 0.0  # 不影响数值但参与计算图
        
    #     # 将扰动同步到两个返回的损失中
    #     base_loss = base_loss + param_dummy_loss
    #     weighted_loss = weighted_loss + param_dummy_loss
    #     # ====================
        
    #     # 对batch求平均
    #     base_loss = base_loss.mean()
    #     weighted_loss = weighted_loss.mean()
        
    #     return base_loss, weighted_loss

    def multiscale_chi2_loss(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        error: torch.Tensor,
        weights: Optional[Union[float, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算多尺度chi2损失
        
        通过对输入、重建和误差图像进行逐级的bin2操作，计算不同尺度下的chi2损失，然后求平均。
        自动计算变换的尺度级别，直到最小尺寸。
        
        参数:
            inputs: 输入图像 [B, C, H, W]
            reconstructions: 重建图像 [B, C, H, W]
            error: sigma误差图 [B, C, H, W]
            weights: 可选的权重
            
        返回:
            多尺度chi2损失和加权多尺度chi2损失
        """
        # 初始化损失列表
        chi2_losses = []
        
        # 当前尺度的输入、重建和误差
        curr_inputs = inputs
        curr_reconstructions = reconstructions
        curr_error = error
        curr_mask = self.mask
        
        # 调试打印
        # print(f"Original shapes - inputs: {curr_inputs.shape}, error: {curr_error.shape}, mask: {curr_mask.shape}")
        
        # 添加epsilon防止除零
        epsilon = 1e-8
        
        # 计算原始尺度的chi2损失
        rec_loss = curr_inputs - curr_reconstructions
        if curr_mask is not None:
            masked_rec_loss = rec_loss * curr_mask + torch.logical_not(curr_mask) * rec_loss.detach() * 0.0
            # 对于masked_error，在无效区域使用一个大的sigma值（相当于权重为0）
            masked_error = torch.where(curr_mask, curr_error, torch.ones_like(curr_error))
        else:
            masked_rec_loss = rec_loss
            masked_error = curr_error
            
        # chi2 = (residual/sigma)^2
        # 添加epsilon确保分母不为0
        nll_loss = (torch.abs(masked_rec_loss) / (masked_error + epsilon))**2
        chi2_loss = self.mask_loss(nll_loss)
        chi2_losses.append(chi2_loss)
        
        # 自动计算可以进行的最大尺度级别
        # 找到输入图像的最小尺寸
        min_size = min(curr_inputs.shape[2], curr_inputs.shape[3])
        # 计算可以进行的最大bin2操作次数
        max_levels = int(np.floor(np.log2(min_size)))
        # 限制最大级别数
        levels = max_levels
        
        # 逐级进行bin2操作
        for level in range(1, levels):
            # 对输入、重建和误差进行bin2操作
            # 使用平均池化实现bin2
            curr_inputs = F.avg_pool2d(curr_inputs, kernel_size=2, stride=2)
            curr_reconstructions = F.avg_pool2d(curr_reconstructions, kernel_size=2, stride=2)
            
            # 对sigma的处理：sigma在bin2后需要除以2
            # 因为方差的传播：Var(avg(X1,X2,X3,X4)) = Var(X)/4，所以sigma = sigma_orig / sqrt(4) = sigma_orig / 2
            curr_error = F.avg_pool2d(curr_error, kernel_size=2, stride=2) / 2.0
            
            # 计算当前尺度的chi2损失
            curr_rec_loss = curr_inputs - curr_reconstructions
            
            # 如果有掩码，创建当前尺度的掩码
            if curr_mask is not None:
                # 使用平均池化来降采样掩码，保留有效像素的比例信息
                curr_mask_float = curr_mask.float()  # 转换为浮点数
                curr_mask_ratio = F.avg_pool2d(curr_mask_float, kernel_size=2, stride=2)
                
                # 调试打印
                # print(f"Level {level} shapes - inputs: {curr_inputs.shape}, error: {curr_error.shape}, mask_ratio: {curr_mask_ratio.shape}")
                
                # 确保掩码比例与error形状完全匹配
                # 这里采用更安全的方式，直接创建一个与error形状相同的掩码
                safe_mask_ratio = torch.zeros_like(curr_error)
                
                # 处理可能的形状不匹配
                # 获取两个张量的最小尺寸
                min_batch = min(curr_mask_ratio.shape[0], curr_error.shape[0])
                min_channel = min(curr_mask_ratio.shape[1], curr_error.shape[1])
                min_height = min(curr_mask_ratio.shape[2], curr_error.shape[2])
                min_width = min(curr_mask_ratio.shape[3], curr_error.shape[3])
                
                # 复制共同部分
                safe_mask_ratio[:min_batch, :min_channel, :min_height, :min_width] = \
                    curr_mask_ratio[:min_batch, :min_channel, :min_height, :min_width]
                
                # 如果通道数不同，但需要扩展
                if curr_mask_ratio.shape[1] == 1 and curr_error.shape[1] > 1:
                    for c in range(1, curr_error.shape[1]):
                        safe_mask_ratio[:min_batch, c, :min_height, :min_width] = \
                            curr_mask_ratio[:min_batch, 0, :min_height, :min_width]
                
                # 使用安全的掩码比例
                curr_mask_ratio = safe_mask_ratio
                
                # 创建二值mask（大于0.5认为是有效区域）
                curr_binary_mask = curr_mask_ratio > 0.5
                # 对于masked_error，在无效区域使用一个大的sigma值（相当于权重为0）
                masked_curr_error = torch.where(curr_binary_mask, curr_error, torch.ones_like(curr_error))
            else:
                masked_curr_error = curr_error
                
            # chi2 = (residual/sigma)^2
            # 添加epsilon确保分母不为0
            curr_nll_loss = (torch.abs(curr_rec_loss) / (masked_curr_error + epsilon))**2
            
            # 使用自定义的掩码损失函数处理当前尺度的损失
            if curr_mask is not None:
                # 创建一个临时掩码，用于标记至少有部分有效像素的区域
                temp_mask = (curr_mask_ratio > 0)
                # 计算损失时，考虑有效像素的比例
                curr_chi2_loss = self._multiscale_mask_loss(curr_nll_loss, temp_mask, curr_mask_ratio)
            else:
                curr_chi2_loss = torch.mean(curr_nll_loss, dim=(1, 2, 3))
                
            chi2_losses.append(curr_chi2_loss)
        
        # 将所有尺度的损失堆叠起来
        chi2_losses = torch.stack(chi2_losses, dim=0)  # [levels, B]
        
        # 计算所有尺度的平均损失
        avg_chi2_loss = torch.mean(chi2_losses, dim=0)  # [B]
        
        return self.weighted_loss(avg_chi2_loss, weights)
    
    def _multiscale_mask_loss(self, loss: torch.Tensor, mask: torch.Tensor, mask_ratio: torch.Tensor) -> torch.Tensor:
        """
        为多尺度chi2损失计算掩码损失
        
        参数:
            loss: 损失张量 [B, C, H, W]
            mask: 二值掩码，标记有效区域 [B, C, H, W]
            mask_ratio: 有效像素的比例 [B, C, H, W]
            
        返回:
            掩码后的损失 [B]
        """
        # 计算每个样本的损失
        # 只在有效区域计算损失，并根据有效像素比例进行加权
        masked_loss = loss * mask
        
        # 计算每个样本的有效区域总数
        valid_pixels = mask.sum(dim=(1, 2, 3)).clamp(min=1.0)  # [B]
        
        # 计算加权平均损失
        # 先对每个样本的所有像素求和，然后除以有效像素数
        sample_loss = masked_loss.sum(dim=(1, 2, 3)) / valid_pixels  # [B]
        
        return sample_loss
