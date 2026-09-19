from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

# from ...modules.autoencoding.lpips.loss.lpips import LPIPS
from ...modules.encoders.modules import GeneralConditioner
from ...util import append_dims, instantiate_from_config
from .denoiser import Denoiser
from astropy.cosmology import LambdaCDM
import numpy as np
from scipy.stats import norm
def double_gaussian(x, mu1, sigma1, amp1, mu2, sigma2, amp2):
    return amp1 * norm.pdf(x, mu1, sigma1) + amp2 * norm.pdf(x, mu2, sigma2)
def generate_double_gaussian_data(mu1, sigma1, amp1, mu2, sigma2, amp2, h, w):
    x = np.linspace(-1, 1, h*w)
    y = double_gaussian(x, mu1, sigma1, amp1, mu2, sigma2, amp2)
    y /= y.sum()  # Normalize
    data = np.random.choice(x, size=h*w, p=y)
    return data
def get_background(params_matrix, h, w, batch_size):
    batch_data = []
    for _ in range(batch_size):
        channel_data = []
        for params in params_matrix:
            channel = generate_double_gaussian_data(*params, h=h, w=w).reshape(h, w)
            channel_data.append(channel)
        batch_data.append(np.stack(channel_data))
    return np.stack(batch_data)
params_matrix = np.load('/data/public/renhaoye/DESI_background_gaussian.npy')
class StandardDiffusionLoss(nn.Module):
    def __init__(
        self,
        sigma_sampler_config: dict, 
        loss_weighting_config: dict,
        loss_type: str = "l2",
        offset_noise_level: float = 0.0,
        batch2model_keys: Optional[Union[str, List[str]]] = None,
        invvar_as_weight: bool = False,
        redshifting: bool = False,
    ):
        super().__init__()

        assert loss_type in ["l2", "l1", "lpips"]
        self.invvar_as_weight = invvar_as_weight
        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)
        self.loss_weighting = instantiate_from_config(loss_weighting_config)
        self.redshifting = redshifting
        self.loss_type = loss_type
        self.offset_noise_level = offset_noise_level # 噪声偏移量

        # if loss_type == "lpips":
        #     self.lpips = LPIPS().eval()

        if not batch2model_keys:
            batch2model_keys = []

        if isinstance(batch2model_keys, str):
            batch2model_keys = [batch2model_keys]

        self.batch2model_keys = set(batch2model_keys)

    def get_noised_input(
        self, sigmas_bc: torch.Tensor, noise: torch.Tensor, input: torch.Tensor
    ) -> torch.Tensor:
        noised_input = input + noise * sigmas_bc
        return noised_input

    def forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        conditioner: GeneralConditioner,
        input: torch.Tensor,
        batch: Dict,
        invvar: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cond = conditioner(batch) # 如果无条件则cond就是一个空字典
        return self._forward(network, denoiser, cond, input, batch) if invvar is None else self._forward(network, denoiser, cond, input, batch, invvar)

    def _forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        cond: Dict,
        input: torch.Tensor,
        batch: Dict,
        invvar: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }
        # 这里实现的就是对x_t=x_{t-1}+sigma_t*noise_t的模拟
        sigmas = self.sigma_sampler(input.shape[0], z).to(input) # 用sigma_sampler随机抽取batch_size个sigma并转换到input的device上
        # 这里标准差的数量级大概是0.0几到十几
        noise = torch.randn_like(input) # 生成和input同样shape的标准正态分布
        if self.offset_noise_level > 0.0:
            offset_shape = (
                (input.shape[0], 1, input.shape[2])
                if self.n_frames is not None
                else (input.shape[0], input.shape[1])
            )
            noise = noise + self.offset_noise_level * append_dims(
                torch.randn(offset_shape, device=input.device),
                input.ndim,
            )
        sigmas_bc = append_dims(sigmas, input.ndim) # 扩成每个batch_size的星系给一个sigma，shape=(batch_size, 1, 1, 1), 当为红移的情况时，sigmas代表噪声上需要乘的系数
        # if self.redshifting:
        #     first_part = 
        noised_input = self.get_noised_input(sigmas_bc, noise, input) # x_t=x_{t-1}+sigma_t*noise_t

        model_output = denoiser(
            network, noised_input, sigmas, cond, **additional_model_inputs
        )

        w = append_dims(self.loss_weighting(sigmas), input.ndim)
        return self.get_loss(model_output, input, w) if invvar is None else self.get_loss(model_output, input, w, invvar)

    def get_loss(self, model_output, target, w, invvar=None):
        if self.loss_type == "l2":
            if self.invvar_as_weight:
                return torch.mean((w * (model_output - target) ** 2 * invvar).reshape(target.shape[0], -1), 1)
            else:
                return torch.mean((w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1)
        elif self.loss_type == "l1":
            if self.invvvar_as_weight:
                return torch.mean((w * (model_output - target).abs() * invvar).reshape(target.shape[0], -1), 1)
            else:
                return torch.mean((w * (model_output - target).abs()).reshape(target.shape[0], -1), 1)
        # elif self.loss_type == "lpips":
        #     loss = self.lpips(model_output, target).reshape(-1)
        #     return loss
        else:
            raise NotImplementedError(f"Unknown loss type {self.loss_type}")

class RedshiftDiffusionLoss(nn.Module):
    def __init__(
        self,
        sigma_sampler_config: dict, 
        loss_weighting_config: dict,
        loss_type: str = "l2",
        offset_noise_level: float = 0.0,
        batch2model_keys: Optional[Union[str, List[str]]] = None,
        invvar_as_weight: bool = False,
        redshifting: bool = True,
        H0=70, Om0=0.3, Ode0=0.7
    ):
        super().__init__()

        assert loss_type in ["l2", "l1", "lpips"]
        self.cosmo = LambdaCDM(H0=H0, Om0=Om0, Ode0=Ode0)
        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)
        self.loss_weighting = instantiate_from_config(loss_weighting_config)
        self.redshifting = redshifting
        self.loss_type = loss_type
        self.offset_noise_level = offset_noise_level # 噪声偏移量

        # if loss_type == "lpips":
        #     self.lpips = LPIPS().eval()

        if not batch2model_keys:
            batch2model_keys = []

        if isinstance(batch2model_keys, str):
            batch2model_keys = [batch2model_keys]

        self.batch2model_keys = set(batch2model_keys)

    def get_noised_input(
        self, sigmas_bc: torch.Tensor, noise: torch.Tensor, input: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        # input是原图，sigmas_bc是每个batch_size的星系给一个sigma，noise是标准正态分布
        def cosmic_dimming(z_low, z_high):
            device = z_low.device
            Dl_local = self.cosmo.luminosity_distance(z_low.detach().cpu().numpy()).value
            Dl_high = self.cosmo.luminosity_distance(z_high.detach().cpu().numpy()).value
            dimming_factor = (Dl_local / Dl_high)**2
            return torch.Tensor(dimming_factor).to(device)
        # def dimmer(self, batch_z_low, batch_z_high):
        #     batch_z_low = batch_z_low.cpu().numpy()
        #     batch_z_high = batch_z_high.cpu().numpy()
        #     dimming_factor = torch.zeros_like(batch_z_low)
        #     for i in range(len(batch_z_low)):
        #         dimming_factor[i] = self.cosmic_dimming(batch_z_low[i], batch_z_high[i])
        #     return dimming_factor
        # def A2(self, step):
        #     return 
        
        # noised_input = cosmic_dimming(z,sigmas_bc)*input + noise * sigmas_bc # h(z-1,z)x_{z-1} + [1-h(z-1,z)]\xi
        noised_input = cosmic_dimming(z,sigmas_bc)*input + noise * (1-cosmic_dimming(z,sigmas_bc))
        return noised_input

    def forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        conditioner: GeneralConditioner,
        input: torch.Tensor,
        batch: Dict,
        invvar: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cond = conditioner(batch) # 如果无条件则cond就是一个空字典
        return self._forward(network, denoiser, cond, input, batch) if invvar is None else self._forward(network, denoiser, cond, input, batch, invvar)

    def _forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        cond: Dict,
        input: torch.Tensor,
        batch: Dict,
        invvar: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }
        # 这里实现的就是对x_t=x_{t-1}+sigma_t*noise_t的模拟
        z = batch["redshift"]["z"]
        sigmas = self.sigma_sampler(input.shape[0], z).to(input) # 用sigma_sampler随机抽取batch_size个sigma并转换到input的device上
        # 这里标准差的数量级大概是0.0几到十几
        noise = torch.randn_like(input) # 生成和input同样shape的标准正态分布
        if self.offset_noise_level > 0.0:
            offset_shape = (
                (input.shape[0], 1, input.shape[2])
                if self.n_frames is not None
                else (input.shape[0], input.shape[1])
            )
            noise = noise + self.offset_noise_level * append_dims(
                torch.randn(offset_shape, device=input.device),
                input.ndim,
            )
        z_bc = append_dims(z, input.ndim)
        sigmas_bc = append_dims(sigmas, input.ndim) # 扩成每个batch_size的星系给一个sigma，shape=(batch_size, 1, 1, 1), 当为红移的情况时，sigmas代表噪声上需要乘的系数
        # if self.redshifting:
        #     first_part = 
        noised_input = self.get_noised_input(sigmas_bc, noise, input, z_bc) # x_t=x_{t-1}+sigma_t*noise_t
        model_output = denoiser(
                network, noised_input, sigmas, cond, **additional_model_inputs
            )
        if batch.get("global_step") % 100 == 0:
            from torchvision.utils import make_grid
            from astropy.io import fits
            input_grid = make_grid(input.detach().cpu())
            fits.writeto(f'logs/test/input_{batch.get("global_step")}.fits', input_grid.numpy(), overwrite=True)
            
            noised_input_grid = make_grid(noised_input.detach().cpu())
            fits.writeto(f'logs/test/noised_input_{batch.get("global_step")}.fits', noised_input_grid.numpy(), overwrite=True)
            
            model_output_grid = make_grid(model_output.detach().cpu())
            fits.writeto(f'logs/test/model_output_{batch.get("global_step")}.fits', model_output_grid.numpy(), overwrite=True)
        w = append_dims(self.loss_weighting(sigmas), input.ndim)
        return self.get_loss(model_output, input, w) if invvar is None else self.get_loss(model_output, input, w, invvar)

    def get_loss(self, model_output, target, w, invvar=None):
        if self.loss_type == "l2":
            
            return torch.mean(
                (w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1
            ) if invvar is None else torch.mean(
                (w * (model_output - target) ** 2 * invvar).reshape(target.shape[0], -1), 1
            )
        elif self.loss_type == "l1":
            return torch.mean(
                (w * (model_output - target).abs()).reshape(target.shape[0], -1), 1
            ) if invvar is None else torch.mean(
                (w * (model_output - target).abs() * invvar).reshape(target.shape[0], -1), 1
            )
        # elif self.loss_type == "lpips":
        #     loss = self.lpips(model_output, target).reshape(-1)
        #     return loss
        else:
            raise NotImplementedError(f"Unknown loss type {self.loss_type}")
