from typing import Dict, Union

import torch
import torch.nn as nn

from ...util import append_dims, instantiate_from_config
from .denoiser_scaling import DenoiserScaling
from .discretizer import Discretization


class Denoiser(nn.Module):
    def __init__(self, scaling_config: Dict):
        super().__init__()
        self.scaling: DenoiserScaling = instantiate_from_config(scaling_config)

    def possibly_quantize_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma

    def possibly_quantize_c_noise(self, c_noise: torch.Tensor) -> torch.Tensor:
        return c_noise

    def forward(
        self,
        network: nn.Module,
        input: torch.Tensor,
        sigma: torch.Tensor, # 给原图加的噪声
        cond: Dict,
        **additional_model_inputs,
    ) -> torch.Tensor:
        sigma = self.possibly_quantize_sigma(sigma) # 这里的sigma是红移
        sigma_shape = sigma.shape
        sigma = append_dims(sigma, input.ndim)
        c_skip, c_out, c_in, c_noise = self.scaling(sigma) # c_skip是1，c_out是-sigma，c_in是1/(sigma**2+1)**0.5，c_noise是sigma
        c_noise = self.possibly_quantize_c_noise(c_noise.reshape(sigma_shape)) # 步长
        # from torchvision.utils import make_grid
        # from astropy.io import fits
        # denoised = network(input * c_in, c_noise, cond, **additional_model_inputs)
        # input_grid = make_grid(denoised.detach().cpu())
        # fits.writeto(f'logs/test/denoised.fits', input_grid.numpy(), overwrite=True)
        # model_output_grid = make_grid((denoised * c_out).detach().cpu())
        # fits.writeto(f'logs/test/denoised_cout.fits', model_output_grid.numpy(), overwrite=True)
        # output = make_grid((denoised * c_out + input * c_skip).detach().cpu())
        # fits.writeto(f'logs/test/denoised_cout_input_cskip.fits', output.numpy(), overwrite=True)
        return (
            network(input * c_in, c_noise, cond, **additional_model_inputs) * c_out # c_noise就是之前的sigma的步长，来告诉被加噪的input是哪一步。,也就是，c_in对应的是反向过程中的1/(sigma**2+1)**0.5，c_out对应的是-beta_t
            + input * c_skip 
        ) # network 是OpenAIWrapper下的扩散模型，input*c_skip就是对原图加噪后的结果*1
        
        
class DiscreteDenoiser(Denoiser):
    def __init__(
        self,
        scaling_config: Dict,
        num_idx: int,
        discretization_config: Dict,
        do_append_zero: bool = False,
        quantize_c_noise: bool = True,
        flip: bool = True,
    ):
        super().__init__(scaling_config)
        self.discretization: Discretization = instantiate_from_config(
            discretization_config
        )
        sigmas = self.discretization(num_idx, do_append_zero=do_append_zero, flip=flip)
        self.register_buffer("sigmas", sigmas)
        self.quantize_c_noise = quantize_c_noise
        self.num_idx = num_idx

    def sigma_to_idx(self, sigma: torch.Tensor) -> torch.Tensor:
        dists = sigma - self.sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape)

    def idx_to_sigma(self, idx: Union[torch.Tensor, int]) -> torch.Tensor:
        return self.sigmas[idx]

    def possibly_quantize_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        return self.idx_to_sigma(self.sigma_to_idx(sigma))

    def possibly_quantize_c_noise(self, c_noise: torch.Tensor) -> torch.Tensor:
        if self.quantize_c_noise:
            return self.sigma_to_idx(c_noise)
        else:
            return c_noise
        