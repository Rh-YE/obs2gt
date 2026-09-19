from abc import abstractmethod
from functools import partial

import numpy as np
import torch

from ...modules.diffusionmodules.util import make_beta_schedule
from ...util import append_zero

from astropy.cosmology import LambdaCDM

def generate_roughly_equally_spaced_steps(
    num_substeps: int, max_step: int
) -> np.ndarray:
    return np.linspace(max_step - 1, 0, num_substeps, endpoint=False).astype(int)[::-1]

# def generate_redshift_steps(
#     num_substeps: int, max_step: int
# )->np.ndarray:
#     num_points = 1000
#     split_point = 0.3
#     num_points_left = int(num_points * 0.7)  # 比如 70% 的点在左侧
#     num_points_right = num_points - num_points_left
#     x_left = np.linspace(0, split_point, num_points_left)
#     x_right = np.linspace(split_point, 3, num_points_right)
#     x = np.concatenate((x_left, x_right))
#     return x

class Discretization:
    def __call__(self, n, do_append_zero=True, device="cpu", flip=False):
        sigmas = self.get_sigmas(n, device=device)
        sigmas = append_zero(sigmas) if do_append_zero else sigmas
        return sigmas if not flip else torch.flip(sigmas, (0,))

    @abstractmethod
    def get_sigmas(self, n, device):
        pass


class EDMDiscretization(Discretization):
    def __init__(self, sigma_min=0.002, sigma_max=80.0, rho=7.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def get_sigmas(self, n, device="cpu"):
        ramp = torch.linspace(0, 1, n, device=device)
        min_inv_rho = self.sigma_min ** (1 / self.rho)
        max_inv_rho = self.sigma_max ** (1 / self.rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
        return sigmas


class LegacyDDPMDiscretization(Discretization): # 调用时就会生成一个sigmas
    def __init__(
        self,
        linear_start=0.00085,
        linear_end=0.0120,
        num_timesteps=1000,
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        betas = make_beta_schedule(
            "linear", num_timesteps, linear_start=linear_start, linear_end=linear_end
        )
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0) # 这里计算的是每个时间步长的alpha的累积乘积，即\bar{alpha}
        self.to_torch = partial(torch.tensor, dtype=torch.float32)

    def get_sigmas(self, n, device="cpu"):
        if n < self.num_timesteps:
            timesteps = generate_roughly_equally_spaced_steps(n, self.num_timesteps)
            alphas_cumprod = self.alphas_cumprod[timesteps]
        elif n == self.num_timesteps:
            alphas_cumprod = self.alphas_cumprod
        else:
            raise ValueError

        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        sigmas = to_torch((1 - alphas_cumprod) / alphas_cumprod) ** 0.5 # \sqrt{\frac{1-\bar{\alpha}}{\bar{\alpha}}}
        return torch.flip(sigmas, (0,))

class RedshiftDDPMDiscretization(Discretization): # 调用时就会生成一个sigmas
    def __init__(
        self,
        num_timesteps=1000,
        sample_redshift=0.3,
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.redshift = make_beta_schedule(
            "redshifting", num_timesteps, sample_redshift=sample_redshift
        )
        self.to_torch = partial(torch.tensor, dtype=torch.float32)

    def cosmic_dimming(self, z_low, z_high, cosmo=LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)):
        Dl_local = cosmo.luminosity_distance(z_low).value
        Dl_high = cosmo.luminosity_distance(z_high).value
        dimming_factor = (Dl_local / Dl_high)**2
        return dimming_factor
    # def dimmer_A(self, batch_z, z_high):
    #     batch_z_low = batch_z.cpu().numpy()
    #     not_add_A = torch.zeros_like(batch_z_low)
    #     for i in range(len(batch_z_low)):
    #         not_add_A[i] = self.cosmic_dimming(batch_z_low[i], z_high)
    #     return not_add_A
    def get_sigmas(self, n, device="cpu"):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        redshift_result = to_torch(self.redshift)
        # factor = self.dimmer_A(redshift_result, 3)
        return torch.flip(redshift_result, (0,))
