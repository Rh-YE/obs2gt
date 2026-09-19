import torch

from ...util import default, instantiate_from_config


class EDMSampling:
    def __init__(self, p_mean=-1.2, p_std=1.2):
        self.p_mean = p_mean
        self.p_std = p_std

    def __call__(self, n_samples, rand=None):
        log_sigma = self.p_mean + self.p_std * default(rand, torch.randn((n_samples,)))
        return log_sigma.exp()


class DiscreteSampling: # 随机采样一部分sigma进行训练
    def __init__(self, discretization_config, num_idx, do_append_zero=False, flip=True):
        self.num_idx = num_idx # 总步长
        self.sigmas = instantiate_from_config(discretization_config)(
            num_idx, do_append_zero=do_append_zero, flip=flip
        )

    def idx_to_sigma(self, idx):
        return self.sigmas[idx]

    def __call__(self, n_samples, rand=None): # 采样batch_size(n_samples)个sigma, 要么传入rand，要么使用默认的torch.rand
        idx = default(
            rand,
            torch.randint(0, self.num_idx, (n_samples,)),
        ) # 生成n_samples个0到num_idx-1的随机整数用来索引sigmas里已经生成好的标准差
        return self.idx_to_sigma(idx)

class RedshiftDiscreteSampling: # 随机采样一部分sigma进行训练
    def __init__(self, discretization_config, num_idx, do_append_zero=False, flip=True):
        self.num_idx = num_idx # 总步长
        self.sigmas = instantiate_from_config(discretization_config)(
            num_idx, do_append_zero=do_append_zero, flip=flip
        )

    def idx_to_sigma(self, idx):
        return self.sigmas[idx]

    def __call__(self, n_samples, z, rand=None):
        z = z.cpu()
        idx = default(
            rand,
            torch.randint(0, self.num_idx, (n_samples,)),
        )
        sigma_vals = self.idx_to_sigma(idx)

        while True:
            mask = sigma_vals < z
            if not mask.any():
                break
            idx[mask] = torch.randint(0, self.num_idx, (mask.sum(),))
            sigma_vals = self.idx_to_sigma(idx)
            
        return sigma_vals