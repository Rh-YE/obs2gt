from abc import abstractmethod
from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....modules.distributions.distributions import \
    DiagonalGaussianDistribution
from .base import AbstractRegularizer
from .ae_regularizer import AERegularizer, L2Regularizer


class DiagonalGaussianRegularizer(AbstractRegularizer):
    def __init__(self, sample: bool = True):
        super().__init__()
        self.sample = sample

    def get_trainable_parameters(self) -> Any:
        yield from ()

    def forward(self, z: torch.Tensor, single_dim: bool =False) -> Tuple[torch.Tensor, dict]:
        log = dict()
        posterior = DiagonalGaussianDistribution(z)
        if self.sample:
            z = posterior.sample()
        else:
            z = posterior.mode()
        kl_loss = posterior.kl(single_dim)
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        log["kl_loss"] = kl_loss
        return z, log
