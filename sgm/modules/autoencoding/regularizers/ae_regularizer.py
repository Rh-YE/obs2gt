"""
Regularizer for Autoencoder (AE) that doesn't require probabilistic modeling.
"""

import torch
from typing import Tuple, Any
from . import AbstractRegularizer


class AERegularizer(AbstractRegularizer):
    """
    Simple regularizer for Autoencoder that doesn't perform probabilistic sampling.
    Just passes through the latent representation without modification.
    """
    def __init__(self, sample: bool = False):
        super().__init__()
        self.sample = sample  # Always False for AE
    
    def get_trainable_parameters(self) -> Any:
        yield from ()
    
    def forward(self, z: torch.Tensor, single_dim: bool = False) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            z: latent representation tensor
            single_dim: ignored for AE
        Returns:
            z: unchanged latent representation
            log: empty log dict
        """
        log = dict()
        log["kl_loss"] = torch.tensor(0.0, device=z.device)  # No KL loss for AE
        return z, log


class L2Regularizer(AbstractRegularizer):
    """
    L2 regularizer for Autoencoder latent representations.
    """
    def __init__(self, weight: float = 0.001):
        super().__init__()
        self.weight = weight
    
    def get_trainable_parameters(self) -> Any:
        yield from ()
    
    def forward(self, z: torch.Tensor, single_dim: bool = False) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            z: latent representation tensor
            single_dim: ignored for AE
        Returns:
            z: unchanged latent representation
            log: log dict with L2 regularization loss
        """
        log = dict()
        l2_loss = self.weight * torch.mean(z ** 2)
        log["l2_loss"] = l2_loss
        log["kl_loss"] = torch.tensor(0.0, device=z.device)  # Compatibility with VAE loss
        return z, log
