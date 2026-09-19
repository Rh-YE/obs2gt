__all__ = [
    "GeneralLPIPSWithDiscriminator",
    "LatentLPIPS",
]

# from .discriminator_loss import GeneralLPIPSWithDiscriminator
from .vae_loss import GeneralVAELoss
from .disc_loss import DiscVAELoss
from .lpips import LatentLPIPS
# from .toy_loss import ToyVAELoss