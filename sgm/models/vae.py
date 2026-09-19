import math
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from omegaconf import ListConfig, OmegaConf
from safetensors.torch import load_file as load_safetensors
from torch.optim.lr_scheduler import LambdaLR

from ..modules import UNCONDITIONAL_CONFIG
from ..modules.autoencoding.temporal_ae import VideoDecoder
from ..modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER
from ..modules.ema import LitEma
from ..util import (default, disabled_train, get_obj_from_str,
                    instantiate_from_config, log_txt_as_img)

import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F


class VanillaVAE(BaseVAE):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()
        self.latent_dim = latent_dim
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)
        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
class VAEEngine(pl.LightningModule):
    def __init__(
        self,
        network_config,
        denoiser_config,
        conditioner_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        sampler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        optimizer_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        scheduler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        loss_fn_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        network_wrapper: Union[None, str] = None,
        ckpt_path: Union[None, str] = None,
        use_ema: bool = False,
        ema_decay_rate: float = 0.9999,
        scale_factor: float = 1.0,
        disable_first_stage_autocast=False,
        input_key: str = "jpg",
        log_keys: Union[List, None] = None,
        no_cond_log: bool = False,
        compile_model: bool = False,
        en_and_decode_n_samples_a_time: Optional[int] = None,
    ):
        super().__init__()
        self.log_keys = log_keys
        self.input_key = input_key
        self.optimizer_config = default(
            optimizer_config, {"target": "torch.optim.AdamW"}
        )
        model = instantiate_from_config(network_config)
        self.model = get_obj_from_str(default(network_wrapper, OPENAIUNETWRAPPER))(
            model, compile_model=compile_model
        )

        self.denoiser = instantiate_from_config(denoiser_config)
        self.sampler = (
            instantiate_from_config(sampler_config)
            if sampler_config is not None
            else None
        )
        self.conditioner = instantiate_from_config(
            default(conditioner_config, UNCONDITIONAL_CONFIG)
        )
        self.scheduler_config = scheduler_config
        # self._init_first_stage(first_stage_config)

        self.loss_fn = (
            instantiate_from_config(loss_fn_config)
            if loss_fn_config is not None
            else None
        )

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model, decay=ema_decay_rate)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.scale_factor = scale_factor
        self.disable_first_stage_autocast = disable_first_stage_autocast
        self.no_cond_log = no_cond_log

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time

    def init_from_ckpt(
        self,
        path: str,
    ) -> None:
        if path.endswith("ckpt"):
            sd = torch.load(path, map_location="cpu")["state_dict"]
        elif path.endswith("safetensors"):
            sd = load_safetensors(path)
        else:
            raise NotImplementedError

        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(
            f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")


    def get_input(self, batch):
        # assuming unified data format, dataloader returns a dict.
        # image tensors should be scaled to -1 ... 1 and in bchw format
        return batch[self.input_key]

    @torch.no_grad()
    def decode_first_stage(self, z):
        return z
        # z = 1.0 / self.scale_factor * z
        # n_samples = default(self.en_and_decode_n_samples_a_time, z.shape[0])

        # n_rounds = math.ceil(z.shape[0] / n_samples)
        # all_out = []
        # with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
        #     for n in range(n_rounds):
        #         if isinstance(self.first_stage_model.decoder, VideoDecoder):
        #             kwargs = {"timesteps": len(z[n * n_samples : (n + 1) * n_samples])}
        #         else:
        #             kwargs = {}
        #         out = self.first_stage_model.decode(
        #             z[n * n_samples : (n + 1) * n_samples], **kwargs
        #         )
        #         all_out.append(out)
        # out = torch.cat(all_out, dim=0)
        # return out

    @torch.no_grad()
    def encode_first_stage(self, x): # 图像进来的预处理。
        return x
        # n_samples = default(self.en_and_decode_n_samples_a_time, x.shape[0])
        # n_rounds = math.ceil(x.shape[0] / n_samples)
        # all_out = []
        # with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
        #     for n in range(n_rounds):
        #         out = self.first_stage_model.encode(
        #             x[n * n_samples : (n + 1) * n_samples]
        #         )
        #         all_out.append(out)
        # z = torch.cat(all_out, dim=0)
        # z = self.scale_factor * z # config中的缩放因子
        # return z

    def forward(self, x, batch):
        loss = self.loss_fn(self.model, self.denoiser, self.conditioner, x, batch)
        loss_mean = loss.mean()
        loss_dict = {"loss": loss_mean}
        return loss_mean, loss_dict

    def shared_step(self, batch: Dict) -> Any:
        x = self.get_input(batch)
        # x = self.encode_first_stage(x)
        batch["global_step"] = self.global_step
        loss, loss_dict = self(x, batch)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(
            loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False
        )

        self.log(
            "global_step",
            self.global_step,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,sync_dist=True
        )

        if self.scheduler_config is not None:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log(
                "lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False,sync_dist=True
            )

        return loss

    def on_train_start(self, *args, **kwargs):
        if self.sampler is None or self.loss_fn is None:
            raise ValueError("Sampler and loss function need to be set for training.")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def instantiate_optimizer_from_config(self, params, cfg):
        return get_obj_from_str(cfg["target"])(
            params, **cfg.get("params", dict())
        )

    def configure_optimizers(self):
        # lr = self.learning_rate
        params = list(self.model.parameters())
        for embedder in self.conditioner.embedders:
            if embedder.is_trainable:
                params = params + list(embedder.parameters())
        opt = self.instantiate_optimizer_from_config(params, self.optimizer_config)
        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [opt], scheduler
        return opt

    @torch.no_grad()
    def sample(
        self,
        cond: Dict,
        uc: Union[Dict, None] = None,
        batch_size: int = 16,
        shape: Union[None, Tuple, List] = None,
        **kwargs,
    ):
        randn = torch.randn(batch_size, *shape).to(self.device)

        denoiser = lambda input, sigma, c: self.denoiser(
            self.model, input, sigma, c, **kwargs
        )
        samples = self.sampler(denoiser, randn, cond, uc=uc)
        return samples

    @torch.no_grad()
    def log_conditionings(self, batch: Dict, n: int) -> Dict:
        """
        Defines heuristics to log different conditionings.
        These can be lists of strings (text-to-image), tensors, ints, ...
        """
        image_h, image_w = batch[self.input_key].shape[2:]
        log = dict()

        for embedder in self.conditioner.embedders:
            if (
                (self.log_keys is None) or (embedder.input_key in self.log_keys)
            ) and not self.no_cond_log:
                x = batch[embedder.input_key][:n]
                if isinstance(x, torch.Tensor):
                    if x.dim() == 1:
                        # class-conditional, convert integer to string
                        x = [str(x[i].item()) for i in range(x.shape[0])]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 4)
                    elif x.dim() == 2:
                        # size and crop cond and the like
                        x = [
                            "x".join([str(xx) for xx in x[i].tolist()])
                            for i in range(x.shape[0])
                        ]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                elif isinstance(x, (List, ListConfig)):
                    if isinstance(x[0], str):
                        # strings
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
                log[embedder.input_key] = xc
        return log

    @torch.no_grad()
    def log_images(self, batch: Dict, N: int = 8, sample: bool = True, ucg_keys: List[str] = None, **kwargs) -> Dict:
        log = dict()
        x = self.get_input(batch)
        N = min(x.shape[0], N)
        x = x.to(self.device)[:N]
        log["inputs"] = x
        log["reconstructions"] = x  # 直接返回输入作为重建图像

        if sample:
            with self.ema_scope("Plotting"):
                c, uc = self.conditioner.get_unconditional_conditioning(
                    batch,
                    force_uc_zero_embeddings=ucg_keys if len(self.conditioner.embedders) > 0 else [],
                )
                sampling_kwargs = {}
                samples = self.sample(c, shape=x.shape[1:], uc=uc, batch_size=N, **sampling_kwargs)
            log["samples"] = samples
        return log