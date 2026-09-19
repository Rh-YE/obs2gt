import logging
import math
import os
import re
from abc import abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torchvision
import pytorch_lightning as pl
import torch
import torch.nn as nn
from einops import rearrange
from packaging import version
from astropy.cosmology import LambdaCDM
from ..modules.autoencoding.regularizers import AbstractRegularizer
from ..modules.ema import LitEma
from ..util import (default, get_nested_attribute, get_obj_from_str,
                    instantiate_from_config)

logpy = logging.getLogger(__name__)

class AbstractAutoencoder(pl.LightningModule):
    """
    This is the base class for all autoencoders, including image autoencoders, image autoencoders with discriminators,
    unCLIP models, etc. Hence, it is fairly general, and specific features
    (e.g. discriminator training, encoding, decoding) must be implemented in subclasses.
    """

    def __init__(
        self,
        ema_decay: Union[None, float] = None,
        monitor: Union[None, str] = None,
        input_key: str = "jpg",
    ):
        super().__init__()

        self.input_key = input_key
        self.use_ema = ema_decay is not None
        if monitor is not None:
            self.monitor = monitor

        if self.use_ema:
            self.model_ema = LitEma(self, decay=ema_decay)
            logpy.info(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if version.parse(torch.__version__) >= version.parse("2.0.0"):
            self.automatic_optimization = False

    def apply_ckpt(self, ckpt: Union[None, str, dict]):
        if ckpt is None:
            return
        if isinstance(ckpt, str):
            ckpt = {
                "target": "sgm.modules.checkpoint.CheckpointEngine",
                "params": {"ckpt_path": ckpt},
            }
        engine = instantiate_from_config(ckpt)
        engine(self)

    @abstractmethod
    def get_input(self, batch) -> Any:
        raise NotImplementedError()

    def on_train_batch_end(self, *args, **kwargs):
        # 如果有未更新的梯度，强制更新
        if self.global_step % self.accumulate_grad_batches != 0:
            opts = self.optimizers()
            if not isinstance(opts, list):
               opts = [opts]
            for opt in opts:
                opt.step()
        # for EMA computation
        if self.use_ema:
            self.model_ema(self)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                logpy.info(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    logpy.info(f"{context}: Restored training weights")

    @abstractmethod
    def encode(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("encode()-method of abstract base class called")

    @abstractmethod
    def decode(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("decode()-method of abstract base class called")

    def instantiate_optimizer_from_config(self, params, lr, cfg):
        logpy.info(f"loading >>> {cfg['target']} <<< optimizer from config")
        return get_obj_from_str(cfg["target"])(
            params, lr=lr, **cfg.get("params", dict())
        )

    def configure_optimizers(self) -> Any:
        raise NotImplementedError()


class AutoencodingEngine(AbstractAutoencoder):
    """
    Base class for all image autoencoders that we train, like VQGAN or AutoencoderKL
    (we also restore them explicitly as special cases for legacy reasons).
    Regularizations such as KL or VQ are moved to the regularizer class.
    """

    def __init__(
        self,
        *args,
        encoder_config: Dict,
        decoder_config: Dict,
        loss_config: Dict,
        regularizer_config: Dict,
        optimizer_config: Union[Dict, None] = None,
        lr_g_factor: float = 1.0,
        trainable_ae_params: Optional[List[List[str]]] = None,
        ae_optimizer_args: Optional[List[dict]] = None,
        trainable_disc_params: Optional[List[List[str]]] = None,
        disc_optimizer_args: Optional[List[dict]] = None,
        disc_start_iter: int = 0,
        diff_boost_factor: float = 3.0,
        ckpt_engine: Union[None, str, dict] = None,
        ckpt_path: Optional[str] = None,
        additional_decode_keys: Optional[List[str]] = None,
        with_invvar: bool = False,
        no_invvar_but_output_invvar: bool = False,
        single_dim: bool = False,
        latent_dim: int = 80,
        color_map: bool = False,
        predict_invvar: bool = False,
        with_psf: bool = False,
        expand_psf: bool = False,
        random_psf: bool = False,
        use_mask: bool = False,
        accumulate_grad_batches: int = 1,
        cat_invvar: bool = False,
        use_sigma: bool = False,
        sigma_film: bool = False,
        grad_clip_norm: Optional[float] = None,
        target_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # target_key (paper_plan_v2 三范式共享同一 VAE 骨干的开关):
        #   None    -> AE-self, 训练目标 = 网络输入自身 (旧行为不变, 瓶颈是唯一
        #              的防抄袭机制)
        #   'gt'    -> N2C, 训练目标 = 无噪真值 (仅仿真数据合法)
        #   'obs_b' -> N2N, 训练目标 = 同一场景独立第二次噪声实现
        # 三个范式共用完全相同的 encoder/decoder/regularizer/loss 配置, 只切
        # 这一个字段, 避免像 direct_reconstruction.py 的 _FixedTargetEngine
        # 体系那样为每个范式单独建子类——VAE 引擎的 forward/loss 路径本就统一。
        self.target_key = target_key
        self.random_psf = random_psf
        self.with_psf = with_psf
        self.with_invvar = with_invvar
        self.no_invvar_but_output_invvar = no_invvar_but_output_invvar
        self.automatic_optimization = False  # pytorch lightning
        self.accumulate_grad_batches = accumulate_grad_batches
        self.color_map = color_map
        self.predict_invvar = predict_invvar
        self.use_mask = use_mask
        self.encoder: torch.nn.Module = instantiate_from_config(encoder_config)
        self.decoder: torch.nn.Module = instantiate_from_config(decoder_config)
        self.loss: torch.nn.Module = instantiate_from_config(loss_config)
        self.single_dim = single_dim
        self.latent_dim = latent_dim
        self.regularization: AbstractRegularizer = instantiate_from_config(
            regularizer_config
        )
        self.optimizer_config = default(
            optimizer_config, {"target": "torch.optim.Adam"}
        )
        self.diff_boost_factor = diff_boost_factor
        self.disc_start_iter = disc_start_iter
        self.lr_g_factor = lr_g_factor
        self.trainable_ae_params = trainable_ae_params
        self.expand_psf = expand_psf
        self.cat_invvar = cat_invvar
        self.sigma_film = sigma_film  # 用 sigma 做 decoder spatial FiLM 条件化 (与 cat_invvar 互斥)
        self.use_sigma = use_sigma  # 是否使用sigma代替invvar
        # 梯度裁剪阈值 (max_norm); None = 不裁剪 (保持旧行为向后兼容)。
        # 手动优化路径下 Lightning trainer 级别的 gradient_clip_val 不会自动
        # 生效, 见 training_step 里的实际调用点。
        self.grad_clip_norm = grad_clip_norm
        if self.trainable_ae_params is not None:
            self.ae_optimizer_args = default(
                ae_optimizer_args,
                [{} for _ in range(len(self.trainable_ae_params))],
            )
            assert len(self.ae_optimizer_args) == len(self.trainable_ae_params)
        else:
            self.ae_optimizer_args = [{}]  # makes type consitent

        self.trainable_disc_params = trainable_disc_params
        if self.trainable_disc_params is not None:
            self.disc_optimizer_args = default(
                disc_optimizer_args,
                [{} for _ in range(len(self.trainable_disc_params))],
            )
            assert len(self.disc_optimizer_args) == len(self.trainable_disc_params)
        else:
            self.disc_optimizer_args = [{}]  # makes type consitent

        if ckpt_path is not None:
            assert ckpt_engine is None, "Can't set ckpt_engine and ckpt_path"
            logpy.warn("Checkpoint path is deprecated, use `checkpoint_egnine` instead")
        self.apply_ckpt(default(ckpt_path, ckpt_engine))
        self.additional_decode_keys = set(default(additional_decode_keys, []))
        
        # 用于收集验证集的直方图数据
        self.val_histogram_inputs = []
        self.val_histogram_reconstructions = []
        self.val_histogram_chi2 = []
        self.val_histogram_target_samples = 1000

    def get_input(self, batch: Dict) -> torch.Tensor:
        # assuming unified data format, dataloader returns a dict.
        # image tensors should be scaled to -1 ... 1 and in channels-first
        # format (e.g., bchw instead if bhwc)
        return batch[self.input_key]

    def get_autoencoder_params(self) -> list:
        params = []
        if hasattr(self.loss, "get_trainable_autoencoder_parameters"):
            params += list(self.loss.get_trainable_autoencoder_parameters())
        if hasattr(self.regularization, "get_trainable_parameters"):
            params += list(self.regularization.get_trainable_parameters())
        params = params + list(self.encoder.parameters())
        params = params + list(self.decoder.parameters())
        return params

    def get_discriminator_params(self) -> list:
        if hasattr(self.loss, "get_trainable_parameters"):
            params = list(self.loss.get_trainable_parameters())  # e.g., discriminator
        else:
            params = []
        return params

    def get_last_layer(self):
        return self.decoder.get_last_layer()

    def encode(
        self,
        x: torch.Tensor,
        return_reg_log: bool = False,
        unregularized: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        z = self.encoder(x)
        if unregularized:
            return z, dict()
        z, reg_log = self.regularization(z, self.single_dim)
        if return_reg_log:
            return z, reg_log
        return z

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.decoder(z, **kwargs)
        return x

    def forward(
        self, x: torch.Tensor, sigma: torch.Tensor = None, **additional_decode_kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:

        z, reg_log = self.encode(x, return_reg_log=True)  # 这里的reglog就是kl_loss
        # sigma_film 模式: 把 sigma 传给 decoder 做 spatial FiLM 条件化
        if self.sigma_film and sigma is not None:
            dec = self.decode(z, sigma=sigma, **additional_decode_kwargs)
        else:
            dec = self.decode(z, **additional_decode_kwargs)
        return z, dec, reg_log
    
    def get_error(self, batch: Dict) -> torch.Tensor:
        """
        获取error map（sigma）
        
        注意：error已经在数据加载时转换为sigma并裁剪到[1e-6, 1]范围
        这里直接返回batch中的error即可
        """
        error = batch["error"]
        return error

    def inner_training_step(
        self, batch: dict, batch_idx: int, optimizer_idx: int = 0
    ) -> torch.Tensor:
        raw_img = self.get_input(batch)  # 原始图像, 恒为网络输入 (OBS)
        try:
            gt = batch["gt"]
        except:
            gt = raw_img
        # 训练目标: target_key=None 时为 AE-self (目标=网络输入自身, 瓶颈防抄袭);
        # 'gt'/'obs_b' 时为 N2C/N2N (目标=独立字段, 见构造函数注释)。只有目标
        # 换了, 网络输入、误差图、正则化路径全部不变——这保证三范式共享同一
        # 骨干与同一公平协议, 差异只落在损失看见的答案上。
        target_img = raw_img if self.target_key is None else batch[self.target_key]
        img = raw_img
        error_for_loss = None

        sigma_cond = None
        if self.with_invvar:
            error = self.get_error(batch)
            if self.cat_invvar:
                img = torch.cat([img, error], dim=1)
            elif self.sigma_film:
                sigma_cond = error  # 通过 decoder FiLM 注入, 不 cat 到输入
        x = img
        additional_decode_kwargs = {
            key: batch[key] for key in self.additional_decode_keys.intersection(batch)
        }
        z, xrec, regularization_log = self(x, sigma=sigma_cond, **additional_decode_kwargs)

        # x_img: 梯度回传对着的答案, 由 target_key 决定 (AE-self/N2C/N2N 的唯一
        # 差异点)。x_log: 诊断用, 恒对 GT 算一份 (不参与梯度, detach), 使
        # train/loss/rec_to_gt 在三个范式间可比——即便训练目标不是 GT, 也能在
        # 训练曲线里直接看到"离真值多远", 不必等到验证/测试阶段。
        x_img = target_img
        x_log = gt

        if self.with_invvar and self.cat_invvar:
            n_img_channels = raw_img.shape[1]
            error_for_loss = img[:, n_img_channels:, :, :]

        if hasattr(self.loss, "forward_keys"):
            extra_info = {
                "z": z,
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
        if optimizer_idx == 0:
            # 梯度用 x_img (训练答案, 由 target_key 决定); out_loss_log 额外对
            # GT 算一份纯诊断值 (no_grad, 不影响优化), 提供"训练中就能看见的
            # 离真值距离"——率-失真实时绘图脚本读的正是这份 log。
            if self.with_invvar:
                sig = error_for_loss if error_for_loss is not None else error
                out_loss = self.loss(x_img, xrec, sig, **extra_info)
                with torch.no_grad():
                    out_loss_log = self.loss(x_log, xrec, sig, **extra_info)
            else:
                out_loss = self.loss(x_img, xrec, **extra_info)
                with torch.no_grad():
                    out_loss_log = self.loss(x_log, xrec, **extra_info)

            # aeloss 参与梯度 (对训练目标 x_img 算); log_dict_ae 记录训练损失本身
            # 的分解 (mse/mae/chi2/kl 等, 对 x_img 算); log_dict_gt 是纯诊断
            # (对 GT 算的同一组指标, key 加 _to_gt 后缀, 不与训练损失同名, 不
            # 参与任何选择/优化决策——只是让训练曲线里能直接看到离真值多远)。
            if isinstance(out_loss, tuple):
                aeloss, log_dict_ae = out_loss
            else:
                aeloss = out_loss
                log_dict_ae = {"train/loss/rec": aeloss.detach()}

            if isinstance(out_loss_log, tuple):
                _, log_dict_gt_raw = out_loss_log
                log_dict_gt = {f"{k}_to_gt": v for k, v in log_dict_gt_raw.items()}
            else:
                log_dict_gt = {}

            self.log_dict(
                log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True,batch_size=x.shape[0]
            )
            if log_dict_gt:
                self.log_dict(
                    log_dict_gt, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True,batch_size=x.shape[0]
                )
            self.log(
                "loss", aeloss.mean().detach(), prog_bar=True, logger=False, on_epoch=False, on_step=True,sync_dist=True,batch_size=x.shape[0]
            )
            return aeloss
        elif optimizer_idx == 1:
            # 对判别器loss也做同样处理
            if self.with_invvar:
                if error_for_loss is not None:
                    discloss = self.loss(x_img, xrec, error_for_loss, **extra_info)
                    discloss_log = self.loss(x_log, xrec, error_for_loss, **extra_info)
                else:
                    discloss = self.loss(x_img, xrec, error, **extra_info)
                    discloss_log = self.loss(x_log, xrec, error, **extra_info)
            else:
                discloss = self.loss(x_img, xrec, **extra_info)
                discloss_log = self.loss(x_log, xrec, **extra_info)

            # log_dict_disc 用gt生成
            if isinstance(discloss_log, tuple):
                _, log_dict_disc = discloss_log
            else:
                log_dict_disc = {}

            self.log_dict(
                log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True,batch_size=x.shape[0]
            )
            return discloss if not isinstance(discloss, tuple) else discloss[0]
        else:
            raise NotImplementedError(f"Unknown optimizer {optimizer_idx}")

    def training_step(self, batch: dict, batch_idx: int):
        opts = self.optimizers()
        if not isinstance(opts, list):
            opts = [opts]
        optimizer_idx = batch_idx % len(opts)
        if self.global_step < self.disc_start_iter:
            optimizer_idx = 0
        opt = opts[optimizer_idx]
        
        # 只在每10个批次的第一个批次清零梯度
        if self.global_step % self.accumulate_grad_batches == 0:
            opt.zero_grad()
            
        with opt.toggle_model():
            loss = self.inner_training_step(
                batch, batch_idx, optimizer_idx=optimizer_idx
            )
            # 将损失除以10来实现梯度累积
            scaled_loss = loss / self.accumulate_grad_batches
            self.manual_backward(scaled_loss)

        # 梯度裁剪 (对应审计条目 A7): 手动优化 (automatic_optimization=False) 下
        # PyTorch Lightning trainer 级别的 gradient_clip_val 不会自动生效, 必须
        # 显式调用。chi2 类损失的逐像素权重 1/sigma^2 可跨数个量级, 梯度尺度远
        # 大于 mse/mae/huber, 若不裁剪, "名义上相同的学习率"在不同损失间并不
        # 等效 (chi2 更容易发散或需要更保守的有效步长)。裁剪对全部损失一视同
        # 仁地生效, 是"公平比较协议"的一部分, 而不是针对某个损失的特殊处理。
        # 只裁剪当前 optimizer_idx 对应的参数子集 (与 opt.toggle_model() 的范围
        # 一致), 生成器/判别器 (optimizer_idx=1, GAN 场景) 分别裁剪, 不互相影响。
        if self.grad_clip_norm is not None and (self.global_step + 1) % self.accumulate_grad_batches == 0:
            clip_params = (self.get_autoencoder_params() if optimizer_idx == 0
                           else self.get_discriminator_params())
            torch.nn.utils.clip_grad_norm_(clip_params, self.grad_clip_norm)

        if (self.global_step + 1) % self.accumulate_grad_batches == 0:
            opt.step()

    def validation_step(self, batch: dict, batch_idx: int) -> Dict:
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, postfix="_ema")
            log_dict.update(log_dict_ema)
        return log_dict
    
    def on_validation_epoch_end(self):
        """在验证周期结束时保存直方图数据"""
        if self.trainer.global_rank != 0:
            return
        
        if len(self.val_histogram_inputs) == 0:
            return
        all_inputs = torch.cat(self.val_histogram_inputs, dim=0).numpy()
        all_reconstructions = torch.cat(self.val_histogram_reconstructions, dim=0).numpy()
        bins = np.arange(0, 50, 0.1)
        input_hist, _ = np.histogram(all_inputs.flatten(), bins=bins)
        recon_hist, _ = np.histogram(all_reconstructions.flatten(), bins=bins)
        
        # 准备保存的数据字典
        save_data = {
            'bins': bins,
            'input_hist': input_hist,
            'recon_hist': recon_hist,
            'n_samples': len(all_inputs),
            'epoch': self.current_epoch,
            'global_step': self.global_step,
        }
        
        # 如果有reduced chi2数据，也计算reduced chi2的直方图
        has_chi2 = False
        if len(self.val_histogram_chi2) > 0:
            has_chi2 = True
            all_reduced_chi2 = torch.cat(self.val_histogram_chi2, dim=0).numpy()  # 已经是reduced chi2
            reduced_chi2_bins = np.arange(0, 10, 0.05)
            reduced_chi2_hist, _ = np.histogram(all_reduced_chi2, bins=reduced_chi2_bins)
            save_data['reduced_chi2_bins'] = reduced_chi2_bins
            save_data['reduced_chi2_hist'] = reduced_chi2_hist
            save_data['mean_reduced_chi2'] = float(np.mean(all_reduced_chi2))
            save_data['median_reduced_chi2'] = float(np.median(all_reduced_chi2))
            save_data['std_reduced_chi2'] = float(np.std(all_reduced_chi2))
        
        save_dir = os.path.join(self.logger.save_dir, "histograms")
        os.makedirs(save_dir, exist_ok=True)
        
        filename = f"hist_epoch-{self.current_epoch:06d}_step-{self.global_step:09d}.npz"
        save_path = os.path.join(save_dir, filename)
        
        np.savez(save_path, **save_data)
        
        logpy.info(f"保存了 {len(all_inputs)} 个样本的直方图数据到: {save_path}")
        
        if has_chi2:
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                ax.hist(all_reduced_chi2, bins=reduced_chi2_bins, histtype='step', 
                       label=f'n={len(all_inputs)}, mean={save_data["mean_reduced_chi2"]:.3f}')
                ax.axvline(x=1, color='red', linestyle='--', linewidth=1, label='Ideal χ²=1')
                ax.set_xlabel('Reduced Chi-square Value', fontsize=14)
                ax.set_ylabel('Count', fontsize=14)
                ax.set_title(f'Reduced χ² Distribution', fontsize=16)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper right', fontsize=12)
                fig.text(0.99, 0.01, f'Epoch {self.current_epoch} | Step {self.global_step}', 
                        ha='right', va='bottom', fontsize=10, alpha=0.7)
                plt.tight_layout()
                plot_filename = f"reduced_chi2_hist_epoch-{self.current_epoch:06d}_step-{self.global_step:09d}.png"
                plot_path = os.path.join(save_dir, plot_filename)
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                logpy.info(f"保存了reduced chi2直方图图像到: {plot_path}")
            except Exception as e:
                logpy.warning(f"绘制reduced chi2直方图时出错: {e}")
                import traceback
                traceback.print_exc()
        
        self.val_histogram_inputs = []
        self.val_histogram_reconstructions = []
        self.val_histogram_chi2 = []

    def _validation_step(self, batch: dict, batch_idx: int, postfix: str = "") -> Dict:
        optimizer_idx = 0
        # 损失/反向传播相关
        input_img = self.get_input(batch)
        img = input_img
        error_for_loss = None

        sigma_cond = None
        if self.with_invvar:
            error = self.get_error(batch)
            if self.cat_invvar:
                img = torch.cat([img, error], dim=1)
            elif self.sigma_film:
                sigma_cond = error  # 通过 decoder FiLM 注入, 不 cat 到输入
        x = img
        additional_decode_kwargs = {
            key: batch[key] for key in self.additional_decode_keys.intersection(batch)
        }
        z, xrec, regularization_log = self(x, sigma=sigma_cond, **additional_decode_kwargs)

        # 验证损失用与训练一致的答案 (target_key), 保证 val 曲线与训练目标同构;
        # gt_img 恒为真值, 用于展示/直方图/率-失真诊断 log, 与训练目标是否为
        # GT 无关。
        x_img = input_img if self.target_key is None else batch[self.target_key]
        try:
            gt_img = batch["gt"]
        except:
            gt_img = x_img

        if self.with_invvar and self.cat_invvar:
            n_img_channels = input_img.shape[1]
            error_for_loss = img[:, n_img_channels:, :, :]

        # 展示和统计：全部收集gt为真值的数据
        if postfix == "" and len(self.val_histogram_inputs) < self.val_histogram_target_samples:
            batch_size = gt_img.shape[0]
            remaining = self.val_histogram_target_samples - len(self.val_histogram_inputs)
            samples_to_take = min(batch_size, remaining)
            self.val_histogram_inputs.append(gt_img[:samples_to_take].detach().cpu())
            self.val_histogram_reconstructions.append(xrec[:samples_to_take].detach().cpu())
            if self.with_invvar:
                error_for_chi2 = error_for_loss if error_for_loss is not None else error
                epsilon = 1e-8
                chi2_per_pixel = ((gt_img - xrec) / (error_for_chi2 + epsilon)) ** 2
                chi2_per_pixel = chi2_per_pixel[:samples_to_take]
                C, H, W = chi2_per_pixel.shape[1:]
                reduced_chi2 = chi2_per_pixel.sum(dim=(1, 2, 3)) / (C * H * W)
                self.val_histogram_chi2.append(reduced_chi2.detach().cpu())
        
        if hasattr(self.loss, "forward_keys"):
            extra_info = {
                "z": z,
                "optimizer_idx": optimizer_idx,
                "global_step": self.global_step,
                "last_layer": self.get_last_layer(),
                "split": "val" + postfix,
                "regularization_log": regularization_log,
                "autoencoder": self,
            }
            extra_info = {k: extra_info[k] for k in self.loss.forward_keys}
        else:
            extra_info = dict()

        # 验证损失: 对训练目标 (x_img) 算一份 (与训练曲线同构), 再对 GT
        # (gt_img) 算一份纯诊断 (key 加 _to_gt, 不影响任何选择——checkpoint
        # 选择走 UnifiedMetricCheckpoint 的独立 val MAE-to-GT, 不读这里)。
        sig = error_for_loss if error_for_loss is not None else (error if self.with_invvar else None)
        if self.with_invvar:
            out_loss = self.loss(x_img, xrec, sig, **extra_info)
            with torch.no_grad():
                out_loss_gt = self.loss(gt_img, xrec, sig, **extra_info)
        else:
            out_loss = self.loss(x_img, xrec, **extra_info)
            with torch.no_grad():
                out_loss_gt = self.loss(gt_img, xrec, **extra_info)

        if isinstance(out_loss, tuple):
            aeloss, log_dict_ae = out_loss
        else:
            aeloss = out_loss
            log_dict_ae = {f"val{postfix}/loss/mse": aeloss.detach()}

        if isinstance(out_loss_gt, tuple):
            _, log_dict_gt_raw = out_loss_gt
            log_dict_gt = {f"{k}_to_gt": v for k, v in log_dict_gt_raw.items()}
        else:
            log_dict_gt = {}

        log_dict_ae = {k: v.to(x.device) if isinstance(v, torch.Tensor) else v for k, v in log_dict_ae.items()}
        log_dict_gt = {k: v.to(x.device) if isinstance(v, torch.Tensor) else v for k, v in log_dict_gt.items()}
        full_log_dict = {**log_dict_ae, **log_dict_gt}

        have_disc = len(self.get_discriminator_params()) > 0

        if have_disc and "optimizer_idx" in extra_info:
            try:
                orig_extra_info = extra_info.copy()
                extra_info["optimizer_idx"] = 1
                # 判别器损失也用get_input(batch)
                if self.with_invvar:
                    if error_for_loss is not None:
                        discloss, log_dict_disc = self.loss(x_img, xrec, error_for_loss, **extra_info)
                    else:
                        discloss, log_dict_disc = self.loss(x_img, xrec, error, **extra_info)
                else:
                    discloss, log_dict_disc = self.loss(x_img, xrec, **extra_info)

                log_dict_disc = {k: v.to(x.device) if isinstance(v, torch.Tensor) else v for k, v in log_dict_disc.items()}
                full_log_dict.update(log_dict_disc)
            except NotImplementedError as e:
                logpy.info(f"判别器损失计算在验证阶段不可用: {str(e)}")
                extra_info = orig_extra_info
            except Exception as e:
                logpy.warning(f"计算判别器损失时出现错误: {str(e)}")
                extra_info = orig_extra_info
        
        self.log(
            f"val{postfix}/loss/mse",
            log_dict_ae[f"val{postfix}/loss/mse"],
            sync_dist=True,batch_size=x.shape[0]
        )
        self.log_dict(full_log_dict, sync_dist=True,batch_size=x.shape[0])
        return full_log_dict

    def get_param_groups(
        self, parameter_names: List[List[str]], optimizer_args: List[dict]
    ) -> Tuple[List[Dict[str, Any]], int]:
        groups = []
        num_params = 0
        for names, args in zip(parameter_names, optimizer_args):
            params = []
            for pattern_ in names:
                pattern_params = []
                pattern = re.compile(pattern_)
                for p_name, param in self.named_parameters():
                    if re.match(pattern, p_name):
                        pattern_params.append(param)
                        num_params += param.numel()
                if len(pattern_params) == 0:
                    logpy.warn(f"Did not find parameters for pattern {pattern_}")
                params.extend(pattern_params)
            groups.append({"params": params, **args})
        return groups, num_params

    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        if self.trainable_ae_params is None:
            ae_params = self.get_autoencoder_params()
        else:
            ae_params, num_ae_params = self.get_param_groups(
                self.trainable_ae_params, self.ae_optimizer_args
            )
            logpy.info(f"Number of trainable autoencoder parameters: {num_ae_params:,}")
        if self.trainable_disc_params is None:
            disc_params = self.get_discriminator_params()
        else:
            disc_params, num_disc_params = self.get_param_groups(
                self.trainable_disc_params, self.disc_optimizer_args
            )
            logpy.info(
                f"Number of trainable discriminator parameters: {num_disc_params:,}"
            )
        opt_ae = self.instantiate_optimizer_from_config(
            ae_params,
            default(self.lr_g_factor, 1.0) * self.learning_rate,
            self.optimizer_config,
        )
        opts = [opt_ae]
        if len(disc_params) > 0:
            opt_disc = self.instantiate_optimizer_from_config(
                disc_params, self.learning_rate, self.optimizer_config
            )
            opts.append(opt_disc)

        return opts

    @torch.no_grad()
    def log_images(
        self, batch: dict, additional_log_kwargs: Optional[Dict] = None, **kwargs
    ) -> dict:
        log = dict()
        additional_decode_kwargs = {}
        # 重建直接用 get_input(batch) 做网络前向，不参与梯度，不会影响梯度流/训练
        input_img = self.get_input(batch)

        # 展示和分析类全部用batch["gt"]，但网络推理只用get_input(batch)
        try:
            x_gt = batch["gt"]
        except:
            x_gt = input_img
        error = self.get_error(batch)
        additional_decode_kwargs.update(
            {key: batch[key] for key in self.additional_decode_keys.intersection(batch)}
        )

        if self.cat_invvar:
            _, raw_rec, _ = self(torch.cat([input_img, error], dim=1), **additional_decode_kwargs)
        elif self.sigma_film:
            _, raw_rec, _ = self(input_img, sigma=error, **additional_decode_kwargs)
        else:
            _, raw_rec, _ = self(input_img, **additional_decode_kwargs)
        batch_rec = raw_rec

        # 展示用真值和重建(both detached from graph, no grad)
        if self.with_psf:
            batch_rec = batch_rec[:, :x_gt.shape[1] // 2, :, :].contiguous()
            if self.with_psf:
                log["raw_input-rec"] = torch.cat([input_img,x_gt, input_img-raw_rec, raw_rec, batch_rec, x_gt - batch_rec], dim=-1)
        else:
            log["raw_input-rec"] = torch.cat([input_img, x_gt, input_img-raw_rec, raw_rec, x_gt - raw_rec], dim=-1)
        if self.with_invvar:
            error = self.get_error(batch)
            epsilon = 1e-8
            chi2 = ((x_gt - batch_rec) / (error + epsilon)) ** 2
            B, C, H, W = chi2.shape
            reduced_chi2 = chi2.sum(dim=(2, 3)) / (H * W)
            chi2_display = chi2.clone()
            chi2_display[:, :, :5, :5] = 0
            for b in range(B):
                for c in range(C):
                    chi2_display[b, c, :3, :3] = reduced_chi2[b, c]
            log["chi2"] = chi2_display

        if hasattr(self.loss, "log_images"):
            log.update(self.loss.log_images(x_gt, batch_rec))
        if additional_log_kwargs:
            additional_decode_kwargs.update(additional_log_kwargs)
            # 这里依然从 get_input(batch) 做前向推理
            if self.cat_invvar:
                _, batch_rec_add, _ = self(torch.cat([input_img, error], dim=1), **additional_decode_kwargs)
            else:
                _, batch_rec_add, _ = self(input_img, **additional_decode_kwargs)
            log_str = "reconstructions-" + "-".join(
                [f"{key}={additional_log_kwargs[key]}" for key in additional_log_kwargs]
            )
            log[log_str] = batch_rec_add

        is_ae_mode = (
            hasattr(self.regularization, 'sample')
            and not getattr(self.regularization, 'sample', True)
        ) or (
            'AERegularizer' in str(type(self.regularization))
        )
        if not is_ae_mode:
            if self.single_dim:
                samples = self.sample(36, self.latent_dim // 2, self.single_dim, self.decoder.z_shape[2], input_img.device)
            else:
                if len(self.decoder.z_shape) == 4:
                    latent_channels = self.decoder.z_shape[1]
                    spatial_size = self.decoder.z_shape[2]
                else:
                    latent_channels = self.decoder.z_shape[0] if len(self.decoder.z_shape) == 3 else self.decoder.z_shape[1]
                    spatial_size = self.decoder.z_shape[-1]
                samples = self.sample(36, latent_channels, self.single_dim, spatial_size, input_img.device)
            log["samples"] = torchvision.utils.make_grid(samples, nrow=6, padding=0)
        else:
            log["samples"] = torchvision.utils.make_grid(batch_rec[:36], nrow=6, padding=0)
        return log
    
    @torch.no_grad()
    def sample(self, num_samples: int, latent_dim: int, single_dim: bool, spatial_dim: int, device: torch.device = torch.device("cuda")) -> torch.Tensor:
        """
        Generate samples from the autoencoder.
        
        Args:
            num_samples (int): Number of samples to generate.
            device (torch.device): The device to perform the sampling on.
        
        Returns:
            torch.Tensor: Generated samples.
        """
        self.eval()
        with torch.no_grad():
            # Sample latent vectors from a standard normal distribution
            if single_dim:
                z = torch.randn(num_samples, latent_dim).to(device)
            else:
                z = torch.randn(num_samples, latent_dim, spatial_dim, spatial_dim).to(device)
            # Decode the latent vectors to generate samples
            samples = self.decode(z)
        return samples


class AutoencodingEngineLegacy(AutoencodingEngine):
    def __init__(self, embed_dim: int, **kwargs):
        self.max_batch_size = kwargs.pop("max_batch_size", None)
        ddconfig = kwargs.pop("ddconfig")
        ckpt_path = kwargs.pop("ckpt_path", None)
        ckpt_engine = kwargs.pop("ckpt_engine", None)
        super().__init__(
            encoder_config={
                "target": "sgm.modules.diffusionmodules.model.Encoder",
                "params": ddconfig,
            },
            decoder_config={
                "target": "sgm.modules.diffusionmodules.model.Decoder",
                "params": ddconfig,
            },
            **kwargs,
        )
        self.quant_conv = torch.nn.Conv2d(
            (1 + ddconfig["double_z"]) * ddconfig["z_channels"],
            (1 + ddconfig["double_z"]) * embed_dim,
            1,
        )
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim

        self.apply_ckpt(default(ckpt_path, ckpt_engine))

    def get_autoencoder_params(self) -> list:
        params = super().get_autoencoder_params()
        return params

    def encode(
        self, x: torch.Tensor, return_reg_log: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        if self.max_batch_size is None:
            z = self.encoder(x)
            z = self.quant_conv(z)
        else:
            N = x.shape[0]
            bs = self.max_batch_size
            n_batches = int(math.ceil(N / bs))
            z = list()
            for i_batch in range(n_batches):
                z_batch = self.encoder(x[i_batch * bs : (i_batch + 1) * bs])
                z_batch = self.quant_conv(z_batch)
                z.append(z_batch)
            z = torch.cat(z, 0)

        z, reg_log = self.regularization(z)
        if return_reg_log:
            return z, reg_log
        return z

    def decode(self, z: torch.Tensor, **decoder_kwargs) -> torch.Tensor:
        if self.max_batch_size is None:
            dec = self.post_quant_conv(z)
            dec = self.decoder(dec, **decoder_kwargs)
        else:
            N = z.shape[0]
            bs = self.max_batch_size
            n_batches = int(math.ceil(N / bs))
            dec = list()
            for i_batch in range(n_batches):
                dec_batch = self.post_quant_conv(z[i_batch * bs : (i_batch + 1) * bs])
                dec_batch = self.decoder(dec_batch, **decoder_kwargs)
                dec.append(dec_batch)
            dec = torch.cat(dec, 0)

        return dec


class AutoencoderKL(AutoencodingEngineLegacy):
    def __init__(self, **kwargs):
        if "lossconfig" in kwargs:
            kwargs["loss_config"] = kwargs.pop("lossconfig")
        super().__init__(
            regularizer_config={
                "target": (
                    "sgm.modules.autoencoding.regularizers"
                    ".DiagonalGaussianRegularizer"
                )
            },
            **kwargs,
        )


class AutoencoderLegacyVQ(AutoencodingEngineLegacy):
    def __init__(
        self,
        embed_dim: int,
        n_embed: int,
        sane_index_shape: bool = False,
        **kwargs,
    ):
        if "lossconfig" in kwargs:
            logpy.warn(f"Parameter `lossconfig` is deprecated, use `loss_config`.")
            kwargs["loss_config"] = kwargs.pop("lossconfig")
        super().__init__(
            regularizer_config={
                "target": (
                    "sgm.modules.autoencoding.regularizers.quantize" ".VectorQuantizer"
                ),
                "params": {
                    "n_e": n_embed,
                    "e_dim": embed_dim,
                    "sane_index_shape": sane_index_shape,
                },
            },
            **kwargs,
        )


class IdentityFirstStage(AbstractAutoencoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_input(self, x: Any) -> Any:
        return x

    def encode(self, x: Any, *args, **kwargs) -> Any:
        return x

    def decode(self, x: Any, *args, **kwargs) -> Any:
        return x


class AEIntegerWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        shape: Union[None, Tuple[int, int], List[int]] = (16, 16),
        regularization_key: str = "regularization",
        encoder_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.model = model
        assert hasattr(model, "encode") and hasattr(
            model, "decode"
        ), "Need AE interface"
        self.regularization = get_nested_attribute(model, regularization_key)
        self.shape = shape
        self.encoder_kwargs = default(encoder_kwargs, {"return_reg_log": True})

    def encode(self, x) -> torch.Tensor:
        assert (
            not self.training
        ), f"{self.__class__.__name__} only supports inference currently"
        _, log = self.model.encode(x, **self.encoder_kwargs)
        assert isinstance(log, dict)
        inds = log["min_encoding_indices"]
        return rearrange(inds, "b ... -> b (...)")

    def decode(
        self, inds: torch.Tensor, shape: Union[None, tuple, list] = None
    ) -> torch.Tensor:
        # expect inds shape (b, s) with s = h*w
        shape = default(shape, self.shape)  # Optional[(h, w)]
        if shape is not None:
            assert len(shape) == 2, f"Unhandeled shape {shape}"
            inds = rearrange(inds, "b (h w) -> b h w", h=shape[0], w=shape[1])
        h = self.regularization.get_codebook_entry(inds)  # (b, h, w, c)
        h = rearrange(h, "b h w c -> b c h w")
        return self.model.decode(h)


class AutoencoderKLModeOnly(AutoencodingEngineLegacy):
    def __init__(self, **kwargs):
        if "lossconfig" in kwargs:
            kwargs["loss_config"] = kwargs.pop("lossconfig")
        super().__init__(
            regularizer_config={
                "target": (
                    "sgm.modules.autoencoding.regularizers"
                    ".DiagonalGaussianRegularizer"
                ),
                "params": {"sample": False},
            },
            **kwargs,
        )
