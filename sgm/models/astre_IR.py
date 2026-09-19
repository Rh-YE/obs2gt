import logging
import math
import re
from abc import abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union
import torchvision
import pytorch_lightning as pl
import torch
import torch.nn as nn
from einops import rearrange
from packaging import version
from omegaconf import ListConfig, OmegaConf

from safetensors.torch import load_file as load_safetensors
from torch.optim.lr_scheduler import LambdaLR
from ..modules import UNCONDITIONAL_CONFIG
from ..modules.autoencoding.temporal_ae import VideoDecoder
from ..modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER
from ..modules.ema import LitEma
from ..modules.autoencoding.regularizers import AbstractRegularizer
from ..modules.ema import LitEma
from ..util import (default, get_nested_attribute, get_obj_from_str,
                    instantiate_from_config)

logpy = logging.getLogger(__name__)

        
def convolve(x, psf, dp=30):
    # 归一化PSF并重塑为卷积核格式
    psf_kernel = psf / psf.sum(dim=(2,3), keepdim=True)
    psf_kernel = psf_kernel.view(-1, 1, psf.shape[2], psf.shape[3])
    
    # 自动检测非零区域并裁剪PSF核
    k0 = psf_kernel[0, 0]
    nonzero_idx = (k0 != 0).nonzero()
    if nonzero_idx.size(0) > 0:
        row_min = nonzero_idx[:, 0].min().item()
        row_max = nonzero_idx[:, 0].max().item()
        col_min = nonzero_idx[:, 1].min().item()
        col_max = nonzero_idx[:, 1].max().item()
        psf_kernel = psf_kernel[:, :, row_min:row_max+1, col_min:col_max+1]
    
    # 执行带填充的卷积操作
    output_padded = torch.nn.functional.pad(x, (dp, dp, dp, dp), mode='constant', value=0)
    convolved_output = torch.nn.functional.conv2d(
        output_padded.view(1, -1, output_padded.shape[2], output_padded.shape[3]),
        psf_kernel,
        padding='same',
        groups=psf_kernel.shape[0]
    )[:,:,dp:-dp,dp:-dp].view_as(x)
    
    return convolved_output


class MultiModalVAEDiffusion(pl.LightningModule):
    def __init__(
        self,
        euclid_vae_config: Dict,
        desi_vae_config: Dict,
        denoiser_config: Dict,
        network_config: Dict,
        conditioner_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        sampler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        optimizer_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        scheduler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        loss_fn_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        network_wrapper: Union[None, str] = None,
        ckpt_path: Union[None, str] = None,
        euclid_vae_ckpt: Union[None, str] = None,
        desi_vae_ckpt: Union[None, str] = None,
        psf_vae_ckpt: Union[None, str] = None,
        use_ema: bool = False,
        ema_decay_rate: float = 0.9999,
        scale_factor: float = 1.0,
        disable_first_stage_autocast=False,
        psf_vae_config: Dict = None,
        lr_g_factor: float = 1.0,
        log_keys: Union[List, None] = None,
        no_cond_log: bool = False,
        compile_model: bool = False,
        ema_decay: Union[None, float] = None,
        input_key_euclid: str = "euclid_img",
        input_key_desi: str = "desi_img",
        input_key_euclid_psf: str = "euclid_psf",
        input_key_desi_psf: str = "desi_psf",
        **kwargs
    ):
        super().__init__()
        
        self.input_key_euclid = input_key_euclid
        self.input_key_desi = input_key_desi
        self.input_key_euclid_psf = input_key_euclid_psf
        self.input_key_desi_psf = input_key_desi_psf
    
        # 保存配置以供后续加载模型
        self.euclid_vae_config = euclid_vae_config
        self.desi_vae_config = desi_vae_config
        self.psf_vae_config = psf_vae_config if psf_vae_config is not None else None
        
        # 初始化VAE模型
        self.euclid_vae = instantiate_from_config(self.euclid_vae_config)
        self.desi_vae = instantiate_from_config(self.desi_vae_config)
        
        # 如果提供了PSF VAE配置，则初始化它
        self.psf_vae = None
        if self.psf_vae_config is not None:
            self.psf_vae = instantiate_from_config(self.psf_vae_config)
        
        # 加载VAE模型权重
        self.load_vae_weights(euclid_vae_ckpt, desi_vae_ckpt, psf_vae_ckpt)
            
        # self._add_numerical_stability_fixes()
        # 设置所有VAE为评估模式并冻结参数
        for vae in [self.euclid_vae, self.desi_vae]:
            if vae is not None:
                vae.eval()
                for param in vae.parameters():
                    param.requires_grad = False
                    
        if self.psf_vae is not None:
            self.psf_vae.eval()
            for param in self.psf_vae.parameters():
                param.requires_grad = False
        
        # 初始化扩散模型
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

        # 添加sigma采样器
        self.sigma_sampler = None
        if loss_fn_config is not None and "sigma_sampler_config" in loss_fn_config.get("params", {}):
            self.sigma_sampler = instantiate_from_config(loss_fn_config["params"]["sigma_sampler_config"])
        elif self.loss_fn is not None and hasattr(self.loss_fn, 'sigma_sampler'):
            self.sigma_sampler = self.loss_fn.sigma_sampler

        # EMA设置 - 统一处理
        self.use_ema = use_ema or (ema_decay is not None)
        if self.use_ema:
            # 优先使用ema_decay，如果没有则使用ema_decay_rate
            decay_rate = ema_decay if ema_decay is not None else ema_decay_rate
            self.model_ema = LitEma(self.model, decay=decay_rate)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")
        else:
            self.model_ema = None

        self.scale_factor = scale_factor
        self.disable_first_stage_autocast = disable_first_stage_autocast
        self.no_cond_log = no_cond_log
        
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)
        
        logpy.info("所有VAE模型已加载")
        
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
            
    def load_vae_weights(self, euclid_vae_ckpt=None, desi_vae_ckpt=None, psf_vae_ckpt=None):
        """
        加载VAE模型的预训练权重
        
        Args:
            euclid_vae_ckpt: Euclid VAE权重路径
            desi_vae_ckpt: DESI VAE权重路径
            psf_vae_ckpt: PSF VAE权重路径
        """
        # 加载Euclid VAE权重
        if euclid_vae_ckpt is not None:
            self._load_vae_checkpoint(self.euclid_vae, euclid_vae_ckpt, "Euclid VAE")
        elif hasattr(self.euclid_vae, "apply_ckpt") and hasattr(self.euclid_vae_config, "params"):
            # 检查配置中是否有ckpt_path或ckpt_engine
            ckpt = self.euclid_vae_config.params.get("ckpt_path", None) or self.euclid_vae_config.params.get("ckpt_engine", None)
            if ckpt is not None:
                logpy.info(f"Euclid VAE使用配置中的检查点: {ckpt}")
        else:
            logpy.warning("未提供Euclid VAE权重路径，使用随机初始化的权重")
            
        # 加载DESI VAE权重
        if desi_vae_ckpt is not None:
            self._load_vae_checkpoint(self.desi_vae, desi_vae_ckpt, "DESI VAE")
        elif hasattr(self.desi_vae, "apply_ckpt") and hasattr(self.desi_vae_config, "params"):
            # 检查配置中是否有ckpt_path或ckpt_engine
            ckpt = self.desi_vae_config.params.get("ckpt_path", None) or self.desi_vae_config.params.get("ckpt_engine", None)
            if ckpt is not None:
                logpy.info(f"DESI VAE使用配置中的检查点: {ckpt}")
        else:
            logpy.warning("未提供DESI VAE权重路径，使用随机初始化的权重")
            
        # 如果有PSF VAE，加载其权重
        if self.psf_vae is not None and psf_vae_ckpt is not None:
            self._load_vae_checkpoint(self.psf_vae, psf_vae_ckpt, "PSF VAE")
        elif self.psf_vae is not None and hasattr(self.psf_vae, "apply_ckpt") and hasattr(self.psf_vae_config, "params"):
            # 检查配置中是否有ckpt_path或ckpt_engine
            ckpt = self.psf_vae_config.params.get("ckpt_path", None) or self.psf_vae_config.params.get("ckpt_engine", None)
            if ckpt is not None:
                logpy.info(f"PSF VAE使用配置中的检查点: {ckpt}")
        elif self.psf_vae is not None:
            logpy.warning("未提供PSF VAE权重路径，使用随机初始化的权重")
    
    def _load_vae_checkpoint(self, model, checkpoint_path, model_name="VAE"):
        """
        加载单个VAE模型的检查点
        
        Args:
            model: 要加载权重的模型
            checkpoint_path: 检查点路径
            model_name: 模型名称（用于日志）
        """
        if checkpoint_path.endswith(".ckpt"):
            try:
                state_dict = torch.load(checkpoint_path, map_location="cpu")
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                
                # 尝试直接加载状态字典
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                
                if len(missing) > 0:
                    logpy.warning(f"{model_name}加载时缺少键: {missing}")
                if len(unexpected) > 0:
                    logpy.warning(f"{model_name}加载时有意外键: {unexpected}")
                
                logpy.info(f"成功从{checkpoint_path}加载{model_name}权重")
                
            except Exception as e:
                logpy.error(f"加载{model_name}权重时出错: {str(e)}")
        
        elif checkpoint_path.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file as load_safetensors
                state_dict = load_safetensors(checkpoint_path)
                
                # 尝试直接加载状态字典
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                
                if len(missing) > 0:
                    logpy.warning(f"{model_name}加载时缺少键: {missing}")
                if len(unexpected) > 0:
                    logpy.warning(f"{model_name}加载时有意外键: {unexpected}")
                
                logpy.info(f"成功从{checkpoint_path}加载{model_name}权重")
                
            except Exception as e:
                logpy.error(f"加载{model_name}权重时出错: {str(e)}")
        
        else:
            logpy.error(f"不支持的检查点格式: {checkpoint_path}")
    
    def load_diffusion_model(self):
        """
        加载扩散模型
        """
        self.diffusion_model = instantiate_from_config(self.diffusion_config)
        
        # 如果使用EMA，初始化EMA模型
        if self.use_ema:
            self.model_ema = LitEma(self.diffusion_model, decay=self.ema_decay)
            logpy.info(f"为扩散模型创建EMA，保持 {len(list(self.model_ema.buffers()))} 个缓冲区")
            
        logpy.info("扩散模型已加载")
    
    def get_input(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        从批次中获取输入数据（暂时只处理图像和invvar）
        
        Args:
            batch: 包含输入数据的字典
            
        Returns:
            Euclid图像、DESI图像、Euclid invvar、DESI invvar、额外信息
        """
        # 获取数据源信息
        folder_key = batch.get("folder_key", None)
        extra_info = {"folder_key": folder_key}
        
        # 获取图像数据
        euclid_img = batch.get(self.input_key_euclid, None)
        desi_img = batch.get(self.input_key_desi, None) 
        
        # 获取误差数据（支持 'error' 和 'invvar' 两种键名，优先使用 'error'）
        euclid_invvar = batch.get("euclid_error", batch.get("euclid_invvar", None))
        desi_invvar = batch.get("desi_error", batch.get("desi_invvar", None))
        
        # 检查是否所有必要的数据都存在
        missing_keys = []
        if euclid_img is None:
            missing_keys.append(self.input_key_euclid)
        if desi_img is None:
            missing_keys.append(self.input_key_desi)
        if euclid_invvar is None:
            missing_keys.append("euclid_error/euclid_invvar")
        if desi_invvar is None:
            missing_keys.append("desi_error/desi_invvar")
        
        if missing_keys:
            raise ValueError(f"缺少必要的输入数据: {missing_keys}. 可用的键: {list(batch.keys())}")
        
        return euclid_img, desi_img, euclid_invvar, desi_invvar, extra_info
    
    @torch.no_grad()
    def encode_inputs(self, euclid_input: torch.Tensor, desi_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        # PyTorch Lightning在训练时会自动启用autocast（混合精度），但是：
        # 在训练步骤中：Lightning可能会启用autocast上下文
        # VAE在autocast环境中：某些VAE操作（特别是GroupNorm、注意力机制）在fp16精度下可能产生数值不稳定
        # 您的配置没有禁用autocast：在configs/generation/astroIR.yaml中没有设置disable_first_stage_autocast: true
        # 显式禁用autocast以确保VAE在full precision下运行
        with torch.amp.autocast('cuda', enabled=False):
            # 编码Euclid图像
            euclid_z, euclid_rec, euclid_reg_log = self.euclid_vae(euclid_input)
            
            # 编码DESI图像  
            desi_z, desi_rec, desi_reg_log = self.desi_vae(desi_input)
    
        return {
            "euclid_z": euclid_z,
            "euclid_rec": euclid_rec, 
            "euclid_reg_log": euclid_reg_log,
            "desi_z": desi_z,
            "desi_rec": desi_rec,
            "desi_reg_log": desi_reg_log,
        }
    
    def process_and_concat_latents(self, encoded_outputs: Dict[str, torch.Tensor], extra_info: Dict = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        处理隐变量，返回要扩散的目标（Euclid）和条件（DESI）
        """
        # 从编码输出中提取隐变量
        euclid_z = encoded_outputs["euclid_z"]
        desi_z = encoded_outputs["desi_z"]
        
        # 返回Euclid作为扩散目标，DESI作为条件
        return euclid_z, desi_z
    
    def forward(self, batch: Dict) -> Dict:
        # 获取输入数据和额外信息
        euclid_img, desi_img, euclid_invvar, desi_invvar, extra_info = self.get_input(batch)
        
        # 编码输入
        euclid_input = torch.cat([euclid_img, euclid_invvar], dim=1) if euclid_invvar is not None else euclid_img
        desi_input = torch.cat([desi_img, desi_invvar], dim=1) if desi_invvar is not None else desi_img
        encoded_outputs = self.encode_inputs(euclid_input, desi_input)
        
        # 处理隐变量：Euclid作为扩散目标，DESI作为条件
        euclid_z, desi_z = self.process_and_concat_latents(encoded_outputs, extra_info)
        
        # 为了使用标准的扩散框架，我们需要创建一个包含条件信息的新batch
        # 将DESI隐变量作为条件添加到batch中
        diffusion_batch = batch.copy()
        diffusion_batch["desi_z"] = desi_z
        
        # 扩散训练：对Euclid隐变量添加噪声并训练去噪
        if self.training and self.loss_fn is not None:
            # 使用标准扩散损失进行训练
            loss = self.loss_fn(
                self.model, 
                self.denoiser, 
                self.conditioner, 
                euclid_z, 
                diffusion_batch
            )
            diffusion_output = loss
        else:
            # 推理模式或没有损失函数时：直接返回编码结果
            diffusion_output = euclid_z
        
        # 构建输出字典
        outputs = {
            "encoded_outputs": encoded_outputs,
            "euclid_z": euclid_z,
            "desi_z": desi_z,
            "diffusion_output": diffusion_output,
            "extra_info": extra_info,
            "euclid_invvar": euclid_invvar,
            "desi_invvar": desi_invvar
        }
        
        return outputs
    
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """
        训练步骤
        
        Args:
            batch: 输入批次
            batch_idx: 批次索引
            
        Returns:
            损失值
        """
        outputs = self(batch)
        
        # 获取扩散损失
        loss = outputs["diffusion_output"]
        
        # 确保损失是标量值，如果是张量则取平均值
        if loss.dim() > 0:
            loss_scalar = loss.mean()
        else:
            loss_scalar = loss
        
        # 计算reduced chi-squared
        with torch.no_grad():
            # 获取真实的Euclid图像和invvar
            euclid_img, _, euclid_invvar, _, _ = self.get_input(batch)
            
            # 获取重建的Euclid图像
            euclid_rec = outputs["encoded_outputs"]["euclid_rec"]
            
            # 计算chi-squared: sum((obs - model)^2 * invvar)
            chi2 = ((euclid_img - euclid_rec) ** 2 * euclid_invvar).sum()
            
            # 计算自由度 (N_pixels - N_params)
            # 这里假设N_params约等于潜在空间维度
            n_pixels = euclid_img.numel()
            n_params = outputs["euclid_z"].numel()
            dof = n_pixels - n_params
            
            # 计算reduced chi-squared
            reduced_chi2 = chi2 / dof if dof > 0 else chi2
        
        # 记录多种格式的损失，确保ModelCheckpoint能找到监控指标
        self.log("train_loss", loss_scalar, prog_bar=True, on_step=True, on_epoch=True,sync_dist=True)
        self.log("train/loss", loss_scalar, on_step=True, on_epoch=True,sync_dist=True)
        self.log("train/loss/nll", loss_scalar, on_step=True, on_epoch=True,sync_dist=True)
        self.log("train/reduced_chi2", reduced_chi2, prog_bar=True, on_step=True, on_epoch=True,sync_dist=True)
        
        return loss_scalar
    
    def on_train_start(self, *args, **kwargs):
        if self.sampler is None or self.loss_fn is None:
            raise ValueError("Sampler and loss function need to be set for training.")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)
    
    def validation_step(self, batch: Dict, batch_idx: int) -> Dict:
        """
        验证步骤
        
        Args:
            batch: 输入批次
            batch_idx: 批次索引
            
        Returns:
            包含验证指标的字典
        """
        outputs = self(batch)
        
        # TODO: 实现验证指标计算逻辑
        val_loss = outputs["diffusion_output"]  # 假设diffusion_output是损失值
        
        # 确保损失是标量值
        if val_loss.dim() > 0:
            val_loss_scalar = val_loss.mean()
        else:
            val_loss_scalar = val_loss
        
        # 计算reduced chi-squared
        with torch.no_grad():
            # 获取真实的Euclid图像和invvar
            euclid_img, _, euclid_invvar, _, _ = self.get_input(batch)
            
            # 获取重建的Euclid图像
            euclid_rec = outputs["encoded_outputs"]["euclid_rec"]
            
            # 计算chi-squared: sum((obs - model)^2 * invvar)
            chi2 = ((euclid_img - euclid_rec) ** 2 * euclid_invvar).sum()
            
            # 计算自由度 (N_pixels - N_params)
            n_pixels = euclid_img.numel()
            n_params = outputs["euclid_z"].numel()
            dof = n_pixels - n_params
            
            # 计算reduced chi-squared
            reduced_chi2 = chi2 / dof if dof > 0 else chi2
        
        # 记录验证损失
        self.log("val_loss", val_loss_scalar, on_step=False, on_epoch=True, sync_dist=True,sync_dist=True)
        self.log("val/loss", val_loss_scalar, on_step=False, on_epoch=True, sync_dist=True,sync_dist=True)
        self.log("val/reduced_chi2", reduced_chi2, on_step=False, on_epoch=True, sync_dist=True,sync_dist=True)
        
        return {"val_loss": val_loss_scalar, "val_reduced_chi2": reduced_chi2}
    
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
    def log_images(self, batch: Dict, **kwargs) -> Dict:
        """
        记录图像用于可视化
        
        可视化内容：
        - 第i列：DESI图像（4通道，48x48）
        - 第i+1列：真实Euclid图像（1通道复制成4通道）
        - 第i+2列：扩散生成的Euclid图像（1通道复制成4通道）
        
        Args:
            batch: 输入批次
            
        Returns:
            包含图像的字典
        """
        log = dict()
        # 获取输入数据
        euclid_img, desi_img, euclid_invvar, desi_invvar, extra_info = self.get_input(batch)
        
        # 获取批次大小
        batch_size = euclid_img.shape[0]
        
        # 处理隐变量
        euclid_input = torch.cat([euclid_img, euclid_invvar], dim=1) if euclid_invvar is not None else euclid_img
        desi_input = torch.cat([desi_img, desi_invvar], dim=1) if desi_invvar is not None else desi_img
        encoded_outputs = self.encode_inputs(euclid_input, desi_input)
        euclid_z, desi_z = self.process_and_concat_latents(encoded_outputs, extra_info)
        
        # 准备条件信息用于采样
        diffusion_batch = batch.copy()
        diffusion_batch["desi_z"] = desi_z
        
        # 使用扩散模型生成样本
        if self.sampler is not None:
            with self.ema_scope("Sampling"):
                # 使用conditioner处理条件
                cond = self.conditioner(diffusion_batch)
                
                # 从噪声开始采样
                noise = torch.randn_like(euclid_z)
                
                # 修正denoiser函数调用方式
                def denoiser_fn(x, sigma, cond, uc=None):
                    return self.denoiser(self.model, x, sigma, cond)
                
                # 进行采样
                samples = self.sampler(denoiser_fn, noise, cond)
                
                # 解码采样结果到图像空间
                generated_euclid = self.euclid_vae.decode(samples)
        else:
            # 如果没有采样器，使用原始的euclid隐变量
            generated_euclid = self.euclid_vae.decode(euclid_z)
            print("Warning: No sampler available, using original Euclid latents")
        
        # 将Euclid图像（1通道）复制成4通道以便可视化
        euclid_img_4ch = euclid_img.repeat(1, 3, 1, 1)  # [B, 1, 48, 48] -> [B, 4, 48, 48]
        generated_euclid_4ch = generated_euclid.repeat(1, 3, 1, 1)  # [B, 1, 48, 48] -> [B, 4, 48, 48]
        euclid_rec_ch = encoded_outputs["euclid_rec"].repeat(1, 3, 1, 1)  # [B, 1, 48, 48] -> [B, 4, 48, 48]
        
        # 创建可视化：按照autoencoder.py的风格，在水平方向拼接
        # 每个样本：DESI图像 | 真实Euclid图像 | 生成的Euclid图像
        log["comparison"] = torch.cat([desi_img, encoded_outputs["desi_rec"], euclid_img_4ch, euclid_rec_ch, generated_euclid_4ch], dim=-1)  # 在宽度维度拼接
        
        # 添加以星系为单位归一化的结果
        def normalize_per_galaxy(images):
            """对每个星系图像进行单独归一化"""
            normalized = torch.zeros_like(images)
            for i in range(images.shape[0]):  # 遍历batch中的每个星系
                img = images[i]  # [C, H, W]
                # 计算每个星系图像的最小值和最大值
                img_min = img.min()
                img_max = img.max()
                # 避免除零错误
                if img_max > img_min:
                    normalized[i] = (img - img_min) / (img_max - img_min)
                else:
                    normalized[i] = img
            return normalized
        
        # 对每种图像类型进行归一化
        desi_img_normalized = normalize_per_galaxy(desi_img)
        desi_rec_normalized = normalize_per_galaxy(encoded_outputs["desi_rec"])
        euclid_img_4ch_normalized = normalize_per_galaxy(euclid_img_4ch)
        generated_euclid_4ch_normalized = normalize_per_galaxy(generated_euclid_4ch)
        euclid_rec_ch_normalized = normalize_per_galaxy(euclid_rec_ch)
        # 创建归一化后的比较图像
        log["comparison_normalized"] = torch.cat([
            desi_img_normalized, 
            desi_rec_normalized,
            euclid_img_4ch_normalized, 
            euclid_rec_ch_normalized,
            generated_euclid_4ch_normalized,
        ], dim=-1)  # 在宽度维度拼接
        
        return log
