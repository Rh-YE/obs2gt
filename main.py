#!/data1/public/renhaoye/miniforge-pypy3/envs/ai4galaxy/bin/python
import argparse
import datetime
import glob
import inspect
import os
import sys
import traceback
from inspect import Parameter
from typing import Union, Callable, Dict, Any, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
import wandb
from matplotlib import pyplot as plt
from natsort import natsorted # natural sort
from omegaconf import OmegaConf
from packaging import version
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.utilities import rank_zero_only
from astropy.io import fits
from sgm.util import exists, instantiate_from_config, isheatmap
import importlib.util

MULTINODE_HACKS = True
def arcsinh_rgb(imgs, mode="CHW", m = 0.03, clip=True):
    bands = ["g", "r", "z"]
    rgbscales =dict(g=(2,6.0), r=(1,3.4), z=(0,2.2))

    I = 0
    for img,band in zip(imgs, bands):
        plane,scale = rgbscales[band]
        img = np.maximum(0, img * scale + m)
        I = I + img
    I /= len(bands)

    Q = 50
    fI = np.arcsinh(Q * I) / np.sqrt(Q)
    I += (I == 0.) * 1e-9
    H,W = I.shape
    if mode == "HWC":
        rgb = np.zeros((H,W,3), np.float32)
        for img,band in zip(imgs, bands):
            plane,scale = rgbscales[band]
            rgb[:,:,plane] = (img * scale + m) * fI / I
    elif mode == "CHW":
        rgb = np.zeros((3,H,W), np.float32)
        for img,band in zip(imgs, bands):
            plane,scale = rgbscales[band]
            rgb[plane] = (img * scale + m) * fI / I
    if clip:
        rgb = np.clip(rgb, 0, 1)
    return rgb
def scale(img, Q=2000):
    return np.arcsinh(img*Q)/np.sqrt(Q)

def default_trainer_args():
    # 获取 Trainer 类构造函数的参数签名
    argspec = dict(inspect.signature(Trainer.__init__).parameters)
    
    # 移除 'self' 参数，因为 'self' 只是实例方法的占位符，不是实际的参数
    argspec.pop("self")
    
    # 创建一个字典，包含每个参数及其默认值
    default_args = {
        param: argspec[param].default
        for param in argspec
        if argspec[param].default != Parameter.empty  # 仅包含有默认值的参数
    }
    
    # 返回包含默认参数的字典
    return default_args

def load_custom_function(function_path: str) -> Callable:
    """
    从给定路径加载自定义函数。
    
    Args:
        function_path: 格式为 "module.submodule:function_name" 的函数路径
        
    Returns:
        加载的函数
    """
    if not function_path or ":" not in function_path:
        raise ValueError(f"Invalid function path: {function_path}. Expected format: 'module.path:function_name'")
    
    module_path, function_name = function_path.split(":")
    try:
        module = importlib.import_module(module_path)
        function = getattr(module, function_name)
        if not callable(function):
            raise TypeError(f"Object {function_name} in {module_path} is not callable")
        return function
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not import {function_name} from {module_path}: {e}")

class ImageLogger(Callback):  # 定义一个继承自Callback的类ImageLogger
    def __init__(  # 初始化方法
        self,
        batch_frequency,  # 批次频率
        max_images,  # 最大图像数
        clamp=False,  # 是否限制图像值
        increase_log_steps=False,  # 是否增加日志记录步骤
        rescale=True,  # 是否重新调整图像
        disabled=False,  # 是否禁用日志记录
        log_on_batch_idx=False,  # 是否在批次索引上记录日志
        log_first_step=False,  # 是否记录第一步
        log_images_kwargs=None,  # 记录图像的参数
        log_before_first_step=False,  # 是否在第一步之前记录
        enable_autocast=True,  # 是否启用自动混合精度
        rgb_transform="default",  # RGB图像转换函数路径或"default"使用默认arcsinh_rgb
        scale_transform="default",  # 缩放图像转换函数路径或"default"使用默认scale
        transform_kwargs=None,  # 转换函数的额外参数
    ):
        super().__init__()  # 调用父类的初始化方法
        self.enable_autocast = enable_autocast  # 设置是否启用自动混合精度
        self.rescale = rescale  # 设置是否重新调整图像
        self.batch_freq = batch_frequency  # 设置批次频率
        self.max_images = max_images  # 设置最大图像数
        self.log_steps = [2**n for n in range(int(np.log2(self.batch_freq)) + 1)]  # 设置日志记录步骤
        if not increase_log_steps:  # 如果不增加日志记录步骤
            self.log_steps = [self.batch_freq]  # 仅使用批次频率
        self.clamp = clamp  # 设置是否限制图像值
        self.disabled = disabled  # 设置是否禁用日志记录
        self.log_on_batch_idx = log_on_batch_idx  # 设置是否在批次索引上记录日志
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}  # 设置记录图像的参数
        self.log_first_step = log_first_step  # 设置是否记录第一步
        self.log_before_first_step = log_before_first_step  # 设置是否在第一步之前记录
        self.last_val_batch = None  # 存储最后一个验证批次
        self.last_val_batch_idx = None  # 存储最后一个验证批次的索引
        self.transform_kwargs = transform_kwargs or {}  # 转换函数的额外参数
        
        # 设置RGB转换函数
        if rgb_transform == "default":
            self.rgb_transform = arcsinh_rgb
        else:
            try:
                self.rgb_transform = load_custom_function(rgb_transform)
            except Exception as e:
                print(f"警告：无法加载自定义RGB转换函数 {rgb_transform}，使用默认函数。错误：{e}")
                self.rgb_transform = arcsinh_rgb
        
        # 设置缩放转换函数
        if scale_transform == "default":
            self.scale_transform = scale
        else:
            try:
                self.scale_transform = load_custom_function(scale_transform)
            except Exception as e:
                print(f"警告：无法加载自定义缩放转换函数 {scale_transform}，使用默认函数。错误：{e}")
                self.scale_transform = scale

    @rank_zero_only  # 仅在进程0上执行
    def log_local(  # 本地日志记录方法
        self,
        save_dir,  # 保存目录
        split,  # 数据集划分
        images,  # 图像数据
        global_step,  # 全局步骤
        current_epoch,  # 当前周期
        batch_idx,  # 批次索引
        pl_module: Union[None, pl.LightningModule] = None,  # PyTorch Lightning 模块
    ):
        root = os.path.join(save_dir, "images", split)  # 设置保存路径
        for k in images:  # 遍历图像字典
            if isheatmap(images[k]):  # 如果是热图
                pass
            else:  # 如果不是热图
                # 检查是否为光谱数据 (BCL格式)
                is_spectrum = False
                if isinstance(images[k], torch.Tensor):
                    # 检查数据维度，光谱数据通常是BCL格式(Batch, Channel, Length)
                    # 且Length远大于Channel，这里我们假设Length > 100 * Channel作为判断依据
                    if len(images[k].shape) == 3:
                        B, C, L = images[k].shape
                        if L > 100 * C:  # 这是一个光谱数据的启发式判断
                            is_spectrum = True
                
                # 先保存原始fits文件
                grid = torchvision.utils.make_grid(images[k], nrow=4, padding=0)  # 创建图像网格
                if self.rescale:  # 如果需要重新调整图像
                    grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                grid = grid.numpy()  # 转换为numpy数组
                filename = "{}_gs-{:06}_e-{:06}_b-{:06}.fits".format(
                    k, global_step, current_epoch, batch_idx
                )  # 设置文件名
                path = os.path.join(root, filename)  # 设置文件路径
                os.makedirs(os.path.split(path)[0], exist_ok=True)  # 创建目录
                img = np.array(grid)
                fits.writeto(path, img, overwrite=True)  # 保存图像
                
                if is_spectrum:
                    try:
                        num_samples = min(B, self.max_images)
                        if "raw_input-rec" in k and L % 3 == 0:
                            section_length = L // 3
                            fig, axes = plt.subplots(num_samples, 3, figsize=(18, 5*num_samples))
                            
                            if num_samples == 1:
                                axes = axes.reshape(1, -1)
                                
                            for b_idx in range(num_samples):
                                sample = images[k][b_idx].cpu().numpy()  # [C, L]
                                input_data = sample[:, :section_length]
                                recon_data = sample[:, section_length:2*section_length]
                                resid_data = sample[:, 2*section_length:]
                                
                                x_axis = np.arange(section_length)
                                
                                # 第一列：输入光谱
                                for c_idx in range(C):
                                    axes[b_idx, 0].plot(x_axis, input_data[c_idx], label=f'Ch{c_idx}')
                                axes[b_idx, 0].set_xlabel('Wavelength')
                                axes[b_idx, 0].set_ylabel('Intensity')
                                if b_idx == 0:  # 只在第一行显示图例
                                    axes[b_idx, 0].legend()
                                axes[b_idx, 0].grid(True, alpha=0.3)
                                
                                # 第二列：重建光谱
                                for c_idx in range(C):
                                    axes[b_idx, 1].plot(x_axis, recon_data[c_idx], label=f'Ch{c_idx}')
                                axes[b_idx, 1].set_xlabel('Wavelength')
                                axes[b_idx, 1].grid(True, alpha=0.3)
                                
                                # 第三列：残差
                                for c_idx in range(C):
                                    axes[b_idx, 2].plot(x_axis, resid_data[c_idx], label=f'Ch{c_idx}')
                                axes[b_idx, 2].set_xlabel('Wavelength')
                                axes[b_idx, 2].grid(True, alpha=0.3)
                                axes[b_idx, 2].axhline(y=0, color='r', linestyle='-', alpha=0.3)  # 添加零线
                                
                                # 确保输入和重建图有相同的Y轴范围
                                y_min = min(axes[b_idx, 0].get_ylim()[0], axes[b_idx, 1].get_ylim()[0])
                                y_max = max(axes[b_idx, 0].get_ylim()[1], axes[b_idx, 1].get_ylim()[1])
                                axes[b_idx, 0].set_ylim(y_min, y_max)
                                axes[b_idx, 1].set_ylim(y_min, y_max)
                                
                                # 为残差图设置合适的Y轴范围
                                res_max = max(abs(axes[b_idx, 2].get_ylim()[0]), abs(axes[b_idx, 2].get_ylim()[1]))
                                axes[b_idx, 2].set_ylim(-res_max, res_max)
                            
                            plt.tight_layout()
                            
                            # 保存为单个PNG文件
                            spec_filename = "{}_all_samples_gs-{:06}_e-{:06}_b-{:06}.png".format(
                                k, global_step, current_epoch, batch_idx
                            )
                            spec_path = os.path.join(root, "spectra", spec_filename)
                            os.makedirs(os.path.dirname(spec_path), exist_ok=True)
                            plt.savefig(spec_path, dpi=150, bbox_inches='tight')
                            plt.close()
                            
                            # 如果有wandb，也记录到wandb
                            if exists(pl_module) and isinstance(pl_module.logger, WandbLogger):
                                # 重新读取图像用于wandb记录
                                spec_img = plt.imread(spec_path)
                                pl_module.logger.log_image(
                                    key=f"{split}/{k}_spectrum_all",
                                    images=[spec_img],
                                    step=pl_module.global_step,
                                )
                                
                        elif "chi2" in k:
                            fig, axes = plt.subplots(num_samples, 1, figsize=(12, 4*num_samples))
                            
                            # 处理单样本情况
                            if num_samples == 1:
                                axes = np.array([axes])
                                
                            for b_idx in range(num_samples):
                                sample = images[k][b_idx].cpu().numpy()  # [C, L]
                                mean_chi2 = np.mean(sample)
                                reduced_chi2 = sample[0, 0]  # 前几个点存储的是reduced chi2
                                
                                for c_idx in range(C):
                                    axes[b_idx].plot(np.arange(L), sample[c_idx], label=f'Chi2 Ch{c_idx}')
                                    
                                # 添加均值线
                                axes[b_idx].axhline(y=mean_chi2, color='r', linestyle='--', 
                                              label=f'Mean Chi2: {mean_chi2:.4f}')
                                
                                axes[b_idx].set_xlabel("Wavelength")
                                axes[b_idx].set_ylabel("Chi-square Value")
                                axes[b_idx].set_yscale('log')  # 使用对数刻度更容易看到变化
                                if b_idx == 0:  # 只在第一行显示图例
                                    axes[b_idx].legend()
                                axes[b_idx].grid(True, alpha=0.3)
                            
                            plt.tight_layout()
                            
                            # 保存为单个PNG文件
                            spec_filename = "{}_all_samples_gs-{:06}_e-{:06}_b-{:06}.png".format(
                                k, global_step, current_epoch, batch_idx
                            )
                            spec_path = os.path.join(root, "spectra", spec_filename)
                            os.makedirs(os.path.dirname(spec_path), exist_ok=True)
                            plt.savefig(spec_path, dpi=150, bbox_inches='tight')
                            plt.close()
                            
                            # 如果有wandb，也记录到wandb
                            if exists(pl_module) and isinstance(pl_module.logger, WandbLogger):
                                spec_img = plt.imread(spec_path)
                                pl_module.logger.log_image(
                                    key=f"{split}/{k}_spectrum_all",
                                    images=[spec_img],
                                    step=pl_module.global_step,
                                )
                                
                        elif "samples" in k or True:  # 对于样本数据和其他所有光谱数据
                            # 创建一个大图，每个样本一行
                            fig, axes = plt.subplots(num_samples, 1, figsize=(12, 4*num_samples))
                            
                            # 处理单样本情况
                            if num_samples == 1:
                                axes = np.array([axes])
                                
                            for b_idx in range(num_samples):
                                sample = images[k][b_idx].cpu().numpy()  # [C, L]
                                
                                for c_idx in range(C):
                                    axes[b_idx].plot(np.arange(L), sample[c_idx], label=f'Channel {c_idx}')
                                axes[b_idx].set_xlabel("Wavelength")
                                axes[b_idx].set_ylabel("Intensity")
                                if b_idx == 0:  # 只在第一行显示图例
                                    axes[b_idx].legend()
                                axes[b_idx].grid(True, alpha=0.3)
                            
                            plt.tight_layout()
                            
                            # 保存为单个PNG文件
                            spec_filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                                k, global_step, current_epoch, batch_idx
                            )
                            spec_path = os.path.join(root, "spectra", spec_filename)
                            os.makedirs(os.path.dirname(spec_path), exist_ok=True)
                            plt.savefig(spec_path, dpi=150, bbox_inches='tight')
                            plt.close()
                            
                            # 如果有wandb，也记录到wandb
                            if exists(pl_module) and isinstance(pl_module.logger, WandbLogger):
                                spec_img = plt.imread(spec_path)
                                pl_module.logger.log_image(
                                    key=f"{split}/{k}_spectrum_all",
                                    images=[spec_img],
                                    step=pl_module.global_step,
                                )
                    except Exception as e:
                        print(f"光谱可视化生成失败: {e}")
                        import traceback
                        traceback.print_exc()
                
                # 原始wandb记录逻辑
                if exists(pl_module):
                    assert isinstance(
                        pl_module.logger, WandbLogger
                    ), "logger_log_image 目前仅支持 WandbLogger"
                    
                    # 使用自定义转换函数处理图像
                    if img.shape[0] == 3:
                        # 对于3通道图像使用RGB转换
                        try:
                            transformed_img = self.rgb_transform(img, mode="CHW", **self.transform_kwargs).transpose(1,2,0)
                        except Exception as e:
                            print(f"RGB转换错误，使用默认方法：{e}")
                            transformed_img = arcsinh_rgb(img, mode="CHW").transpose(1,2,0)
                    else:
                        # 对于非3通道图像使用缩放转换
                        try:
                            middle_slice = img[img.shape[0]//2-1:img.shape[0]//2+2,:,:]
                            transformed_img = self.scale_transform(middle_slice, **self.transform_kwargs).transpose(1,2,0)
                        except Exception as e:
                            print(f"缩放转换错误，使用默认方法：{e}")
                            transformed_img = scale(img[img.shape[0]//2-1:img.shape[0]//2+2,:,:]).transpose(1,2,0)
                    if not is_spectrum:
                        pl_module.logger.log_image(
                            key=f"{split}/{k}",
                            images=[transformed_img],
                            step=pl_module.global_step,
                        )

    @rank_zero_only  # 仅在进程0上执行
    def log_img(self, pl_module, batch, batch_idx, split="train"):  # 记录图像方法
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step  # 检查索引
        if (
            self.check_frequency(check_idx)  # 检查频率
            and hasattr(pl_module, "log_images")  # 批次索引 % 批次频率 == 0
            and callable(pl_module.log_images)
            and
            # batch_idx > 5
            self.max_images > 0
        ):
            logger = type(pl_module.logger)  # 获取logger类型
            is_train = pl_module.training  # 是否为训练模式
            if is_train:
                pl_module.eval()  # 设置为评估模式

            gpu_autocast_kwargs = {  # 设置自动混合精度参数
                "enabled": self.enable_autocast,
                "dtype": torch.get_autocast_gpu_dtype(),
                "cache_enabled": torch.is_autocast_cache_enabled(),
            }
            with torch.no_grad(), torch.amp.autocast('cuda', **gpu_autocast_kwargs):  # 禁用梯度计算，启用自动混合精度
                images = pl_module.log_images(
                    batch, split=split, **self.log_images_kwargs
                )  # 获取图像

            for k in images:  # 遍历图像字典
                N = min(images[k].shape[0], self.max_images)  # 获取最小图像数
                if not isheatmap(images[k]):  # 如果不是热图
                    images[k] = images[k][:N]  # 截取前N个图像
                if isinstance(images[k], torch.Tensor):  # 如果图像是Tensor类型
                    images[k] = images[k].detach().float().cpu()  # 转换为CPU上的float类型
                    # if self.clamp and not isheatmap(images[k]):  # 如果需要限制图像值
                    #     images[k] = torch.clamp(images[k], -1.0, 1.0)  # 限制图像值在-1.0到1.0之间

            self.log_local(  # 记录本地图像
                pl_module.logger.save_dir,
                split,
                images,
                pl_module.global_step,
                pl_module.current_epoch,
                batch_idx,
                pl_module=pl_module
                if isinstance(pl_module.logger, WandbLogger)
                else None,
            )

            if is_train:  # 如果为训练模式
                pl_module.train()  # 设置为训练模式

    def check_frequency(self, check_idx):  # 检查频率方法
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
            check_idx > 0 or self.log_first_step
        ):
            try:
                self.log_steps.pop(0)  # 弹出日志记录步骤
            except IndexError as e:  # 捕捉索引错误
                print(e)
                pass
            return True
        return False

    @rank_zero_only  # 仅在进程0上执行
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):  # 训练批次结束时调用
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):  # 如果未禁用日志记录且全局步骤大于0或记录第一步
            self.log_img(pl_module, batch, batch_idx, split="train")  # 记录图像

    @rank_zero_only  # 仅在进程0上执行
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):  # 训练批次开始时调用
        if self.log_before_first_step and pl_module.global_step == 0:  # 如果在第一步之前记录且全局步骤为0
            print(f"{self.__class__.__name__}: logging before training")  # 打印日志记录消息
            self.log_img(pl_module, batch, batch_idx, split="train")  # 记录图像

    @rank_zero_only
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs
    ):
        if not self.disabled and pl_module.global_step > 0:  # 如果未禁用日志记录且全局步骤大于0
            self.log_img(pl_module, batch, batch_idx, split="val")  # 记录图像
        if hasattr(pl_module, "calibrate_grad_norm"):  # 如果pl_module有calibrate_grad_norm属性
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:  # 如果需要校准梯度范数且批次索引为25的倍数且大于0
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)  # 记录梯度
        
        # 存储最后一个验证批次，无论check_frequency如何
        self.last_val_batch = batch
        self.last_val_batch_idx = batch_idx

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        """在验证周期结束时记录最后一个批次的图像，无论是否满足频率要求"""
        if self.disabled or pl_module.global_step <= 0 or self.last_val_batch is None:
            return
        
        # 直接记录最后一个验证批次的图像，绕过check_frequency检查
        if hasattr(pl_module, "log_images") and callable(pl_module.log_images) and self.max_images > 0:
            logger = type(pl_module.logger)
            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            gpu_autocast_kwargs = {
                "enabled": self.enable_autocast,
                "dtype": torch.get_autocast_gpu_dtype(),
                "cache_enabled": torch.is_autocast_cache_enabled(),
            }
            with torch.no_grad(), torch.amp.autocast('cuda', **gpu_autocast_kwargs):
                images = pl_module.log_images(
                    self.last_val_batch, split="val_last", **self.log_images_kwargs
                )

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                if not isheatmap(images[k]):
                    images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().float().cpu()

            self.log_local(
                pl_module.logger.save_dir,
                "val_last",
                images,
                pl_module.global_step,
                pl_module.current_epoch,
                self.last_val_batch_idx,
                pl_module=pl_module
                if isinstance(pl_module.logger, WandbLogger)
                else None,
            )

            if is_train:
                pl_module.train()
            
            # 清除存储的批次，避免内存泄漏
            self.last_val_batch = None
            self.last_val_batch_idx = None



def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-n", "--name", type=str, const=True, default="", nargs="?", help="postfix for logdir")
    parser.add_argument("--no_date", type=str2bool, nargs="?", const=True, default=False, help="if True, skip date generation for logdir and only use naming via opt.base or opt.name (+ opt.postfix, optionally)")
    parser.add_argument("-r", "--resume", type=str, const=True, default="", nargs="?", help="resume from logdir or checkpoint in logdir")
    parser.add_argument("--tags", type=str, default="", help="wandb tags, separated by commas, for example 'experiment1,test'")
    parser.add_argument("-b", "--base", nargs="*", metavar="base_config.yaml", help="paths to base configs. Loaded from left-to-right. Parameters can be overwritten or added with command-line options of the form `--key value`.",
                        # default=["/data/public/renhaoye/ai4galmorph/configs/generation/para2img.yaml"])
                        default=["/data1/public/renhaoye/code/astre_train/configs/generation/ICML2026_DESI.yaml"])
    parser.add_argument("-t", "--train", type=str2bool, const=True, default=True, nargs="?", help="train")
    parser.add_argument("--no-test", type=str2bool, const=True, default=False, nargs="?", help="disable test")
    parser.add_argument("-p", "--project", help="name of new or path to existing project")
    parser.add_argument("-d", "--debug", type=str2bool, nargs="?", const=True, default=False, help="enable post-mortem debugging")
    parser.add_argument("-s", "--seed", type=int, default=1024, help="seed for seed_everything")
    parser.add_argument("-f", "--postfix", type=str, default="", help="post-postfix for default name")
    parser.add_argument("-w", "--projectname", type=str, default="ICML2026")
    parser.add_argument("-l", "--logdir", type=str, default="logs", help="directory for logging dat shit")
    parser.add_argument("--scale_lr", type=str2bool, nargs="?", const=True, default=False, help="scale base-lr by ngpu * batch_size * n_accumulate")
    parser.add_argument("--legacy_naming", type=str2bool, nargs="?", const=True, default=False, help="name run based on config file name if true, else by whole path")
    parser.add_argument("--enable_tf32", type=str2bool, nargs="?", const=True, default=True, help="enables the TensorFloat32 format both for matmuls and cuDNN for pytorch 1.12")
    parser.add_argument("--startup", type=str, default=None, help="Startuptime from distributed script")
    parser.add_argument("--wandb", type=str2bool, nargs="?", const=True, default=True, help="log to wandb")
    parser.add_argument("--no_base_name", type=str2bool, nargs="?", const=True, default=False, help="log to wandb")
    if version.parse(torch.__version__) >= version.parse("2.0.0"):
        parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="single checkpoint file to resume from")

    default_args = default_trainer_args()
    for key in default_args:
        parser.add_argument("--" + key, default=default_args[key])
    return parser

@rank_zero_only
def init_wandb(save_dir, opt, config, group_name, name_str):
    print(f"setting WANDB_DIR to {save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    os.environ["WANDB_DIR"] = save_dir
    if opt.debug:
        wandb.init(project=opt.projectname, mode="offline", group=group_name)
    else:
        wandb.init(
            project=opt.projectname,
            config=config,
            settings=wandb.Settings(code_dir="./sgm"),
            group=group_name,
            name=name_str,
        )

class SetupCallback(Callback):
    def __init__(
        self,
        resume,
        now,
        logdir,
        ckptdir,
        cfgdir,
        config,
        lightning_config,
        debug,
        ckpt_name=None,
    ):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config
        self.debug = debug
        self.ckpt_name = ckpt_name

    def on_exception(self, trainer: pl.Trainer, pl_module, exception): # 出现异常时保存checkpoint
        if not self.debug and trainer.global_rank == 0:
            print("Summoning checkpoint.")
            if self.ckpt_name is None:
                ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            else:
                ckpt_path = os.path.join(self.ckptdir, self.ckpt_name)
            trainer.save_checkpoint(ckpt_path)

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if (
                    "metrics_over_trainsteps_checkpoint"
                    in self.lightning_config["callbacks"]
                ):
                    os.makedirs(
                        os.path.join(self.ckptdir, "trainstep_checkpoints"), # 存放checkpoint的文件夹
                        exist_ok=True,
                    )
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            if MULTINODE_HACKS:
                import time

                time.sleep(5)
            OmegaConf.save(
                self.config,
                os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)), # 保存配置文件
            )

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(
                OmegaConf.create({"lightning": self.lightning_config}),
                os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)), # 保存配置文件
            )

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not MULTINODE_HACKS and not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass

def get_checkpoint_name(logdir):
    ckpt = os.path.join(logdir, "checkpoints", "last**.ckpt")
    ckpt = natsorted(glob.glob(ckpt))
    print('available "last" checkpoints:')
    print(ckpt)
    if len(ckpt) > 1:
        print("got most recent checkpoint")
        ckpt = sorted(ckpt, key=lambda x: os.path.getmtime(x))[-1]
        print(f"Most recent ckpt is {ckpt}")
        with open(os.path.join(logdir, "most_recent_ckpt.txt"), "w") as f:
            f.write(ckpt + "\n")
        try:
            version = int(ckpt.split("/")[-1].split("-v")[-1].split(".")[0])
        except Exception as e:
            print("version confusion but not bad")
            print(e)
            version = 1
        # version = last_version + 1
    else:
        # in this case, we only have one "last.ckpt"
        ckpt = ckpt[0]
        version = 1
    melk_ckpt_name = f"last-v{version}.ckpt"
    print(f"Current melk ckpt name: {melk_ckpt_name}")
    return ckpt, melk_ckpt_name

# 将 CheckpointLinker 类定义移动到文件顶部（在 SetupCallback 类之后）
class CheckpointLinker(Callback):
    def __init__(self, link_interval=1000):  # 添加初始化参数
        super().__init__()
        self.link_interval = link_interval

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.link_interval == 0:  # 使用配置参数
            src = trainer.checkpoint_callback.best_model_path
            if os.path.exists(src):
                dst = os.path.join(os.path.dirname(src), "last.ckpt")
                if os.path.lexists(dst):
                    os.remove(dst)
                os.link(src, dst)

    def on_train_epoch_end(self, trainer, pl_module):
        # 在epoch结束时强制更新链接
        src = trainer.checkpoint_callback.best_model_path
        if os.path.exists(src):
            dst = os.path.join(os.path.dirname(src), "last.ckpt")
            os.replace(src, dst)  # 原子操作替换

if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())
    import time
    # time.sleep(4.5 * 60 * 60)  # 休眠5小时
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    if opt.name and opt.resume: # 文件名和恢复训练不能同时指定
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    melk_ckpt_name = None
    name = None
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume)) # 无法找到继续训练的文件
        if os.path.isfile(opt.resume): # 如果是文件
            paths = opt.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
            _, melk_ckpt_name = get_checkpoint_name(logdir) # 获取最新的checkpoint
        else:
            assert os.path.isdir(opt.resume), opt.resume # 如果是目录
            logdir = opt.resume.rstrip("/")
            ckpt, melk_ckpt_name = get_checkpoint_name(logdir)

        print("#" * 100)
        print(f'Resuming from checkpoint "{ckpt}"')
        print("#" * 100)

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name: # 要么就是自己指定名字
            name = "_" + opt.name
        elif opt.base: # 要么就是用yaml文件名（base）作为名字
            if opt.no_base_name:
                name = ""
            else:
                if opt.legacy_naming:
                    cfg_fname = os.path.split(opt.base[0])[-1] # 从绝对路径提取文件名
                    cfg_name = os.path.splitext(cfg_fname)[0] # 去掉后缀
                else:
                    assert "configs" in os.path.split(opt.base[0])[0], os.path.split(opt.base[0])[0]
                    cfg_path = os.path.split(opt.base[0])[0].split(os.sep)[
                        os.path.split(opt.base[0])[0].split(os.sep).index("configs")
                        + 1 :
                    ]  # cut away the first one (we assert all configs are in "configs") configs子文件夹的路径作为日期后的名字
                    cfg_name = os.path.splitext(os.path.split(opt.base[0])[-1])[0] # 从绝对路径提取文件名并去掉后缀
                    cfg_name = "-".join(cfg_path) + f"-{cfg_name}"
                name = "_" + cfg_name # 作为日期_{cfg_name}
        else:
            name = ""
        if not opt.no_date: # 如果不是不加日期
            nowname = now + name + opt.postfix # 日期+名字+后缀
        else:
            nowname = name + opt.postfix
            if nowname.startswith("_"):
                nowname = nowname[1:]
        logdir = os.path.join(opt.logdir, nowname) # 这个是相对路径
        print(f"LOGDIR: {logdir}")

    ckptdir = os.path.join(logdir, "checkpoints") # checkpoint: 日期+名字+后缀/checkpoints
    cfgdir = os.path.join(logdir, "configs") # 配置文件: 日期+名字+后缀/configs
    seed_everything(opt.seed, workers=True)
    # 设置精度
    if opt.enable_tf32:
        # pt_version = version.parse(torch.__version__)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"Enabling TF32 for PyTorch {torch.__version__}")
    else:
        print(f"Using default TF32 settings for PyTorch {torch.__version__}:")
        print(
            f"torch.backends.cuda.matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32}"
        )
        print(f"torch.backends.cudnn.allow_tf32={torch.backends.cudnn.allow_tf32}")
    try:
        configs = [OmegaConf.load(cfg) for cfg in opt.base] # 加载配置文件
        cli = OmegaConf.from_dotlist(unknown) # 从命令行参数中加载配置
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create()) # 确保 lightning_config 总是存在，哪怕原始配置中没有 lightning 配置项，则新建一个空的

        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        trainer_config["accelerator"] = "gpu"
        
        standard_args = default_trainer_args() # 获取 Trainer 类构造函数的参数签名
        for k in standard_args: # k是参数名
            if getattr(opt, k) != standard_args[k]: # 如果opt中的参数值不等于standard_args[k]的默认值
                trainer_config[k] = getattr(opt, k) # 将opt中的参数值赋给trainer_config[k]
        
        ckpt_resume_path = opt.resume_from_checkpoint
        if not "devices" in trainer_config and trainer_config["accelerator"] != "gpu":
            del trainer_config["accelerator"]
            cpu = True
        else:
            gpuinfo = trainer_config["devices"]
            print(f"Running on GPUs {gpuinfo}")
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # 模型初始化
        model = instantiate_from_config(config.model)
                # trainer and callbacks
        trainer_kwargs = dict()

        # default logger configs
        default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "name": nowname,
                    "save_dir": logdir,
                    "offline": opt.debug,
                    "id": nowname,
                    "project": opt.projectname,
                    "log_model": False,
                    "tags": opt.tags.split(",") if opt.tags else None,  # 添加tags支持
                    # "dir": logdir,
                },
            },
            "csv": {
                "target": "pytorch_lightning.loggers.CSVLogger",
                "params": {
                    "name": "testtube",  # hack for sbord fanatics
                    "save_dir": logdir,
                },
            },
        }
        default_logger_cfg = default_logger_cfgs["wandb" if opt.wandb else "csv"] # 选择记录方式是wandb还是csv
        if "logger" in lightning_config:
            logger_cfg = lightning_config.logger
        else:
            logger_cfg = OmegaConf.create()
            
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)
        
        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:03}-{step:09}",
                "verbose": True,
                "save_last": True,
            },
        }
        if hasattr(model, "monitor"):
            print(f"Monitoring {model.monitor} as checkpoint metric.")
            default_modelckpt_cfg["params"]["monitor"] = model.monitor
            default_modelckpt_cfg["params"]["save_top_k"] = -1

        if "modelcheckpoint" in lightning_config:
            modelckpt_cfg = lightning_config.modelcheckpoint
        else:
            modelckpt_cfg = OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")

        # https://pytorch-lightning.readthedocs.io/en/stable/extensions/strategy.html
        # default to ddp if not further specified
        default_strategy_config = {
                                    "target": "pytorch_lightning.strategies.DDPStrategy",
                                    "params": {
                                        "find_unused_parameters": True,
                                        # 其他参数...
                                    }
                                }

        if "strategy" in lightning_config:
            strategy_cfg = lightning_config.strategy
        else:
            strategy_cfg = OmegaConf.create()
            default_strategy_config["params"] = {
                "find_unused_parameters": True,
                # "static_graph": True,
                # "ddp_comm_hook": default.fp16_compress_hook  # TODO: experiment with this, also for DDPSharded
            }
        strategy_cfg = OmegaConf.merge(default_strategy_config, strategy_cfg)
        print(f"strategy config: \n ++++++++++++++ \n {strategy_cfg} \n ++++++++++++++ ")
        trainer_kwargs["strategy"] = instantiate_from_config(strategy_cfg)

        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "main.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                    "debug": opt.debug,
                    "ckpt_name": melk_ckpt_name,
                },
            },
            "image_logger": {
                "target": "main.ImageLogger",
                "params": {"batch_frequency": 1000, "max_images": 4, "clamp": True},
            },
            "learning_rate_logger": {
                "target": "pytorch_lightning.callbacks.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                    # "log_momentum": True
                },
            },
            "checkpoint_linker": {
                "target": "__main__.CheckpointLinker",
                "params": {
                    "link_interval": lightning_config.callbacks.checkpoint_linker.params.link_interval
                }
            },
            "unified_metric_checkpoint": {
                "target": "sgm.callbacks.UnifiedMetricCheckpoint",
                "params": {},
            },
            "own_loss_best_checkpoint": {
                "target": "sgm.callbacks.OwnLossBestCheckpoint",
                "params": {},
            },
        }
        if version.parse(pl.__version__) >= version.parse("1.4.0"):
            default_callbacks_cfg.update({"checkpoint_callback": modelckpt_cfg})

        if "callbacks" in lightning_config:
            callbacks_cfg = lightning_config.callbacks
        else:
            callbacks_cfg = OmegaConf.create()

        if "metrics_over_trainsteps_checkpoint" in callbacks_cfg:
            print(
                "Caution: Saving checkpoints every n train steps without deleting. This might require some free space."
            )
            default_metrics_over_trainsteps_ckpt_dict = {
                "metrics_over_trainsteps_checkpoint": {
                    "target": "pytorch_lightning.callbacks.ModelCheckpoint",
                    "params": {
                        "dirpath": os.path.join(ckptdir, "trainstep_checkpoints"),
                        "filename": "{epoch:06}-{step:09}",
                        "verbose": True,
                        "save_top_k": -1,
                        "every_n_train_steps": 10000,
                        "save_weights_only": True,
                    },
                }
            }
            default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        if "ignore_keys_callback" in callbacks_cfg and ckpt_resume_path is not None:
            callbacks_cfg.ignore_keys_callback.params["ckpt_path"] = ckpt_resume_path
        elif "ignore_keys_callback" in callbacks_cfg:
            del callbacks_cfg["ignore_keys_callback"]

        trainer_kwargs["callbacks"] = [
            instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg
        ]
        if not "plugins" in trainer_kwargs:
            trainer_kwargs["plugins"] = list()

        # cmd line trainer args (which are in trainer_opt) have always priority over config-trainer-args (which are in trainer_kwargs)
        trainer_opt = vars(trainer_opt)
        trainer_kwargs = {
            key: val for key, val in trainer_kwargs.items() if key not in trainer_opt
        }
            
            
        # 设置训练器
        trainer = Trainer(**trainer_opt, **trainer_kwargs)
        # trainer = Trainer(detect_anomaly=True,**trainer_opt, **trainer_kwargs)

        trainer.logdir = logdir  ###

        # 数据加载和准备
        data = instantiate_from_config(config.data)
        data.prepare_data()

        print("#### Data #####")
        try:
            for k in data.datasets:
                print(
                    f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}"
                )
        except:
            print("datasets not yet initialized.")

        # configure learning rate
        if "batch_size" in config.data.params:
            bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        else:
            bs, base_lr = (
                config.data.params.train.loader.batch_size,
                config.model.base_learning_rate,
            )
        if not cpu:
            ngpu = len(lightning_config.trainer.devices.strip(",").split(","))
        else:
            ngpu = 1
        if "accumulate_grad_batches" in lightning_config.trainer:
            accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
        else:
            accumulate_grad_batches = 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        if opt.scale_lr:
            model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
            print(
                "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                    model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr
                )
            )
        else:
            model.learning_rate = base_lr
            print("++++ NOT USING LR SCALING ++++")
            print(f"Setting learning rate to {model.learning_rate:.2e}")

        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                if melk_ckpt_name is None:
                    ckpt_path = os.path.join(ckptdir, "last.ckpt")
                else:
                    ckpt_path = os.path.join(ckptdir, melk_ckpt_name)
                trainer.save_checkpoint(ckpt_path)

        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb

                pudb.set_trace()

        import signal

        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        # run
        if opt.train:
            try:
                trainer.fit(model, data, ckpt_path=ckpt_resume_path) # 训练主入口
            except Exception:
                if not opt.debug:
                    melk()
                raise
        # if not opt.no_test and not trainer.interrupted:
        #     trainer.test(model, data)
    except RuntimeError as err:
        # if MULTINODE_HACKS:
        #     import datetime
        #     import os
        #     import socket

        #     import requests

        #     device = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
        #     hostname = socket.gethostname()
        #     ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        #     resp = requests.get("http://169.254.169.254/latest/meta-data/instance-id")
        #     print(
        #         f"ERROR at {ts} on {hostname}/{resp.text} (CUDA_VISIBLE_DEVICES={device}): {type(err).__name__}: {err}",
        #         flush=True,
        #     )
        raise err
    except Exception:
        if opt.debug and trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank == 0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)

        if opt.wandb:
            wandb.finish()
        # if trainer.global_rank == 0:
        #    print(trainer.profiler.summary())
