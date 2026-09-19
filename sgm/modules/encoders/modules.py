import math
from contextlib import nullcontext
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

# import kornia
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from omegaconf import ListConfig
from torch.utils.checkpoint import checkpoint

from ...modules.autoencoding.regularizers import DiagonalGaussianRegularizer
from ...modules.diffusionmodules.model import Encoder
from ...modules.diffusionmodules.openaimodel import Timestep
from ...modules.diffusionmodules.util import (extract_into_tensor,
                                              make_beta_schedule)
from ...modules.distributions.distributions import DiagonalGaussianDistribution
from ...util import (append_dims, autocast, count_params, default,
                     disabled_train, expand_dims_like, instantiate_from_config)


class AbstractEmbModel(nn.Module):  # 定义一个抽象嵌入模型类，继承自 nn.Module
    def __init__(self):
        super().__init__()
        self._is_trainable = None  # 初始化可训练标志
        self._ucg_rate = None  # 初始化 UCG 速率
        self._input_key = None  # 初始化输入键

    @property
    def is_trainable(self) -> bool:  # 定义可训练属性的 getter 方法
        return self._is_trainable

    @property
    def ucg_rate(self) -> Union[float, torch.Tensor]:  # 定义 UCG 速率属性的 getter 方法
        return self._ucg_rate

    @property
    def input_key(self) -> str:  # 定义输入键属性的 getter 方法
        return self._input_key

    @is_trainable.setter
    def is_trainable(self, value: bool):  # 定义可训练属性的 setter 方法
        self._is_trainable = value

    @ucg_rate.setter
    def ucg_rate(self, value: Union[float, torch.Tensor]):  # 定义 UCG 速率属性的 setter 方法
        self._ucg_rate = value

    @input_key.setter
    def input_key(self, value: str):  # 定义输入键属性的 setter 方法
        self._input_key = value

    @is_trainable.deleter
    def is_trainable(self):  # 定义可训练属性的 deleter 方法
        del self._is_trainable

    @ucg_rate.deleter
    def ucg_rate(self):  # 定义 UCG 速率属性的 deleter 方法
        del self._ucg_rate

    @input_key.deleter
    def input_key(self):  # 定义输入键属性的 deleter 方法
        del self._input_key


class GeneralConditioner(nn.Module):  # 定义一个通用条件器类，继承自 nn.Module
    # shape=[batch, embedding]=vector, [] 
    OUTPUT_DIM2KEYS = {2: "vector", 3: "crossattn", 4: "concat", 5: "concat"} # 根据embedding的结果维度，将其映射到对应的输出类型
    KEY2CATDIM = {"vector": 1, "crossattn": 2, "concat": 1} # 根据输出类型，确定拼接的维度

    def __init__(self, emb_models: Union[List, ListConfig]):  # 接收嵌入模型配置信息的列表
        super().__init__()
        embedders = []
        for n, embconfig in enumerate(emb_models):  # 遍历嵌入模型配置
            embedder = instantiate_from_config(embconfig)  # 从配置实例化嵌入模型
            assert isinstance(
                embedder, AbstractEmbModel
            ), f"embedder model {embedder.__class__.__name__} has to inherit from AbstractEmbModel"  # 确保嵌入模型继承自 AbstractEmbModel
            embedder.is_trainable = embconfig.get("is_trainable", False)  # 设置可训练属性，默认为 False
            embedder.ucg_rate = embconfig.get("ucg_rate", 0.0)  # 设置 UCG 速率，默认为 0.0
            if not embedder.is_trainable:  # 如果不可训练
                embedder.train = disabled_train  # 禁用训练方法
                for param in embedder.parameters():  # 禁用所有参数的梯度计算
                    param.requires_grad = False
                embedder.eval()  # 设置模型为评估模式
            print(
                f"Initialized embedder #{n}: {embedder.__class__.__name__} "
                f"with {count_params(embedder, False)} params. Trainable: {embedder.is_trainable}"
            )

            if "input_key" in embconfig:  # 检查配置中是否有输入键
                embedder.input_key = embconfig["input_key"]
            elif "input_keys" in embconfig:  # 检查配置中是否有输入键列表
                embedder.input_keys = embconfig["input_keys"]
            else:
                raise KeyError(
                    f"need either 'input_key' or 'input_keys' for embedder {embedder.__class__.__name__}"
                )

            embedder.legacy_ucg_val = embconfig.get("legacy_ucg_value", None)  # 获取遗留的 UCG 值
            if embedder.legacy_ucg_val is not None:
                embedder.ucg_prng = np.random.RandomState()  # 初始化随机数生成器

            embedders.append(embedder)  # 将嵌入模型添加到列表中
        self.embedders = nn.ModuleList(embedders)  # 将嵌入模型列表转换为 ModuleList
        
    def possibly_get_ucg_val(self, embedder: AbstractEmbModel, batch: Dict) -> Dict:  # 获取可能的 UCG 值
        assert embedder.legacy_ucg_val is not None
        p = embedder.ucg_rate
        val = embedder.legacy_ucg_val
        for i in range(len(batch[embedder.input_key])):  # 遍历批处理中的每个输入键
            if embedder.ucg_prng.choice(2, p=[1 - p, p]):  # 根据 UCG 速率随机选择
                batch[embedder.input_key][i] = val  # 设置输入键为遗留的 UCG 值
        return batch

    def forward(
        self, batch: Dict, force_zero_embeddings: Optional[List] = None
    ) -> Dict:  # 前向传播方法
        output = dict()
        if force_zero_embeddings is None:
            force_zero_embeddings = []
        for embedder in self.embedders:  # 遍历每个嵌入模型
            embedding_context = nullcontext if embedder.is_trainable else torch.no_grad  # 根据是否可训练设置上下文
            with embedding_context():
                if hasattr(embedder, "input_key") and (embedder.input_key is not None):  # 如果有输入键属性
                    if embedder.legacy_ucg_val is not None:
                        batch = self.possibly_get_ucg_val(embedder, batch).astype  # 获取可能的 UCG 值
                        
                    emb_out = embedder(batch[embedder.input_key])  # 获取嵌入输出
                elif hasattr(embedder, "input_keys"):  # 如果有输入键列表属性
                    emb_out = embedder(*[batch[k] for k in embedder.input_keys])  # 获取嵌入输出
            assert isinstance(
                emb_out, (torch.Tensor, list, tuple)
            ), f"encoder outputs must be tensors or a sequence, but got {type(emb_out)}"  # 确保嵌入输出是张量或序列
            if not isinstance(emb_out, (list, tuple)):  # 如果嵌入输出不是列表或元组
                emb_out = [emb_out]
            for emb in emb_out:
                out_key = self.OUTPUT_DIM2KEYS[emb.dim()]  # 获取输出键
                if embedder.ucg_rate > 0.0 and embedder.legacy_ucg_val is None:
                    emb = (
                        expand_dims_like(
                            torch.bernoulli(
                                (1.0 - embedder.ucg_rate)
                                * torch.ones(emb.shape[0], device=emb.device)
                            ),
                            emb,
                        )
                        * emb
                    )  # 根据 UCG 速率对嵌入输出进行调整
                if (
                    hasattr(embedder, "input_key")
                    and embedder.input_key in force_zero_embeddings
                ):
                    emb = torch.zeros_like(emb)  # 如果在强制零嵌入列表中，将嵌入输出置零
                if out_key in output: # 检查是否已经存在相同类型的嵌入输出例如，如果 out_key 是 "concat"，那么这一步是检查 output 字典中是否已经有 "concat" 类型的嵌入输出。
                    output[out_key] = torch.cat(
                        (output[out_key], emb), self.KEY2CATDIM[out_key]
                    )  # 将嵌入输出拼接到已有输出中
                else:
                    output[out_key] = emb  # 将嵌入输出添加到输出字典中
        return output

    def get_unconditional_conditioning(
        self,
        batch_c: Dict,
        batch_uc: Optional[Dict] = None,
        force_uc_zero_embeddings: Optional[List[str]] = None,
        force_cond_zero_embeddings: Optional[List[str]] = None,
    ):  # 获取无条件条件方法
        if force_uc_zero_embeddings is None:
            force_uc_zero_embeddings = []
        ucg_rates = list()
        for embedder in self.embedders:
            ucg_rates.append(embedder.ucg_rate)
            embedder.ucg_rate = 0.0  # 暂时将 UCG 速率设置为 0
        c = self(batch_c, force_cond_zero_embeddings)  # 获取条件嵌入输出
        uc = self(batch_c if batch_uc is None else batch_uc, force_uc_zero_embeddings)  # 获取无条件嵌入输出

        for embedder, rate in zip(self.embedders, ucg_rates):  # 恢复 UCG 速率
            embedder.ucg_rate = rate
        return c, uc  # 返回条件和无条件嵌入输出
    
class NumericLabelEmbedder(AbstractEmbModel):
    def __init__(self, num_labels: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_labels, embed_dim)
    
    def forward(self, labels):
        return self.embedding(labels)

    def encode(self, labels):
        return self(labels)


# class InceptionV3(nn.Module):
#     """Wrapper around the https://github.com/mseitzer/pytorch-fid inception
#     port with an additional squeeze at the end"""

#     def __init__(self, normalize_input=False, **kwargs):
#         super().__init__()
#         from pytorch_fid import inception

#         kwargs["resize_input"] = True
#         self.model = inception.InceptionV3(normalize_input=normalize_input, **kwargs)

#     def forward(self, inp):
#         outp = self.model(inp)

#         if len(outp) == 1:
#             return outp[0].squeeze()

#         return outp


class IdentityEncoder(AbstractEmbModel):
    def encode(self, x):
        return x

    def forward(self, x):
        return x


class ClassEmbedder(AbstractEmbModel):  # 类别嵌入器类，继承自AbstractEmbModel
    def __init__(self, embed_dim, n_classes=1000, add_sequence_dim=False):  # 初始化方法，接收嵌入维度、类别数量和是否添加序列维度
        super().__init__()  # 调用父类的初始化方法
        self.embedding = nn.Embedding(n_classes, embed_dim)  # 创建嵌入层
        self.n_classes = n_classes  # 设置类别数量
        self.add_sequence_dim = add_sequence_dim  # 设置是否添加序列维度

    def forward(self, c):  # 前向传播方法
        c = self.embedding(c)  # 获取嵌入层的输出
        if self.add_sequence_dim:  # 如果需要添加序列维度
            c = c[:, None, :]  # 在第二个维度上添加一个维度
        return c  # 返回嵌入层的输出

    def get_unconditional_conditioning(self, bs, device="cuda"):  # 获取无条件条件的方法
        uc_class = (
            self.n_classes - 1
        )  # 1000 classes --> 0 ... 999, one extra class for ucg (class 1000)  # 设置无条件类别为最后一个类别
        uc = torch.ones((bs,), device=device) * uc_class  # 创建无条件类别的张量
        uc = {self.key: uc.long()}  # 将张量转换为字典格式
        return uc  # 返回无条件类别的字典


class ClassEmbedderForMultiCond(ClassEmbedder):  # 多条件类别嵌入器类，继承自ClassEmbedder
    def forward(self, batch, key=None, disable_dropout=False):  # 前向传播方法
        out = batch  # 初始化输出为输入的批量数据
        key = default(key, self.key)  # 获取默认键值
        islist = isinstance(batch[key], list)  # 判断批量数据的键值是否为列表
        if islist:  # 如果是列表
            batch[key] = batch[key][0]  # 获取列表的第一个元素
        c_out = super().forward(batch[key])  # 调用父类的前向传播方法获取输出
        out[key] = [c_out] if islist else c_out  # 根据输入类型设置输出
        return out  # 返回输出


class ContinuousValueEmbedder(AbstractEmbModel):  # 连续值嵌入器类，继承自 AbstractEmbModel
    def __init__(self, outdim, n_classes=1, add_sequence_dim=False):  # 初始化方法，接收嵌入维度、类别数量和是否添加序列维度
        super().__init__()  # 调用父类的初始化方法
        self.embedding = nn.Linear(n_classes, outdim)  # 创建嵌入层
        self.n_classes = n_classes  # 设置类别数量
        self.add_sequence_dim = add_sequence_dim  # 设置是否添加序列维度

    def forward(self, c):  # 前向传播方法
        if c.dtype == torch.float64:  # 如果输入数据类型为 float32
            c = c.float()  # 转换为 float32
        if c.dim() == 1:
            c = c.unsqueeze(-1)  # 增加一个维度，使其形状变为 (batch_size, 1)
        c = self.embedding(c)  # 获取嵌入层的输出
        if self.add_sequence_dim:  # 如果需要添加序列维度
            c = c[:, None, :]  # 在第二个维度上添加一个维度
        return c  # 返回嵌入层的输出

    def get_unconditional_conditioning(self, bs, device="cuda"):  # 获取无条件条件的方法
        uc = torch.ones((bs,), device=device) * -5  # 创建无条件类别的张量
        uc = {self.key: uc.long()}  # 将张量转换为字典格式
        return uc  # 返回无条件类别的字典

class ContinuousValueEmbedderForMultiCond(ContinuousValueEmbedder):  # 多条件连续值嵌入器类
    def __init__(self, embed_dim, n_classes=1, add_sequence_dim=False, input_keys=None):
        super().__init__(embed_dim, n_classes, add_sequence_dim)
        self.input_keys = input_keys  # 设置输入键列表
        if input_keys is not None and len(input_keys) > 0:
            self.key = input_keys[0]  # 设置默认键值为输入键列表的第一个键
    def forward(self, *batch, key=None, disable_dropout=False):  # 前向传播方法
        out = batch  # 初始化输出为输入的批量数据
        key = default(key, self.key)  # 获取默认键值
        islist = isinstance(batch[key], list)  # 判断批量数据的键值是否为列表
        if islist:  # 如果是列表
            batch[key] = batch[key][0]  # 获取列表的第一个元素
        c_out = super().forward(batch[key])  # 调用父类的前向传播方法获取输出
        out[key] = [c_out] if islist else c_out  # 根据输入类型设置输出
        return out  # 返回输出

class SpatialRescaler(nn.Module):  # 空间重缩放器类，继承自nn.Module
    def __init__(
        self,
        n_stages=1,  # 重缩放阶段数，默认为1
        method="bilinear",  # 插值方法，默认为双线性
        multiplier=0.5,  # 缩放倍数，默认为0.5
        in_channels=3,  # 输入通道数，默认为3
        out_channels=None,  # 输出通道数，默认为None
        bias=False,  # 是否使用偏置，默认为False
        wrap_video=False,  # 是否处理视频数据，默认为False
        kernel_size=1,  # 卷积核大小，默认为1
        remap_output=False,  # 是否重新映射输出，默认为False
    ):
        super().__init__()  # 调用父类的初始化方法
        self.n_stages = n_stages  # 设置重缩放阶段数
        assert self.n_stages >= 0  # 断言重缩放阶段数大于等于0
        assert method in [
            "nearest",
            "linear",
            "bilinear",
            "trilinear",
            "bicubic",
            "area",
        ]  # 断言插值方法在允许范围内
        self.multiplier = multiplier  # 设置缩放倍数
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)  # 创建插值函数
        self.remap_output = out_channels is not None or remap_output  # 设置是否重新映射输出
        if self.remap_output:  # 如果需要重新映射输出
            print(
                f"Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing."
            )  # 打印映射信息
            self.channel_mapper = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                bias=bias,
                padding=kernel_size // 2,
            )  # 创建卷积层进行通道映射
        self.wrap_video = wrap_video  # 设置是否处理视频数据

    def forward(self, x):  # 前向传播方法
        if self.wrap_video and x.ndim == 5:  # 如果处理视频数据且输入维度为5
            B, C, T, H, W = x.shape  # 获取输入数据的形状
            x = rearrange(x, "b c t h w -> b t c h w")  # 重新排列数据
            x = rearrange(x, "b t c h w -> (b t) c h w")  # 重新排列数据

        for stage in range(self.n_stages):  # 遍历重缩放阶段
            x = self.interpolator(x, scale_factor=self.multiplier)  # 进行插值操作

        if self.wrap_video:  # 如果处理视频数据
            x = rearrange(x, "(b t) c h w -> b t c h w", b=B, t=T, c=C)  # 重新排列数据
            x = rearrange(x, "b t c h w -> b c t h w")  # 重新排列数据
        if self.remap_output:  # 如果需要重新映射输出
            x = self.channel_mapper(x)  # 进行通道映射
        return x  # 返回输出

    def encode(self, x):  # 编码方法
        return self(x)  # 调用前向传播方法并返回结果


class LowScaleEncoder(nn.Module):  # 低尺度编码器类，继承自nn.Module
    def __init__(
        self,
        model_config,  # 模型配置
        linear_start,  # 线性起始值
        linear_end,  # 线性结束值
        timesteps=1000,  # 时间步数，默认为1000
        max_noise_level=250,  # 最大噪声级别，默认为250
        output_size=64,  # 输出大小，默认为64
        scale_factor=1.0,  # 缩放因子，默认为1.0
    ):
        super().__init__()  # 调用父类的初始化方法
        self.max_noise_level = max_noise_level  # 设置最大噪声级别
        self.model = instantiate_from_config(model_config)  # 从配置中实例化模型
        self.augmentation_schedule = self.register_schedule(
            timesteps=timesteps, linear_start=linear_start, linear_end=linear_end
        )  # 注册扩增计划
        self.out_size = output_size  # 设置输出大小
        self.scale_factor = scale_factor  # 设置缩放因子

    def register_schedule(
        self,
        beta_schedule="linear",  # beta计划，默认为线性
        timesteps=1000,  # 时间步数，默认为1000
        linear_start=1e-4,  # 线性起始值
        linear_end=2e-2,  # 线性结束值
        cosine_s=8e-3,  # 余弦参数，默认为8e-3
    ):
        betas = make_beta_schedule(
            beta_schedule,
            timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )  # 创建beta计划
        alphas = 1.0 - betas  # 计算alpha值
        alphas_cumprod = np.cumprod(alphas, axis=0)  # 计算alpha累积乘积
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])  # 计算前一个alpha累积乘积

        (timesteps,) = betas.shape  # 获取时间步数
        self.num_timesteps = int(timesteps)  # 设置时间步数
        self.linear_start = linear_start  # 设置线性起始值
        self.linear_end = linear_end  # 设置线性结束值
        assert (
            alphas_cumprod.shape[0] == self.num_timesteps
        ), "alphas have to be defined for each timestep"  # 断言alpha累积乘积的形状与时间步数一致

        to_torch = partial(torch.tensor, dtype=torch.float32)  # 部分应用，转换为torch张量

        self.register_buffer("betas", to_torch(betas))  # 注册beta缓冲区
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))  # 注册alpha累积乘积缓冲区
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))  # 注册前一个alpha累积乘积缓冲区

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))  # 注册sqrt alpha累积乘积缓冲区
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )  # 注册sqrt (1-alpha)累积乘积缓冲区
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )  # 注册log (1-alpha)累积乘积缓冲区
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        )  # 注册sqrt recip alpha累积乘积缓冲区
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )  # 注册sqrt recipm1 alpha累积乘积缓冲区

    def q_sample(self, x_start, t, noise=None):  # q样本方法
        noise = default(noise, lambda: torch.randn_like(x_start))  # 获取噪声
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )  # 计算q样本

    def forward(self, x):  # 前向传播方法
        z = self.model.encode(x)  # 编码输入数据
        if isinstance(z, DiagonalGaussianDistribution):  # 如果z是对角高斯分布
            z = z.sample()  # 采样
        z = z * self.scale_factor  # 缩放z
        noise_level = torch.randint(
            0, self.max_noise_level, (x.shape[0],), device=x.device
        ).long()  # 随机生成噪声级别
        z = self.q_sample(z, noise_level)  # 计算q样本
        if self.out_size is not None:  # 如果输出大小不为空
            z = torch.nn.functional.interpolate(z, size=self.out_size, mode="nearest")  # 插值
        return z, noise_level  # 返回z和噪声级别

    def decode(self, z):  # 解码方法
        z = z / self.scale_factor  # 缩放z
        return self.model.decode(z)  # 解码并返回


class ConcatTimestepEmbedderND(AbstractEmbModel):  # 拼接时间步嵌入器ND类，继承自AbstractEmbModel
    """embeds each dimension independently and concatenates them"""  # 类的文档字符串，说明其功能

    def __init__(self, outdim):  # 初始化方法，接收输出维度
        super().__init__()  # 调用父类的初始化方法
        self.timestep = Timestep(outdim)  # 创建时间步实例
        self.outdim = outdim  # 设置输出维度

    def forward(self, x):  # 前向传播方法
        if x.ndim == 1:  # 如果输入维度为1
            x = x[:, None]  # 在第二个维度上添加一个维度
        assert len(x.shape) == 2  # 断言输入的形状为2维
        b, dims = x.shape[0], x.shape[1]  # 获取输入的形状
        x = rearrange(x, "b d -> (b d)")  # 重新排列输入
        emb = self.timestep(x)  # 获取时间步嵌入
        emb = rearrange(emb, "(b d) d2 -> b (d d2)", b=b, d=dims, d2=self.outdim)  # 重新排列嵌入
        return emb  # 返回嵌入


class GaussianEncoder(Encoder, AbstractEmbModel):  # 高斯编码器类，继承自Encoder和AbstractEmbModel
    def __init__(
        self, weight: float = 1.0, flatten_output: bool = True, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)  # 调用父类的初始化方法
        self.posterior = DiagonalGaussianRegularizer()  # 创建对角高斯正则化器
        self.weight = weight  # 设置权重
        self.flatten_output = flatten_output  # 设置是否展平输出

    def forward(self, x) -> Tuple[Dict, torch.Tensor]:  # 前向传播方法
        z = super().forward(x)  # 调用父类的前向传播方法获取z
        z, log = self.posterior(z)  # 获取z和日志信息
        log["loss"] = log["kl_loss"]  # 设置日志信息中的损失
        log["weight"] = self.weight  # 设置日志信息中的权重
        if self.flatten_output:  # 如果需要展平输出
            z = rearrange(z, "b c h w -> b (h w ) c")  # 重新排列z
        return log, z  # 返回日志信息和z



class FrozenOpenCLIPImagePredictionEmbedder(AbstractEmbModel):  # 冻结的OpenCLIP图像预测嵌入器类，继承自AbstractEmbModel
    def __init__(
        self,
        open_clip_embedding_config: Dict,  # OpenCLIP嵌入配置
        n_cond_frames: int,  # 条件帧数
        n_copies: int,  # 复制次数
    ):
        super().__init__()  # 调用父类的初始化方法

        self.n_cond_frames = n_cond_frames  # 设置条件帧数
        self.n_copies = n_copies  # 设置复制次数
        self.open_clip = instantiate_from_config(open_clip_embedding_config)  # 从配置中实例化OpenCLIP

    def forward(self, vid):  # 前向传播方法
        vid = self.open_clip(vid)  # 获取OpenCLIP的输出
        vid = rearrange(vid, "(b t) d -> b t d", t=self.n_cond_frames)  # 重新排列输出
        vid = repeat(vid, "b t d -> (b s) t d", s=self.n_copies)  # 重复输出

        return vid  # 返回输出
