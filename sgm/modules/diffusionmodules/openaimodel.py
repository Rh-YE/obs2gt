import logging
import math
from abc import abstractmethod
from typing import Iterable, List, Optional, Tuple, Union

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from ...modules.attention import SpatialTransformer
from ...modules.video_attention import SpatialVideoTransformer
from ...modules.diffusionmodules.util import (avg_pool_nd, conv_nd, linear,
                                              normalization,
                                              timestep_embedding, zero_module)
from ...util import exists

logpy = logging.getLogger(__name__)


class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: Optional[int] = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim**2 + 1) / embed_dim**0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x: th.Tensor) -> th.Tensor:
        b, c, _ = x.shape
        x = x.reshape(b, c, -1)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x: th.Tensor, emb: th.Tensor):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(
        self,
        x: th.Tensor,
        emb: th.Tensor,
        context: Optional[th.Tensor] = None,
        image_only_indicator: Optional[th.Tensor] = None,
        time_context: Optional[int] = None,
        num_video_frames: Optional[int] = None,
    ):
        from ...modules.diffusionmodules.video_model import VideoResBlock

        for layer in self:
            module = layer

            if isinstance(module, TimestepBlock) and not isinstance(
                module, VideoResBlock
            ):
                x = layer(x, emb)
            elif isinstance(module, VideoResBlock):
                x = layer(x, emb, num_video_frames, image_only_indicator)
            elif isinstance(module, SpatialVideoTransformer):
                x = layer(
                    x,
                    context,
                    time_context,
                    num_video_frames,
                    image_only_indicator,
                )
            elif isinstance(module, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool,
        dims: int = 2,
        out_channels: Optional[int] = None,
        padding: int = 1,
        third_up: bool = False,
        kernel_size: int = 3,
        scale_factor: int = 2,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.third_up = third_up
        self.scale_factor = scale_factor
        if use_conv:
            self.conv = conv_nd(
                dims, self.channels, self.out_channels, kernel_size, padding=padding
            )

    def forward(self, x: th.Tensor) -> th.Tensor:
        assert x.shape[1] == self.channels

        if self.dims == 3:
            t_factor = 1 if not self.third_up else self.scale_factor
            x = F.interpolate(
                x,
                (
                    t_factor * x.shape[2],
                    x.shape[3] * self.scale_factor,
                    x.shape[4] * self.scale_factor,
                ),
                mode="nearest",
            )
        else:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool,
        dims: int = 2,
        out_channels: Optional[int] = None,
        padding: int = 1,
        third_down: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else ((1, 2, 2) if not third_down else (2, 2, 2))
        if use_conv:
            logpy.info(f"Building a Downsample layer with {dims} dims.")
            logpy.info(
                f"  --> settings are: \n in-chn: {self.channels}, out-chn: {self.out_channels}, "
                f"kernel-size: 3, stride: {stride}, padding: {padding}"
            )
            if dims == 3:
                logpy.info(f"  --> Downsampling third axis (time): {third_down}")
            self.op = conv_nd(
                dims,
                self.channels,
                self.out_channels,
                3,
                stride=stride,
                padding=padding,
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x: th.Tensor) -> th.Tensor:
        assert x.shape[1] == self.channels

        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float,
        out_channels: Optional[int] = None,
        use_conv: bool = False,
        use_scale_shift_norm: bool = False,
        dims: int = 2,
        use_checkpoint: bool = False,
        up: bool = False,
        down: bool = False,
        kernel_size: int = 3,
        exchange_temb_dims: bool = False,
        skip_t_emb: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.exchange_temb_dims = exchange_temb_dims

        if isinstance(kernel_size, Iterable):
            padding = [k // 2 for k in kernel_size]
        else:
            padding = kernel_size // 2

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, kernel_size, padding=padding),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.skip_t_emb = skip_t_emb
        self.emb_out_channels = (
            2 * self.out_channels if use_scale_shift_norm else self.out_channels
        )
        if self.skip_t_emb:
            logpy.info(f"Skipping timestep embedding in {self.__class__.__name__}")
            assert not self.use_scale_shift_norm
            self.emb_layers = None
            self.exchange_temb_dims = False
        else:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                linear(
                    emb_channels,
                    self.emb_out_channels,
                ),
            )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(
                    dims,
                    self.out_channels,
                    self.out_channels,
                    kernel_size,
                    padding=padding,
                )
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, kernel_size, padding=padding
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x: th.Tensor, emb: th.Tensor) -> th.Tensor:
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.use_checkpoint:
            return checkpoint(self._forward, x, emb)
        else:
            return self._forward(x, emb)

    def _forward(self, x: th.Tensor, emb: th.Tensor) -> th.Tensor:
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        if self.skip_t_emb:
            emb_out = th.zeros_like(h)
        else:
            emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape): # 调整embedding的形状然后保证最后能加到h上
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            if self.exchange_temb_dims:
                emb_out = rearrange(emb_out, "b t c ... -> b c t ...")
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 1,
        num_head_channels: int = -1,
        use_checkpoint: bool = False,
        use_new_attention_order: bool = False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x: th.Tensor, **kwargs) -> th.Tensor:
        return checkpoint(self._forward, x)

    def _forward(self, x: th.Tensor) -> th.Tensor:
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv: th.Tensor) -> th.Tensor:
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv: th.Tensor) -> th.Tensor:
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)


class Timestep(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: th.Tensor) -> th.Tensor:
        return timestep_embedding(t, self.dim)

class UNetModel(nn.Module):
    """
    带有注意力机制和时间步嵌入的完整UNet模型。
    :param in_channels: 输入Tensor的通道数。
    :param model_channels: 模型的基础通道数。
    :param out_channels: 输出Tensor的通道数。
    :param num_res_blocks: 每个降采样阶段的残差块数量。
    :param attention_resolutions: 在哪些降采样率下进行注意力机制操作。可以是集合、列表或元组。
        例如，如果包含4，则在4倍降采样时使用注意力机制。
    :param dropout: dropout概率。
    :param channel_mult: UNet每个层级的通道倍增数。
    :param conv_resample: 如果为True，则使用卷积进行上采样和下采样。
    :param dims: 决定信号是1D，2D还是3D。
    :param num_classes: 如果指定（如整数），则此模型将是带有`num_classes`类别的类条件模型。
    :param use_checkpoint: 使用梯度检查点以减少内存使用。
    :param num_heads: 每个注意力层中的注意力头数。
    :param num_heads_channels: 如果指定，则忽略num_heads，而是使用每个注意力头的固定通道宽度。
    :param num_heads_upsample: 配合num_heads设置上采样的头数。已弃用。
    :param use_scale_shift_norm: 使用类似FiLM的调节机制。
    :param resblock_updown: 使用残差块进行上采样/下采样。
    :param use_new_attention_order: 使用不同的注意力模式以潜在提高效率。
    """

    def __init__(
        self,
        in_channels: int,  # 输入通道数
        model_channels: int,  # 模型通道数
        out_channels: int,  # 输出通道数
        num_res_blocks: int,  # 每层的残差块数量
        attention_resolutions: int,  # 注意力机制的分辨率
        dropout: float = 0.0,  # dropout概率
        channel_mult: Union[List, Tuple] = (1, 2, 4, 8),  # 通道倍增系数
        conv_resample: bool = True,  # 是否使用卷积重采样
        dims: int = 2,  # 信号维度（1D, 2D, 3D）
        num_classes: Optional[Union[int, str]] = None,  # 类别数量
        use_checkpoint: bool = False,  # 是否使用梯度检查点
        num_heads: int = -1,  # 注意力头数量
        num_head_channels: int = -1,  # 每个注意力头的通道数
        num_heads_upsample: int = -1,  # 上采样时的注意力头数量（已弃用）
        use_scale_shift_norm: bool = False,  # 是否使用Scale-Shift归一化
        resblock_updown: bool = False,  # 是否使用残差块进行上/下采样
        transformer_depth: int = 1,  # Transformer的深度
        context_dim: Optional[int] = None,  # 上下文维度
        disable_self_attentions: Optional[List[bool]] = None,  # 禁用自注意力机制
        num_attention_blocks: Optional[List[int]] = None,  # 注意力块数量
        disable_middle_self_attn: bool = False,  # 禁用中间自注意力机制
        disable_middle_transformer: bool = False,  # 禁用中间Transformer
        use_linear_in_transformer: bool = False,  # 是否在线性Transformer中使用线性变换
        spatial_transformer_attn_type: str = "softmax",  # 空间Transformer的注意力类型
        adm_in_channels: Optional[int] = None,  # ADM输入通道数
    ):
        super().__init__()

        # 如果未指定num_heads_upsample，则使用num_heads的值
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        # 确保num_heads和num_head_channels其中之一被设置
        if num_heads == -1:
            assert (
                num_head_channels != -1
            ), "必须设置num_heads或num_head_channels其中之一"

        if num_head_channels == -1:
            assert (
                num_heads != -1
            ), "必须设置num_heads或num_head_channels其中之一"

        self.in_channels = in_channels  # 输入通道数
        self.model_channels = model_channels  # 模型通道数
        self.out_channels = out_channels  # 输出通道数

        # 如果transformer_depth是整数，则将其扩展为与channel_mult相同长度的列表
        if isinstance(transformer_depth, int):
            transformer_depth = len(channel_mult) * [transformer_depth]
        transformer_depth_middle = transformer_depth[-1]  # 中间层的transformer深度

        # 如果num_res_blocks是整数，则将其扩展为与channel_mult相同长度的列表
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError(
                    "num_res_blocks应该是一个整数（全局一致）或与channel_mult相同长度的列表"
                )
            self.num_res_blocks = num_res_blocks

        # 检查disable_self_attentions的长度是否正确
        if disable_self_attentions is not None:
            assert len(disable_self_attentions) == len(channel_mult)

        # 检查num_attention_blocks的长度是否正确
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(
                map(
                    lambda i: self.num_res_blocks[i] >= num_attention_blocks[i],
                    range(len(num_attention_blocks)),
                )
            )
            logpy.info(
                f"UNetModel构造函数接收到num_attention_blocks={num_attention_blocks}。"
                f"此选项优先级低于attention_resolutions {attention_resolutions}，"
                f"即如果num_attention_blocks[i] > 0但2**i不在attention_resolutions中，"
                f"仍然不会设置注意力机制。"
            )

        self.attention_resolutions = attention_resolutions  # 注意力机制的分辨率
        self.dropout = dropout  # dropout概率
        self.channel_mult = channel_mult  # 通道倍增系数
        self.conv_resample = conv_resample  # 是否使用卷积重采样
        self.num_classes = num_classes  # 类别数量
        self.use_checkpoint = use_checkpoint  # 是否使用梯度检查点
        self.num_heads = num_heads  # 注意力头数量
        self.num_head_channels = num_head_channels  # 每个注意力头的通道数
        self.num_heads_upsample = num_heads_upsample  # 上采样时的注意力头数量

        # 时间步嵌入维度
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # 类别嵌入层
        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                logpy.info("设置线性c_adm嵌入层")
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "timestep":
                self.label_emb = nn.Sequential(
                    Timestep(model_channels),
                    nn.Sequential(
                        linear(model_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    ),
                )
            elif self.num_classes == "sequential":
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        linear(adm_in_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    )
                )
            else:
                raise ValueError

        # 输入块
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels  # 特征大小
        input_block_chans = [model_channels]  # 输入块通道列表
        ch = model_channels  # 当前通道数
        ds = 1  # 当前降采样率

        # 构建每层的残差块和注意力块
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels

                    if context_dim is not None and exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if (
                        not exists(num_attention_blocks)
                        or nr < num_attention_blocks[level]
                    ):
                        layers.append(
                            SpatialTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth[level],
                                context_dim=context_dim,
                                disable_self_attn=disabled_sa,
                                use_linear=use_linear_in_transformer,
                                attn_type=spatial_transformer_attn_type,
                                use_checkpoint=use_checkpoint,
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        # 中间块
        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                out_channels=ch,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            SpatialTransformer(
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth_middle,
                context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn,
                use_linear=use_linear_in_transformer,
                attn_type=spatial_transformer_attn_type,
                use_checkpoint=use_checkpoint,
            )
            if not disable_middle_transformer
            else th.nn.Identity(),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        # 输出块
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels

                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if (
                        not exists(num_attention_blocks)
                        or i < num_attention_blocks[level]
                    ):
                        layers.append(
                            SpatialTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth[level],
                                context_dim=context_dim,
                                disable_self_attn=disabled_sa,
                                use_linear=use_linear_in_transformer,
                                attn_type=spatial_transformer_attn_type,
                                use_checkpoint=use_checkpoint,
                            )
                        )
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        # 最终输出层
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def forward(
        self,
        x: th.Tensor,  # 输入Tensor
        timesteps: Optional[th.Tensor] = None,  # 时间步Tensor
        context: Optional[th.Tensor] = None,  # 上下文Tensor
        y: Optional[th.Tensor] = None,  # 类别标签
        **kwargs,
    ) -> th.Tensor:
        """
        将模型应用于输入批次。
        :param x: [N x C x ...]形状的输入Tensor。
        :param timesteps: 一维的时间步批次。
        :param context: 通过交叉注意力提供的条件
        :param y: [N]形状的标签，如果是类条件模型。
        :return: [N x C x ...]形状的输出Tensor。
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "如果且仅当模型是类条件时，必须指定y"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False) # tiesteps是时间步t，t_emb是时间步嵌入
        emb = self.time_embed(t_emb) # 生成更高维、更具表达能力的嵌入向量 emb

        if self.num_classes is not None: # 如果需要嵌入类别信息，在这里接在embedding后面
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        h = x
        for module in self.input_blocks: # UNet下采样
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context) # UNet中间块
        for module in self.output_blocks: # UNet上采样
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        h = h.type(x.dtype)

        return self.out(h)




