# pytorch_diffusion + derived encoder decoder
import logging
import math
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from packaging import version

logpy = logging.getLogger(__name__)

try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILABLE = True
except:
    XFORMERS_IS_AVAILABLE = False
    logpy.warning("no module 'xformers'. Processing without...")

from ...modules.attention import LinearAttention, MemoryEfficientCrossAttention
from ...modules.diffusionmodules.filtered_lrelu import filtered_lrelu

import scipy.signal
import scipy.optimize

from scipy.signal.windows import kaiser
from scipy.special import j1

from torch.nn import functional as F

f_settings={}
f_settings['kernel_size']=3
f_settings['kaiser_beta']=2
f_settings['omega_c_down'] =np.pi/2
f_settings['omega_c_up'] = np.pi/2


def custom_downsample( x, jinc_filter,factor=2):
    # Apply the Jinc filter before downsampling
    jinc_filter = jinc_filter[None, None, :, :].to(x.device)  # Shape (1, 1, filter_size, filter_size)
    jinc_filter = jinc_filter.repeat(x.size(1), 1, 1, 1)  # Match number of channels
    x = F.conv2d(x, jinc_filter, padding='same', groups=x.size(1))
    x = x[:, :, ::factor, ::factor]
    return x

def custom_upsample(x, sinc_filter,factor=2):
    # Upsample using zero padding followed by applying the sinc filter
    # Get the original dimensions
    batch_size, channels, height, width = x.shape

    # Create a new tensor with double the height and width filled with zeros
    upsampled = torch.zeros(batch_size, channels, height * factor, width * factor, device=x.device)

    # Assign the original values to the correct positions
    upsampled[:, :, ::factor, ::factor] = x
    x=upsampled
    # Apply the sinc filter (low-pass filter)
    sinc_filter = sinc_filter[None, None, :, :].to(x.device)  # Shape (1, 1, filter_size, filter_size)
    sinc_filter = sinc_filter.repeat(x.size(1), 1, 1, 1)  # Match number of channels
    x = F.conv2d(x, sinc_filter, padding='same', groups=x.size(1))
    return x  

def jinc_filter_2d(size=6, beta=14):
    # Similar to the sinc filter, create a 2D jinc filter (simplified)
    sinc_filter_1d = np.sinc(np.linspace(-size / 2, size / 2, size))
    window = kaiser(size, beta)
    jinc_filter_2d = np.outer(sinc_filter_1d * window, sinc_filter_1d * window)
    # Normalize the kernel
    jinc_filter_2d = jinc_filter_2d / np.sum(jinc_filter_2d)
    return torch.tensor(jinc_filter_2d, dtype=torch.float32)

def circularLowpassKernel(omega_c=np.pi, N=6,beta=None):  # omega = cutoff frequency in radians (pi is max), N = horizontal size of the kernel, also its vertical size.
    # 此处使用 np.errstate 来暂时忽略浮点运算中出现的除零和无效操作的警告，
    # 以确保在计算过程中不会因数学异常而中断代码执行。
    with np.errstate(divide='ignore', invalid='ignore'):
        kernel = np.fromfunction(lambda x, y: omega_c*j1(omega_c*np.sqrt((x - (N - 1)/2)**2 + (y - (N - 1)/2)**2))/(2*np.pi*np.sqrt((x - (N - 1)/2)**2 + (y - (N - 1)/2)**2)), [N, N])
    if N % 2:
        kernel[(N - 1)//2, (N - 1)//2] = omega_c**2/(4*np.pi)
    
    if beta is not None:
        # Create a 1D Kaiser window
        kaiser_window_1d = np.kaiser(N, beta)

        # Generate a 2D Kaiser window by outer product
        kaiser_window_2d = np.outer(kaiser_window_1d, kaiser_window_1d)

        # Apply the Kaiser window to the kernel
        kernel *= kaiser_window_2d
    # Normalize the kernel
    kernel=kernel/ np.sum(kernel)
    return torch.tensor(kernel, dtype=torch.float32)

def design_lowpass_filter(numtaps, cutoff, width, fs, radial=False):
    assert numtaps >= 1

    # Identity filter.
    if numtaps == 1:
        return None

    # Separable Kaiser low-pass filter.
    if not radial:
        f = scipy.signal.firwin(numtaps=numtaps, cutoff=cutoff, width=width, fs=fs)
        return torch.as_tensor(f, dtype=torch.float32)

    # Radially symmetric jinc-based filter. config R
    x = (np.arange(numtaps) - (numtaps - 1) / 2) / fs
    r = np.hypot(*np.meshgrid(x, x))
    f = scipy.special.j1(2 * cutoff * (np.pi * r)) / (np.pi * r)
    beta = scipy.signal.kaiser_beta(scipy.signal.kaiser_atten(numtaps, width / (fs / 2)))
    w = np.kaiser(numtaps, beta)
    f *= np.outer(w, w)
    f /= np.sum(f)
    return torch.as_tensor(f, dtype=torch.float32)

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # sinc_filter = circularLowpassKernel(omega_c=f_settings['omega_c_up'],  # Creating a Sinc filter
    #                                              N=f_settings['kernel_size'], 
    #                                              beta=f_settings['kaiser_beta'])  # Using filter settings
    # jinc_filter = circularLowpassKernel(omega_c=f_settings['omega_c_up'],  # Creating a Sinc filter
    #                                              N=f_settings['kernel_size'], 
    #                                              beta=f_settings['kaiser_beta'])  # Using filter settings
    # x = custom_upsample(x, sinc_filter)  # Upsampling using the Sinc filter
    x = x*torch.sigmoid(x)  # Applying GELU activation
    # x = custom_downsample(x, jinc_filter)  # Downsampling using the Jinc filter
    return x

# 新增代码：自定义的激活模块，用于替换原来的 nonlinearity 实现

class FilteredLReluActivation(nn.Module):
    """
    一个"即插即用"版本的 FilteredLRelu 激活模块，
    在 forward 时自动根据输入的空间尺寸计算滤波器参数，
    并在第一次 forward 时根据输入通道数自动创建可训练参数 bias。

    参数说明：
      in_sampling_rate, out_sampling_rate : 采样率（可以理解为像素采样间隔的模拟值），
                                             默认可取 1.0，即输入与输出采样率相同。
      in_cutoff, out_cutoff               : 截止频率（根据实际需求调整）
      in_half_width, out_half_width       : 过渡带一半宽度
      conv_kernel                         : 激活前的卷积核尺寸（默认为 3）
      filter_size                         : 滤波器大小因子（默认 6）
      lrelu_upsampling                    : leaky ReLU 内部上采样倍率（默认 2）
      use_radial_filters                  : 是否使用径向对称滤波器（默认 False）
      conv_clamp                          : clamp 参数（例如 256）
      is_critically_sampled, use_fp16, is_torgb : 其他可能影响滤波器设计的超参数
    """
    def __init__(self, 
                 in_sampling_rate=1.0, 
                 out_sampling_rate=1.0, 
                 in_cutoff=0.05, 
                 out_cutoff=0.9,
                 in_half_width=0.15, 
                 out_half_width=0.05,
                 conv_kernel=3,
                 filter_size=6, 
                 lrelu_upsampling=2,
                 use_radial_filters=True, 
                 conv_clamp=65536,
                 is_critically_sampled=False, 
                 use_fp16=False, 
                 is_torgb=False):
        super().__init__()
        self.in_sampling_rate = in_sampling_rate
        self.out_sampling_rate = out_sampling_rate
        self.in_cutoff = in_cutoff
        self.out_cutoff = out_cutoff
        self.in_half_width = in_half_width
        self.out_half_width = out_half_width
        self.conv_kernel = conv_kernel
        self.filter_size = filter_size
        self.lrelu_upsampling = lrelu_upsampling
        self.use_radial_filters = use_radial_filters
        self.conv_clamp = conv_clamp
        self.is_critically_sampled = is_critically_sampled
        self.use_fp16 = use_fp16
        self.is_torgb = is_torgb

        # 使用额外的标志记录 bias 是否已初始化
        self._bias_initialized = False

    def forward(self, x):
        # x: Tensor shape (B, C, H, W)
        B, C, H, W = x.shape
        # 如果 bias 尚未初始化，则根据当前通道数创建并注册
        if not self._bias_initialized:
            bias = torch.zeros(C, device=x.device, dtype=x.dtype)
            self.register_parameter('bias', nn.Parameter(bias))
            self._bias_initialized = True
        else:
            # 检查已注册的 bias 通道数是否与当前输入匹配
            if self.bias.shape[0] != C:
                # 若不匹配，则调整 bias 参数的尺寸，保留之前学到的权重
                old_bias = self.bias.data.clone()
                old_channels = old_bias.shape[0]
                if C > old_channels:
                    new_bias = torch.cat([
                        old_bias,
                        torch.zeros(C - old_channels, device=x.device, dtype=x.dtype)
                    ], dim=0)
                else:
                    new_bias = old_bias[:C]
                self.bias = nn.Parameter(new_bias)

        # 使用输入尺寸作为 in_size 和 out_size（若分辨率不改变则二者相同）
        in_size = np.array((H, W))
        out_size = np.array((H, W))
        # 根据 lrelu_upsampling 计算临时采样率
        tmp_sampling_rate = max(self.in_sampling_rate, self.out_sampling_rate) * self.lrelu_upsampling
        # 计算上采样与下采样因子
        up_factor = int(np.rint(tmp_sampling_rate / self.in_sampling_rate))
        down_factor = int(np.rint(tmp_sampling_rate / self.out_sampling_rate))
        # 计算滤波器 tap 数（若 up_factor==1 则使用 1 tap，即身份滤波）
        up_taps = self.filter_size * up_factor if (up_factor > 1 and not self.is_torgb) else 1
        down_taps = self.filter_size * down_factor if (down_factor > 1 and not self.is_torgb) else 1
        
        pad_total = (out_size - 1) * down_factor + 1 # Desired output size before downsampling.
        pad_total -= (in_size + self.conv_kernel - 1) * up_factor # Input size after upsampling.
        pad_total += up_taps + down_taps - 2 # Size reduction caused by the filters.
        pad_lo = (pad_total + up_factor) // 2 # Shift sample locations according to the symmetric interpretation (Appendix C.3).
        pad_hi = pad_total - pad_lo
        padding = [int(pad_lo[0]) + 2, int(pad_hi[0]) + 2, int(pad_lo[1]) + 2, int(pad_hi[1]) + 2] 
        
        # 动态设计上采样与下采样滤波器
        up_filter = design_lowpass_filter(numtaps=up_taps, cutoff=self.in_cutoff, width=self.in_half_width * 2, fs=tmp_sampling_rate)
        down_filter = design_lowpass_filter(numtaps=down_taps, cutoff=self.out_cutoff, width=self.out_half_width * 2, fs=tmp_sampling_rate, radial=(self.use_radial_filters and not self.is_critically_sampled))
        # 确保滤波器在相同设备上，如果它们非空的话
        if up_filter is not None:
            up_filter = up_filter.to(x.device)
        if down_filter is not None:
            down_filter = down_filter.to(x.device)
        
        # if up_filter is None:
        #     up_filter = torch.tensor([1.], dtype=torch.float32, device=x.device)
        # if down_filter is None:
        #     down_filter = torch.tensor([1.], dtype=torch.float32, device=x.device)
        # 调用 filtered_lrelu，将所有参数传入
        return filtered_lrelu(
            x,
            fu=up_filter,
            fd=down_filter,
            b=self.bias,
            up=up_factor,
            down=down_factor,
            padding=padding,
            gain=np.sqrt(2),
            slope=0.2,
            clamp=self.conv_clamp,
            flip_filter=False,
            impl='cuda' if x.is_cuda else 'ref'
        )


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )
            # self.conv1 = torch.nn.Conv2d(
            #     in_channels, in_channels, kernel_size=1, stride=1, padding=0
            # )
        # self.f_settings = f_settings  # Storing filter settings
        # # Generate the 2D Jinc filter with Kaiser window
        # self.jinc_filter = circularLowpassKernel(omega_c=self.f_settings['omega_c_down'],  # Creating a Jinc filter
        #                                          N=self.f_settings['kernel_size'], 
        #                                          beta=self.f_settings['kaiser_beta'])  # Using filter settings
    def forward(self, x):
        # x = custom_upsample(x, self.jinc_filter)
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
            # x = self.conv1(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )
        # self.f_settings = f_settings  # Storing filter settings
        # # Generate the 2D Jinc filter with Kaiser window
        # self.jinc_filter = circularLowpassKernel(omega_c=self.f_settings['omega_c_down'],  # Creating a Jinc filter
        #                                          N=self.f_settings['kernel_size'], 
        #                                          beta=self.f_settings['kaiser_beta'])  # Using filter settings

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            # x = custom_downsample(x, self.jinc_filter)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            # x = custom_downsample(x, self.jinc_filter)
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
        stylegan_act: False,
    ):
        super().__init__()
        self.stylegan_act = stylegan_act
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

        # 替换非线性操作：不再直接调用 nonlinearity，而是创建一个 lazy 版本的激活模块
        self.act = FilteredLReluActivation()

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = self.act(h) if self.stylegan_act else nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = self.act(h) if self.stylegan_act else nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""

    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def attention(self, h_: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q, k, v = map(
            lambda x: rearrange(x, "b c h w -> b 1 (h w) c").contiguous(), (q, k, v)
        )
        h_ = torch.nn.functional.scaled_dot_product_attention(
            q, k, v
        )  # scale is dim ** -0.5 per default
        # compute attention

        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def forward(self, x, **kwargs):
        h_ = x
        h_ = self.attention(h_)
        h_ = self.proj_out(h_)
        return x + h_


class MemoryEfficientAttnBlock(nn.Module):
    """
    Uses xformers efficient implementation,
    see https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    Note: this is a single-head self-attention operation
    """

    #
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.attention_op: Optional[Any] = None

    def attention(self, h_: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        B, C, H, W = q.shape
        q, k, v = map(lambda x: rearrange(x, "b c h w -> b (h w) c"), (q, k, v))

        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(B, t.shape[1], 1, C)
            .permute(0, 2, 1, 3)
            .reshape(B * 1, t.shape[1], C)
            .contiguous(),
            (q, k, v),
        )
        out = xformers.ops.memory_efficient_attention(
            q, k, v, attn_bias=None, op=self.attention_op
        )

        out = (
            out.unsqueeze(0)
            .reshape(B, 1, out.shape[1], C)
            .permute(0, 2, 1, 3)
            .reshape(B, out.shape[1], C)
        )
        return rearrange(out, "b (h w) c -> b c h w", b=B, h=H, w=W, c=C)

    def forward(self, x, **kwargs):
        h_ = x
        h_ = self.attention(h_)
        h_ = self.proj_out(h_)
        return x + h_


class MemoryEfficientCrossAttentionWrapper(MemoryEfficientCrossAttention):
    def forward(self, x, context=None, mask=None, **unused_kwargs):
        b, c, h, w = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        out = super().forward(x, context=context, mask=mask)
        out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w, c=c)
        return x + out


def make_attn(in_channels, attn_type="vanilla", attn_kwargs=None):
    assert attn_type in [
        "vanilla",
        "vanilla-xformers",
        "memory-efficient-cross-attn",
        "linear",
        "none",
    ], f"attn_type {attn_type} unknown"
    if (
        version.parse(torch.__version__) < version.parse("2.0.0")
        and attn_type != "none"
    ):
        assert XFORMERS_IS_AVAILABLE, (
            f"We do not support vanilla attention in {torch.__version__} anymore, "
            f"as it is too expensive. Please install xformers via e.g. 'pip install xformers==0.0.16'"
        )
        attn_type = "vanilla-xformers"
    logpy.info(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        assert attn_kwargs is None
        return AttnBlock(in_channels)
    elif attn_type == "vanilla-xformers":
        logpy.info(
            f"building MemoryEfficientAttnBlock with {in_channels} in_channels..."
        )
        return MemoryEfficientAttnBlock(in_channels)
    elif attn_type == "memory-efficient-cross-attn":
        attn_kwargs["query_dim"] = in_channels
        return MemoryEfficientCrossAttentionWrapper(**attn_kwargs)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        return LinAttnBlock(in_channels)


class Model(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=False,
        in_channels,
        resolution,
        use_timestep=True,
        use_linear_attn=False,
        attn_type="vanilla",
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.use_timestep = use_timestep
        if self.use_timestep:
            # timestep embedding
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList(
                [
                    torch.nn.Linear(self.ch, self.temb_ch),
                    torch.nn.Linear(self.temb_ch, self.temb_ch),
                ]
            )

        # downsampling
        self.conv_in = torch.nn.Conv2d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(
                    ResnetBlock(
                        in_channels=block_in + skip_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in, out_ch, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x, t=None, context=None):
        # assert x.shape[2] == x.shape[3] == self.resolution
        if context is not None:
            # assume aligned context, cat along channel axis
            x = torch.cat((x, context), dim=1)
        if self.use_timestep:
            # timestep embedding
            assert t is not None
            temb = get_timestep_embedding(t, self.ch)
            temb = self.temb.dense[0](temb)
            temb = nonlinearity(temb)
            temb = self.temb.dense[1](temb)
        else:
            temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb
                )
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

    def get_last_layer(self):
        return self.conv_out.weight


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=False,
        in_channels,
        resolution,
        z_channels,
        double_z=True,
        use_linear_attn=False,
        attn_type="vanilla",
        single_dim = False,
        latent_dim = 40,
        stylegan_act = False,
        **ignore_kwargs,
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.stylegan_act = stylegan_act
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.single_dim = single_dim
        self.latent_dim = latent_dim
        # downsampling
        self.conv_in = torch.nn.Conv2d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        self.curr_res = resolution
        self.in_ch_mult = (1,) + tuple(ch_mult)
        for i_level in range(self.num_resolutions):
            if i_level != self.num_resolutions - 1:
                self.curr_res = self.curr_res // 2
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        stylegan_act=self.stylegan_act,
                    )
                )
                block_in = block_out
                # print(curr_res)
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            stylegan_act=self.stylegan_act,
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            stylegan_act=self.stylegan_act,
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        if self.single_dim:
            self.final_z = 2 * z_channels if double_z else z_channels
            self.fc_out = nn.Linear(self.final_z*self.curr_res*self.curr_res, self.latent_dim)
            
        self.act = FilteredLReluActivation()
    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = self.act(h) if self.stylegan_act else nonlinearity(h)
        h = self.conv_out(h)
        if self.single_dim:
            h = h.view(h.size(0), -1)
            h = self.fc_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=False,
        in_channels,
        resolution,
        z_channels,
        give_pre_end=False,
        tanh_out=False,
        use_linear_attn=False,
        attn_type="vanilla",
        single_dim = False,
        latent_dim = 40,
        double_z = True,
        stylegan_act = False,
        **ignorekwargs,
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.stylegan_act = stylegan_act
        self.ch = ch
        self.latent_dim = latent_dim
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.single_dim = single_dim
        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        if self.single_dim:
            self.fc_in = nn.Linear(self.latent_dim//2, np.prod(self.z_shape))
        logpy.info(
            "Working with z of shape {} = {} dimensions.".format(
                self.z_shape, np.prod(self.z_shape)
            )
        )

        make_attn_cls = self._make_attn()
        make_resblock_cls = self._make_resblock()
        make_conv_cls = self._make_conv()
        # z to block_in
        self.conv_in = torch.nn.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = make_resblock_cls(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            stylegan_act=self.stylegan_act,
        )
        self.mid.attn_1 = make_attn_cls(block_in, attn_type=attn_type)
        self.mid.block_2 = make_resblock_cls(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            stylegan_act=self.stylegan_act,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    make_resblock_cls(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        stylegan_act=self.stylegan_act,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn_cls(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = make_conv_cls(
            block_in, out_ch, kernel_size=3, stride=1, padding=1
        )
        self.act = FilteredLReluActivation()
                
    def _make_attn(self) -> Callable:
        return make_attn

    def _make_resblock(self) -> Callable:
        return ResnetBlock

    def _make_conv(self) -> Callable:
        return torch.nn.Conv2d

    def get_last_layer(self, **kwargs):
        return self.conv_out.weight

    def forward(self, z, **kwargs):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        if self.single_dim:
            z = self.fc_in(z)
            z = z.view(z.size(0), *self.z_shape[1:])
            
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb, **kwargs)
        h = self.mid.attn_1(h, **kwargs)
        h = self.mid.block_2(h, temb, **kwargs)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb, **kwargs)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, **kwargs)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = self.act(h) if self.stylegan_act else nonlinearity(h)
        # h = self.act(h)
        h = self.conv_out(h, **kwargs)
        if self.tanh_out:
            h = torch.tanh(h)
        return h

# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# # 显示VAE模型的参数量
# encoder = Encoder(
#     attn_type=None,
#     double_z=True,
#     z_channels=40,
#     resolution=64,
#     in_channels=8,
#     out_ch=4,
#     ch=128,
#     ch_mult=(1, 2, 4),
#     num_res_blocks=2,
#     attn_resolutions=[],
#     dropout=0.0,
#     single_dim=True,
#     latent_dim=80,
# )

# decoder = Decoder(
#     attn_type=None,
#     double_z=True,
#     z_channels=40,
#     resolution=64,
#     in_channels=8,
#     out_ch=4,
#     ch=128,
#     ch_mult=(1, 2, 4),
#     num_res_blocks=2,
#     attn_resolutions=[],
#     dropout=0.0,
#     single_dim=True,
#     latent_dim=80,
# )

# print(f"Encoder参数量: {count_parameters(encoder)}")
# print(f"Decoder参数量: {count_parameters(decoder)}")
