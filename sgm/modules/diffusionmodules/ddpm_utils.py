import torch  # Importing the PyTorch library for tensor operations
import torch.nn as nn  # Importing the neural network module from PyTorch
import torch.nn.functional as F  # Importing functional operations for neural networks
from modules.utils import *  # Importing utility functions from the utils module
from modules.filtrs import *  # Importing filter functions from the filtrs module
import logging  # Importing the logging module for logging messages
from tqdm import tqdm  # Importing tqdm for progress bar functionality
from torch import optim  # Importing the optim module for optimization algorithms

def jinc_filter_2d(size=6, beta=14):  # Defining a function to create a 2D Jinc filter
    # Similar to the sinc filter, create a 2D jinc filter (simplified)
    sinc_filter_1d = np.sinc(np.linspace(-size / 2, size / 2, size))  # Creating a 1D sinc filter
    window = kaiser(size, beta)  # Creating a Kaiser window
    jinc_filter_2d = np.outer(sinc_filter_1d * window, sinc_filter_1d * window)  # Creating a 2D Jinc filter
    # Normalize the kernel
    jinc_filter_2d = jinc_filter_2d / np.sum(jinc_filter_2d)  # Normalizing the filter
    return torch.tensor(jinc_filter_2d, dtype=torch.float32)  # Returning the filter as a PyTorch tensor

def circularLowpassKernel(omega_c=np.pi, N=6, beta=None):  # Defining a function for a circular lowpass kernel
    # omega = cutoff frequency in radians (pi is max), N = horizontal size of the kernel, also its vertical size.
    # 此处使用 np.errstate 来暂时忽略浮点运算中出现的除零和无效操作的警告，
    # 以确保在计算过程中不会因数学异常而中断代码执行。
    with np.errstate(divide='ignore', invalid='ignore'):  # Suppressing warnings for invalid operations
        kernel = np.fromfunction(lambda x, y: omega_c * j1(omega_c * np.sqrt((x - (N - 1) / 2) ** 2 + (y - (N - 1) / 2) ** 2)) / (2 * np.pi * np.sqrt((x - (N - 1) / 2) ** 2 + (y - (N - 1) / 2) ** 2)), [N, N])  # Creating the kernel using a mathematical function
    if N % 2:  # Checking if N is odd
        kernel[(N - 1) // 2, (N - 1) // 2] = omega_c ** 2 / (4 * np.pi)  # Adjusting the center value for odd N
    
    if beta is not None:  # Checking if beta is provided
        # Create a 1D Kaiser window
        kaiser_window_1d = np.kaiser(N, beta)  # Creating a 1D Kaiser window

        # Generate a 2D Kaiser window by outer product
        kaiser_window_2d = np.outer(kaiser_window_1d, kaiser_window_1d)  # Creating a 2D Kaiser window

        # Apply the Kaiser window to the kernel
        kernel *= kaiser_window_2d  # Modifying the kernel with the Kaiser window
    # Normalize the kernel
    kernel = kernel / np.sum(kernel)  # Normalizing the kernel
    return torch.tensor(kernel, dtype=torch.float32)  # Returning the kernel as a PyTorch tensor

class DoubleConv(nn.Module):  # Defining a class for double convolutional layers
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):  # Initializing the class
        super().__init__()  # Calling the parent class initializer
        self.residual = residual  # Storing the residual flag
        if not mid_channels:  # Checking if mid_channels is not provided
            mid_channels = out_channels  # Setting mid_channels to out_channels
        self.double_conv = nn.Sequential(  # Creating a sequential container for double convolution
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),  # First convolution layer
            nn.GroupNorm(1, mid_channels),  # Group normalization layer
            nn.GELU(),  # GELU activation function
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),  # Second convolution layer
            nn.GroupNorm(1, out_channels),  # Group normalization layer
        )

    def forward(self, x):  # Defining the forward pass
        if self.residual:  # Checking if residual connections are enabled
            return F.gelu(x + self.double_conv(x))  # Returning the output with residual connection
        else:
            return self.double_conv(x)  # Returning the output without residual connection

class DoubleConv_F(nn.Module):  # Defining a class for double convolution with filters
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False, f_settings=None):  # Initializing the class
        super().__init__()  # Calling the parent class initializer
        self.residual = residual  # Storing the residual flag
        self.f_settings = f_settings  # Storing filter settings
        self.jinc_filter = circularLowpassKernel(omega_c=self.f_settings['omega_c_down'],  # Creating a Jinc filter
                                                 N=self.f_settings['kernel_size'], 
                                                 beta=self.f_settings['kaiser_beta'])  # Using filter settings
        self.sinc_filter = circularLowpassKernel(omega_c=self.f_settings['omega_c_up'],  # Creating a Sinc filter
                                                 N=self.f_settings['kernel_size'], 
                                                 beta=self.f_settings['kaiser_beta'])  # Using filter settings

        if not mid_channels:  # Checking if mid_channels is not provided
            mid_channels = out_channels  # Setting mid_channels to out_channels

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)  # First convolution layer
        self.norm1 = nn.GroupNorm(1, mid_channels)  # Group normalization layer
        self.gelu = nn.GELU()  # GELU activation function
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)  # Second convolution layer
        self.norm2 = nn.GroupNorm(1, out_channels)  # Group normalization layer

    def forward(self, x):  # Defining the forward pass
        if self.residual:  # Checking if residual connections are enabled
            residual = x  # Storing the input for residual connection
            x = self.conv1(x)  # Applying the first convolution
            x = self.norm1(x)  # Applying normalization
            x = custom_upsample(x, self.sinc_filter)  # Upsampling using the Sinc filter
            x = self.gelu(x)  # Applying GELU activation
            x = custom_downsample(x, self.jinc_filter)  # Downsampling using the Jinc filter
            x = self.conv2(x)  # Applying the second convolution
            x = self.norm2(x)  # Applying normalization
            x = x + residual  # Adding the residual connection
            x = custom_upsample(x, self.sinc_filter)  # Upsampling again using the Sinc filter
            x = F.gelu(x)  # Applying GELU activation
            x = custom_downsample(x, self.jinc_filter)  # Downsampling again using the Jinc filter
            return x  # Returning the output
            # return F.gelu(x + self.double_conv(x))
        else:
            x = self.conv1(x)  # Applying the first convolution
            x = self.norm1(x)  # Applying normalization
            x = custom_upsample(x, self.sinc_filter)  # Upsampling using the Sinc filter
            x = self.gelu(x)  # Applying GELU activation
            x = custom_downsample(x, self.jinc_filter)  # Downsampling using the Jinc filter
            x = self.conv2(x)  # Applying the second convolution
            x = self.norm2(x)  # Applying normalization
            return x  # Returning the output
            # return self.double_conv(x)

class DoubleConv_F4(nn.Module):  # Defining a class for double convolution with filters (4)
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False, f_settings=None):  # Initializing the class
        super().__init__()  # Calling the parent class initializer
        self.residual = residual  # Storing the residual flag
        self.f_settings = f_settings  # Storing filter settings
        self.jinc_filter = circularLowpassKernel(omega_c=self.f_settings['omega_c_down'],  # Creating a Jinc filter
                                                 N=self.f_settings['kernel_size'], 
                                                 beta=self.f_settings['kaiser_beta'])  # Using filter settings
        self.sinc_filter = circularLowpassKernel(omega_c=self.f_settings['omega_c_up'],  # Creating a Sinc filter
                                                 N=self.f_settings['kernel_size'], 
                                                 beta=self.f_settings['kaiser_beta'])  # Using filter settings

        if not mid_channels:  # Checking if mid_channels is not provided
            mid_channels = out_channels  # Setting mid_channels to out_channels

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)  # First convolution layer
        self.norm1 = nn.GroupNorm(1, mid_channels)  # Group normalization layer
        self.gelu = nn.GELU()  # GELU activation function
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)  # Second convolution layer
        self.norm2 = nn.GroupNorm(1, out_channels)  # Group normalization layer

    def forward(self, x):  # Defining the forward pass
        if self.residual:  # Checking if residual connections are enabled
            residual = x  # Storing the input for residual connection
            x = self.conv1(x)  # Applying the first convolution
            # x = self.norm1(x)  # Applying normalization (commented out)
            x = custom_upsample(x, self.sinc_filter)  # Upsampling using the Sinc filter
            x = self.norm1(x)  # Applying normalization (added)
            x = self.gelu(x)  # Applying GELU activation
            x = custom_downsample(x, self.jinc_filter)  # Downsampling using the Jinc filter
            # x = self.norm1(x)  # Applying normalization (commented out)
            x = self.conv2(x)  # Applying the second convolution
            x = self.norm2(x)  # Applying normalization
            x = x + residual  # Adding the residual connection
            x = custom_upsample(x, self.sinc_filter)  # Upsampling again using the Sinc filter
            x = self.norm2(x)  # Applying normalization (added)
            x = F.gelu(x)  # Applying GELU activation
            x = custom_downsample(x, self.jinc_filter)  # Downsampling again using the Jinc filter
            # x = self.norm2(x)  # Applying normalization (commented out)
            return x  # Returning the output
            # return F.gelu(x + self.double_conv(x))
        else:
            x = self.conv1(x)  # Applying the first convolution
            # x = self.norm1(x)  # Applying normalization (commented out)
            x = custom_upsample(x, self.sinc_filter)  # Upsampling using the Sinc filter
            x = self.norm1(x)  # Applying normalization (added)
            x = self.gelu(x)  # Applying GELU activation
            x = custom_downsample(x, self.jinc_filter)  # Downsampling using the Jinc filter
            # x = self.norm1(x)  # Applying normalization (commented out)
            x = self.conv2(x)  # Applying the second convolution
            x = self.norm2(x)  # Applying normalization
            return x  # Returning the output
            # return self.double_conv(x)

class Down(nn.Module):  # Defining a class for downsampling
    def __init__(self, in_channels, out_channels, emb_dim=256):  # Initializing the class
        super().__init__()  # Calling the parent class initializer
        self.maxpool_conv = nn.Sequential(  # Creating a sequential container for max pooling and convolution
            nn.MaxPool2d(2),  # Max pooling layer with a kernel size of 2
            DoubleConv(in_channels, in_channels, residual=True),  # First double convolution layer
            DoubleConv(in_channels, out_channels),  # Second double convolution layer
        )

        self.emb_layer = nn.Sequential(  # Creating a sequential container for embedding layer
            nn.SiLU(),  # SiLU activation function
            nn.Linear(  # Linear layer
                emb_dim,  # Input dimension
                out_channels  # Output dimension
            ),
        )

    def forward(self, x, t):  # Defining the forward pass
        x = self.maxpool_conv(x)  # Applying max pooling and convolution
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])  # Creating embeddings
        return x + emb  # Returning the output with added embeddings


class Up(nn.Module):  # Defining a class for upsampling
    def __init__(self, in_channels, out_channels, emb_dim=256):  # Initializing the class
        super().__init__()  # Calling the parent class initializer

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)  # Upsampling layer
        self.conv = nn.Sequential(  # Creating a sequential container for convolution
            DoubleConv(in_channels, in_channels, residual=True),  # First double convolution layer
            DoubleConv(in_channels, out_channels, in_channels // 2),  # Second double convolution layer
        )

        self.emb_layer = nn.Sequential(  # Creating a sequential container for embedding layer
            nn.SiLU(),  # SiLU activation function
            nn.Linear(  # Linear layer
                emb_dim,  # Input dimension
                out_channels  # Output dimension
            ),
        )

    def forward(self, x, skip_x, t):  # Defining the forward pass
        x = self.up(x)  # Applying upsampling
        x = torch.cat([skip_x, x], dim=1)  # Concatenating skip connections
        x = self.conv(x)  # Applying convolution
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])  # Creating embeddings
        return x + emb  # Returning the output with added embeddings

"""
Down- orignal
Down_F- uses DoubleConv_F
Down_FF- uses filteres during downsampling with normal DoubleConv
Down_FFF- uses filter with DoubleConv_F
"""
class Down_F(nn.Module):  # Defining a class for downsampling with filters
    def __init__(self, in_channels, out_channels, emb_dim=256, f_settings=None):  # Initializing the class
        super().__init__()  # Calling the parent class initializer
        self.f_settings = f_settings  # Storing filter settings
        self.maxpool_conv = nn.Sequential(  # Creating a sequential container for max pooling and convolution
            nn.MaxPool2d(2),  # Max pooling layer with a kernel size of 2
            DoubleConv_F(in_channels, in_channels, residual=True, f_settings=self.f_settings),  # First double convolution layer with filters
            DoubleConv_F(in_channels, out_channels, f_settings=self.f_settings),  # Second double convolution layer with filters
        )

        self.emb_layer = nn.Sequential(  # Creating a sequential container for embedding layer
            nn.SiLU(),  # SiLU activation function
            nn.Linear(  # Linear layer
                emb_dim,  # Input dimension
                out_channels  # Output dimension
            ),
        )

    def forward(self, x, t):  # Defining the forward pass
        x = self.maxpool_conv(x)  # Applying max pooling and convolution
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])  # Creating embeddings
        return x + emb  # Returning the output with added embeddings

class Up_F(nn.Module):  # Defining a class for upsampling with filters
    def __init__(self, in_channels, out_channels, emb_dim=256, f_settings=None):  # Initializing the class
        super().__init__()  # Calling the parent class initializer
        self.f_settings = f_settings  # Storing filter settings
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)  # Upsampling layer
        self.conv = nn.Sequential(  # Creating a sequential container for convolution
            DoubleConv_F(in_channels, in_channels, residual=True, f_settings=self.f_settings),  # First double convolution layer with filters
            DoubleConv_F(in_channels, out_channels, in_channels // 2, f_settings=self.f_settings),  # Second double convolution layer with filters
        )

        self.emb_layer = nn.Sequential(  # Creating a sequential container for embedding layer
            nn.SiLU(),  # SiLU activation function
            nn.Linear(  # Linear layer
                emb_dim,  # Input dimension
                out_channels  # Output dimension
            ),
        )

    def forward(self, x, skip_x, t):  # Defining the forward pass
        x = self.up(x)  # Applying upsampling
        x = torch.cat([skip_x, x], dim=1)  # Concatenating skip connections
        x = self.conv(x)  # Applying convolution
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])  # Creating embeddings
        return x + emb  # Returning the output with added embeddings

class Down_FF(nn.Module):  # Defining a class for downsampling with filters (FF)
    def __init__(self, in_channels, out_channels, emb_dim=256, f_settings=None):  # Initializing the class
        super().__init__()  # Calling the parent class initializer
        self.f_settings = f_settings  # Storing filter settings
        # Generate the 2D Jinc filter with Kaiser window
        self.jinc_filter = circularLowpassKernel(omega_c=self.f_settings['omega_c_down'],  # Creating a Jinc filter
                                                 N=self.f_settings['kernel_size'], 
                                                 beta=self.f_settings['kaiser_beta'])  # Using filter settings

        self.conv = nn.Sequential(  # Creating a sequential container for convolution
            DoubleConv(in_channels, in_channels, residual=True),  # First double convolution layer
            DoubleConv(in_channels, out_channels),  # Second double convolution layer
        )

        self.emb_layer = nn.Sequential(  # Creating a sequential container for embedding layer
            nn.SiLU(),  # SiLU activation function
            nn.Linear(  # Linear layer
                emb_dim,  # Input dimension
                out_channels  # Output dimension
            ),
        )

    def forward(self, x, t):  # Defining the forward pass
        # Downsample using the custom Jinc-based filter
        x = custom_downsample(x, self.jinc_filter)  # Downsampling using the Jinc filter
        x = self.conv(x)  # Applying convolution
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])  # Creating embeddings
        return x + emb  # Returning the output with added embeddings

class Up_FF(nn.Module):  # Defining a class for upsampling with filters (FF)
    def __init__(self, in_channels, out_channels, emb_dim=256, f_settings=None):  # Initializing the class
        super().__init__()  # Calling the parent class initializer
        self.f_settings = f_settings  # Storing filter settings
        # Generate the 2D sinc filter with Kaiser window
        self.sinc_filter = circularLowpassKernel(omega_c=self.f_settings['omega_c_up'],  # Creating a Sinc filter
                                                 N=self.f_settings['kernel_size'], 
                                                 beta=self.f_settings['kaiser_beta'])  # Using filter settings

        self.conv = nn.Sequential(  # Creating a sequential container for convolution
            DoubleConv(in_channels, in_channels, residual=True),  # First double convolution layer
            DoubleConv(in_channels, out_channels, in_channels // 2),  # Second double convolution layer
        )

        self.emb_layer = nn.Sequential(  # Creating a sequential container for embedding layer
            nn.SiLU(),  # SiLU activation function
            nn.Linear(  # Linear layer
                emb_dim,  # Input dimension
                out_channels  # Output dimension
            ),
        )

    def forward(self, x, skip_x, t):  # Defining the forward pass
        # Upsample using the custom filter
        x = custom_upsample(x, self.sinc_filter)  # Upsampling using the Sinc filter
        x = torch.cat([skip_x, x], dim=1)  # Concatenating skip connections
        x = self.conv(x)  # Applying convolution
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])  # Creating embeddings
        return x + emb  # Returning the output with added embeddings

class Down_FFF(nn.Module):  # Defining a class for downsampling with filters (FFF)
    def __init__(self, in_channels, out_channels, emb_dim=256, f_settings=None):  # Initializing the class
        super().__init__()  # Calling the parent class initializer
        self.f_settings = f_settings  # Storing filter settings
        # Generate the 2D Jinc filter with Kaiser window
        self.jinc_filter = circularLowpassKernel(omega_c=self.f_settings['omega_c_down'],  # Creating a Jinc filter
                                                 N=self.f_settings['kernel_size'], 
                                                 beta=self.f_settings['kaiser_beta'])  # Using filter settings

        self.conv = nn.Sequential(  # Creating a sequential container for convolution
            DoubleConv_F(in_channels, in_channels, residual=True, f_settings=self.f_settings),  # First double convolution layer with filters
            DoubleConv_F(in_channels, out_channels, f_settings=self.f_settings),  # Second double convolution layer with filters
        )

        self.emb_layer = nn.Sequential(  # Creating a sequential container for embedding layer
            nn.SiLU(),  # SiLU activation function
            nn.Linear(  # Linear layer
                emb_dim,  # Input dimension
                out_channels  # Output dimension
            ),
        )

    def forward(self, x, t):  # Defining the forward pass
        # Downsample using the custom Jinc-based filter
        x = custom_downsample(x, self.jinc_filter)  # Downsampling using the Jinc filter
        x = self.conv(x)  # Applying convolution
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])  # Creating embeddings
        return x + emb  # Returning the output with added embeddings

class Up_FFF(nn.Module):  # Defining a class for upsampling with filters (FFF)
    def __init__(self, in_channels, out_channels, emb_dim=256, f_settings=None):  # Initializing the class
        super().__init__()  # Calling the parent class initializer
        self.f_settings = f_settings  # Storing filter settings
        # Generate the 2D sinc filter with Kaiser window
        self.sinc_filter = circularLowpassKernel(omega_c=self.f_settings['omega_c_up'],  # Creating a Sinc filter
                                                 N=self.f_settings['kernel_size'], 
                                                 beta=self.f_settings['kaiser_beta'])  # Using filter settings

        self.conv = nn.Sequential(  # Creating a sequential container for convolution
            DoubleConv_F(in_channels, in_channels, residual=True, f_settings=self.f_settings),  # First double convolution layer with filters
            DoubleConv_F(in_channels, out_channels, in_channels // 2, f_settings=self.f_settings),  # Second double convolution layer with filters
        )

        self.emb_layer = nn.Sequential(  # Creating a sequential container for embedding layer
            nn.SiLU(),  # SiLU activation function
            nn.Linear(  # Linear layer
                emb_dim,  # Input dimension
                out_channels  # Output dimension
            ),
        )

    def forward(self, x, skip_x, t):  # Defining the forward pass
        # Upsample using the custom filter
        x = custom_upsample(x, self.sinc_filter)  # Upsampling using the Sinc filter
        x = torch.cat([skip_x, x], dim=1)  # Concatenating skip connections
        x = self.conv(x)  # Applying convolution
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])  # Creating embeddings
        return x + emb  # Returning the output with added embeddings

class Down_F4(nn.Module):  # Defining a class for downsampling with filters (F4)
    def __init__(self, in_channels, out_channels, emb_dim=256, f_settings=None):  # Initializing the class
        super().__init__()  # Calling the parent class initializer
        self.f_settings = f_settings  # Storing filter settings
        # Generate the 2D Jinc filter with Kaiser window
        self.jinc_filter = circularLowpassKernel(omega_c=self.f_settings['omega_c_down'],  # Creating a Jinc filter
                                                 N=self.f_settings['kernel_size'], 
                                                 beta=self.f_settings['kaiser_beta'])  # Using filter settings

        self.conv = nn.Sequential(  # Creating a sequential container for convolution
            DoubleConv_F4(in_channels, in_channels, residual=True, f_settings=self.f_settings),  # First double convolution layer with filters
            DoubleConv_F4(in_channels, out_channels, f_settings=self.f_settings),  # Second double convolution layer with filters
        )

        self.emb_layer = nn.Sequential(  # Creating a sequential container for embedding layer
            nn.SiLU(),  # SiLU activation function
            nn.Linear(  # Linear layer
                emb_dim,  # Input dimension
                out_channels  # Output dimension
            ),
        )
        self.norm1 = nn.GroupNorm(1, in_channels)  # Group normalization layer

    def forward(self, x, t):  # Defining the forward pass
        # Downsample using the custom Jinc-based filter
        x = custom_downsample(x, self.jinc_filter)  # Downsampling using the Jinc filter
        # x = self.norm1(x)  # Applying normalization (commented out)
        x = self.conv(x)  # Applying convolution
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])  # Creating embeddings
        return x + emb  # Returning the output with added embeddings

class Up_F4(nn.Module):  # Defining a class for upsampling with filters (F4)
    def __init__(self, in_channels, out_channels, emb_dim=256, f_settings=None):  # Initializing the class
        super().__init__()  # Calling the parent class initializer
        self.f_settings = f_settings  # Storing filter settings
        # Generate the 2D sinc filter with Kaiser window
        self.sinc_filter = circularLowpassKernel(omega_c=self.f_settings['omega_c_up'],  # Creating a Sinc filter
                                                 N=self.f_settings['kernel_size'], 
                                                 beta=self.f_settings['kaiser_beta'])  # Using filter settings

        self.conv = nn.Sequential(  # Creating a sequential container for convolution
            DoubleConv_F4(in_channels, in_channels, residual=True, f_settings=self.f_settings),  # First double convolution layer with filters
            DoubleConv_F4(in_channels, out_channels, in_channels // 2, f_settings=self.f_settings),  # Second double convolution layer with filters
        )

        self.emb_layer = nn.Sequential(  # Creating a sequential container for embedding layer
            nn.SiLU(),  # SiLU activation function
            nn.Linear(  # Linear layer
                emb_dim,  # Input dimension
                out_channels  # Output dimension
            ),
        )
        self.norm1 = nn.GroupNorm(1, in_channels // 2)  # Group normalization layer

    def forward(self, x, skip_x, t):  # Defining the forward pass
        # Upsample using the custom filter
        x = custom_upsample(x, self.sinc_filter)  # Upsampling using the Sinc filter
        # x = self.norm1(x)  # Applying normalization (commented out)
        x = torch.cat([skip_x, x], dim=1)  # Concatenating skip connections
        x = self.conv(x)  # Applying convolution
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])  # Creating embeddings
        return x + emb  # Returning the output with added embeddings
