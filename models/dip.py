"""
Deep Image Prior Network Architecture.

Paper: "Deep Image Prior" (Ulyanov, Vedaldi, & Lempitsky, CVPR 2018)

Architecture: 5-level U-Net encoder-decoder with skip connections
- ~2 million parameters
- Strided convolution for downsampling
- Bilinear upsampling + convolution for upsampling
- LeakyReLU activation, BatchNorm, reflection padding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


def get_activation(act_fun: str = 'LeakyReLU') -> nn.Module:
    """Get activation function by name."""
    activations = {
        'LeakyReLU': nn.LeakyReLU(0.2, inplace=True),
        'ReLU': nn.ReLU(inplace=True),
        'ELU': nn.ELU(inplace=True),
    }
    return activations.get(act_fun, nn.LeakyReLU(0.2, inplace=True))


class DownsampleBlock(nn.Module):
    """
    Downsampling block using strided convolution.
    
    Structure: Conv(stride=2) → BN → Act → Conv → BN → Act
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
        pad: str = 'reflection',
        act_fun: str = 'LeakyReLU'
    ):
        super().__init__()
        
        pad_fn = nn.ReflectionPad2d if pad == 'reflection' else nn.ZeroPad2d
        padding = kernel_size // 2
        
        self.block = nn.Sequential(
            # Strided conv for downsampling
            pad_fn(padding),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=0, bias=bias),
            nn.BatchNorm2d(out_channels),
            get_activation(act_fun),
            # Second conv
            pad_fn(padding),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=0, bias=bias),
            nn.BatchNorm2d(out_channels),
            get_activation(act_fun),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpsampleBlock(nn.Module):
    """
    Upsampling block using bilinear interpolation + convolution.
    
    Structure: Bilinear↑ → Conv → BN → Act → Conv(1x1) → BN → Act
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
        pad: str = 'reflection',
        act_fun: str = 'LeakyReLU',
        upsample_mode: str = 'bilinear'
    ):
        super().__init__()
        
        self.upsample_mode = upsample_mode
        pad_fn = nn.ReflectionPad2d if pad == 'reflection' else nn.ZeroPad2d
        padding = kernel_size // 2
        
        self.conv_block = nn.Sequential(
            pad_fn(padding),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=bias),
            nn.BatchNorm2d(out_channels),
            get_activation(act_fun),
        )
        
        # 1x1 convolution (paper: need1x1_up=True)
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, stride=1, padding=0, bias=bias),
            nn.BatchNorm2d(out_channels),
            get_activation(act_fun),
        )

    def forward(self, x: torch.Tensor, output_size: tuple) -> torch.Tensor:
        x = F.interpolate(x, size=output_size, mode=self.upsample_mode, align_corners=False)
        x = self.conv_block(x)
        x = self.conv_1x1(x)
        return x


class SkipConnection(nn.Module):
    """
    Skip connection using 1x1 convolution for channel reduction.
    
    Structure: Conv(1x1) → BN → Act
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        act_fun: str = 'LeakyReLU'
    ):
        super().__init__()
        
        if out_channels > 0:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=bias),
                nn.BatchNorm2d(out_channels),
                get_activation(act_fun),
            )
        else:
            self.skip = None

    def forward(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        return self.skip(x) if self.skip is not None else None


class DeepImagePriorNet(nn.Module):
    """
    Deep Image Prior Network - Paper-correct implementation.
    
    Default architecture (from paper):
    - num_channels_down = num_channels_up = [128, 128, 128, 128, 128]
    - num_channels_skip = [4, 4, 4, 4, 4]
    - ~2 million parameters
    
    Args:
        num_input_channels: Input noise channels (default: 32)
        num_output_channels: Output image channels (default: 3 for RGB)
        num_channels_down: Channels at each encoder level
        num_channels_up: Channels at each decoder level
        num_channels_skip: Skip connection channels at each level
        filter_size_down: Kernel size for encoder
        filter_size_up: Kernel size for decoder
        need_sigmoid: Apply sigmoid to output
        upsample_mode: Upsampling method ('bilinear' or 'nearest')
        act_fun: Activation function name
    """
    def __init__(
        self,
        num_input_channels: int = 32,
        num_output_channels: int = 3,
        num_channels_down: List[int] = None,
        num_channels_up: List[int] = None,
        num_channels_skip: List[int] = None,
        filter_size_down: int = 3,
        filter_size_up: int = 3,
        need_sigmoid: bool = True,
        need_bias: bool = True,
        pad: str = 'reflection',
        upsample_mode: str = 'bilinear',
        act_fun: str = 'LeakyReLU'
    ):
        super().__init__()
        
        # Default paper architecture
        if num_channels_down is None:
            num_channels_down = [128, 128, 128, 128, 128]
        if num_channels_up is None:
            num_channels_up = [128, 128, 128, 128, 128]
        if num_channels_skip is None:
            num_channels_skip = [4, 4, 4, 4, 4]
        
        self.num_levels = len(num_channels_down)
        self.need_sigmoid = need_sigmoid
        
        # Input layer
        pad_fn = nn.ReflectionPad2d if pad == 'reflection' else nn.ZeroPad2d
        padding = filter_size_down // 2
        
        self.input_conv = nn.Sequential(
            pad_fn(padding),
            nn.Conv2d(num_input_channels, num_channels_down[0], filter_size_down,
                      stride=1, padding=0, bias=need_bias),
            nn.BatchNorm2d(num_channels_down[0]),
            get_activation(act_fun),
            pad_fn(padding),
            nn.Conv2d(num_channels_down[0], num_channels_down[0], filter_size_down,
                      stride=1, padding=0, bias=need_bias),
            nn.BatchNorm2d(num_channels_down[0]),
            get_activation(act_fun),
        )
        
        # Encoder
        self.encoders = nn.ModuleList([
            DownsampleBlock(
                num_channels_down[i-1], num_channels_down[i],
                kernel_size=filter_size_down, bias=need_bias, pad=pad, act_fun=act_fun
            )
            for i in range(1, self.num_levels)
        ])
        
        # Skip connections
        self.skips = nn.ModuleList([
            SkipConnection(
                num_channels_down[0] if i == 0 else num_channels_down[i],
                num_channels_skip[i], bias=need_bias, act_fun=act_fun
            )
            for i in range(self.num_levels)
        ])
        
        # Decoder
        self.decoders = nn.ModuleList([
            UpsampleBlock(
                num_channels_up[i] + num_channels_skip[i],
                num_channels_up[i-1],
                kernel_size=filter_size_up, bias=need_bias, pad=pad,
                act_fun=act_fun, upsample_mode=upsample_mode
            )
            for i in range(self.num_levels - 1, 0, -1)
        ])
        
        # Output layer
        final_in_ch = num_channels_up[0] + num_channels_skip[0]
        self.output_conv = nn.Conv2d(final_in_ch, num_output_channels, 1, stride=1, padding=0, bias=need_bias)
        
        self.num_channels_up = num_channels_up
        self.num_channels_skip = num_channels_skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder_outputs = []
        sizes = []
        
        # Input convolution
        x = self.input_conv(x)
        encoder_outputs.append(x)
        sizes.append(x.shape[-2:])
        
        # Encoder path
        for encoder in self.encoders:
            x = encoder(x)
            encoder_outputs.append(x)
            sizes.append(x.shape[-2:])
        
        # Skip connection at deepest level
        skip_out = self.skips[-1](encoder_outputs[-1])
        if skip_out is not None:
            x = torch.cat([x, skip_out], dim=1)
        
        # Decoder path
        for i, decoder in enumerate(self.decoders):
            enc_idx = self.num_levels - 2 - i
            x = decoder(x, sizes[enc_idx])
            
            skip_out = self.skips[enc_idx](encoder_outputs[enc_idx])
            if skip_out is not None:
                x = torch.cat([x, skip_out], dim=1)
        
        # Output
        x = self.output_conv(x)
        
        if self.need_sigmoid:
            x = torch.sigmoid(x)
        
        return x


def get_dip_network(
    input_depth: int = 32,
    output_depth: int = 3,
    num_channels: int = 128,
    num_levels: int = 5,
    skip_channels: int = 4
) -> DeepImagePriorNet:
    """
    Create Deep Image Prior network with paper-correct defaults.
    
    Args:
        input_depth: Input noise channels
        output_depth: Output image channels
        num_channels: Channels at each encoder/decoder level
        num_levels: Number of encoder-decoder levels
        skip_channels: Channels for skip connections
    
    Returns:
        Configured DeepImagePriorNet
    """
    return DeepImagePriorNet(
        num_input_channels=input_depth,
        num_output_channels=output_depth,
        num_channels_down=[num_channels] * num_levels,
        num_channels_up=[num_channels] * num_levels,
        num_channels_skip=[skip_channels] * num_levels,
    )


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
