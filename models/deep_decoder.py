"""
Deep Decoder Network Architecture.

Paper: "Deep Decoder: Concise Image Representations from Untrained 
        Non-convolutional Networks" (Heckel & Hand, ICLR 2019)

Architecture: Decoder-only with 1x1 convolutions
- ~60-100K parameters (20-30x smaller than DIP)
- Only 1x1 convolutions (channel mixing, no spatial filtering)
- Tiny learnable input (k × 4 × 4)
- Bilinear upsampling between layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import math


class DecoderBlock(nn.Module):
    """
    Single decoder block: Upsample → 1x1 Conv → BN → ReLU
    
    Uses only 1x1 convolutions for channel mixing (no spatial filtering).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample_factor: int = 2,
        upsample_mode: str = 'bilinear'
    ):
        super().__init__()
        
        self.upsample_factor = upsample_factor
        self.upsample_mode = upsample_mode
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample_factor > 1:
            x = F.interpolate(
                x, 
                scale_factor=self.upsample_factor, 
                mode=self.upsample_mode, 
                align_corners=False
            )
        return self.conv(x)


class DeepDecoder(nn.Module):
    """
    Deep Decoder Network.
    
    Key differences from Deep Image Prior:
    - No encoder (decoder only)
    - Only 1x1 convolutions (no spatial filtering)
    - Tiny input tensor (k × 4 × 4) that is learnable
    - ~60-100K parameters vs ~2M for DIP
    
    The network structure enforces smoothness through:
    1. Tiny input cannot encode pixel-level details
    2. Bilinear upsampling produces smooth interpolation
    3. 1x1 convolutions only mix channels, no spatial patterns
    
    Args:
        input_channels: Channels in the tiny input tensor
        input_spatial_size: Spatial size of input (e.g., 4 for 4x4)
        channels: List of channel counts for each decoder layer
        output_channels: Output image channels (3 for RGB)
        output_size: Target output resolution (H, W)
        learnable_input: Whether to optimize the input tensor
    """
    def __init__(
        self,
        input_channels: int = 256,
        input_spatial_size: int = 4,
        channels: List[int] = None,
        output_channels: int = 3,
        output_size: tuple = (256, 256),
        learnable_input: bool = True,
        upsample_mode: str = 'bilinear'
    ):
        super().__init__()
        
        if channels is None:
            channels = [256, 256, 128, 128, 64, 64, 32, 32]
        
        self.input_spatial_size = input_spatial_size
        self.output_size = output_size
        self.learnable_input = learnable_input
        
        # Calculate number of upsampling stages needed
        num_upsamples = int(math.log2(output_size[0] // input_spatial_size))
        
        # Adjust channels list to match required upsamples
        if len(channels) < num_upsamples:
            # Extend with last channel value
            channels = channels + [channels[-1]] * (num_upsamples - len(channels))
        channels = channels[:num_upsamples]
        
        # Create decoder blocks
        self.blocks = nn.ModuleList()
        in_ch = input_channels
        
        for out_ch in channels:
            self.blocks.append(
                DecoderBlock(in_ch, out_ch, upsample_factor=2, upsample_mode=upsample_mode)
            )
            in_ch = out_ch
        
        # Output layer (1x1 conv to RGB)
        self.output_conv = nn.Sequential(
            nn.Conv2d(channels[-1], output_channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )
        
        # Initialize input tensor
        self.register_buffer('_input_tensor', torch.rand(1, input_channels, input_spatial_size, input_spatial_size) * 0.1)
        
        if learnable_input:
            self.input_tensor = nn.Parameter(
                torch.rand(1, input_channels, input_spatial_size, input_spatial_size) * 0.1
            )
        else:
            self.input_tensor = None
    
    def get_input(self, batch_size: int = 1) -> torch.Tensor:
        """Get the input tensor (learnable or fixed)."""
        if self.learnable_input and self.input_tensor is not None:
            z = self.input_tensor
        else:
            z = self._input_tensor
        
        if batch_size > 1:
            z = z.expand(batch_size, -1, -1, -1)
        
        return z
    
    def forward(self, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            z: Input tensor. If None, uses internal (learnable) input.
        
        Returns:
            Output image tensor
        """
        if z is None:
            z = self.get_input()
        
        x = z
        for block in self.blocks:
            x = block(x)
        
        # Ensure output matches target size
        if x.shape[-2:] != self.output_size:
            x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False)
        
        x = self.output_conv(x)
        return x


def get_deep_decoder_network(
    output_size: tuple = (256, 256),
    input_channels: int = 256,
    input_spatial_size: int = 4,
    channels: List[int] = None,
    output_channels: int = 3,
    learnable_input: bool = True
) -> DeepDecoder:
    """
    Create Deep Decoder network.
    
    Args:
        output_size: Target output resolution (H, W)
        input_channels: Channels in tiny input tensor
        input_spatial_size: Spatial size of input
        channels: Channel progression for decoder layers
        output_channels: Output channels (3 for RGB)
        learnable_input: Whether to optimize input tensor
    
    Returns:
        Configured DeepDecoder
    """
    if channels is None:
        channels = [256, 256, 128, 128, 64, 64, 32, 32]
    
    return DeepDecoder(
        input_channels=input_channels,
        input_spatial_size=input_spatial_size,
        channels=channels,
        output_channels=output_channels,
        output_size=output_size,
        learnable_input=learnable_input,
    )
