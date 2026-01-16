"""
Transformer-based Image Prior Architectures.

This module explores whether Transformers can serve as image priors.
Includes three architectures:
1. PureTransformer: Standard ViT with global self-attention
2. LocalTransformer: Swin-style with window-based attention
3. HybridCNNTransformer: CNN encoder/decoder with Transformer bottleneck

Hypothesis: CNN inductive biases (locality, translation equivariance) are
essential for image priors. Transformers lack these biases and may overfit.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


# =============================================================================
# Building Blocks
# =============================================================================

class PatchEmbed(nn.Module):
    """Convert image to patch embeddings."""
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 8,
        in_channels: int = 32,
        embed_dim: int = 256
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) → (B, num_patches, embed_dim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class PatchUnembed(nn.Module):
    """Convert patch embeddings back to image."""
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 8,
        out_channels: int = 3,
        embed_dim: int = 256
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.out_channels = out_channels
        self.proj = nn.Linear(embed_dim, patch_size * patch_size * out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        x = self.proj(x)
        x = x.reshape(B, self.grid_size, self.grid_size, 
                      self.patch_size, self.patch_size, self.out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.reshape(B, self.out_channels, self.img_size, self.img_size)
        return x


class MultiHeadAttention(nn.Module):
    """Standard multi-head self-attention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention (Swin-style)."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: int,
        dropout: float = 0.0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
        # Create relative position index
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), 
            torch.arange(window_size), 
            indexing='ij'
        ))
        coords_flatten = coords.flatten(1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1).permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    """Feed-forward network."""
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Standard Transformer block with global attention."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class WindowTransformerBlock(nn.Module):
    """Transformer block with window attention (Swin-style)."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        shift: bool = False
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2 if shift else 0
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = WindowAttention(embed_dim, num_heads, window_size, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
    
    def window_partition(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x = x.view(B, H, W, C)
        
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        
        x = x.view(B, H // self.window_size, self.window_size,
                   W // self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(-1, self.window_size * self.window_size, C)
        return x
    
    def window_reverse(self, x: torch.Tensor, H: int, W: int, B: int) -> torch.Tensor:
        x = x.view(B, H // self.window_size, W // self.window_size,
                   self.window_size, self.window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H, W, -1)
        
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        
        x = x.view(B, H * W, -1)
        return x
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        shortcut = x
        x = self.norm1(x)
        
        x = self.window_partition(x, H, W)
        x = self.attn(x)
        x = self.window_reverse(x, H, W, B)
        
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


# =============================================================================
# Full Architectures
# =============================================================================

class PureTransformer(nn.Module):
    """
    Pure Vision Transformer for image prior.
    
    WARNING: Expected to overfit to noise quickly due to lack of locality bias.
    """
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 8,
        in_channels: int = 32,
        out_channels: int = 3,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        mlp_ratio: float = 2.0
    ):
        super().__init__()
        
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.patch_unembed = PatchUnembed(img_size, patch_size, out_channels, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x + self.pos_embed
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        x = self.patch_unembed(x)
        x = torch.sigmoid(x)
        return x


class LocalTransformer(nn.Module):
    """
    Transformer with local (window) attention - Swin-style.
    
    Reintroduces some locality bias through windowed attention.
    """
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 8,
        in_channels: int = 32,
        out_channels: int = 3,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        mlp_ratio: float = 2.0,
        window_size: int = 4
    ):
        super().__init__()
        
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        self.grid_size = img_size // patch_size
        num_patches = self.patch_embed.num_patches
        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Alternating window attention blocks
        self.blocks = nn.ModuleList([
            WindowTransformerBlock(
                embed_dim, num_heads, window_size, mlp_ratio,
                shift=(i % 2 == 1)
            )
            for i in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.patch_unembed = PatchUnembed(img_size, patch_size, out_channels, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x + self.pos_embed
        
        for block in self.blocks:
            x = block(x, self.grid_size, self.grid_size)
        
        x = self.norm(x)
        x = self.patch_unembed(x)
        x = torch.sigmoid(x)
        return x


class HybridCNNTransformer(nn.Module):
    """
    Hybrid: CNN encoder/decoder with Transformer bottleneck.
    
    Keeps CNN's locality bias while adding global context via Transformer.
    This is expected to work best among transformer variants for image priors.
    """
    def __init__(
        self,
        img_size: int = 256,
        in_channels: int = 32,
        out_channels: int = 3,
        base_channels: int = 64,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        mlp_ratio: float = 2.0
    ):
        super().__init__()
        
        def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            )
        
        # CNN Encoder
        self.enc1 = conv_block(in_channels, base_channels)
        self.enc2 = conv_block(base_channels, base_channels * 2)
        self.enc3 = conv_block(base_channels * 2, base_channels * 4)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Bottleneck size after 3 poolings
        bottleneck_channels = base_channels * 4
        
        # Project to transformer dimension
        self.to_transformer = nn.Conv2d(bottleneck_channels, embed_dim, 1)
        
        # Transformer bottleneck
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])
        
        # Project back
        self.from_transformer = nn.Conv2d(embed_dim, bottleneck_channels, 1)
        
        # CNN Decoder
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = conv_block(base_channels * 4 + base_channels * 4, base_channels * 4)
        self.dec2 = conv_block(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = conv_block(base_channels * 2 + base_channels, base_channels)
        
        self.final = nn.Conv2d(base_channels, out_channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # Bottleneck with transformer
        b = self.pool(e3)
        b = self.to_transformer(b)
        
        B, C, H, W = b.shape
        b = b.flatten(2).transpose(1, 2)
        
        for block in self.transformer_blocks:
            b = block(b)
        
        b = b.transpose(1, 2).view(B, C, H, W)
        b = self.from_transformer(b)
        
        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        
        out = torch.sigmoid(self.final(d1))
        return out


def get_transformer_network(
    architecture: str = 'hybrid',
    img_size: int = 256,
    in_channels: int = 32,
    out_channels: int = 3,
    **kwargs
) -> nn.Module:
    """
    Create transformer-based network.
    
    Args:
        architecture: 'pure', 'local', or 'hybrid'
        img_size: Input/output image size
        in_channels: Input channels
        out_channels: Output channels
        **kwargs: Architecture-specific parameters
    
    Returns:
        Configured transformer network
    """
    if architecture == 'pure':
        return PureTransformer(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            **kwargs
        )
    elif architecture == 'local':
        return LocalTransformer(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            **kwargs
        )
    elif architecture == 'hybrid':
        return HybridCNNTransformer(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
