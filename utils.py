"""
Utility functions for image processing and training.

Includes:
- Image loading/saving
- PSNR computation
- Noise generation
- Memory management
- Optimizer creation
"""

import os
import gc
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import Tuple, Optional, List


# =============================================================================
# Device and Memory Management
# =============================================================================

def get_device() -> str:
    """Get available device (CUDA or CPU)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def clear_gpu_memory():
    """Clear GPU memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# =============================================================================
# PSNR Computation
# =============================================================================

def compute_psnr_from_paths(path1: str, path2: str) -> float:
    """
    Compute PSNR between two images from file paths.
    
    Args:
        path1: Path to first image
        path2: Path to second image
    
    Returns:
        PSNR value in dB
    """
    img1 = Image.open(path1).convert("RGB")
    img2 = Image.open(path2).convert("RGB")
    
    if img1.size != img2.size:
        img2 = img2.resize(img1.size, Image.BICUBIC)
    
    x = np.asarray(img1).astype(np.float32) / 255.0
    y = np.asarray(img2).astype(np.float32) / 255.0
    
    mse = np.mean((x - y) ** 2)
    if mse == 0:
        return float("inf")
    
    return 10 * math.log10(1.0 / mse)


def compute_psnr_torch(x: torch.Tensor, y: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Compute PSNR between two tensors.
    
    Args:
        x: First tensor [B, C, H, W] in [0, max_val]
        y: Second tensor [B, C, H, W] in [0, max_val]
        max_val: Maximum pixel value
    
    Returns:
        PSNR value in dB
    """
    assert x.shape == y.shape, f"Shape mismatch: {x.shape} vs {y.shape}"
    mse = torch.mean((x.float() - y.float()) ** 2)
    if mse.item() == 0:
        return float("inf")
    return 10 * torch.log10(torch.tensor(max_val**2) / mse).item()


# =============================================================================
# Image Loading and Saving
# =============================================================================

def resize_to_max(img: Image.Image, max_size: int) -> Image.Image:
    """Resize image so largest dimension equals max_size."""
    if max_size is None:
        return img
    
    w, h = img.size
    if max(w, h) <= max_size:
        return img
    
    if w > h:
        new_w = max_size
        new_h = int(h * max_size / w)
    else:
        new_h = max_size
        new_w = int(w * max_size / h)
    
    # Make divisible by 32 for network compatibility
    new_w = max((new_w // 32) * 32, 32)
    new_h = max((new_h // 32) * 32, 32)
    
    return img.resize((new_w, new_h), Image.BICUBIC)


def load_image(
    path: str,
    max_size: Optional[int] = None,
    target_size: Optional[Tuple[int, int]] = None,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Load image as tensor.
    
    Args:
        path: Image file path
        max_size: Maximum dimension size
        target_size: Exact target size (H, W)
        device: Target device
    
    Returns:
        Image tensor [1, 3, H, W] in [0, 1]
    """
    img = Image.open(path).convert("RGB")
    
    if target_size is not None:
        img = img.resize((target_size[1], target_size[0]), Image.BICUBIC)
    elif max_size is not None:
        img = resize_to_max(img, max_size)
    
    tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
    return tensor


def load_mask(
    path: str,
    max_size: Optional[int] = None,
    target_size: Optional[Tuple[int, int]] = None,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Load binary mask as tensor.
    
    Args:
        path: Mask file path
        max_size: Maximum dimension size
        target_size: Exact target size (H, W)
        device: Target device
    
    Returns:
        Mask tensor [1, 1, H, W] with values {0, 1}
    """
    mask = Image.open(path).convert("L")
    
    if target_size is not None:
        mask = mask.resize((target_size[1], target_size[0]), Image.NEAREST)
    elif max_size is not None:
        mask = resize_to_max(mask, max_size)
    
    tensor = transforms.ToTensor()(mask).unsqueeze(0).to(device)
    tensor = (tensor > 0.5).float()
    return tensor


def save_image(tensor: torch.Tensor, path: str):
    """
    Save tensor as image file.
    
    Args:
        tensor: Image tensor [1, C, H, W] in [0, 1]
        path: Output file path
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    img = tensor.squeeze(0).clamp(0, 1).cpu()
    img = transforms.ToPILImage()(img)
    img.save(path)


# =============================================================================
# Noise Generation
# =============================================================================

def get_noise_input(
    channels: int,
    size: Tuple[int, int],
    device: str = "cpu",
    noise_type: str = "uniform"
) -> torch.Tensor:
    """
    Generate noise input tensor for network.
    
    Args:
        channels: Number of channels
        size: Spatial size (H, W)
        device: Target device
        noise_type: "uniform" for U(0, 0.1) or "normal" for N(0, 1)
    
    Returns:
        Noise tensor [1, channels, H, W]
    """
    shape = (1, channels, size[0], size[1])
    
    if noise_type == "uniform":
        # Paper uses U(0, 0.1)
        noise = torch.rand(shape, device=device) * 0.1
    else:
        noise = torch.randn(shape, device=device)
    
    return noise


# =============================================================================
# Downsampling for Super-Resolution
# =============================================================================

def lanczos_downsample(img: torch.Tensor, factor: int) -> torch.Tensor:
    """
    Downsample image (approximation of Lanczos).
    
    Args:
        img: Image tensor [B, C, H, W]
        factor: Downsampling factor
    
    Returns:
        Downsampled tensor
    """
    _, _, h, w = img.shape
    new_h, new_w = h // factor, w // factor
    return F.interpolate(img, size=(new_h, new_w), mode='bicubic', align_corners=False)


# =============================================================================
# Optimizer Creation
# =============================================================================

def get_optimizer(
    model: nn.Module,
    optimizer_type: str = "adam",
    lr: float = 0.01,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
    betas: Tuple[float, float] = (0.9, 0.999),
    extra_params: Optional[List[torch.Tensor]] = None
) -> Tuple[torch.optim.Optimizer, bool]:
    """
    Create optimizer for training.
    
    Args:
        model: Neural network
        optimizer_type: "adam", "adamw", "sgd", "rmsprop", "lbfgs"
        lr: Learning rate
        weight_decay: L2 regularization
        momentum: Momentum for SGD/RMSprop
        betas: Beta parameters for Adam/AdamW
        extra_params: Additional parameters to optimize (e.g., learnable input)
    
    Returns:
        (optimizer, is_lbfgs) tuple
    """
    params = list(model.parameters())
    if extra_params:
        params = params + extra_params
    
    optimizer_type = optimizer_type.lower()
    
    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)
        is_lbfgs = False
    elif optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)
        is_lbfgs = False
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        is_lbfgs = False
    elif optimizer_type == "rmsprop":
        optimizer = torch.optim.RMSprop(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        is_lbfgs = False
    elif optimizer_type == "lbfgs":
        optimizer = torch.optim.LBFGS(params, lr=lr, history_size=10, max_iter=20, line_search_fn='strong_wolfe')
        is_lbfgs = True
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    return optimizer, is_lbfgs


# =============================================================================
# Model Utilities
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_model(model: nn.Module):
    """Freeze all model parameters."""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model: nn.Module):
    """Unfreeze all model parameters."""
    for param in model.parameters():
        param.requires_grad = True
