"""
Super-resolution task implementation.

Loss: E(x; x0) = ||d(f_θ(z)) - x0||²

The network generates a high-resolution image that, when downsampled,
matches the low-resolution input.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from typing import Tuple, Optional, List, Dict, Any

from ..utils import (
    get_device, clear_gpu_memory, load_image, save_image,
    get_noise_input, compute_psnr_torch, get_optimizer, lanczos_downsample
)


def run_superresolution(
    model: nn.Module,
    lowres_img_path: str,
    output_path: str,
    clean_img_path: Optional[str] = None,
    up_factor: int = 4,
    num_iter: int = 5000,
    lr: float = 0.01,
    input_depth: int = 32,
    input_noise_std: float = 1/30,
    max_size: int = 512,
    log_every: int = 100,
    use_amp: bool = True,
    optimizer_type: str = "adam",
    exp_weight: float = 0.99,
    seed: int = 42,
    model_type: str = "dip",
    **optimizer_kwargs
) -> Tuple[Optional[float], Dict[str, List]]:
    """
    Run super-resolution optimization.
    
    Args:
        model: Neural network (DIP, DeepDecoder, or Transformer)
        lowres_img_path: Path to low-resolution input image
        output_path: Path to save high-resolution output
        clean_img_path: Path to clean HR reference (for PSNR computation)
        up_factor: Upscaling factor (e.g., 4 for 4x)
        num_iter: Number of optimization iterations
        lr: Learning rate
        input_depth: Input noise channels (for DIP/Transformer)
        input_noise_std: Input perturbation std
        max_size: Maximum output image size
        log_every: Logging interval
        use_amp: Use mixed precision training
        optimizer_type: Optimizer type
        exp_weight: Exponential averaging weight
        seed: Random seed
        model_type: "dip", "deep_decoder", or "transformer"
        **optimizer_kwargs: Additional optimizer arguments
    
    Returns:
        (final_psnr, logs_dict) tuple
    """
    device = get_device()
    torch.manual_seed(seed)
    clear_gpu_memory()
    
    # Load low-res image
    lr_img = Image.open(lowres_img_path).convert("RGB")
    lr_w, lr_h = lr_img.size
    
    # Calculate high-res size
    hr_h, hr_w = lr_h * up_factor, lr_w * up_factor
    
    # Limit to max size
    if max_size and max(hr_h, hr_w) > max_size:
        scale = max_size / max(hr_h, hr_w)
        hr_h = int(hr_h * scale)
        hr_w = int(hr_w * scale)
        # Make divisible by 32
        hr_h = max((hr_h // 32) * 32, 32)
        hr_w = max((hr_w // 32) * 32, 32)
        # Recalculate LR size
        lr_h, lr_w = hr_h // up_factor, hr_w // up_factor
    
    print(f"  LR: {lr_h}x{lr_w} -> HR: {hr_h}x{hr_w}")
    
    # Load and resize LR image to target LR size
    y_lr = load_image(lowres_img_path, target_size=(lr_h, lr_w), device=device)
    
    # Load clean HR image for PSNR
    x_clean = None
    if clean_img_path and os.path.exists(clean_img_path):
        x_clean = load_image(clean_img_path, target_size=(hr_h, hr_w), device=device)
    
    # Setup model
    model = model.to(device)
    
    # Setup input
    if model_type == "deep_decoder":
        net_input = None
    else:
        net_input = get_noise_input(input_depth, (hr_h, hr_w), device, noise_type='uniform')
    
    # Setup optimizer
    optimizer, is_lbfgs = get_optimizer(model, optimizer_type, lr, **optimizer_kwargs)
    criterion = nn.MSELoss()
    
    # Mixed precision
    use_scaler = use_amp and device == 'cuda' and not is_lbfgs
    scaler = torch.amp.GradScaler('cuda') if use_scaler else None
    
    # Tracking
    out_avg = None
    logs = {'iterations': [], 'loss': [], 'psnr': []}
    
    for i in range(num_iter):
        # Get input
        if model_type == "deep_decoder":
            current_input = None
        else:
            current_input = net_input + input_noise_std * torch.randn_like(net_input)
        
        # Forward pass
        if is_lbfgs:
            def closure():
                optimizer.zero_grad()
                out_hr = model(current_input) if current_input is not None else model()
                out_lr = lanczos_downsample(out_hr, up_factor)
                loss = criterion(out_lr, y_lr)
                loss.backward()
                return loss
            loss = optimizer.step(closure)
            with torch.no_grad():
                out_hr = model(current_input) if current_input is not None else model()
        elif scaler is not None:
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                out_hr = model(current_input) if current_input is not None else model()
                out_lr = lanczos_downsample(out_hr, up_factor)
                loss = criterion(out_lr, y_lr)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.zero_grad()
            out_hr = model(current_input) if current_input is not None else model()
            out_lr = lanczos_downsample(out_hr, up_factor)
            loss = criterion(out_lr, y_lr)
            loss.backward()
            optimizer.step()
        
        # Exponential averaging
        with torch.no_grad():
            if out_avg is None:
                out_avg = out_hr.detach().clone()
            else:
                out_avg = out_avg * exp_weight + out_hr.detach() * (1 - exp_weight)
        
        # Logging
        if (i + 1) % log_every == 0 or i == 0:
            logs['iterations'].append(i + 1)
            logs['loss'].append(loss.item())
            
            if x_clean is not None:
                psnr = compute_psnr_torch(out_avg, x_clean)
                logs['psnr'].append(psnr)
            else:
                logs['psnr'].append(None)
    
    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_image(out_avg, output_path)
    
    # Final PSNR
    final_psnr = None
    if x_clean is not None:
        final_psnr = compute_psnr_torch(out_avg, x_clean)
        print(f"  Final PSNR: {final_psnr:.2f} dB")
    
    # Cleanup
    del model, optimizer, y_lr, out_hr, out_avg
    if x_clean is not None:
        del x_clean
    if net_input is not None:
        del net_input
    clear_gpu_memory()
    
    return final_psnr, logs
