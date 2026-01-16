"""
Denoising task implementation.

Loss: E(x; x0) = ||f_θ(z) - x0||²

The network learns to fit the noisy image, but due to CNN's inductive bias,
it learns the clean structure before fitting the noise.
"""

import os
import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Dict, Any

from ..utils import (
    get_device, clear_gpu_memory, load_image, save_image,
    get_noise_input, compute_psnr_torch, get_optimizer
)


def run_denoising(
    model: nn.Module,
    noisy_img_path: str,
    output_path: str,
    clean_img_path: Optional[str] = None,
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
    Run denoising optimization.
    
    Args:
        model: Neural network (DIP, DeepDecoder, or Transformer)
        noisy_img_path: Path to noisy input image
        output_path: Path to save denoised output
        clean_img_path: Path to clean reference (for PSNR computation)
        num_iter: Number of optimization iterations
        lr: Learning rate
        input_depth: Input noise channels (for DIP/Transformer)
        input_noise_std: Input perturbation std (paper: 1/30)
        max_size: Maximum image size
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
    
    # Load noisy image
    x0 = load_image(noisy_img_path, max_size=max_size, device=device)
    _, c, h, w = x0.shape
    print(f"  Processing at resolution: {h}x{w}")
    
    # Load clean image for PSNR
    x_clean = None
    if clean_img_path and os.path.exists(clean_img_path):
        x_clean = load_image(clean_img_path, target_size=(h, w), device=device)
    
    # Setup model
    model = model.to(device)
    
    # Setup input based on model type
    if model_type == "deep_decoder":
        # Deep Decoder uses learnable input
        net_input = None
        extra_params = [p for p in model.parameters() if p.requires_grad]
        extra_params = None  # Already included in model
    else:
        # DIP and Transformer use fixed noise input with perturbation
        net_input = get_noise_input(input_depth, (h, w), device, noise_type='uniform')
        extra_params = None
    
    # Setup optimizer
    optimizer, is_lbfgs = get_optimizer(
        model, optimizer_type, lr, 
        extra_params=extra_params,
        **optimizer_kwargs
    )
    criterion = nn.MSELoss()
    
    # Mixed precision
    use_scaler = use_amp and device == 'cuda' and not is_lbfgs
    scaler = torch.amp.GradScaler('cuda') if use_scaler else None
    
    # Tracking
    out_avg = None
    logs = {'iterations': [], 'loss': [], 'psnr': []}
    
    for i in range(num_iter):
        # Get input (with perturbation for DIP/Transformer)
        if model_type == "deep_decoder":
            current_input = None
        else:
            current_input = net_input + input_noise_std * torch.randn_like(net_input)
        
        # Forward pass
        if is_lbfgs:
            def closure():
                optimizer.zero_grad()
                out = model(current_input) if current_input is not None else model()
                loss = criterion(out, x0)
                loss.backward()
                return loss
            loss = optimizer.step(closure)
            with torch.no_grad():
                out = model(current_input) if current_input is not None else model()
        elif scaler is not None:
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                out = model(current_input) if current_input is not None else model()
                loss = criterion(out, x0)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.zero_grad()
            out = model(current_input) if current_input is not None else model()
            loss = criterion(out, x0)
            loss.backward()
            optimizer.step()
        
        # Exponential averaging
        with torch.no_grad():
            if out_avg is None:
                out_avg = out.detach().clone()
            else:
                out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)
        
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
    del model, optimizer, x0, out, out_avg
    if x_clean is not None:
        del x_clean
    if net_input is not None:
        del net_input
    clear_gpu_memory()
    
    return final_psnr, logs
