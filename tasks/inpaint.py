"""
Inpainting task implementation.

Loss: E(x; x0) = ||(f_θ(z) - x0) ⊙ m||²

The network is trained only on visible pixels. The network's prior
causes it to fill missing regions with coherent content.
"""

import os
import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Dict, Any

from ..utils import (
    get_device, clear_gpu_memory, load_image, load_mask, save_image,
    get_noise_input, compute_psnr_torch, get_optimizer
)


def run_inpainting(
    model: nn.Module,
    corrupted_img_path: str,
    mask_path: str,
    output_path: str,
    clean_img_path: Optional[str] = None,
    num_iter: int = 8000,
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
    Run inpainting optimization.
    
    Args:
        model: Neural network (DIP, DeepDecoder, or Transformer)
        corrupted_img_path: Path to image with missing regions
        mask_path: Path to binary mask (1 = known pixels, 0 = missing)
        output_path: Path to save inpainted output
        clean_img_path: Path to clean reference (for PSNR computation)
        num_iter: Number of optimization iterations
        lr: Learning rate
        input_depth: Input noise channels (for DIP/Transformer)
        input_noise_std: Input perturbation std
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
    
    # Load corrupted image and mask
    y = load_image(corrupted_img_path, max_size=max_size, device=device)
    _, c, h, w = y.shape
    print(f"  Processing at resolution: {h}x{w}")
    
    mask = load_mask(mask_path, target_size=(h, w), device=device)
    
    # Expand mask to 3 channels
    mask_3ch = mask.expand(-1, c, -1, -1)
    
    # Load clean image for PSNR
    x_clean = None
    if clean_img_path and os.path.exists(clean_img_path):
        x_clean = load_image(clean_img_path, target_size=(h, w), device=device)
    
    # Setup model
    model = model.to(device)
    
    # Setup input
    if model_type == "deep_decoder":
        net_input = None
    else:
        net_input = get_noise_input(input_depth, (h, w), device, noise_type='uniform')
    
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
                out = model(current_input) if current_input is not None else model()
                loss = criterion(out * mask_3ch, y * mask_3ch)
                loss.backward()
                return loss
            loss = optimizer.step(closure)
            with torch.no_grad():
                out = model(current_input) if current_input is not None else model()
        elif scaler is not None:
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                out = model(current_input) if current_input is not None else model()
                loss = criterion(out * mask_3ch, y * mask_3ch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.zero_grad()
            out = model(current_input) if current_input is not None else model()
            loss = criterion(out * mask_3ch, y * mask_3ch)
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
    del model, optimizer, y, mask, mask_3ch, out, out_avg
    if x_clean is not None:
        del x_clean
    if net_input is not None:
        del net_input
    clear_gpu_memory()
    
    return final_psnr, logs
