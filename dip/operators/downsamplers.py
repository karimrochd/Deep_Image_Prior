import torch
import torch.nn.functional as F

def differentiable_downsample(x, scale):
    # Bilinear downsampling (differentiable)
    if scale == 1:
        return x
    H, W = x.shape[-2:]
    h, w = H//scale, W//scale
    return F.interpolate(x, size=(h,w), mode='bilinear', align_corners=False)
