import torch
import torch.nn.functional as F
import math

def psnr(x, y, max_val=1.0, eps=1e-8):
    mse = F.mse_loss(x, y)
    if mse.item() == 0:
        return torch.tensor(99.0, device=x.device)
    return 20.0*torch.log10(torch.tensor(max_val, device=x.device)) - 10.0*torch.log10(mse + eps)

def ssim_simple(x, y, max_val=1.0):
    # Minimal SSIM (Gaussian window); not optimized
    # expects [1,C,H,W], C in {1,3}
    K1, K2 = 0.01, 0.03
    C1 = (K1*max_val)**2
    C2 = (K2*max_val)**2
    mu_x = F.avg_pool2d(x, 7, 1, 3)
    mu_y = F.avg_pool2d(y, 7, 1, 3)
    sigma_x = F.avg_pool2d(x*x, 7, 1, 3) - mu_x*mu_x
    sigma_y = F.avg_pool2d(y*y, 7, 1, 3) - mu_y*mu_y
    sigma_xy = F.avg_pool2d(x*y, 7, 1, 3) - mu_x*mu_y
    ssim_map = ((2*mu_x*mu_y + C1)*(2*sigma_xy + C2))/((mu_x*mu_x + mu_y*mu_y + C1)*(sigma_x + sigma_y + C2))
    return ssim_map.mean()

def total_variation_l1(x):
    # anisotropic TV L1
    dh = torch.abs(x[...,1:,:] - x[...,:-1,:]).mean()
    dw = torch.abs(x[...,:,1:] - x[...,:,:-1]).mean()
    return dh + dw

def high_freq_energy(x):
    # crude measure: energy above Nyquist/4 via FFT magnitude
    # expects [1,C,H,W] in [0,1]
    X = torch.fft.rfftn(x, dim=(-2,-1))
    mag = torch.abs(X)
    H, W = x.shape[-2:]
    # radial mask: keep frequencies with radius > 0.25 * max radius
    yy = torch.fft.fftfreq(H, d=1.0).to(x.device).unsqueeze(1)
    xx = torch.fft.rfftfreq(W, d=1.0).to(x.device).unsqueeze(0)
    rr = torch.sqrt(yy**2 + xx**2)
    mask = (rr > 0.25*rr.max()).float()
    while mask.ndim < mag.ndim:
        mask = mask.unsqueeze(0)
    e = (mag * mask).mean()
    return e
