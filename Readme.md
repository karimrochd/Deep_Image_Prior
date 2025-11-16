# Deep Image Prior (DIP) – Denoising, Super-Resolution & Inpainting

This repo/notebook implements **Deep Image Prior (DIP)** in PyTorch for:

- **Denoising**
- **4× Super-Resolution**
- **Inpainting**

All three tasks share the same U-Net–like network; only the **loss** changes.

---

## Method

We use a randomly initialized convolutional network `f_θ` and a fixed noise input `z`.

- **Denoising**  
  Minimize the MSE between the output and the noisy image:
  \[
  \| f_\theta(z) - x_{\text{noisy}} \|^2
  \]

- **Super-Resolution (x4)**  
  The network outputs a high-res image; we downsample it and match the low-res input:
  \[
  \| D(f_\theta(z)) - y_{\text{LR}} \|^2
  \]

- **Inpainting**  
  With a mask \(M\) (1 = known, 0 = hole), we only enforce the loss on known pixels:
  \[
  \| M \odot f_\theta(z) - M \odot y \|^2
  \]

Here \(D\) is bilinear downsampling and ⊙ is elementwise product.

---

## Implementation Highlights

- Network: small U-Net–style `DIPUNet` (encoder–decoder with skip connections).
- Input: fixed noise tensor `net_input` with 32 channels.
- Optimizer: Adam, `lr=0.01`.
- Training: typically `10,000` iterations.
- Every `log_every` iterations:
  - Save current reconstruction image.
  - Log **loss** and **PSNR** vs the clean image (if available).

At the end of each run, we:

- Save the final reconstruction:
  - `dip_denoised.png`
  - `dip_superres_x4.png`
  - `dip_inpainted.png`
- Plot **Loss & PSNR vs iteration** on a twin-axis plot, e.g.:
  - `dip_denoised_curves.png`
  - `dip_superres_x4_curves.png`
  - `dip_inpainted_curves.png`

---

## PSNR Behavior (Important Observation)

Across **denoising**, **super-resolution**, and **inpainting**:

- PSNR **rises quickly** in the early iterations as the network captures image structure.
- Around a few thousand iterations, we observe a **drop in PSNR** (and a spike in loss).
- After this drop, PSNR increases again and then **plateaus**.

This shows that **PSNR is not monotonic** during training:
- The “best” reconstruction may occur **before the final iteration**.
- Monitoring PSNR (or using early stopping) is important when using Deep Image Prior.
