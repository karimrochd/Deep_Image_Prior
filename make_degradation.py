# make_degradation.py
# Create standard degradations for DIP-style experiments:
#   - Gaussian noise (for denoising)
#   - Low-resolution image (for super-resolution input)
#   - Inpainting corruption + mask (for masked regression)
#
# Usage examples are at the bottom.

import argparse
from pathlib import Path
import numpy as np
from PIL import Image

def load_rgb(path):
    img = Image.open(path).convert("RGB")
    return img

def save_rgb(img, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)

def to_float(img):
    # PIL -> float32 [0,1], shape HxWxC
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr

def from_float(arr):
    # float32 [0,1] -> PIL
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr)

def add_gaussian_noise(img, sigma):
    """
    img: PIL RGB
    sigma: noise std (if >1 => 0..255 scale; else 0..1 scale)
    """
    arr = to_float(img)
    sig = sigma / 255.0 if sigma > 1 else float(sigma)
    noisy = arr + np.random.normal(0.0, sig, size=arr.shape).astype(np.float32)
    return from_float(noisy)

def make_low_res(img, scale, return_up=False):
    """
    Downsample by 'scale' with bicubic to create LR input.
    If return_up=True, also return the LR image re-upsampled to original size (for visualization).
    """
    w, h = img.size
    lr = img.resize((max(1, w // scale), max(1, h // scale)), resample=Image.BICUBIC)
    if return_up:
        up = lr.resize((w, h), resample=Image.BICUBIC)
        return lr, up
    return lr

def make_inpainting(img, mode, *, box_frac=0.4, drop_prob=0.5, seed=0):
    """
    Create corrupted image and binary mask m where 1=known, 0=hole (matches DIP eq. (6) semantics).
    - mode='box': removes a centered box occupying box_frac of min(H,W)
    - mode='random': drops each pixel independently with prob 'drop_prob'
    """
    rng = np.random.RandomState(seed)
    arr = to_float(img)  # HxWxC
    H, W, _ = arr.shape

    if mode == "box":
        side = int(min(H, W) * box_frac)
        y0 = (H - side) // 2
        x0 = (W - side) // 2
        m = np.ones((H, W), dtype=np.float32)
        m[y0:y0+side, x0:x0+side] = 0.0

    elif mode == "random":
        m = (rng.rand(H, W) > drop_prob).astype(np.float32)

    else:
        raise ValueError("mode must be 'box' or 'random'")

    m3 = np.repeat(m[..., None], 3, axis=2)  # HxWx3
    corrupted = arr * m3  # set holes to 0; DIP ignores them via mask anyway
    return from_float(corrupted), Image.fromarray((m * 255).astype(np.uint8))

def main():
    p = argparse.ArgumentParser(description="Create degraded inputs for DIP experiments")
    p.add_argument("--input", required=True, help="Path to clean RGB image")
    p.add_argument("--task", required=True, choices=["noise", "lr", "inpaint"], help="Which degradation to apply")

    # Noise
    p.add_argument("--sigma", type=float, default=25.0, help="Gaussian noise std; >1 means 0..255 scale, else 0..1")

    # Low-res
    p.add_argument("--scale", type=int, default=4, help="Downscale factor for LR creation")
    p.add_argument("--save-upsampled", action="store_true", help="Also save LR re-upsampled to original size")

    # Inpainting
    p.add_argument("--mode", choices=["box", "random"], default="box", help="Inpainting mode")
    p.add_argument("--box-frac", type=float, default=0.4, help="Box side as fraction of min(H,W) for mode=box")
    p.add_argument("--drop-prob", type=float, default=0.5, help="Random drop probability for mode=random")
    p.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")

    # Outputs
    p.add_argument("--out", required=True, help="Output path (image).")
    p.add_argument("--mask-out", default=None, help="Where to save mask for inpainting (PNG, white=known, black=hole).")

    args = p.parse_args()
    img = load_rgb(args.input)

    if args.task == "noise":
        noisy = add_gaussian_noise(img, args.sigma)
        save_rgb(noisy, args.out)
        print(f"Saved noisy image → {args.out}")

    elif args.task == "lr":
        if args.save_upsampled:
            lr, up = make_low_res(img, args.scale, return_up=True)
            stem = Path(args.out)
            save_rgb(lr, args.out)
            save_rgb(up, str(stem.with_name(stem.stem + "_upsampled.png")))
            print(f"Saved LR → {args.out}")
        else:
            lr = make_low_res(img, args.scale, return_up=False)
            save_rgb(lr, args.out)
            print(f"Saved LR → {args.out}")

    else:  # inpaint
        corrupted, mask = make_inpainting(
            img, args.mode, box_frac=args.box_frac, drop_prob=args.drop_prob, seed=args.seed
        )
        save_rgb(corrupted, args.out)
        if args.mask_out is None:
            raise SystemExit("Please pass --mask-out to save the binary mask (required for training/evaluation).")
        save_rgb(mask.convert("L"), args.mask_out)
        print(f"Saved corrupted image → {args.out}")
        print(f"Saved mask (1=known, 0=hole) → {args.mask_out}")

if __name__ == "__main__":
    main()
