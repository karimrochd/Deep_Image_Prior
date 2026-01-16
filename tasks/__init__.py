"""
Image reconstruction tasks for Deep Image Prior experiments.

Available tasks:
- denoise: Remove noise from images
- superres: Super-resolution (upscaling)
- inpaint: Fill in missing regions
"""

from .denoise import run_denoising
from .superres import run_superresolution
from .inpaint import run_inpainting

__all__ = [
    "run_denoising",
    "run_superresolution",
    "run_inpainting",
]
