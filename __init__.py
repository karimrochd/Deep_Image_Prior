"""
Deep Image Prior Project
========================

A comprehensive implementation of image prior methods for image reconstruction tasks.

This package implements:
1. Deep Image Prior (DIP) - Ulyanov et al., CVPR 2018
2. Deep Decoder - Heckel & Hand, ICLR 2019
3. Transformer-based priors (experimental)

Tasks supported:
- Denoising
- Super-resolution (4x)
- Inpainting

Usage
-----
```python
from deep_image_prior_project import run_experiment, get_default_config

# Run DIP experiment with default settings
results = run_experiment(model_type="dip")

# Run Deep Decoder experiment
results = run_experiment(model_type="deep_decoder")

# Run with custom config
config = get_default_config()
config.dip.learning_rate = 0.005
results = run_experiment(model_type="dip", config=config)
```

For more control, use individual modules:
```python
from deep_image_prior_project.models import get_dip_network
from deep_image_prior_project.tasks import run_denoising

model = get_dip_network()
psnr, logs = run_denoising(model, "noisy.jpg", "output.png")
```
"""

__version__ = "1.0.0"
__author__ = "Deep Image Prior Project"

from .config import (
    ExperimentConfig,
    DIPConfig,
    DeepDecoderConfig,
    TransformerConfig,
    PathConfig,
    get_default_config,
)

from .experiment import (
    run_experiment,
    create_model,
    plot_psnr_curves,
    print_summary,
)

from .models import (
    DeepImagePriorNet,
    get_dip_network,
    DeepDecoder,
    get_deep_decoder_network,
    HybridCNNTransformer,
    get_transformer_network,
)

from .tasks import (
    run_denoising,
    run_superresolution,
    run_inpainting,
)

from .utils import (
    get_device,
    clear_gpu_memory,
    load_image,
    save_image,
    compute_psnr_torch,
    count_parameters,
)

__all__ = [
    # Config
    "ExperimentConfig",
    "DIPConfig", 
    "DeepDecoderConfig",
    "TransformerConfig",
    "PathConfig",
    "get_default_config",
    # Experiment
    "run_experiment",
    "create_model",
    "plot_psnr_curves",
    "print_summary",
    # Models
    "DeepImagePriorNet",
    "get_dip_network",
    "DeepDecoder",
    "get_deep_decoder_network",
    "HybridCNNTransformer",
    "get_transformer_network",
    # Tasks
    "run_denoising",
    "run_superresolution", 
    "run_inpainting",
    # Utils
    "get_device",
    "clear_gpu_memory",
    "load_image",
    "save_image",
    "compute_psnr_torch",
    "count_parameters",
]
