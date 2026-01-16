# Deep Image Prior Project

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)


A comprehensive implementation of **Deep Image Prior** and related untrained neural network methods for image reconstruction tasks.

## 📋 Overview

This project implements three neural network architectures that can reconstruct images **without any training data**:

| Architecture | Parameters | Description |
|-------------|------------|-------------|
| **Deep Image Prior (DIP)** | ~2M | U-Net encoder-decoder with skip connections |
| **Deep Decoder** | ~68K | Decoder-only with 1×1 convolutions |
| **Hybrid CNN-Transformer** | ~1.5M | CNN encoder/decoder with Transformer bottleneck |

### Key Insight

The structure of a neural network itself captures natural image statistics. CNNs learn smooth, natural patterns **faster** than noise, enabling reconstruction through optimization alone.

## 🚀 Quick Start

### Installation

```bash
# Clone or extract the project
cd deep_image_prior_project

# Install dependencies
pip install torch torchvision numpy matplotlib pillow
```

### Basic Usage

```python
from deep_image_prior_project import run_experiment

# Run Deep Image Prior on your images
results = run_experiment(model_type="dip")

# Run Deep Decoder (faster, fewer parameters)
results = run_experiment(model_type="deep_decoder")

# Run Transformer variant (experimental)
results = run_experiment(model_type="transformer")
```

### Single Image Reconstruction

```python
from deep_image_prior_project import get_dip_network, run_denoising

# Create model
model = get_dip_network()

# Denoise a single image
psnr, logs = run_denoising(
    model=model,
    noisy_img_path="path/to/noisy.jpg",
    output_path="path/to/output.png",
    clean_img_path="path/to/clean.jpg",  # Optional, for PSNR
    num_iter=5000,
    lr=0.01
)

print(f"Final PSNR: {psnr:.2f} dB")
```

## 📁 Project Structure

```
deep_image_prior_project/
│
├── __init__.py           # Package exports and documentation
├── config.py             # Configuration dataclasses
├── experiment.py         # Main experiment runner
├── utils.py              # Utility functions
├── README.md             # This file
├── report.tex            # LaTeX report source
│
├── models/
│   ├── __init__.py       # Model exports
│   ├── dip.py            # Deep Image Prior (U-Net)
│   ├── deep_decoder.py   # Deep Decoder
│   └── transformer.py    # Transformer variants
│
└── tasks/
    ├── __init__.py       # Task exports
    ├── denoise.py        # Denoising task
    ├── superres.py       # Super-resolution task
    └── inpaint.py        # Inpainting task
```

## 🎯 Supported Tasks

### 1. Denoising
Remove additive Gaussian noise from images.

```python
from deep_image_prior_project import get_dip_network, run_denoising

model = get_dip_network()
psnr, logs = run_denoising(model, "noisy.jpg", "denoised.png")
```

### 2. Super-Resolution (4×)
Upscale low-resolution images by 4×.

```python
from deep_image_prior_project import get_dip_network, run_superresolution

model = get_dip_network()
psnr, logs = run_superresolution(model, "lowres.jpg", "highres.png", up_factor=4)
```

### 3. Inpainting
Fill in missing or corrupted regions.

```python
from deep_image_prior_project import get_dip_network, run_inpainting

model = get_dip_network()
psnr, logs = run_inpainting(model, "corrupted.jpg", "mask.jpg", "inpainted.png")
```

## ⚙️ Configuration

### Using Configuration Classes

```python
from deep_image_prior_project import get_default_config, run_experiment

# Get default configuration
config = get_default_config()

# Modify DIP settings
config.dip.learning_rate = 0.005
config.dip.num_iter_denoise = 10000
config.dip.optimizer = "adam"

# Modify paths
config.paths.base_path = "/path/to/your/data"

# Run experiment
results = run_experiment(model_type="dip", config=config)
```

### Available Configuration Options

#### DIP Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 0.01 | Optimizer learning rate |
| `num_iter_denoise` | 5000 | Iterations for denoising |
| `num_iter_superres` | 5000 | Iterations for super-resolution |
| `num_iter_inpaint` | 8000 | Iterations for inpainting |
| `input_depth` | 32 | Input noise channels |
| `num_channels` | 128 | Channels per encoder/decoder level |
| `num_levels` | 5 | Number of U-Net levels |
| `skip_channels` | 4 | Skip connection channels |
| `input_noise_std` | 1/30 | Input perturbation σ |
| `optimizer` | "adam" | Optimizer type |
| `max_image_size` | 512 | Maximum image dimension |
| `use_mixed_precision` | True | Use FP16 for faster training |

#### Deep Decoder Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_channels` | 256 | Channels in tiny input tensor |
| `input_spatial_size` | 4 | Spatial size of input (4×4) |
| `channels` | [256,256,128,128,64,64,32,32] | Channel progression |
| `learnable_input` | True | Optimize input tensor |

#### Transformer Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `architecture` | "hybrid" | "pure", "local", or "hybrid" |
| `embed_dim` | 256 | Transformer embedding dimension |
| `num_heads` | 8 | Number of attention heads |
| `num_layers` | 4 | Number of transformer layers |
| `max_image_size` | 256 | Max size (transformers need smaller) |

## 📊 Expected Results

### PSNR Benchmarks (Higher is Better)

| Model | Denoise | Super-Res | Inpaint |
|-------|---------|-----------|---------|
| **DIP (Adam, LR=0.01)** | **32.4 dB** | **21.6 dB** | **22.9 dB** |
| DIP (Adam, LR=0.005) | 32.1 dB | 21.8 dB | 22.9 dB |
| DIP (SGD, LR=0.01) | 30.1 dB | 20.8 dB | 21.5 dB |
| Deep Decoder | 26.5 dB | 21.0 dB | 20.1 dB |
| Hybrid Transformer | 32→25 dB* | 15.2 dB | 22.1 dB |

*Transformer peaks at 32 dB then drops to 25 dB due to overfitting.

### Learning Rate Effects

| Learning Rate | Stability | Quality | Recommendation |
|--------------|-----------|---------|----------------|
| 0.005 | Very stable | Good | Safe choice |
| **0.01** | Stable | **Best** | **Recommended** |
| 0.05 | Unstable | Collapse | Avoid |

## 📂 Data Format

### Expected Folder Structure

```
images/
├── highres/
│   ├── im1/
│   │   ├── clean.jpg      # Ground truth
│   │   ├── noisy.jpg      # Noisy version (σ=25)
│   │   ├── lr_x4.jpg      # 4× downsampled
│   │   ├── corrupted.jpg  # With missing regions
│   │   └── mask.jpg       # Binary mask (white=known)
│   ├── im2/
│   └── ...
└── lowres/
    ├── im1/
    └── ...
```

### Mask Format
- White pixels (255): Known/visible regions
- Black pixels (0): Missing/to-be-filled regions

## 🔧 Advanced Usage

### Custom Model Creation

```python
from deep_image_prior_project.models import DeepImagePriorNet, DeepDecoder

# Custom DIP
model = DeepImagePriorNet(
    num_input_channels=32,
    num_output_channels=3,
    num_channels_down=[64, 128, 256, 512, 512],
    num_channels_up=[64, 128, 256, 512, 512],
    num_channels_skip=[4, 4, 4, 4, 4],
)

# Custom Deep Decoder
model = DeepDecoder(
    input_channels=128,
    input_spatial_size=8,
    channels=[128, 64, 64, 32],
    output_size=(256, 256),
)
```

### Plotting PSNR Curves

```python
from deep_image_prior_project import run_experiment, plot_psnr_curves

results = run_experiment(model_type="dip")
plot_psnr_curves(results, "psnr_curves.png")
```

### Using Different Optimizers

```python
from deep_image_prior_project import get_dip_network, run_denoising

model = get_dip_network()

# With SGD
psnr, logs = run_denoising(
    model, "noisy.jpg", "output.png",
    optimizer_type="sgd",
    lr=0.01,
    momentum=0.9
)

# With AdamW
psnr, logs = run_denoising(
    model, "noisy.jpg", "output.png",
    optimizer_type="adamw",
    lr=0.01,
    weight_decay=0.01
)
```



## 📚 References

1. **Deep Image Prior**  
   Ulyanov, D., Vedaldi, A., & Lempitsky, V. (2018).  
   *Deep Image Prior.* CVPR 2018.  
   [Paper](https://arxiv.org/abs/1711.10925) | [Code](https://github.com/DmitryUlyanov/deep-image-prior)

2. **Deep Decoder**  
   Heckel, R., & Hand, P. (2019).  
   *Deep Decoder: Concise Image Representations from Untrained Non-convolutional Networks.* ICLR 2019.  
   [Paper](https://arxiv.org/abs/1810.03982)

3. **Vision Transformer**  
   Dosovitskiy, A., et al. (2020).  
   *An Image is Worth 16x16 Words.* ICLR 2021.  
   [Paper](https://arxiv.org/abs/2010.11929)
