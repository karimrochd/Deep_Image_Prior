"""
Configuration settings for Deep Image Prior experiments.

This module contains all hyperparameters and settings for:
- Deep Image Prior (DIP)
- Deep Decoder
- Transformer-based priors
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import os


@dataclass
class PathConfig:
    """File and directory paths."""
    base_path: str = "/content/drive/MyDrive/Deep_Image_Prior"
    
    @property
    def images_path(self) -> str:
        return os.path.join(self.base_path, "images")
    
    @property
    def scores_path(self) -> str:
        return os.path.join(self.base_path, "scores")
    
    @property
    def output_path(self) -> str:
        return os.path.join(self.base_path, "output")
    
    resolutions: List[str] = field(default_factory=lambda: ["highres", "lowres"])
    image_folders: List[str] = field(default_factory=lambda: ["im1", "im2", "im3", "im4", "im5"])


@dataclass
class DIPConfig:
    """Deep Image Prior configuration (paper-correct)."""
    # Architecture
    input_depth: int = 32
    num_channels: int = 128
    num_levels: int = 5
    skip_channels: int = 4
    
    # Training
    learning_rate: float = 0.01
    num_iter_denoise: int = 5000
    num_iter_superres: int = 5000
    num_iter_inpaint: int = 8000
    
    # Input perturbation (paper: σ_p = 1/30)
    input_noise_std: float = 1.0 / 30.0
    
    # Memory
    max_image_size: int = 512
    use_mixed_precision: bool = True
    
    # Optimizer
    optimizer: str = "adam"  # "adam" or "sgd"
    weight_decay: float = 0.0
    momentum: float = 0.9
    betas: Tuple[float, float] = (0.9, 0.999)


@dataclass
class DeepDecoderConfig:
    """Deep Decoder configuration."""
    # Architecture
    input_channels: int = 256
    input_spatial_size: int = 4
    channels: List[int] = field(default_factory=lambda: [256, 256, 128, 128, 64, 64, 32, 32])
    
    # Training
    learning_rate: float = 0.01
    num_iter_denoise: int = 5000
    num_iter_superres: int = 5000
    num_iter_inpaint: int = 8000
    
    # Whether input z is learnable
    learnable_input: bool = True
    
    # Memory
    max_image_size: int = 512
    use_mixed_precision: bool = True


@dataclass
class TransformerConfig:
    """Transformer-based prior configuration."""
    # Architecture type: "pure", "local", "hybrid"
    architecture: str = "hybrid"
    
    # Transformer dimensions
    embed_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    mlp_ratio: float = 2.0
    
    # Patch settings
    patch_size: int = 8
    window_size: int = 4  # For local attention
    
    # CNN settings (for hybrid)
    base_channels: int = 64
    
    # Training
    learning_rate: float = 0.001
    num_iter_denoise: int = 5000
    num_iter_superres: int = 5000
    num_iter_inpaint: int = 8000
    
    # Input
    input_channels: int = 32
    
    # Memory (transformers need smaller images)
    max_image_size: int = 256
    use_mixed_precision: bool = True


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    paths: PathConfig = field(default_factory=PathConfig)
    dip: DIPConfig = field(default_factory=DIPConfig)
    deep_decoder: DeepDecoderConfig = field(default_factory=DeepDecoderConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    
    # Common settings
    up_factor: int = 4  # Super-resolution factor
    log_every: int = 100
    seed: int = 42
    exp_weight: float = 0.99  # Exponential averaging weight


def get_default_config() -> ExperimentConfig:
    """Get default experiment configuration."""
    return ExperimentConfig()
