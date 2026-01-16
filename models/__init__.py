"""
Neural network architectures for image prior experiments.

Available models:
- DeepImagePriorNet: U-Net encoder-decoder (Ulyanov et al., 2018)
- DeepDecoder: Decoder-only with 1x1 convolutions (Heckel & Hand, 2019)
- HybridCNNTransformer: CNN encoder/decoder with Transformer bottleneck
"""

from .dip import DeepImagePriorNet, get_dip_network
from .deep_decoder import DeepDecoder, get_deep_decoder_network
from .transformer import (
    PureTransformer,
    LocalTransformer, 
    HybridCNNTransformer,
    get_transformer_network
)

__all__ = [
    "DeepImagePriorNet",
    "get_dip_network",
    "DeepDecoder", 
    "get_deep_decoder_network",
    "PureTransformer",
    "LocalTransformer",
    "HybridCNNTransformer",
    "get_transformer_network",
]
