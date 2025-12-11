"""Simple encoder implementation for demonstration purposes."""

import torch
import torch.nn as nn
from typing import Any, Dict

from vlm_eval.core import BaseEncoder, EncoderRegistry


@EncoderRegistry.register("simple_cnn")
class SimpleCNNEncoder(BaseEncoder):
    """A simple CNN encoder for demonstration.
    
    This is a lightweight encoder that can run without pretrained weights,
    useful for testing and demonstrations.
    
    Args:
        variant: Model variant ("tiny", "small", "base")
        pretrained: Whether to load pretrained weights (not implemented for demo)
    """
    
    VARIANTS = {
        "tiny": {"channels": [64, 128, 256], "output_dim": 256},
        "small": {"channels": [64, 128, 256, 512], "output_dim": 512},
        "base": {"channels": [64, 128, 256, 512, 768], "output_dim": 768},
    }
    
    def __init__(self, variant: str = "base", pretrained: bool = False):
        super().__init__()
        self.variant = variant
        self.pretrained = pretrained
        self._patch_size = 16
        
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown variant: {variant}. Choose from {list(self.VARIANTS.keys())}")
        
        config = self.VARIANTS[variant]
        channels = config["channels"]
        self._output_dim = config["output_dim"]
        
        # Build simple CNN
        layers = []
        in_channels = 3
        for out_channels in channels:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            ])
            in_channels = out_channels
        
        # Final projection to output dimension
        layers.append(nn.Conv2d(in_channels, self._output_dim, kernel_size=1))
        
        self.encoder = nn.Sequential(*layers)
        
        if pretrained:
            print(f"Warning: Pretrained weights not available for {self.__class__.__name__}")
    
    @property
    def output_channels(self) -> int:
        """Number of output feature channels."""
        return self._output_dim
    
    @property
    def patch_size(self) -> int:
        """Effective patch size (downsampling factor)."""
        return self._patch_size
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            images: Input images of shape (B, 3, H, W).
        
        Returns:
            Features of shape (B, output_channels, H//patch_size, W//patch_size).
        """
        return self.encoder(images)
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary."""
        return {
            "variant": self.variant,
            "pretrained": self.pretrained,
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SimpleCNNEncoder":
        """Create encoder from configuration."""
        return cls(**config)
