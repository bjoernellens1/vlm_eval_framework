"""Linear probe segmentation head implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict

from vlm_eval.core import BaseEncoder, BaseSegmentationHead, HeadRegistry


@HeadRegistry.register("linear_probe")
class LinearProbeHead(BaseSegmentationHead):
    """Linear probe segmentation head.
    
    A simple 1x1 convolution followed by bilinear upsampling to produce
    segmentation predictions. This is commonly used for evaluating
    pretrained encoders.
    
    Args:
        encoder: The encoder to use for feature extraction.
        num_classes: Number of segmentation classes.
        freeze_encoder: Whether to freeze the encoder parameters.
    """
    
    def __init__(
        self,
        encoder: BaseEncoder,
        num_classes: int,
        freeze_encoder: bool = True
    ):
        super().__init__(encoder, num_classes, freeze_encoder)
        
        # Simple 1x1 convolution for classification
        self.classifier = nn.Conv2d(
            encoder.output_channels,
            num_classes,
            kernel_size=1,
            bias=True
        )
        
        # Initialize weights
        nn.init.kaiming_normal_(self.classifier.weight, mode='fan_out', nonlinearity='linear')
        nn.init.constant_(self.classifier.bias, 0)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            features: Encoder features of shape (B, C, H//patch_size, W//patch_size).
        
        Returns:
            Segmentation logits of shape (B, num_classes, H, W).
        """
        # Classify each spatial location
        logits = self.classifier(features)
        
        # Upsample to original resolution
        logits = F.interpolate(
            logits,
            scale_factor=self.encoder.patch_size,
            mode='bilinear',
            align_corners=False
        )
        
        return logits
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary."""
        return {
            "num_classes": self.num_classes,
            "freeze_encoder": self.freeze_encoder,
        }
    
    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        encoder: BaseEncoder
    ) -> "LinearProbeHead":
        """Create head from configuration."""
        return cls(encoder=encoder, **config)
