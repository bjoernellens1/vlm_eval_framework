"""Abstract base class for heads."""

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
import torch.nn as nn

from vlm_eval.core.base_encoder import BaseEncoder


class BaseHead(ABC, nn.Module):
    """Abstract base class for all heads (segmentation, classification, etc.)."""
    
    def __init__(self, encoder: BaseEncoder) -> None:
        """Initialize the head.
        
        Args:
            encoder: The vision encoder to use.
        """
        super().__init__()
        self.encoder = encoder
        
    @abstractmethod
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary."""
        pass
    
    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any], encoder: BaseEncoder) -> "BaseHead":
        """Create head from configuration."""
        pass
        
    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """Get number of parameters in the head (including encoder)."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def get_head_parameters(self, trainable_only: bool = False) -> int:
        """Get number of parameters in the head only (excluding encoder)."""
        encoder_params = self.encoder.get_num_parameters(trainable_only)
        total_params = self.get_num_parameters(trainable_only)
        return total_params - encoder_params
        
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}(\n"
            f"  encoder={self.encoder.__class__.__name__}\n"
            f")"
        )


class BaseSegmentationHead(BaseHead):
    """Abstract base class for segmentation heads.
    
    Segmentation heads take encoder features and produce per-pixel class
    predictions. They handle upsampling to the original image resolution
    and can optionally freeze the encoder.
    
    Input: Features from encoder (B, C, H//patch_size, W//patch_size)
    Output: Class logits (B, num_classes, H, W)
    """
    
    def __init__(
        self,
        encoder: BaseEncoder,
        num_classes: int,
        freeze_encoder: bool = True
    ) -> None:
        """Initialize the segmentation head.
        
        Args:
            encoder: The vision encoder to use for feature extraction.
            num_classes: Number of segmentation classes.
            freeze_encoder: If True, freeze the encoder parameters.
        """
        super().__init__(encoder)
        self.num_classes = num_classes
        self.freeze_encoder = freeze_encoder
        
        if freeze_encoder:
            self.encoder.freeze()
            
    def train(self, mode: bool = True) -> "BaseSegmentationHead":
        """Set training mode."""
        super().train(mode)
        
        # Keep encoder in eval mode if frozen
        if self.freeze_encoder and mode:
            self.encoder.eval()
        
        return self
        
    def __repr__(self) -> str:
        """String representation of the head."""
        total_params = self.get_num_parameters()
        trainable_params = self.get_num_parameters(trainable_only=True)
        head_params = self.get_head_parameters()
        
        return (
            f"{self.__class__.__name__}(\n"
            f"  num_classes={self.num_classes},\n"
            f"  freeze_encoder={self.freeze_encoder},\n"
            f"  total_parameters={total_params:,},\n"
            f"  trainable_parameters={trainable_params:,},\n"
            f"  head_parameters={head_params:,},\n"
            f"  encoder={self.encoder.__class__.__name__}\n"
            f")"
        )
