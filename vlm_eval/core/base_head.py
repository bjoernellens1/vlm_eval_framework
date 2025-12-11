"""Abstract base class for segmentation heads."""

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
import torch.nn as nn

from vlm_eval.core.base_encoder import BaseEncoder


class BaseSegmentationHead(ABC, nn.Module):
    """Abstract base class for segmentation heads.
    
    Segmentation heads take encoder features and produce per-pixel class
    predictions. They handle upsampling to the original image resolution
    and can optionally freeze the encoder.
    
    Input: Features from encoder (B, C, H//patch_size, W//patch_size)
    Output: Class logits (B, num_classes, H, W)
    
    Example:
        >>> @HeadRegistry.register("linear_probe")
        >>> class LinearProbeHead(BaseSegmentationHead):
        ...     def __init__(self, encoder, num_classes, freeze_encoder=True):
        ...         super().__init__(encoder, num_classes, freeze_encoder)
        ...         # Initialize your head layers
        ...         self.classifier = nn.Conv2d(
        ...             encoder.output_channels,
        ...             num_classes,
        ...             kernel_size=1
        ...         )
        ...
        ...     def forward(self, features: torch.Tensor) -> torch.Tensor:
        ...         # Your forward pass
        ...         logits = self.classifier(features)
        ...         # Upsample to original resolution
        ...         return F.interpolate(logits, scale_factor=self.encoder.patch_size)
        ...
        ...     def get_config(self) -> Dict[str, Any]:
        ...         return {
        ...             "num_classes": self.num_classes,
        ...             "freeze_encoder": self.freeze_encoder
        ...         }
        ...
        ...     @classmethod
        ...     def from_config(cls, config: Dict[str, Any], encoder) -> "LinearProbeHead":
        ...         return cls(encoder=encoder, **config)
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
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.freeze_encoder = freeze_encoder
        
        if freeze_encoder:
            self.encoder.freeze()
    
    @abstractmethod
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass of the segmentation head.
        
        Args:
            features: Encoder features of shape (B, C, H//patch_size, W//patch_size).
        
        Returns:
            Class logits of shape (B, num_classes, H, W) at original resolution.
        """
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary for serialization.
        
        The configuration should contain all necessary parameters to
        reconstruct the head using from_config(). Must be YAML-serializable
        (no torch objects). Do not include the encoder in the config.
        
        Returns:
            Dictionary containing head configuration.
        """
        pass
    
    @classmethod
    @abstractmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        encoder: BaseEncoder
    ) -> "BaseSegmentationHead":
        """Create head instance from configuration dictionary.
        
        Args:
            config: Configuration dictionary from get_config().
            encoder: Encoder instance to use.
        
        Returns:
            Head instance.
        """
        pass
    
    def train(self, mode: bool = True) -> "BaseSegmentationHead":
        """Set training mode.
        
        Respects encoder freezing - if encoder is frozen, it stays in eval mode.
        
        Args:
            mode: If True, set to training mode. If False, set to eval mode.
        
        Returns:
            Self for method chaining.
        """
        super().train(mode)
        
        # Keep encoder in eval mode if frozen
        if self.freeze_encoder and mode:
            self.encoder.eval()
        
        return self
    
    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """Get number of parameters in the head (including encoder).
        
        Args:
            trainable_only: If True, count only trainable parameters.
        
        Returns:
            Number of parameters.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def get_head_parameters(self, trainable_only: bool = False) -> int:
        """Get number of parameters in the head only (excluding encoder).
        
        Args:
            trainable_only: If True, count only trainable parameters.
        
        Returns:
            Number of parameters in head.
        """
        encoder_params = self.encoder.get_num_parameters(trainable_only)
        total_params = self.get_num_parameters(trainable_only)
        return total_params - encoder_params
    
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
