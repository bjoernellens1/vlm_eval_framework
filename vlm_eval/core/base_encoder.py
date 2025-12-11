"""Abstract base class for vision encoders."""

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
import torch.nn as nn


class BaseEncoder(ABC, nn.Module):
    """Abstract base class for vision encoders.
    
    All encoders must inherit from this class and implement the required
    abstract methods and properties. Encoders take images as input and
    produce dense feature maps.
    
    Input shape: (B, 3, H, W) - RGB images
    Output shape: (B, C, H//patch_size, W//patch_size) - Dense features
    
    Example:
        >>> @EncoderRegistry.register("my_encoder")
        >>> class MyEncoder(BaseEncoder):
        ...     def __init__(self, variant: str = "base"):
        ...         super().__init__()
        ...         self.variant = variant
        ...         # Initialize your encoder here
        ...
        ...     @property
        ...     def output_channels(self) -> int:
        ...         return 768
        ...
        ...     @property
        ...     def patch_size(self) -> int:
        ...         return 16
        ...
        ...     def forward(self, images: torch.Tensor) -> torch.Tensor:
        ...         # Your forward pass
        ...         return features
        ...
        ...     def get_config(self) -> Dict[str, Any]:
        ...         return {"variant": self.variant}
        ...
        ...     @classmethod
        ...     def from_config(cls, config: Dict[str, Any]) -> "MyEncoder":
        ...         return cls(**config)
    """
    
    def __init__(self) -> None:
        """Initialize the encoder."""
        super().__init__()
        self._frozen = False
    
    @property
    @abstractmethod
    def output_channels(self) -> int:
        """Number of output feature channels.
        
        Returns:
            Number of channels in the output feature map.
        """
        pass
    
    @property
    @abstractmethod
    def patch_size(self) -> int:
        """Patch size used by the encoder.
        
        The output spatial dimensions will be H//patch_size x W//patch_size.
        
        Returns:
            Patch size as an integer.
        """
        pass
    
    @abstractmethod
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass of the encoder.
        
        Args:
            images: Input images of shape (B, 3, H, W) with values in [0, 1].
        
        Returns:
            Dense feature map of shape (B, C, H//patch_size, W//patch_size).
        """
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary for serialization.
        
        The configuration should contain all necessary parameters to
        reconstruct the encoder using from_config(). Must be YAML-serializable
        (no torch objects).
        
        Returns:
            Dictionary containing encoder configuration.
        """
        pass
    
    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseEncoder":
        """Create encoder instance from configuration dictionary.
        
        Args:
            config: Configuration dictionary from get_config().
        
        Returns:
            Encoder instance.
        """
        pass
    
    def freeze(self) -> None:
        """Freeze all encoder parameters.
        
        This sets requires_grad=False for all parameters and puts the
        encoder in eval mode.
        """
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        self._frozen = True
    
    def unfreeze(self) -> None:
        """Unfreeze all encoder parameters.
        
        This sets requires_grad=True for all parameters.
        """
        for param in self.parameters():
            param.requires_grad = True
        self._frozen = False
    
    def is_frozen(self) -> bool:
        """Check if encoder is frozen.
        
        Returns:
            True if encoder is frozen, False otherwise.
        """
        return self._frozen
    
    def to_device(self, device: torch.device) -> "BaseEncoder":
        """Move encoder to specified device.
        
        Args:
            device: Target device (e.g., 'cuda', 'cpu').
        
        Returns:
            Self for method chaining.
        """
        return self.to(device)
    
    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """Get number of parameters in the encoder.
        
        Args:
            trainable_only: If True, count only trainable parameters.
        
        Returns:
            Number of parameters.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def __repr__(self) -> str:
        """String representation of the encoder."""
        num_params = self.get_num_parameters()
        num_trainable = self.get_num_parameters(trainable_only=True)
        return (
            f"{self.__class__.__name__}(\n"
            f"  output_channels={self.output_channels},\n"
            f"  patch_size={self.patch_size},\n"
            f"  num_parameters={num_params:,},\n"
            f"  trainable_parameters={num_trainable:,},\n"
            f"  frozen={self._frozen}\n"
            f")"
        )
