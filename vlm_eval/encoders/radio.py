"""RADIO encoder from NVIDIA."""

import torch
import torch.nn as nn
from typing import Any, Dict, Optional
from einops import rearrange

from vlm_eval.core import BaseEncoder, EncoderRegistry


@EncoderRegistry.register("radio")
class RADIOEncoder(BaseEncoder):
    """RADIO (Reduce All Domains Into One) encoder from NVIDIA.
    
    Wraps the NVIDIA RADIO model from Hugging Face for semantic segmentation.
    Uses the spatial_features output which is suitable for dense prediction tasks.
    
    Args:
        variant: Model variant (currently only 'base' supported, uses nvidia/RADIO)
        pretrained: Whether to load pretrained weights (default: True)
        input_size: Input image size (default: 518 for segmentation tasks)
        freeze: Whether to freeze encoder weights (default: False)
        hf_token: Optional Hugging Face access token
    
    Note:
        Requires Hugging Face authentication. Run `huggingface-cli login` first.
        The model will be downloaded on first use (~2GB).
    """
    
    def __init__(
        self,
        variant: str = "base",
        pretrained: bool = True,
        input_size: int = 518,
        freeze: bool = False,
        hf_token: Optional[str] = None,
    ):
        super().__init__()
        self.variant = variant
        self.input_size = input_size
        self._freeze = freeze
        
        # Load RADIO model from Hugging Face
        if pretrained:
            try:
                from transformers import AutoModel
                
                # Load the model
                model_name = "nvidia/RADIO"
                # Pin to specific revision to avoid downloading new versions
                # This revision is known to work with transformers 4.50.3
                revision = "10f0448935988a74dd59b4969ac520dbcd7db293"
                self.model = AutoModel.from_pretrained(
                    model_name,
                    revision=revision,
                    trust_remote_code=True,
                    token=hf_token,
                )
                
                # Get patch size from model config
                # RADIO uses patch_size=14 for the base model
                self._patch_size = getattr(self.model.config, 'patch_size', 14)
                
                # Get spatial feature dimension
                # This is the dimension D in the spatial_features output (B, T, D)
                self._output_channels = getattr(
                    self.model.config, 
                    'hidden_size', 
                    1024  # Default for RADIO base
                )
                
            except ImportError:
                raise ImportError(
                    "transformers library is required for RADIO encoder. "
                    "Install with: pip install transformers"
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load RADIO model. Make sure you're logged in to Hugging Face "
                    f"(run 'huggingface-cli login'). Error: {e}"
                )
        else:
            raise ValueError("RADIO encoder requires pretrained=True")
        
        # Freeze weights if requested
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
    
    @property
    def output_channels(self) -> int:
        """Number of output channels from spatial features."""
        return self._output_channels
    
    @property
    def patch_size(self) -> int:
        """Patch size used by the encoder."""
        return self._patch_size
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass through RADIO encoder.
        
        Args:
            images: Input images of shape (B, 3, H, W)
        
        Returns:
            Spatial features of shape (B, C, H', W') where:
                - C is the number of output channels (e.g., 1024)
                - H' = H // patch_size
                - W' = W // patch_size
        """
        # RADIO returns a tuple: (summary, spatial_features)
        # summary: (B, C) - global image representation
        # spatial_features: (B, T, D) - spatial tokens for dense tasks
        _, spatial_features = self.model(images)
        
        # Reshape spatial features from (B, T, D) to (B, D, H, W)
        # T = (H // patch_size) * (W // patch_size)
        B, T, D = spatial_features.shape
        H_in, W_in = images.shape[-2:]
        H_out = H_in // self._patch_size
        W_out = W_in // self._patch_size
        
        # Verify dimensions match
        assert T == H_out * W_out, (
            f"Spatial token count mismatch: expected {H_out * W_out}, got {T}"
        )
        
        # Rearrange to spatial format
        spatial_features = rearrange(
            spatial_features,
            'b (h w) d -> b d h w',
            h=H_out,
            w=W_out
        )
        
        return spatial_features
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary."""
        return {
            "variant": self.variant,
            "pretrained": True,
            "input_size": self.input_size,
            "freeze": self._freeze,
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RADIOEncoder":
        """Create encoder from configuration dictionary."""
        # Remove 'name' if present (from registry)
        config = {k: v for k, v in config.items() if k != 'name'}
        return cls(**config)
    
    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """Get number of parameters in the encoder.
        
        Args:
            trainable_only: If True, only count trainable parameters.
        
        Returns:
            Number of parameters.
        """
        if trainable_only:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.model.parameters())
