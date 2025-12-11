"""Zero-shot classification head."""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional

from vlm_eval.core import BaseHead, HeadRegistry, BaseEncoder


@HeadRegistry.register("zero_shot")
class ZeroShotHead(BaseHead):
    """Zero-shot classification head.
    
    Performs classification by computing similarity between image features
    and pre-computed text embeddings of class names.
    
    Args:
        encoder: The encoder instance (must support encode_text)
        class_names: List of class names to classify
        template: Template for text prompts (default: "a photo of a {}")
    """
    
    def __init__(
        self,
        encoder: BaseEncoder,
        class_names: List[str],
        template: str = "a photo of a {}",
    ):
        super().__init__(encoder)
        self.class_names = class_names
        self.template = template
        
        # Verify encoder has encode_text method
        if not hasattr(encoder, "encode_text"):
            raise ValueError(
                "Encoder must implement 'encode_text' method for ZeroShotHead"
            )
            
        # Pre-compute text embeddings
        self.register_buffer("text_embeddings", self._compute_text_embeddings())
        
    def _compute_text_embeddings(self) -> torch.Tensor:
        """Compute text embeddings for all classes."""
        prompts = [self.template.format(c) for c in self.class_names]
        
        # We need to temporarily move encoder to CPU or current device if needed
        # But usually we assume encoder is already on the right device or we handle it
        # For initialization, we might be on CPU.
        
        # If encoder is on meta device or something, this might fail.
        # We assume standard usage.
        
        with torch.no_grad():
            embeddings = self.encoder.encode_text(prompts)
            
        return embeddings
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            features: Image features (B, C, H, W) or (B, C)
            
        Returns:
            Logits (B, NumClasses)
        """
        # Handle spatial dimensions if present
        if features.dim() == 4:
            # Global average pooling if spatial features are provided
            # (B, C, H, W) -> (B, C)
            features = features.mean(dim=(2, 3))
        
        # Normalize image features (encoder should have done it, but safety first)
        features = features / features.norm(dim=-1, keepdim=True)
        
        # Compute similarity
        # (B, C) @ (NumClasses, C).T -> (B, NumClasses)
        logits = 100.0 * features @ self.text_embeddings.t()
        
        return logits
        
    def get_config(self) -> Dict[str, Any]:
        return {
            "class_names": self.class_names,
            "template": self.template,
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], encoder: BaseEncoder) -> "ZeroShotHead":
        config = {k: v for k, v in config.items() if k != 'name'}
        return cls(encoder=encoder, **config)
        
    def get_num_parameters(self, trainable_only: bool = False) -> int:
        # No trainable parameters in the head itself (embeddings are buffers)
        return 0
