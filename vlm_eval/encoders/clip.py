"""CLIP encoder using open_clip."""

import torch
import torch.nn as nn
from typing import Any, Dict, Optional, List, Union
import open_clip

from vlm_eval.core import BaseEncoder, EncoderRegistry


@EncoderRegistry.register("clip")
class CLIPEncoder(BaseEncoder):
    """CLIP encoder using open_clip.
    
    Args:
        variant: Model variant (e.g., 'ViT-B-32', 'ViT-L-14')
        pretrained: Pretrained weights name (e.g., 'laion2b_s34b_b79k')
        freeze: Whether to freeze encoder weights (default: True)
    """
    
    def __init__(
        self,
        variant: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        freeze: bool = True,
    ):
        super().__init__()
        self.variant = variant
        self.pretrained = pretrained
        self._freeze = freeze
        
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                variant, 
                pretrained=pretrained
            )
            self.model = model
            self.preprocess = preprocess
            
            # Get patch size and output channels
            if hasattr(self.model.visual, 'patch_size'):
                self._patch_size = self.model.visual.patch_size
                # For ViT models, the output width is the embed_dim
                self._output_channels = self.model.visual.output_dim
            else:
                # Fallback for ResNet or other architectures if needed
                # But primarily we support ViT for now
                self._patch_size = 32  # Default guess
                self._output_channels = 512
                
        except ImportError:
            raise ImportError(
                "open_clip library is required for CLIP encoder. "
                "Install with: pip install open-clip-torch"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP model: {e}")
        
        # Freeze weights if requested
        if freeze:
            self.freeze()
            
    @property
    def output_channels(self) -> int:
        return self._output_channels
    
    @property
    def patch_size(self) -> int:
        return self._patch_size
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass to get image features.
        
        Args:
            images: Input images (B, 3, H, W)
            
        Returns:
            Image features (B, C) - Global pooled features for CLIP
            Note: BaseEncoder expects (B, C, H, W), but CLIP is often used for 
            global features. For compatibility with segmentation heads, we might 
            need spatial features, but for ZeroShotHead we need global features.
            
            For now, we return global features reshaped to (B, C, 1, 1) to 
            maintain partial compatibility, or we can just return (B, C) and 
            let the head handle it.
            
            Let's return (B, C, 1, 1) to satisfy the "dense feature map" contract 
            loosely, but really this encoder is best for zero-shot classification.
        """
        # CLIP expects normalized images, usually handled by preprocess
        # But here we assume images are already tensors.
        # We might need to resize or normalize if not done by dataset.
        
        features = self.model.encode_image(images)
        # features is (B, D)
        
        # Normalize features
        features = features / features.norm(dim=-1, keepdim=True)
        
        # Reshape to (B, C, 1, 1)
        return features.unsqueeze(-1).unsqueeze(-1)

    def encode_text(self, text: List[str]) -> torch.Tensor:
        """Encode text prompts.
        
        Args:
            text: List of text strings
            
        Returns:
            Text embeddings (N, C)
        """
        tokenizer = open_clip.get_tokenizer(self.variant)
        text_tokens = tokenizer(text).to(next(self.parameters()).device)
        
        text_features = self.model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features

    def get_config(self) -> Dict[str, Any]:
        return {
            "variant": self.variant,
            "pretrained": self.pretrained,
            "freeze": self._freeze,
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CLIPEncoder":
        config = {k: v for k, v in config.items() if k != 'name'}
        return cls(**config)
