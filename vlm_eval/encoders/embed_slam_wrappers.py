import torch
import numpy as np
from typing import Any, Dict, Optional, List, Union
from pathlib import Path
import os

from vlm_eval.core import BaseEncoder, EncoderRegistry

# Import from the copied files
# We assume the directory structure is vlm_eval/encoders/embed_slam/
try:
    from vlm_eval.encoders.embed_slam.concept_fusion import ConceptFusion
    from vlm_eval.encoders.embed_slam.dino_fusion import DINOFusion
    from vlm_eval.encoders.embed_slam.x_fusion import XFusion
    from vlm_eval.encoders.embed_slam.naradio_fusion import NARadioFusion
except ImportError as e:
    print(f"Warning: Failed to import embed_slam models: {e}")
    ConceptFusion = None
    DINOFusion = None
    XFusion = None
    NARadioFusion = None

@EncoderRegistry.register("concept_fusion")
class ConceptFusionEncoder(BaseEncoder):
    def __init__(
        self, 
        sam_checkpoint: str = "sam_vit_b_01ec64.pth", 
        sam_variant: str = "SAM", 
        device: str = "cuda"
    ):
        super().__init__()
        if ConceptFusion is None:
            raise ImportError("ConceptFusion could not be imported. Check dependencies.")
            
        self.sam_checkpoint = sam_checkpoint
        self.sam_variant = sam_variant
        self._device_str = device
        
        # ConceptFusion expects a Path object for checkpoint
        checkpoint_path = Path(sam_checkpoint) if sam_checkpoint else None
        
        # Check if checkpoint exists, if not check cache
        if checkpoint_path and not checkpoint_path.exists():
            cache_path = Path(os.path.expanduser("~/.cache/torch/hub/checkpoints")) / sam_checkpoint
            if cache_path.exists():
                checkpoint_path = cache_path
        
        self.model = ConceptFusion(
            sam_checkpoint=checkpoint_path,
            sam=sam_variant,
            device=device
        )
        
        # ViT-H-14 output dim
        self._output_channels = 1024 

    @property
    def output_channels(self) -> int:
        return self._output_channels

    @property
    def patch_size(self) -> int:
        return 1

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: (B, 3, H, W)
        outputs = []
        for i in range(images.shape[0]):
            # Convert to numpy (H, W, 3)
            img = images[i].permute(1, 2, 0)
            if img.max() <= 1.0:
                img = img * 255.0
            img_np = img.cpu().numpy().astype(np.uint8)
            
            feat = self.model.process(img_np) # (H, W, D)
            outputs.append(feat.permute(2, 0, 1)) # (D, H, W)
        
        return torch.stack(outputs)

    def encode_text(self, text: List[str]) -> torch.Tensor:
        # text_tokens = self.model.clip_tokenizer(text).to(self.model.device)
        # text_features = self.model.clip_model.encode_text(text_tokens)
        inputs = self.model.clip_processor(text=text, return_tensors="pt", padding=True).to(self.model.device)
        text_features = self.model.clip_model.get_text_features(**inputs)
        return torch.nn.functional.normalize(text_features, dim=-1)

    def get_config(self) -> Dict[str, Any]:
        return {
            "sam_checkpoint": str(self.sam_checkpoint),
            "sam_variant": self.sam_variant,
            "device": self._device_str
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ConceptFusionEncoder":
        return cls(**config)


@EncoderRegistry.register("dino_fusion")
class DINOFusionEncoder(BaseEncoder):
    def __init__(self, device: str = "cuda"):
        super().__init__()
        if DINOFusion is None:
            raise ImportError("DINOFusion could not be imported. Check dependencies.")
            
        self._device_str = device
        self.model = DINOFusion(device=device)
        
        # DINOv3 ViT-L-16 output dim
        self._output_channels = 1024 # Need to verify DINOv3 dim

    @property
    def output_channels(self) -> int:
        return self._output_channels

    @property
    def patch_size(self) -> int:
        return 1

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = []
        for i in range(images.shape[0]):
            img = images[i].permute(1, 2, 0)
            if img.max() <= 1.0:
                img = img * 255.0
            img_np = img.cpu().numpy().astype(np.uint8)
            
            feat = self.model.process(img_np)
            outputs.append(feat.permute(2, 0, 1))
            
        return torch.stack(outputs)

    def encode_text(self, text: List[str]) -> torch.Tensor:
        # tokenized_texts_tensor = self.model.dino_tokenizer.tokenize(text).to(self.model.device)
        # # DINOv3 text encoder usage from dino_fusion.py
        # textfeat = self.model.dino_model.encode_text(tokenized_texts_tensor)[:, 1024:]
        # return torch.nn.functional.normalize(textfeat.float(), p=2, dim=1)
        raise NotImplementedError("Text encoding is not supported for DINOv3 loaded via transformers (vision only).")

    def get_config(self) -> Dict[str, Any]:
        return {"device": self._device_str}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DINOFusionEncoder":
        return cls(**config)


@EncoderRegistry.register("x_fusion")
class XFusionEncoder(BaseEncoder):
    def __init__(
        self, 
        sam_checkpoint: str = "sam_vit_b_01ec64.pth", 
        sam_variant: str = "SAM", 
        device: str = "cuda"
    ):
        super().__init__()
        if XFusion is None:
            raise ImportError("XFusion could not be imported. Check dependencies.")
            
        self.sam_checkpoint = sam_checkpoint
        self.sam_variant = sam_variant
        self._device_str = device
        
        checkpoint_path = Path(sam_checkpoint) if sam_checkpoint else None
        
        # Check if checkpoint exists, if not check cache
        if checkpoint_path and not checkpoint_path.exists():
            cache_path = Path(os.path.expanduser("~/.cache/torch/hub/checkpoints")) / sam_checkpoint
            if cache_path.exists():
                checkpoint_path = cache_path
        
        self.model = XFusion(
            sam_checkpoint=checkpoint_path,
            sam=sam_variant,
            device=device
        )
        
        # CLIP ViT-B-16 output dim
        self._output_channels = 512

    @property
    def output_channels(self) -> int:
        return self._output_channels

    @property
    def patch_size(self) -> int:
        return 1

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = []
        for i in range(images.shape[0]):
            img = images[i].permute(1, 2, 0)
            if img.max() <= 1.0:
                img = img * 255.0
            img_np = img.cpu().numpy().astype(np.uint8)
            
            feat = self.model.process(img_np)
            outputs.append(feat.permute(2, 0, 1))
            
        return torch.stack(outputs)

    def encode_text(self, text: List[str]) -> torch.Tensor:
        # text_tokens = self.model.clip_tokenizer(text).to(self.model.device)
        # text_features = self.model.clip_model.encode_text(text_tokens)
        inputs = self.model.clip_processor(text=text, return_tensors="pt", padding=True).to(self.model.device)
        text_features = self.model.clip_model.get_text_features(**inputs)
        return torch.nn.functional.normalize(text_features, dim=-1)

    def get_config(self) -> Dict[str, Any]:
        return {
            "sam_checkpoint": str(self.sam_checkpoint),
            "sam_variant": self.sam_variant,
            "device": self._device_str
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "XFusionEncoder":
        return cls(**config)


@EncoderRegistry.register("naradio_fusion")
class NARadioFusionEncoder(BaseEncoder):
    def __init__(self, device: str = "cuda"):
        super().__init__()
        if NARadioFusion is None:
            raise ImportError("NARadioFusion could not be imported. Check dependencies.")
            
        self._device_str = device
        self.model = NARadioFusion(device=device)
        
        # NaRadio output dim (SigLIP)
        self._output_channels = 1152 # Need to check SigLIP dim, usually 1152 for SO400M or similar

    @property
    def output_channels(self) -> int:
        return self._output_channels

    @property
    def patch_size(self) -> int:
        return 1

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = []
        for i in range(images.shape[0]):
            img = images[i].permute(1, 2, 0)
            if img.max() <= 1.0:
                img = img * 255.0
            img_np = img.cpu().numpy().astype(np.uint8)
            
            feat = self.model.process(img_np)
            outputs.append(feat.permute(2, 0, 1))
            
        return torch.stack(outputs)

    def encode_text(self, text: List[str]) -> torch.Tensor:
        prompt_embeddings = self.model.naradio.encode_prompts(text)
        return torch.nn.functional.normalize(prompt_embeddings, dim=-1)

    def get_config(self) -> Dict[str, Any]:
        return {"device": self._device_str}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "NARadioFusionEncoder":
        return cls(**config)
