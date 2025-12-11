"""Test utilities and fixtures for VLM evaluation framework."""

from typing import Any, Dict

import pytest
import torch
import torch.nn as nn

from vlm_eval.core import BaseDataset, BaseEncoder, BaseSegmentationHead


class DummyEncoder(BaseEncoder):
    """Dummy encoder for testing purposes.
    
    A simple encoder that returns random features for testing.
    """
    
    def __init__(self, variant: str = "base", output_dim: int = 768, patch_size: int = 16):
        """Initialize dummy encoder.
        
        Args:
            variant: Model variant name.
            output_dim: Number of output channels.
            patch_size: Patch size for downsampling.
        """
        super().__init__()
        self.variant = variant
        self._output_dim = output_dim
        self._patch_size = patch_size
        
        # Simple conv layer for testing
        self.conv = nn.Conv2d(3, output_dim, kernel_size=patch_size, stride=patch_size)
    
    @property
    def output_channels(self) -> int:
        """Number of output channels."""
        return self._output_dim
    
    @property
    def patch_size(self) -> int:
        """Patch size."""
        return self._patch_size
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            images: Input images (B, 3, H, W).
        
        Returns:
            Features (B, C, H//patch_size, W//patch_size).
        """
        return self.conv(images)
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration."""
        return {
            "variant": self.variant,
            "output_dim": self._output_dim,
            "patch_size": self._patch_size,
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DummyEncoder":
        """Create from configuration."""
        return cls(**config)


class DummyHead(BaseSegmentationHead):
    """Dummy segmentation head for testing purposes.
    
    A simple head that upsamples features to produce segmentation logits.
    """
    
    def __init__(
        self,
        encoder: BaseEncoder,
        num_classes: int,
        freeze_encoder: bool = True
    ):
        """Initialize dummy head.
        
        Args:
            encoder: Encoder to use.
            num_classes: Number of classes.
            freeze_encoder: Whether to freeze encoder.
        """
        super().__init__(encoder, num_classes, freeze_encoder)
        
        # Simple 1x1 conv for classification
        self.classifier = nn.Conv2d(encoder.output_channels, num_classes, kernel_size=1)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            features: Encoder features (B, C, H//patch_size, W//patch_size).
        
        Returns:
            Logits (B, num_classes, H, W) at original resolution.
        """
        # Classify
        logits = self.classifier(features)
        
        # Upsample to original resolution
        logits = nn.functional.interpolate(
            logits,
            scale_factor=self.encoder.patch_size,
            mode="bilinear",
            align_corners=False
        )
        
        return logits
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration."""
        return {
            "num_classes": self.num_classes,
            "freeze_encoder": self.freeze_encoder,
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], encoder: BaseEncoder) -> "DummyHead":
        """Create from configuration."""
        return cls(encoder=encoder, **config)


class DummyDataset(BaseDataset):
    """Dummy dataset for testing purposes.
    
    Generates random images and masks.
    """
    
    def __init__(self, num_samples: int = 100, image_size: int = 224, num_classes: int = 21):
        """Initialize dummy dataset.
        
        Args:
            num_samples: Number of samples in dataset.
            image_size: Size of square images.
            num_classes: Number of classes.
        """
        super().__init__()
        self.num_samples = num_samples
        self.image_size = image_size
        self._num_classes = num_classes
    
    @property
    def num_classes(self) -> int:
        """Number of classes."""
        return self._num_classes
    
    @property
    def class_names(self) -> list:
        """Class names."""
        return [f"class_{i}" for i in range(self._num_classes)]
    
    @property
    def ignore_index(self) -> int:
        """Ignore index."""
        return 255
    
    def __len__(self) -> int:
        """Dataset length."""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item.
        
        Args:
            idx: Sample index.
        
        Returns:
            Dictionary with image, mask, filename, and image_id.
        """
        # Random image in [0, 1]
        image = torch.rand(3, self.image_size, self.image_size)
        
        # Random mask with class indices
        mask = torch.randint(0, self._num_classes, (self.image_size, self.image_size))
        
        return {
            "image": image,
            "mask": mask,
            "filename": f"image_{idx:05d}.jpg",
            "image_id": idx,
        }


@pytest.fixture
def dummy_encoder():
    """Fixture providing a dummy encoder."""
    return DummyEncoder(variant="test", output_dim=256, patch_size=16)


@pytest.fixture
def dummy_head(dummy_encoder):
    """Fixture providing a dummy head."""
    return DummyHead(encoder=dummy_encoder, num_classes=21, freeze_encoder=True)


@pytest.fixture
def dummy_dataset():
    """Fixture providing a dummy dataset."""
    return DummyDataset(num_samples=10, image_size=224, num_classes=21)


@pytest.fixture
def sample_batch():
    """Fixture providing a sample batch of images."""
    batch_size = 4
    height, width = 224, 224
    
    images = torch.rand(batch_size, 3, height, width)
    masks = torch.randint(0, 21, (batch_size, height, width))
    
    return {
        "images": images,
        "masks": masks,
    }


@pytest.fixture
def sample_config():
    """Fixture providing sample configuration dictionaries."""
    return {
        "encoder": {
            "name": "dummy",
            "variant": "base",
            "pretrained": False,
        },
        "head": {
            "name": "dummy",
            "num_classes": 21,
            "freeze_encoder": True,
        },
        "dataset": {
            "name": "dummy",
            "split": "val",
            "root": "/tmp/data",
        },
    }
