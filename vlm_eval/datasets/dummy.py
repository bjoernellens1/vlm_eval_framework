"""Dummy dataset for testing and demonstrations."""

import torch
from typing import Any, Dict, List

from vlm_eval.core import BaseDataset, DatasetRegistry


@DatasetRegistry.register("dummy")
class DummyDataset(BaseDataset):
    """Dummy dataset that generates random images and masks.
    
    Useful for testing and demonstrations without requiring actual data.
    
    Args:
        num_samples: Number of samples in the dataset.
        image_size: Size of square images (default: 224).
        num_classes: Number of segmentation classes (default: 21).
        split: Dataset split (ignored for dummy data).
        root: Root directory (ignored for dummy data).
    """
    
    def __init__(
        self,
        num_samples: int = 100,
        image_size: int = 224,
        num_classes: int = 21,
        split: str = "train",
        root: str = "",
    ):
        super().__init__()
        self.num_samples = num_samples
        self.image_size = image_size
        self._num_classes = num_classes
        self.split = split
        self.root = root
    
    @property
    def num_classes(self) -> int:
        """Number of segmentation classes."""
        return self._num_classes
    
    @property
    def class_names(self) -> List[str]:
        """List of class names."""
        return [f"class_{i}" for i in range(self._num_classes)]
    
    @property
    def ignore_index(self) -> int:
        """Index to ignore in loss computation."""
        return 255
    
    def __len__(self) -> int:
        """Number of samples in the dataset."""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample from the dataset.
        
        Args:
            idx: Sample index.
        
        Returns:
            Dictionary containing image, mask, filename, and image_id.
        """
        # Generate random image in [0, 1]
        image = torch.rand(3, self.image_size, self.image_size)
        
        # Generate random segmentation mask
        mask = torch.randint(0, self._num_classes, (self.image_size, self.image_size))
        
        return {
            "image": image,
            "mask": mask,
            "filename": f"dummy_{idx:05d}.jpg",
            "image_id": idx,
        }
