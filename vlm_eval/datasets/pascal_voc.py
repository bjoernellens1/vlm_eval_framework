"""Pascal VOC dataset for semantic segmentation."""

import torch
from typing import Any, Dict, List, Optional
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from torchvision.datasets import VOCSegmentation

from vlm_eval.core import BaseDataset, DatasetRegistry


@DatasetRegistry.register("pascal_voc")
class PascalVOCDataset(BaseDataset):
    """Pascal VOC 2012 semantic segmentation dataset.
    
    Wraps torchvision's VOCSegmentation with additional features:
    - Subset mode for faster testing
    - Consistent tensor output format
    - Proper class names and ignore index
    
    Args:
        root: Root directory where dataset will be downloaded/stored.
        split: Dataset split ('train', 'val', 'trainval').
        year: Dataset year (default: '2012').
        download: Whether to download the dataset if not present.
        subset_size: If specified, only use first N samples (for fast testing).
        image_size: Target image size for resizing (default: 512).
    """
    
    # Pascal VOC class names
    CLASS_NAMES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
        'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
        'sofa', 'train', 'tvmonitor'
    ]
    
    def __init__(
        self,
        root: str = "./data/pascal_voc",
        split: str = "val",
        year: str = "2012",
        download: bool = True,
        subset_size: Optional[int] = None,
        image_size: int = 512,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.year = year
        self.subset_size = subset_size
        self.image_size = image_size
        
        # Create underlying VOC dataset
        # Note: The official VOC download link is no longer available
        # Users need to download manually from alternative sources
        try:
            self.voc_dataset = VOCSegmentation(
                root=str(self.root),
                year=year,
                image_set=split,
                download=False,  # Don't use automatic download (link is dead)
            )
        except RuntimeError as e:
            if "not found" in str(e).lower():
                raise RuntimeError(
                    f"\n{'='*70}\n"
                    f"Pascal VOC 2012 dataset not found at {self.root}\n"
                    f"{'='*70}\n\n"
                    f"The official download link is no longer available.\n"
                    f"Please download the dataset manually from one of these sources:\n\n"
                    f"1. Kaggle (recommended):\n"
                    f"   https://www.kaggle.com/datasets/huanghanchina/pascal-voc-2012\n"
                    f"   - Download and extract to: {self.root}\n\n"
                    f"2. Alternative mirrors:\n"
                    f"   - Check torchvision documentation for updated links\n"
                    f"   - Search for 'VOCtrainval_11-May-2012.tar' on academic mirrors\n\n"
                    f"Expected directory structure:\n"
                    f"  {self.root}/\n"
                    f"    └── VOCdevkit/\n"
                    f"        └── VOC2012/\n"
                    f"            ├── JPEGImages/\n"
                    f"            ├── SegmentationClass/\n"
                    f"            └── ImageSets/\n\n"
                    f"For quick testing, you can use the DummyDataset instead:\n"
                    f"  dataset = DatasetRegistry.get('dummy', num_samples=50)\n"
                    f"{'='*70}\n"
                )
            else:
                raise
        
        # Determine actual dataset size
        self._length = len(self.voc_dataset)
        if subset_size is not None:
            self._length = min(subset_size, self._length)
        
        # Image transforms
        self.image_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])
        
        # Mask transforms (no normalization, keep as PIL for proper resizing)
        self.mask_transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),
        ])
    
    @property
    def num_classes(self) -> int:
        """Number of segmentation classes (21 for Pascal VOC)."""
        return 21
    
    @property
    def class_names(self) -> List[str]:
        """List of class names."""
        return self.CLASS_NAMES
    
    @property
    def ignore_index(self) -> int:
        """Index to ignore in loss computation (255 for Pascal VOC)."""
        return 255
    
    def __len__(self) -> int:
        """Number of samples in the dataset."""
        return self._length
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample from the dataset.
        
        Args:
            idx: Sample index.
        
        Returns:
            Dictionary containing:
                - image: Tensor of shape (3, H, W) in [0, 1]
                - mask: Tensor of shape (H, W) with class indices
                - filename: Original filename
                - image_id: Sample index
        """
        if idx >= self._length:
            raise IndexError(f"Index {idx} out of range for dataset of size {self._length}")
        
        # Get image and mask from VOC dataset
        image, mask = self.voc_dataset[idx]
        
        # Apply transforms
        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        
        # Convert mask to tensor (PIL Image -> Tensor)
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        
        # Get filename from VOC dataset
        filename = self.voc_dataset.images[idx]
        filename = Path(filename).name
        
        return {
            "image": image,
            "mask": mask,
            "filename": filename,
            "image_id": idx,
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary."""
        return {
            "root": str(self.root),
            "split": self.split,
            "year": self.year,
            "subset_size": self.subset_size,
            "image_size": self.image_size,
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PascalVOCDataset":
        """Create dataset from configuration dictionary."""
        return cls(**config)


# Import numpy for mask conversion
import numpy as np
