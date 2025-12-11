"""ScanNet dataset for semantic segmentation."""

import numpy as np
import torch
import torchvision.transforms as T
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image

from vlm_eval.core import BaseDataset, DatasetRegistry


@DatasetRegistry.register("scannet")
class ScanNetDataset(BaseDataset):
    """ScanNet semantic segmentation dataset (20-class subset).
    
    Args:
        root: Root directory containing the dataset.
        split: Dataset split ('train', 'val').
        image_size: Target image size for resizing.
    """
    
    # NYUv2 20-class subset
    CLASS_NAMES = [
        "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", 
        "window", "bookshelf", "picture", "counter", "desk", "curtain", 
        "refrigerator", "shower curtain", "toilet", "sink", "bathtub", "garbagebin"
    ]

    def __init__(
        self,
        root: str = "./data/scannet",
        split: str = "train",
        image_size: int = 512,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        
        self.image_files = []
        self.mask_files = []
        
        # Scan directory for images and masks
        # Expected structure:
        # root/
        #   scene_id/
        #     color/
        #       0.jpg
        #     label/
        #       0.png
        
        # Find all scene directories
        scenes = [d for d in self.root.iterdir() if d.is_dir()]
        
        # Simple split logic (can be improved with actual split files)
        # For now, we'll just use all found scenes as we don't have the split files
        # In a real scenario, we would filter `scenes` based on `split`
        
        for scene in scenes:
            image_dir = scene / "color"
            mask_dir = scene / "label"
            
            if not image_dir.exists() or not mask_dir.exists():
                continue
                
            # Sort to ensure alignment
            images = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
            
            for img_path in images:
                # Assuming mask has same basename but might be png
                mask_path = mask_dir / f"{img_path.stem}.png"
                if mask_path.exists():
                    self.image_files.append(img_path)
                    self.mask_files.append(mask_path)
        
        if not self.image_files:
            print(f"Warning: No images found in {self.root}")
            
        self.image_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])
        
        self.mask_transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),
        ])

    @property
    def num_classes(self) -> int:
        return len(self.CLASS_NAMES)

    @property
    def class_names(self) -> List[str]:
        return self.CLASS_NAMES

    @property
    def ignore_index(self) -> int:
        return 255

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_path = self.image_files[idx]
        mask_path = self.mask_files[idx]
        
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)
        
        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        
        return {
            "image": image,
            "mask": mask,
            "filename": image_path.name,
            "image_id": idx,
        }
