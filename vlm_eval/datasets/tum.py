"""TUM RGB-D dataset for semantic segmentation."""

import csv
import numpy as np
import torch
import torchvision.transforms as T
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image

from vlm_eval.core import BaseDataset, DatasetRegistry


@DatasetRegistry.register("tum")
class TUMDataset(BaseDataset):
    """TUM RGB-D semantic segmentation dataset.
    
    Args:
        root: Root directory containing the dataset.
        split: Dataset split (not typically used for TUM, but kept for compatibility).
        image_size: Target image size for resizing.
    """
    
    def __init__(
        self,
        root: str = "./data/tum",
        split: str = "train",
        image_size: int = 512,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        
        self.image_files = []
        self.mask_files = []
        self._class_names = []
        
        # Load class names from CSV if available
        # Expected structure:
        # root/
        #   rgb/
        #     1305031453.359684.png
        #   gt/
        #     1305031453.359684.png
        #   LabelColorMapping.csv
        
        csv_path = self.root / "LabelColorMapping.csv"
        if csv_path.exists():
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                # Skip header if present, or assume format
                # Format usually: name, r, g, b
                for row in reader:
                    if row and not row[0].startswith('#'):
                        self._class_names.append(row[0])
        else:
            # Fallback if CSV not found - generic placeholder
            self._class_names = [f"class_{i}" for i in range(95)]
            
        image_dir = self.root / "rgb"
        mask_dir = self.root / "gt"
        
        if image_dir.exists() and mask_dir.exists():
            images = sorted(list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")))
            
            for img_path in images:
                mask_path = mask_dir / img_path.name
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
        return len(self._class_names)

    @property
    def class_names(self) -> List[str]:
        return self._class_names

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
