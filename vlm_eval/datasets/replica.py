"""Replica dataset for semantic segmentation."""

import json
import numpy as np
import torch
import torchvision.transforms as T
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image

from vlm_eval.core import BaseDataset, DatasetRegistry


@DatasetRegistry.register("replica")
class ReplicaDataset(BaseDataset):
    """Replica semantic segmentation dataset.
    
    Args:
        root: Root directory containing the dataset.
        split: Dataset split (not used for Replica as it's typically per-scene).
        scene_id: Specific scene ID to load (e.g., 'room_0').
        image_size: Target image size for resizing.
    """
    
    # Placeholder for the 88 class names. 
    # Ideally this should be loaded from semantic.json or a predefined list.
    # Using a generic list for now as the full 88 classes are specific.
    CLASS_NAMES = [
        "undefined", "backpack", "base-cabinet", "basket", "bathtub", "beam", "beanbag", "bed", "bench", "bike",
        "bin", "blanket", "blinds", "book", "bottle", "box", "bowl", "camera", "cabinet", "candle", "chair",
        "chopping-board", "clock", "cloth", "clothing", "coaster", "comforter", "computer-keyboard", "cup",
        "cushion", "curtain", "desk", "dining-table", "dishwasher", "door", "exercise-ball", "faucet", "floor",
        "flower-pot", "fork", "fridge", "glass", "guitar", "hair-dryer", "hamper", "handrail", "headphones",
        "indoor-plant", "knife", "lamp", "laptop", "ledge", "light-switch", "microwave", "monitor", "mouse",
        "nightstand", "painting", "panel", "paper-towel", "phone", "picture", "pillar", "pillow", "pipe",
        "plant-stand", "plate", "pot", "rack", "remote-control", "scarf", "sculpture", "shampoo", "shoes",
        "shower-stall", "sink", "small-appliance", "sofa", "stair", "stool", "switch", "table", "tablet",
        "tissue-box", "toilet", "towel", "tv-screen", "umbrella", "vase", "vent", "wall", "wall-plug",
        "wardrobe", "window", "rug", "logo", "bag", "set-of-clothing"
    ]

    def __init__(
        self,
        root: str = "./data/replica",
        split: str = "train",  # Kept for compatibility, though Replica is often just scenes
        scene_id: Optional[str] = None,
        image_size: int = 512,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.scene_id = scene_id
        self.image_size = image_size
        
        self.image_files = []
        self.mask_files = []
        
        # Scan directory for images and masks
        # Expected structure:
        # root/
        #   scene_id/
        #     images/
        #       frame_00000.jpg
        #     semantic_class/
        #       frame_00000.png
        
        search_path = self.root
        if scene_id:
            search_path = search_path / scene_id
            
        # Find all scene directories if no specific scene is given
        if not scene_id:
            scenes = [d for d in search_path.iterdir() if d.is_dir()]
        else:
            scenes = [search_path]
            
        for scene in scenes:
            image_dir = scene / "images"
            mask_dir = scene / "semantic_class"
            
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
