"""Abstract base class for segmentation datasets."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from torch.utils.data import Dataset


class BaseDataset(ABC, Dataset):
    """Abstract base class for segmentation datasets.
    
    All datasets must inherit from this class and implement the required
    abstract methods and properties. Datasets provide images and segmentation
    masks in a standardized format.
    
    Example:
        >>> @DatasetRegistry.register("pascal_voc")
        >>> class PascalVOCDataset(BaseDataset):
        ...     def __init__(self, root: str, split: str = "val"):
        ...         super().__init__()
        ...         self.root = root
        ...         self.split = split
        ...         # Load dataset metadata
        ...
        ...     @property
        ...     def num_classes(self) -> int:
        ...         return 21
        ...
        ...     @property
        ...     def class_names(self) -> List[str]:
        ...         return ["background", "aeroplane", ...]
        ...
        ...     @property
        ...     def ignore_index(self) -> int:
        ...         return 255
        ...
        ...     def __len__(self) -> int:
        ...         return len(self.image_files)
        ...
        ...     def __getitem__(self, idx: int) -> Dict[str, Any]:
        ...         # Load and process image and mask
        ...         return {
        ...             "image": image,  # (3, H, W) in [0, 1]
        ...             "mask": mask,    # (H, W) class indices
        ...             "filename": filename,
        ...             "image_id": idx
        ...         }
    """
    
    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Number of segmentation classes (including background).
        
        Returns:
            Number of classes.
        """
        pass
    
    @property
    @abstractmethod
    def class_names(self) -> List[str]:
        """List of class names.
        
        Returns:
            List of class names in order (index 0 = class_names[0]).
        """
        pass
    
    @property
    @abstractmethod
    def ignore_index(self) -> int:
        """Index to ignore in loss computation.
        
        Pixels with this value in the mask will be ignored during
        training and evaluation.
        
        Returns:
            Ignore index (commonly 255).
        """
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Number of samples in the dataset.
        
        Returns:
            Dataset size.
        """
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample from the dataset.
        
        Args:
            idx: Sample index.
        
        Returns:
            Dictionary containing:
                - image: torch.Tensor of shape (3, H, W) with values in [0, 1]
                - mask: torch.Tensor of shape (H, W) with class indices
                - filename: str, name of the image file
                - image_id: int, unique identifier for the image
        """
        pass
    
    def __repr__(self) -> str:
        """String representation of the dataset."""
        return (
            f"{self.__class__.__name__}(\n"
            f"  num_samples={len(self)},\n"
            f"  num_classes={self.num_classes},\n"
            f"  ignore_index={self.ignore_index}\n"
            f")"
        )
