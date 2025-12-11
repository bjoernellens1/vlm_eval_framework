"""Core components of the VLM evaluation framework."""

from vlm_eval.core.base_dataset import BaseDataset
from vlm_eval.core.base_encoder import BaseEncoder
from vlm_eval.core.base_head import BaseHead, BaseSegmentationHead
from vlm_eval.core.registry import DatasetRegistry, EncoderRegistry, HeadRegistry

__all__ = [
    "BaseEncoder",
    "BaseHead",
    "BaseSegmentationHead",
    "BaseDataset",
    "EncoderRegistry",
    "HeadRegistry",
    "DatasetRegistry",
]
