"""VLM Evaluation Framework - A modular framework for evaluating vision encoders."""

__version__ = "0.1.0"

from vlm_eval.core import (
    BaseDataset,
    BaseEncoder,
    BaseSegmentationHead,
    DatasetRegistry,
    EncoderRegistry,
    HeadRegistry,
)

# Import implementations to trigger registration
# This ensures all encoders, heads, and datasets are registered when the package is imported
from vlm_eval import encoders, heads, datasets  # noqa: F401

__all__ = [
    "BaseEncoder",
    "BaseSegmentationHead",
    "BaseDataset",
    "EncoderRegistry",
    "HeadRegistry",
    "DatasetRegistry",
]
