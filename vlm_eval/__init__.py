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

__all__ = [
    "BaseEncoder",
    "BaseSegmentationHead",
    "BaseDataset",
    "EncoderRegistry",
    "HeadRegistry",
    "DatasetRegistry",
]
