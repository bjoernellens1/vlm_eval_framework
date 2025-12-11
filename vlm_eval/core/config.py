"""Pydantic configuration models for type-safe configuration."""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class PrecisionType(str, Enum):
    """Precision types for inference."""
    
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


class EncoderConfig(BaseModel):
    """Configuration for vision encoders.
    
    Attributes:
        name: Name of the encoder (must be registered).
        variant: Model variant (e.g., "base", "large").
        pretrained: Whether to load pretrained weights.
        kwargs: Additional encoder-specific parameters.
    """
    
    name: str = Field(..., description="Encoder name")
    variant: str = Field(default="base", description="Model variant")
    pretrained: bool = Field(default=True, description="Load pretrained weights")
    kwargs: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")
    
    class Config:
        """Pydantic config."""
        use_enum_values = True


class HeadConfig(BaseModel):
    """Configuration for segmentation heads.
    
    Attributes:
        name: Name of the head (must be registered).
        num_classes: Number of segmentation classes.
        freeze_encoder: Whether to freeze the encoder.
        kwargs: Additional head-specific parameters.
    """
    
    name: str = Field(..., description="Head name")
    num_classes: int = Field(..., gt=0, description="Number of classes")
    freeze_encoder: bool = Field(default=True, description="Freeze encoder")
    kwargs: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")
    
    class Config:
        """Pydantic config."""
        use_enum_values = True


class DatasetConfig(BaseModel):
    """Configuration for datasets.
    
    Attributes:
        name: Name of the dataset (must be registered).
        split: Dataset split (e.g., "train", "val", "test").
        root: Root directory containing the dataset.
        kwargs: Additional dataset-specific parameters.
    """
    
    name: str = Field(..., description="Dataset name")
    split: str = Field(default="val", description="Dataset split")
    root: str = Field(..., description="Dataset root directory")
    kwargs: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")
    
    class Config:
        """Pydantic config."""
        use_enum_values = True


class MetricsConfig(BaseModel):
    """Configuration for evaluation metrics.
    
    Attributes:
        compute_miou: Compute mean Intersection over Union.
        compute_boundary_f1: Compute boundary F1 score.
        compute_panoptic_quality: Compute panoptic quality.
        boundary_threshold: Threshold for boundary detection (pixels).
    """
    
    compute_miou: bool = Field(default=True, description="Compute mIoU")
    compute_boundary_f1: bool = Field(default=False, description="Compute boundary F1")
    compute_panoptic_quality: bool = Field(default=False, description="Compute PQ")
    boundary_threshold: int = Field(default=2, gt=0, description="Boundary threshold in pixels")
    
    class Config:
        """Pydantic config."""
        use_enum_values = True


class InferenceConfig(BaseModel):
    """Configuration for inference settings.
    
    Attributes:
        batch_size: Batch size for inference.
        device: Device to use ("cuda", "cpu", or specific GPU like "cuda:0").
        precision: Precision type for inference.
        num_workers: Number of data loading workers.
        pin_memory: Pin memory for faster GPU transfer.
    """
    
    batch_size: int = Field(default=16, gt=0, description="Batch size")
    device: str = Field(default="cuda", description="Device")
    precision: PrecisionType = Field(default=PrecisionType.FP32, description="Precision")
    num_workers: int = Field(default=4, ge=0, description="Number of workers")
    pin_memory: bool = Field(default=True, description="Pin memory")
    
    class Config:
        """Pydantic config."""
        use_enum_values = True


class ModelConfig(BaseModel):
    """Configuration for encoder + head pair.
    
    Attributes:
        encoder: Encoder configuration.
        head: Head configuration.
    """
    
    encoder: EncoderConfig = Field(..., description="Encoder config")
    head: HeadConfig = Field(..., description="Head config")
    
    class Config:
        """Pydantic config."""
        use_enum_values = True


class ExperimentConfig(BaseModel):
    """Complete experiment configuration.
    
    Attributes:
        name: Experiment name.
        encoder: Encoder configuration.
        head: Head configuration.
        dataset: Dataset configuration.
        metrics: Metrics configuration.
        inference: Inference configuration.
    """
    
    name: str = Field(default="experiment", description="Experiment name")
    encoder: EncoderConfig = Field(..., description="Encoder config")
    head: HeadConfig = Field(..., description="Head config")
    dataset: DatasetConfig = Field(..., description="Dataset config")
    metrics: MetricsConfig = Field(default_factory=MetricsConfig, description="Metrics config")
    inference: InferenceConfig = Field(
        default_factory=InferenceConfig,
        description="Inference config"
    )
    
    class Config:
        """Pydantic config."""
        use_enum_values = True


class ResultConfig(BaseModel):
    """Configuration for evaluation results.
    
    Attributes:
        experiment_name: Name of the experiment.
        encoder_name: Name of the encoder used.
        head_name: Name of the head used.
        dataset_name: Name of the dataset evaluated on.
        metrics: Dictionary of metric names to values.
        num_parameters: Total number of model parameters.
        num_trainable_parameters: Number of trainable parameters.
    """
    
    experiment_name: str = Field(..., description="Experiment name")
    encoder_name: str = Field(..., description="Encoder name")
    head_name: str = Field(..., description="Head name")
    dataset_name: str = Field(..., description="Dataset name")
    metrics: Dict[str, float] = Field(..., description="Metric results")
    num_parameters: int = Field(..., ge=0, description="Total parameters")
    num_trainable_parameters: int = Field(..., ge=0, description="Trainable parameters")
    
    class Config:
        """Pydantic config."""
        use_enum_values = True
    
    @field_validator("num_trainable_parameters")
    @classmethod
    def validate_trainable_params(cls, v: int, info: Any) -> int:
        """Validate that trainable params <= total params."""
        if "num_parameters" in info.data and v > info.data["num_parameters"]:
            raise ValueError(
                f"Trainable parameters ({v}) cannot exceed total parameters "
                f"({info.data['num_parameters']})"
            )
        return v
