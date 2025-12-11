"""Tests for configuration models."""

import pytest
from pydantic import ValidationError

from vlm_eval.core.config import (
    DatasetConfig,
    EncoderConfig,
    ExperimentConfig,
    HeadConfig,
    InferenceConfig,
    MetricsConfig,
    ModelConfig,
    PrecisionType,
    ResultConfig,
)


class TestEncoderConfig:
    """Tests for EncoderConfig."""
    
    def test_encoder_config_valid(self):
        """Test valid encoder config."""
        config = EncoderConfig(
            name="radio",
            variant="base",
            pretrained=True,
        )
        
        assert config.name == "radio"
        assert config.variant == "base"
        assert config.pretrained is True
    
    def test_encoder_config_defaults(self):
        """Test encoder config defaults."""
        config = EncoderConfig(name="radio")
        
        assert config.variant == "base"
        assert config.pretrained is True
        assert config.kwargs == {}
    
    def test_encoder_config_missing_name(self):
        """Test that missing name raises error."""
        with pytest.raises(ValidationError):
            EncoderConfig()


class TestHeadConfig:
    """Tests for HeadConfig."""
    
    def test_head_config_valid(self):
        """Test valid head config."""
        config = HeadConfig(
            name="linear_probe",
            num_classes=21,
            freeze_encoder=True,
        )
        
        assert config.name == "linear_probe"
        assert config.num_classes == 21
        assert config.freeze_encoder is True
    
    def test_head_config_defaults(self):
        """Test head config defaults."""
        config = HeadConfig(name="linear_probe", num_classes=21)
        
        assert config.freeze_encoder is True
        assert config.kwargs == {}
    
    def test_head_config_invalid_num_classes(self):
        """Test that invalid num_classes raises error."""
        with pytest.raises(ValidationError):
            HeadConfig(name="linear_probe", num_classes=0)
        
        with pytest.raises(ValidationError):
            HeadConfig(name="linear_probe", num_classes=-1)


class TestDatasetConfig:
    """Tests for DatasetConfig."""
    
    def test_dataset_config_valid(self):
        """Test valid dataset config."""
        config = DatasetConfig(
            name="pascal_voc",
            split="val",
            root="/data/pascal",
        )
        
        assert config.name == "pascal_voc"
        assert config.split == "val"
        assert config.root == "/data/pascal"
    
    def test_dataset_config_defaults(self):
        """Test dataset config defaults."""
        config = DatasetConfig(name="pascal_voc", root="/data")
        
        assert config.split == "val"
        assert config.kwargs == {}


class TestMetricsConfig:
    """Tests for MetricsConfig."""
    
    def test_metrics_config_defaults(self):
        """Test metrics config defaults."""
        config = MetricsConfig()
        
        assert config.compute_miou is True
        assert config.compute_boundary_f1 is False
        assert config.compute_panoptic_quality is False
        assert config.boundary_threshold == 2
    
    def test_metrics_config_custom(self):
        """Test custom metrics config."""
        config = MetricsConfig(
            compute_miou=True,
            compute_boundary_f1=True,
            boundary_threshold=3,
        )
        
        assert config.compute_miou is True
        assert config.compute_boundary_f1 is True
        assert config.boundary_threshold == 3


class TestInferenceConfig:
    """Tests for InferenceConfig."""
    
    def test_inference_config_defaults(self):
        """Test inference config defaults."""
        config = InferenceConfig()
        
        assert config.batch_size == 16
        assert config.device == "cuda"
        assert config.precision == PrecisionType.FP32
        assert config.num_workers == 4
        assert config.pin_memory is True
    
    def test_inference_config_custom(self):
        """Test custom inference config."""
        config = InferenceConfig(
            batch_size=32,
            device="cpu",
            precision=PrecisionType.FP16,
            num_workers=8,
        )
        
        assert config.batch_size == 32
        assert config.device == "cpu"
        assert config.precision == PrecisionType.FP16
        assert config.num_workers == 8
    
    def test_inference_config_invalid_batch_size(self):
        """Test that invalid batch size raises error."""
        with pytest.raises(ValidationError):
            InferenceConfig(batch_size=0)


class TestModelConfig:
    """Tests for ModelConfig."""
    
    def test_model_config_valid(self):
        """Test valid model config."""
        encoder_config = EncoderConfig(name="radio")
        head_config = HeadConfig(name="linear_probe", num_classes=21)
        
        config = ModelConfig(encoder=encoder_config, head=head_config)
        
        assert config.encoder.name == "radio"
        assert config.head.name == "linear_probe"


class TestExperimentConfig:
    """Tests for ExperimentConfig."""
    
    def test_experiment_config_valid(self):
        """Test valid experiment config."""
        config = ExperimentConfig(
            name="test_experiment",
            encoder=EncoderConfig(name="radio"),
            head=HeadConfig(name="linear_probe", num_classes=21),
            dataset=DatasetConfig(name="pascal_voc", root="/data"),
        )
        
        assert config.name == "test_experiment"
        assert config.encoder.name == "radio"
        assert config.head.name == "linear_probe"
        assert config.dataset.name == "pascal_voc"
    
    def test_experiment_config_defaults(self):
        """Test experiment config defaults."""
        config = ExperimentConfig(
            encoder=EncoderConfig(name="radio"),
            head=HeadConfig(name="linear_probe", num_classes=21),
            dataset=DatasetConfig(name="pascal_voc", root="/data"),
        )
        
        assert config.name == "experiment"
        assert isinstance(config.metrics, MetricsConfig)
        assert isinstance(config.inference, InferenceConfig)
    
    def test_experiment_config_yaml_serialization(self):
        """Test YAML serialization."""
        config = ExperimentConfig(
            name="test",
            encoder=EncoderConfig(name="radio"),
            head=HeadConfig(name="linear_probe", num_classes=21),
            dataset=DatasetConfig(name="pascal_voc", root="/data"),
        )
        
        # Convert to dict (YAML-serializable)
        config_dict = config.model_dump()
        
        assert isinstance(config_dict, dict)
        assert config_dict["name"] == "test"
        assert config_dict["encoder"]["name"] == "radio"
        
        # Recreate from dict
        new_config = ExperimentConfig(**config_dict)
        assert new_config.name == config.name
        assert new_config.encoder.name == config.encoder.name


class TestResultConfig:
    """Tests for ResultConfig."""
    
    def test_result_config_valid(self):
        """Test valid result config."""
        config = ResultConfig(
            experiment_name="test",
            encoder_name="radio",
            head_name="linear_probe",
            dataset_name="pascal_voc",
            metrics={"miou": 0.75, "accuracy": 0.85},
            num_parameters=1000000,
            num_trainable_parameters=500000,
        )
        
        assert config.experiment_name == "test"
        assert config.metrics["miou"] == 0.75
        assert config.num_parameters == 1000000
        assert config.num_trainable_parameters == 500000
    
    def test_result_config_validation(self):
        """Test that trainable params validation works."""
        # Valid: trainable <= total
        config = ResultConfig(
            experiment_name="test",
            encoder_name="radio",
            head_name="linear_probe",
            dataset_name="pascal_voc",
            metrics={"miou": 0.75},
            num_parameters=1000000,
            num_trainable_parameters=500000,
        )
        assert config.num_trainable_parameters <= config.num_parameters
        
        # Invalid: trainable > total
        with pytest.raises(ValidationError, match="cannot exceed total parameters"):
            ResultConfig(
                experiment_name="test",
                encoder_name="radio",
                head_name="linear_probe",
                dataset_name="pascal_voc",
                metrics={"miou": 0.75},
                num_parameters=500000,
                num_trainable_parameters=1000000,
            )
