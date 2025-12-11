"""Tests for registry system (EncoderRegistry, HeadRegistry, DatasetRegistry)."""

import pytest

from tests.conftest import DummyDataset, DummyEncoder, DummyHead
from vlm_eval.core import DatasetRegistry, EncoderRegistry, HeadRegistry


class TestEncoderRegistry:
    """Tests for EncoderRegistry."""
    
    def setup_method(self):
        """Clear registry before each test."""
        EncoderRegistry.clear()
    
    def test_encoder_registry_registration(self):
        """Test encoder registration."""
        @EncoderRegistry.register("test_encoder")
        class TestEncoder(DummyEncoder):
            pass
        
        assert EncoderRegistry.is_registered("test_encoder")
        assert "test_encoder" in EncoderRegistry.list_available()
    
    def test_encoder_registry_get(self):
        """Test encoder retrieval."""
        @EncoderRegistry.register("test_encoder")
        class TestEncoder(DummyEncoder):
            pass
        
        encoder = EncoderRegistry.get("test_encoder", variant="base")
        assert isinstance(encoder, DummyEncoder)
        assert encoder.variant == "base"
    
    def test_encoder_registry_duplicate_error(self):
        """Test that duplicate registration raises error."""
        @EncoderRegistry.register("test_encoder")
        class TestEncoder1(DummyEncoder):
            pass
        
        with pytest.raises(ValueError, match="already registered"):
            @EncoderRegistry.register("test_encoder")
            class TestEncoder2(DummyEncoder):
                pass
    
    def test_encoder_registry_not_found(self):
        """Test that retrieving unregistered encoder raises error."""
        with pytest.raises(ValueError, match="not found"):
            EncoderRegistry.get("nonexistent_encoder")
    
    def test_encoder_registry_type_checking(self):
        """Test that non-encoder class raises error."""
        with pytest.raises(ValueError, match="must inherit from BaseEncoder"):
            @EncoderRegistry.register("invalid")
            class InvalidClass:
                pass
    
    def test_encoder_registry_from_config(self):
        """Test encoder creation from config."""
        @EncoderRegistry.register("test_encoder")
        class TestEncoder(DummyEncoder):
            pass
        
        config = {
            "name": "test_encoder",
            "variant": "large",
            "output_dim": 1024,
        }
        
        encoder = EncoderRegistry.from_config(config)
        assert isinstance(encoder, DummyEncoder)
        assert encoder.variant == "large"
        assert encoder.output_channels == 1024
    
    def test_encoder_registry_list_available(self):
        """Test listing available encoders."""
        @EncoderRegistry.register("encoder1")
        class Encoder1(DummyEncoder):
            pass
        
        @EncoderRegistry.register("encoder2")
        class Encoder2(DummyEncoder):
            pass
        
        available = EncoderRegistry.list_available()
        assert "encoder1" in available
        assert "encoder2" in available
        assert len(available) == 2


class TestHeadRegistry:
    """Tests for HeadRegistry."""
    
    def setup_method(self):
        """Clear registry before each test."""
        HeadRegistry.clear()
    
    def test_head_registry_registration(self):
        """Test head registration."""
        @HeadRegistry.register("test_head")
        class TestHead(DummyHead):
            pass
        
        assert HeadRegistry.is_registered("test_head")
        assert "test_head" in HeadRegistry.list_available()
    
    def test_head_registry_get(self):
        """Test head retrieval."""
        @HeadRegistry.register("test_head")
        class TestHead(DummyHead):
            pass
        
        encoder = DummyEncoder()
        head = HeadRegistry.get("test_head", encoder=encoder, num_classes=21)
        assert isinstance(head, DummyHead)
        assert head.num_classes == 21
    
    def test_head_registry_duplicate_error(self):
        """Test that duplicate registration raises error."""
        @HeadRegistry.register("test_head")
        class TestHead1(DummyHead):
            pass
        
        with pytest.raises(ValueError, match="already registered"):
            @HeadRegistry.register("test_head")
            class TestHead2(DummyHead):
                pass
    
    def test_head_registry_not_found(self):
        """Test that retrieving unregistered head raises error."""
        encoder = DummyEncoder()
        with pytest.raises(ValueError, match="not found"):
            HeadRegistry.get("nonexistent_head", encoder=encoder)
    
    def test_head_registry_type_checking(self):
        """Test that non-head class raises error."""
        with pytest.raises(ValueError, match="must inherit from BaseSegmentationHead"):
            @HeadRegistry.register("invalid")
            class InvalidClass:
                pass
    
    def test_head_registry_from_config(self):
        """Test head creation from config."""
        @HeadRegistry.register("test_head")
        class TestHead(DummyHead):
            pass
        
        encoder = DummyEncoder()
        config = {
            "name": "test_head",
            "num_classes": 150,
            "freeze_encoder": False,
        }
        
        head = HeadRegistry.from_config(config, encoder)
        assert isinstance(head, DummyHead)
        assert head.num_classes == 150
        assert head.freeze_encoder is False
    
    def test_head_registry_list_available(self):
        """Test listing available heads."""
        @HeadRegistry.register("head1")
        class Head1(DummyHead):
            pass
        
        @HeadRegistry.register("head2")
        class Head2(DummyHead):
            pass
        
        available = HeadRegistry.list_available()
        assert "head1" in available
        assert "head2" in available
        assert len(available) == 2


class TestDatasetRegistry:
    """Tests for DatasetRegistry."""
    
    def setup_method(self):
        """Clear registry before each test."""
        DatasetRegistry.clear()
    
    def test_dataset_registry_registration(self):
        """Test dataset registration."""
        @DatasetRegistry.register("test_dataset")
        class TestDataset(DummyDataset):
            pass
        
        assert DatasetRegistry.is_registered("test_dataset")
        assert "test_dataset" in DatasetRegistry.list_available()
    
    def test_dataset_registry_get(self):
        """Test dataset retrieval."""
        @DatasetRegistry.register("test_dataset")
        class TestDataset(DummyDataset):
            pass
        
        dataset = DatasetRegistry.get("test_dataset", num_samples=50)
        assert isinstance(dataset, DummyDataset)
        assert len(dataset) == 50
    
    def test_dataset_registry_duplicate_error(self):
        """Test that duplicate registration raises error."""
        @DatasetRegistry.register("test_dataset")
        class TestDataset1(DummyDataset):
            pass
        
        with pytest.raises(ValueError, match="already registered"):
            @DatasetRegistry.register("test_dataset")
            class TestDataset2(DummyDataset):
                pass
    
    def test_dataset_registry_not_found(self):
        """Test that retrieving unregistered dataset raises error."""
        with pytest.raises(ValueError, match="not found"):
            DatasetRegistry.get("nonexistent_dataset")
    
    def test_dataset_registry_type_checking(self):
        """Test that non-dataset class raises error."""
        with pytest.raises(ValueError, match="must inherit from BaseDataset"):
            @DatasetRegistry.register("invalid")
            class InvalidClass:
                pass
    
    def test_dataset_registry_from_config(self):
        """Test dataset creation from config."""
        @DatasetRegistry.register("test_dataset")
        class TestDataset(DummyDataset):
            pass
        
        config = {
            "name": "test_dataset",
            "num_samples": 200,
            "num_classes": 50,
        }
        
        dataset = DatasetRegistry.from_config(config)
        assert isinstance(dataset, DummyDataset)
        assert len(dataset) == 200
        assert dataset.num_classes == 50
    
    def test_dataset_registry_list_available(self):
        """Test listing available datasets."""
        @DatasetRegistry.register("dataset1")
        class Dataset1(DummyDataset):
            pass
        
        @DatasetRegistry.register("dataset2")
        class Dataset2(DummyDataset):
            pass
        
        available = DatasetRegistry.list_available()
        assert "dataset1" in available
        assert "dataset2" in available
        assert len(available) == 2
