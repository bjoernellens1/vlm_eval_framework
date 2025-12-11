"""Tests for base classes (BaseEncoder, BaseSegmentationHead, BaseDataset)."""

import pytest
import torch

from tests.conftest import DummyDataset, DummyEncoder, DummyHead


class TestBaseEncoder:
    """Tests for BaseEncoder interface."""
    
    def test_encoder_interface(self, dummy_encoder):
        """Test that encoder implements required interface."""
        # Check properties
        assert hasattr(dummy_encoder, "output_channels")
        assert hasattr(dummy_encoder, "patch_size")
        assert isinstance(dummy_encoder.output_channels, int)
        assert isinstance(dummy_encoder.patch_size, int)
        
        # Check methods
        assert hasattr(dummy_encoder, "forward")
        assert hasattr(dummy_encoder, "get_config")
        assert hasattr(dummy_encoder, "from_config")
        assert hasattr(dummy_encoder, "freeze")
        assert hasattr(dummy_encoder, "unfreeze")
    
    def test_encoder_forward(self, dummy_encoder):
        """Test encoder forward pass."""
        batch_size = 4
        height, width = 224, 224
        images = torch.rand(batch_size, 3, height, width)
        
        # Forward pass
        features = dummy_encoder(images)
        
        # Check output shape
        expected_h = height // dummy_encoder.patch_size
        expected_w = width // dummy_encoder.patch_size
        expected_shape = (batch_size, dummy_encoder.output_channels, expected_h, expected_w)
        
        assert features.shape == expected_shape
    
    def test_encoder_freeze(self, dummy_encoder):
        """Test encoder freezing."""
        # Initially unfrozen
        assert not dummy_encoder.is_frozen()
        assert all(p.requires_grad for p in dummy_encoder.parameters())
        
        # Freeze
        dummy_encoder.freeze()
        assert dummy_encoder.is_frozen()
        assert all(not p.requires_grad for p in dummy_encoder.parameters())
        
        # Unfreeze
        dummy_encoder.unfreeze()
        assert not dummy_encoder.is_frozen()
        assert all(p.requires_grad for p in dummy_encoder.parameters())
    
    def test_encoder_config(self, dummy_encoder):
        """Test encoder configuration serialization."""
        # Get config
        config = dummy_encoder.get_config()
        assert isinstance(config, dict)
        assert "variant" in config
        
        # Recreate from config
        new_encoder = DummyEncoder.from_config(config)
        assert new_encoder.output_channels == dummy_encoder.output_channels
        assert new_encoder.patch_size == dummy_encoder.patch_size
    
    def test_encoder_num_parameters(self, dummy_encoder):
        """Test parameter counting."""
        total_params = dummy_encoder.get_num_parameters()
        assert total_params > 0
        
        # Freeze and check trainable params
        dummy_encoder.freeze()
        trainable_params = dummy_encoder.get_num_parameters(trainable_only=True)
        assert trainable_params == 0
        
        # Unfreeze and check again
        dummy_encoder.unfreeze()
        trainable_params = dummy_encoder.get_num_parameters(trainable_only=True)
        assert trainable_params == total_params


class TestBaseSegmentationHead:
    """Tests for BaseSegmentationHead interface."""
    
    def test_head_interface(self, dummy_head):
        """Test that head implements required interface."""
        # Check attributes
        assert hasattr(dummy_head, "encoder")
        assert hasattr(dummy_head, "num_classes")
        assert hasattr(dummy_head, "freeze_encoder")
        
        # Check methods
        assert hasattr(dummy_head, "forward")
        assert hasattr(dummy_head, "get_config")
        assert hasattr(dummy_head, "from_config")
    
    def test_head_forward(self, dummy_head):
        """Test head forward pass."""
        batch_size = 4
        height, width = 224, 224
        images = torch.rand(batch_size, 3, height, width)
        
        # Get features from encoder
        features = dummy_head.encoder(images)
        
        # Forward through head
        logits = dummy_head(features)
        
        # Check output shape (should be upsampled to original resolution)
        expected_shape = (batch_size, dummy_head.num_classes, height, width)
        assert logits.shape == expected_shape
    
    def test_head_encoder_freezing(self):
        """Test that head respects encoder freezing."""
        encoder = DummyEncoder()
        
        # Create head with frozen encoder
        head_frozen = DummyHead(encoder, num_classes=21, freeze_encoder=True)
        assert head_frozen.encoder.is_frozen()
        assert all(not p.requires_grad for p in head_frozen.encoder.parameters())
        
        # Create head with unfrozen encoder
        encoder2 = DummyEncoder()
        head_unfrozen = DummyHead(encoder2, num_classes=21, freeze_encoder=False)
        assert not head_unfrozen.encoder.is_frozen()
        assert all(p.requires_grad for p in head_unfrozen.encoder.parameters())
    
    def test_head_train_mode(self):
        """Test that head respects encoder freezing in train mode."""
        encoder = DummyEncoder()
        head = DummyHead(encoder, num_classes=21, freeze_encoder=True)
        
        # Set to train mode
        head.train()
        
        # Encoder should still be in eval mode
        assert not head.encoder.training
        assert head.training
    
    def test_head_config(self, dummy_head):
        """Test head configuration serialization."""
        # Get config
        config = dummy_head.get_config()
        assert isinstance(config, dict)
        assert "num_classes" in config
        assert "freeze_encoder" in config
        
        # Recreate from config
        new_head = DummyHead.from_config(config, encoder=dummy_head.encoder)
        assert new_head.num_classes == dummy_head.num_classes
        assert new_head.freeze_encoder == dummy_head.freeze_encoder
    
    def test_head_num_parameters(self, dummy_head):
        """Test parameter counting."""
        total_params = dummy_head.get_num_parameters()
        head_params = dummy_head.get_head_parameters()
        encoder_params = dummy_head.encoder.get_num_parameters()
        
        # Total should equal encoder + head
        assert total_params == encoder_params + head_params
        
        # With frozen encoder, trainable should only be head
        trainable_params = dummy_head.get_num_parameters(trainable_only=True)
        assert trainable_params == head_params


class TestBaseDataset:
    """Tests for BaseDataset interface."""
    
    def test_dataset_interface(self, dummy_dataset):
        """Test that dataset implements required interface."""
        # Check properties
        assert hasattr(dummy_dataset, "num_classes")
        assert hasattr(dummy_dataset, "class_names")
        assert hasattr(dummy_dataset, "ignore_index")
        
        # Check methods
        assert hasattr(dummy_dataset, "__len__")
        assert hasattr(dummy_dataset, "__getitem__")
    
    def test_dataset_length(self, dummy_dataset):
        """Test dataset length."""
        assert len(dummy_dataset) == 10
    
    def test_dataset_getitem(self, dummy_dataset):
        """Test dataset item retrieval."""
        item = dummy_dataset[0]
        
        # Check keys
        assert "image" in item
        assert "mask" in item
        assert "filename" in item
        assert "image_id" in item
        
        # Check types and shapes
        assert isinstance(item["image"], torch.Tensor)
        assert isinstance(item["mask"], torch.Tensor)
        assert isinstance(item["filename"], str)
        assert isinstance(item["image_id"], int)
        
        # Check image shape (3, H, W)
        assert item["image"].ndim == 3
        assert item["image"].shape[0] == 3
        
        # Check mask shape (H, W)
        assert item["mask"].ndim == 2
        
        # Check value ranges
        assert item["image"].min() >= 0.0
        assert item["image"].max() <= 1.0
        assert item["mask"].min() >= 0
        assert item["mask"].max() < dummy_dataset.num_classes
    
    def test_dataset_properties(self, dummy_dataset):
        """Test dataset properties."""
        assert dummy_dataset.num_classes == 21
        assert len(dummy_dataset.class_names) == 21
        assert dummy_dataset.ignore_index == 255
