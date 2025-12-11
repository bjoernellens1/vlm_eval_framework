"""Tests for new datasets (Replica, ScanNet, TUM)."""

import pytest
import torch
import shutil
import csv
from pathlib import Path
from PIL import Image
import numpy as np

from vlm_eval.datasets.replica import ReplicaDataset
from vlm_eval.datasets.scannet import ScanNetDataset
from vlm_eval.datasets.tum import TUMDataset


@pytest.fixture
def mock_replica_data(tmp_path):
    """Create mock Replica dataset structure."""
    root = tmp_path / "replica"
    scene_dir = root / "room_0"
    image_dir = scene_dir / "images"
    mask_dir = scene_dir / "semantic_class"
    
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    
    # Create dummy image and mask
    img = Image.new('RGB', (100, 100), color='red')
    mask = Image.new('L', (100, 100), color=1)
    
    img.save(image_dir / "frame_0.jpg")
    mask.save(mask_dir / "frame_0.png")
    
    return root


@pytest.fixture
def mock_scannet_data(tmp_path):
    """Create mock ScanNet dataset structure."""
    root = tmp_path / "scannet"
    scene_dir = root / "scene0000_00"
    image_dir = scene_dir / "color"
    mask_dir = scene_dir / "label"
    
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    
    img = Image.new('RGB', (100, 100), color='blue')
    mask = Image.new('L', (100, 100), color=2)
    
    img.save(image_dir / "0.jpg")
    mask.save(mask_dir / "0.png")
    
    return root


@pytest.fixture
def mock_tum_data(tmp_path):
    """Create mock TUM dataset structure."""
    root = tmp_path / "tum"
    image_dir = root / "rgb"
    mask_dir = root / "gt"
    
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    
    img = Image.new('RGB', (100, 100), color='green')
    mask = Image.new('L', (100, 100), color=3)
    
    img.save(image_dir / "1.png")
    mask.save(mask_dir / "1.png")
    
    # Create class mapping CSV
    with open(root / "LabelColorMapping.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["class_0", "0", "0", "0"])
        writer.writerow(["class_1", "1", "1", "1"])
        writer.writerow(["class_2", "2", "2", "2"])
    
    return root


def test_replica_dataset(mock_replica_data):
    """Test ReplicaDataset loading."""
    dataset = ReplicaDataset(root=str(mock_replica_data), scene_id="room_0")
    
    assert len(dataset) == 1
    assert dataset.num_classes == 98
    
    item = dataset[0]
    assert item["image"].shape == (3, 512, 512)
    assert item["mask"].shape == (512, 512)
    assert item["filename"] == "frame_0.jpg"


def test_scannet_dataset(mock_scannet_data):
    """Test ScanNetDataset loading."""
    dataset = ScanNetDataset(root=str(mock_scannet_data))
    
    assert len(dataset) == 1
    assert dataset.num_classes == 20
    
    item = dataset[0]
    assert item["image"].shape == (3, 512, 512)
    assert item["mask"].shape == (512, 512)
    assert item["filename"] == "0.jpg"


def test_tum_dataset(mock_tum_data):
    """Test TUMDataset loading."""
    dataset = TUMDataset(root=str(mock_tum_data))
    
    assert len(dataset) == 1
    assert dataset.num_classes == 3  # From mock CSV
    
    item = dataset[0]
    assert item["image"].shape == (3, 512, 512)
    assert item["mask"].shape == (512, 512)
    assert item["filename"] == "1.png"
