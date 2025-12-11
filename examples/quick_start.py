#!/usr/bin/env python3
"""
Quick Start Example - VLM Evaluation Framework

This script demonstrates how to use the framework with the SimpleCNN encoder
and dummy dataset. It can run immediately without downloading data or weights.
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import framework components
from vlm_eval.core import EncoderRegistry, HeadRegistry, DatasetRegistry
from vlm_eval.encoders import SimpleCNNEncoder
from vlm_eval.heads import LinearProbeHead
from vlm_eval.datasets import DummyDataset


def main():
    print("=" * 80)
    print("VLM Evaluation Framework - Quick Start Example")
    print("=" * 80)
    
    # 1. Create encoder
    print("\n[1/5] Creating encoder...")
    encoder = EncoderRegistry.get("simple_cnn", variant="base", pretrained=False)
    print(f"✓ Created encoder: {encoder.__class__.__name__}")
    print(f"  - Output channels: {encoder.output_channels}")
    print(f"  - Patch size: {encoder.patch_size}")
    print(f"  - Parameters: {encoder.get_num_parameters():,}")
    
    # 2. Create segmentation head
    print("\n[2/5] Creating segmentation head...")
    head = HeadRegistry.get(
        "linear_probe",
        encoder=encoder,
        num_classes=21,
        freeze_encoder=False  # Train encoder for this demo
    )
    print(f"✓ Created head: {head.__class__.__name__}")
    print(f"  - Total parameters: {head.get_num_parameters():,}")
    print(f"  - Trainable parameters: {head.get_num_parameters(trainable_only=True):,}")
    
    # 3. Create dataset
    print("\n[3/5] Creating dataset...")
    dataset = DatasetRegistry.get(
        "dummy",
        num_samples=50,
        image_size=224,
        num_classes=21
    )
    print(f"✓ Created dataset: {dataset.__class__.__name__}")
    print(f"  - Number of samples: {len(dataset)}")
    print(f"  - Number of classes: {dataset.num_classes}")
    
    # 4. Create dataloader
    print("\n[4/5] Creating dataloader...")
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )
    print(f"✓ Created dataloader with batch_size=4")
    
    # 5. Run forward pass
    print("\n[5/5] Running forward pass...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  - Using device: {device}")
    
    head = head.to(device)
    head.eval()
    
    # Get one batch
    batch = next(iter(dataloader))
    images = batch["image"].to(device)
    masks = batch["mask"].to(device)
    
    print(f"  - Input shape: {images.shape}")
    print(f"  - Mask shape: {masks.shape}")
    
    # Forward pass
    with torch.no_grad():
        features = encoder(images)
        logits = head(features)
    
    print(f"  - Feature shape: {features.shape}")
    print(f"  - Output shape: {logits.shape}")
    
    # Compute predictions
    predictions = logits.argmax(dim=1)
    print(f"  - Prediction shape: {predictions.shape}")
    print(f"  - Unique predicted classes: {predictions.unique().tolist()}")
    
    # Summary
    print("\n" + "=" * 80)
    print("✓ SUCCESS! Framework is working correctly.")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Try the Jupyter notebook: notebooks/01_quick_start.ipynb")
    print("  2. Implement real encoders (RADIO, DINOv2, CLIP)")
    print("  3. Add real datasets (Pascal VOC, ADE20K)")
    print("  4. Train and evaluate models")
    print("=" * 80)


if __name__ == "__main__":
    main()
