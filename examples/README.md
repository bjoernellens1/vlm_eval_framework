# VLM Evaluation Framework - Examples

This directory contains ready-to-run examples demonstrating the framework.

## Quick Start

The simplest way to get started:

```bash
python quick_start.py
```

This script demonstrates:
- Creating encoders, heads, and datasets using the registry
- Running forward passes
- Basic model usage

## Available Examples

### 1. quick_start.py
Basic example showing framework usage with SimpleCNN encoder and dummy dataset.
No external data or pretrained weights required.

**Run:**
```bash
cd examples
python quick_start.py
```

## Jupyter Notebooks

For interactive exploration, check out the notebooks in `../notebooks/`:

1. **01_quick_start.ipynb** - Interactive introduction with visualizations
2. **02_training_example.ipynb** - Complete training workflow

**Run notebooks:**
```bash
cd notebooks
jupyter notebook
```

## Configuration Files

Example configurations are in `../configs/experiments/`:

- `demo_simple_cnn.yaml` - Ready-to-run demo configuration
- `example.yaml` - Template for real experiments

## Next Steps

After running these examples:

1. Implement real encoders (RADIO, DINOv2, CLIP)
2. Add real datasets (Pascal VOC, ADE20K, Cityscapes)
3. Implement evaluation metrics (mIoU, boundary F1)
4. Run full evaluation pipelines
