# VLM Evaluation Framework

A modular, extensible framework for evaluating vision encoders on segmentation tasks.

## Features

- 🏗️ **Modular Architecture**: Abstract base classes for encoders, heads, and datasets
- 🔌 **Plugin System**: Decorator-based registry for easy model registration
- ⚙️ **Type-Safe Configuration**: Pydantic models with YAML support
- 🧪 **Comprehensive Testing**: Full test coverage with pytest
- 📊 **Flexible Evaluation**: Support for multiple metrics and datasets

## Installation

### From Source

```bash
git clone https://github.com/yourusername/vlm-eval-framework.git
cd vlm-eval-framework
pip install -e ".[dev]"
```

### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- See `requirements.txt` for full dependencies

## Quick Start

### Ready-to-Run Example

Try the framework immediately with no setup required:

```bash
# Run the quick start example
python examples/quick_start.py
```

This uses a simple CNN encoder and dummy dataset - no external data or pretrained weights needed!

### Interactive Notebooks

Explore the framework interactively:

```bash
# Launch Jupyter
cd notebooks
jupyter notebook

# Try these notebooks:
# 1. 01_quick_start.ipynb - Framework introduction with visualizations
# 2. 02_training_example.ipynb - Complete training workflow
# 3. 03_naradio_evaluation.ipynb - NVIDIA RADIO encoder evaluation
# 4. 04_model_comparison.ipynb - Compare SimpleCNN vs RADIO encoders
```

**Note**: Notebooks 3-4 require Hugging Face authentication:
```bash
huggingface-cli login
```

### 1. Register a Custom Encoder

```python
from vlm_eval.core import BaseEncoder, EncoderRegistry
import torch.nn as nn

@EncoderRegistry.register("my_encoder")
class MyEncoder(BaseEncoder):
    def __init__(self, variant: str = "base"):
        super().__init__()
        self.variant = variant
        # Your encoder implementation
        
    @property
    def output_channels(self) -> int:
        return 768
    
    @property
    def patch_size(self) -> int:
        return 16
    
    def forward(self, images):
        # Your forward pass
        pass
    
    def get_config(self):
        return {"variant": self.variant}
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
```

### 2. Register a Segmentation Head

```python
from vlm_eval.core import BaseSegmentationHead, HeadRegistry

@HeadRegistry.register("linear_probe")
class LinearProbeHead(BaseSegmentationHead):
    def __init__(self, encoder, num_classes, freeze_encoder=True):
        super().__init__(encoder, num_classes, freeze_encoder)
        # Your head implementation
        
    def forward(self, features):
        # Your forward pass
        pass
```

### 3. Use Configuration Files

```yaml
# configs/experiments/my_experiment.yaml
encoder:
  name: "my_encoder"
  variant: "base"
  pretrained: true

head:
  name: "linear_probe"
  num_classes: 21
  freeze_encoder: true

dataset:
  name: "pascal_voc"
  split: "val"
  root: "/path/to/data"

inference:
  batch_size: 16
  device: "cuda"
  precision: "fp16"
```

### 4. Run Evaluation

```python
from vlm_eval.core import ExperimentConfig
import yaml

# Load configuration
with open("configs/experiments/my_experiment.yaml") as f:
    config = ExperimentConfig(**yaml.safe_load(f))

# Get encoder and head
encoder = EncoderRegistry.get(config.encoder.name, **config.encoder.dict())
head = HeadRegistry.get(config.head.name, encoder=encoder, **config.head.dict())

# Run evaluation (to be implemented in Week 3)
```

## Architecture

### Core Components

```
vlm_eval/
├── core/
│   ├── base_encoder.py      # Abstract encoder interface
│   ├── base_head.py         # Abstract segmentation head interface
│   ├── base_dataset.py      # Abstract dataset interface
│   ├── registry.py          # Plugin registration system
│   └── config.py            # Pydantic configuration models
├── encoders/                # Concrete encoder implementations
│   ├── simple_cnn.py        # ✓ SimpleCNN (ready to use)
│   └── radio.py             # ✓ NVIDIA RADIO (ready to use)
├── heads/                   # Concrete head implementations
│   └── linear_probe.py      # ✓ Linear probe (ready to use)
├── datasets/                # Concrete dataset implementations
│   ├── dummy.py             # ✓ Dummy dataset (ready to use)
│   └── pascal_voc.py        # ✓ Pascal VOC (ready to use)
├── cli/                     # Command-line interface
└── utils/                   # Utility functions
```

**Included Implementations:**
- **SimpleCNNEncoder**: Lightweight CNN encoder (tiny/small/base variants)
- **RADIOEncoder**: NVIDIA's RADIO foundation model (pretrained)
- **LinearProbeHead**: 1x1 convolution + bilinear upsampling
- **DummyDataset**: Random image/mask generator for testing
- **PascalVOCDataset**: Pascal VOC 2012 segmentation (with subset mode)

### Design Principles

1. **Abstraction**: All components inherit from abstract base classes
2. **Registration**: Models are registered via decorators for easy discovery
3. **Configuration**: Type-safe configs with Pydantic validation
4. **Extensibility**: Add new models by implementing base classes and registering

## Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=vlm_eval --cov-report=html

# Run specific test file
pytest tests/test_base_classes.py -v
```

### Code Formatting

```bash
# Format code
black vlm_eval/ tests/
isort vlm_eval/ tests/

# Check formatting
black --check vlm_eval/ tests/
isort --check vlm_eval/ tests/
```

### Linting

```bash
# Run flake8
flake8 vlm_eval/ tests/

# Run mypy type checking
mypy vlm_eval/
```

## Roadmap

- [x] **Week 1**: Core architecture and registry system
- [x] **Week 2**: RADIO encoder integration and Pascal VOC dataset
- [x] **Week 2**: Example notebooks for evaluation and comparison
- [ ] **Week 3**: Additional model implementations (DINOv2, CLIP, SAM)
- [ ] **Week 3**: Evaluation pipeline and metrics
- [ ] **Week 3-4**: CLI and API interface

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass and code is formatted
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{vlm_eval_framework,
  title = {VLM Evaluation Framework},
  author = {VLM Eval Team},
  year = {2024},
  url = {https://github.com/yourusername/vlm-eval-framework}
}
```
