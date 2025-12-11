<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Give tasks for codeagent. give an overview how to implement and necessary steps

Set up base repo structure this week
Implement BaseEncoder + EncoderRegistry

Perfect! I've created a comprehensive **Week 1 code agent task document** with complete implementation specifications. Let me provide you with a structured overview:

## 📋 Document Overview

The document contains **complete, ready-to-implement code** organized into **5 Task Blocks**:

***

## **Task Block 1: Repository Structure** ✅

- **1.1**: Directory hierarchy (copy-paste folder structure)
- **1.2**: `.gitignore` file (complete)
- **1.3**: `requirements.txt` + `requirements-dev.txt` (all dependencies)
- **1.4**: `setup.py` + `pyproject.toml` (packaging)
- **1.5**: Initial `README.md` template

***

## **Task Block 2: Core Architecture Classes** 🏗️

### **2.1 BaseEncoder** (`vlm_eval/core/base_encoder.py`)

```python
class BaseEncoder(ABC, nn.Module):
    # Abstract properties:
    - @property output_channels → int
    - @property patch_size → int
    
    # Abstract methods:
    - forward(images) → features
    - get_config() → Dict
    - from_config(config) → BaseEncoder
    
    # Utility methods:
    - freeze() / unfreeze()
    - to_device(device)
    - get_num_parameters()
```

**Key Design**:

- Input: `(B, 3, H, W)` images
- Output: `(B, C, H//patch_size, W//patch_size)` dense features
- All configs are YAML-serializable (no torch objects)

***

### **2.2 BaseSegmentationHead** (`vlm_eval/core/base_head.py`)

```python
class BaseSegmentationHead(ABC, nn.Module):
    def __init__(encoder: BaseEncoder, num_classes: int, freeze_encoder: bool)
    
    # Abstract methods:
    - forward(features) → logits (B, num_classes, H, W)
    - get_config() → Dict
    - from_config(config) → BaseSegmentationHead
    
    # Respects encoder freezing in train() mode
```

**Key Design**:

- Always upsamples to original image resolution
- Automatically freezes encoder if `freeze_encoder=True`

***

### **2.3 BaseDataset** (`vlm_eval/core/base_dataset.py`)

```python
class BaseDataset(ABC, Dataset):
    @property num_classes → int
    @property class_names → List[str]
    @property ignore_index → int
    
    def __getitem__() → Dict with:
        - "image": (3, H, W) in [0,1]
        - "mask": (H, W) class indices
        - "filename": str
        - "image_id": int
```


***

## **Task Block 3: Registry System** 🔌

### **3.1 EncoderRegistry** (Plugin system)

```python
@EncoderRegistry.register("radio")
class RadioEncoder(BaseEncoder):
    ...

# Usage:
encoder = EncoderRegistry.get("radio", variant="base")
available = EncoderRegistry.list_available()
```

**Features**:

- Automatic registration via decorator
- Type checking (must subclass BaseEncoder)
- Config-based instantiation
- Error handling with available options

***

### **3.2 HeadRegistry** \& **3.3 DatasetRegistry**

Same pattern as EncoderRegistry, fully implemented

***

## **Task Block 4: Configuration Schema** ⚙️

Pydantic models for type-safe configuration:

```python
EncoderConfig → encoder settings
HeadConfig → head settings
DatasetConfig → dataset settings
MetricsConfig → metric selection
InferenceConfig → batch size, precision, etc.
ModelConfig → encoder + head pair
ExperimentConfig → full experiment specification
ResultConfig → evaluation results metadata
```

**Benefits**:

- Type validation
- YAML serialization/deserialization
- Clear schema documentation
- IDE autocomplete support

***

## **Task Block 5: Testing Framework** 🧪

### **5.1 Test Utilities** (`tests/conftest.py`)

- `DummyEncoder` - Mock encoder for testing
- `DummyHead` - Mock head for testing
- `sample_batch` fixture - Test data
- pytest fixtures for reuse


### **5.2 Base Class Tests** (`tests/test_base_classes.py`)

```python
test_encoder_interface()
test_encoder_forward()
test_encoder_freeze()
test_head_interface()
test_head_forward()
test_head_encoder_freezing()
```


### **5.3 Registry Tests** (`tests/test_registries.py`)

```python
test_encoder_registry_registration()
test_encoder_registry_get()
test_encoder_registry_duplicate_error()
test_head_registry_get()
test_registry_from_config()
```


***

## 🎯 Implementation Checklist

The document includes a **Week 1 Completion Checklist**:

```
Repository Structure ────────────────────
  ✓ All directories created
  ✓ All __init__.py files added
  ✓ .gitignore created
  ✓ Git initialized

Dependency Files ────────────────────
  ✓ requirements.txt
  ✓ requirements-dev.txt
  ✓ setup.py
  ✓ pyproject.toml

Base Classes ────────────────────
  ✓ BaseEncoder
  ✓ BaseSegmentationHead
  ✓ BaseDataset
  ✓ All docstrings

Registries ────────────────────
  ✓ EncoderRegistry
  ✓ HeadRegistry
  ✓ DatasetRegistry
  ✓ All error handling

Configuration ────────────────────
  ✓ Pydantic models
  ✓ Validation
  ✓ from_config tested

Testing ────────────────────
  ✓ Test utilities
  ✓ Base class tests
  ✓ Registry tests
  ✓ All tests passing
```


***

## 📊 Quick Start Commands

```bash
# 1. Setup repo structure
mkdir -p vlm-eval-framework/vlm_eval/{core,encoders,heads,datasets,cli,utils}
mkdir -p configs/{encoders,heads,experiments}
mkdir -p tests docs

# 2. Create all files from document
# (Each file has complete, copy-paste-ready code)

# 3. Install
cd vlm-eval-framework
pip install -e ".[dev]"

# 4. Run tests
pytest tests/ -v

# 5. Format & lint
black vlm_eval/ tests/
isort vlm_eval/ tests/
flake8 vlm_eval/ tests/
```


***

## 🚀 Next Steps After Week 1

Once core framework is done:

**Week 2-3: Model Integration**

- Implement concrete encoders (RADIO, DINOv3, CLIP)
- Implement concrete heads (linear probe, MLP decoder)
- Register all models

**Week 3: Evaluation Pipeline**

- EvaluationNode orchestrator
- Metrics computation (mIoU, PQ, boundary F1)
- Results aggregation

**Week 3-4: CLI/API Interface**

- Click CLI with config loading
- FastAPI REST endpoints
- Batch evaluation support

***

## 📝 Key Document Features

✅ **Complete Code**: No pseudocode, every function is fully implemented
✅ **Type Hints**: Full typing for IDE support
✅ **Docstrings**: Google style, comprehensive
✅ **Error Handling**: Validation, helpful error messages
✅ **Tests**: Unit tests for all components
✅ **Copy-Paste Ready**: Can be pasted directly into files
✅ **Extensible**: Clear patterns for adding new encoders/heads

The document is **ready for a code agent to implement immediately** - no design decisions needed, just follow the specifications!

