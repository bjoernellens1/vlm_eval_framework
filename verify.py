#!/usr/bin/env python3
"""Verification script to test the VLM evaluation framework (no dependencies)."""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("VLM Evaluation Framework - Verification Script")
print("=" * 80)

# Test 1: Test file structure
print("\n[1/3] Verifying file structure...")
try:
    required_files = [
        "vlm_eval/__init__.py",
        "vlm_eval/core/__init__.py",
        "vlm_eval/core/base_encoder.py",
        "vlm_eval/core/base_head.py",
        "vlm_eval/core/base_dataset.py",
        "vlm_eval/core/registry.py",
        "vlm_eval/core/config.py",
        "vlm_eval/encoders/__init__.py",
        "vlm_eval/heads/__init__.py",
        "vlm_eval/datasets/__init__.py",
        "vlm_eval/cli/__init__.py",
        "vlm_eval/utils/__init__.py",
        "tests/__init__.py",
        "tests/conftest.py",
        "tests/test_base_classes.py",
        "tests/test_registries.py",
        "tests/test_config.py",
        "configs/encoders/radio_base.yaml",
        "configs/heads/linear_probe.yaml",
        "configs/experiments/example.yaml",
        "setup.py",
        "pyproject.toml",
        "README.md",
        "requirements.txt",
        "requirements-dev.txt",
        ".gitignore",
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        sys.exit(1)
    
    print(f"✓ All {len(required_files)} required files present")
except Exception as e:
    print(f"✗ File structure verification failed: {e}")
    sys.exit(1)

# Test 2: Test Python syntax (imports without execution)
print("\n[2/3] Testing Python syntax...")
try:
    import ast
    
    python_files = [
        "vlm_eval/__init__.py",
        "vlm_eval/core/__init__.py",
        "vlm_eval/core/base_encoder.py",
        "vlm_eval/core/base_head.py",
        "vlm_eval/core/base_dataset.py",
        "vlm_eval/core/registry.py",
        "vlm_eval/core/config.py",
        "tests/conftest.py",
        "tests/test_base_classes.py",
        "tests/test_registries.py",
        "tests/test_config.py",
    ]
    
    syntax_errors = []
    for file_path in python_files:
        full_path = project_root / file_path
        try:
            with open(full_path, 'r') as f:
                ast.parse(f.read())
        except SyntaxError as e:
            syntax_errors.append(f"{file_path}: {e}")
    
    if syntax_errors:
        print(f"✗ Syntax errors found:")
        for error in syntax_errors:
            print(f"  - {error}")
        sys.exit(1)
    
    print(f"✓ All {len(python_files)} Python files have valid syntax")
except Exception as e:
    print(f"✗ Syntax verification failed: {e}")
    sys.exit(1)

# Test 3: Test YAML configuration files
print("\n[3/3] Testing YAML configuration files...")
try:
    import yaml
    
    yaml_files = [
        "configs/encoders/radio_base.yaml",
        "configs/heads/linear_probe.yaml",
        "configs/experiments/example.yaml",
    ]
    
    yaml_errors = []
    for file_path in yaml_files:
        full_path = project_root / file_path
        try:
            with open(full_path, 'r') as f:
                yaml.safe_load(f)
        except yaml.YAMLError as e:
            yaml_errors.append(f"{file_path}: {e}")
    
    if yaml_errors:
        print(f"✗ YAML errors found:")
        for error in yaml_errors:
            print(f"  - {error}")
        sys.exit(1)
    
    print(f"✓ All {len(yaml_files)} YAML files are valid")
except ImportError:
    print("⚠ PyYAML not installed, skipping YAML validation")
except Exception as e:
    print(f"✗ YAML verification failed: {e}")
    sys.exit(1)

# Summary
print("\n" + "=" * 80)
print("✓ ALL VERIFICATION CHECKS PASSED!")
print("=" * 80)
print("\nFramework structure is complete and valid!")
print("\nNext steps:")
print("  1. Install dependencies:")
print("     pip install -e '.[dev]'")
print("  2. Run tests:")
print("     pytest tests/ -v")
print("  3. Implement concrete encoders:")
print("     - RADIO encoder in vlm_eval/encoders/radio.py")
print("     - DINOv2 encoder in vlm_eval/encoders/dinov2.py")
print("     - CLIP encoder in vlm_eval/encoders/clip.py")
print("  4. Implement concrete heads:")
print("     - Linear probe in vlm_eval/heads/linear_probe.py")
print("     - MLP decoder in vlm_eval/heads/mlp_decoder.py")
print("  5. Implement datasets:")
print("     - Pascal VOC in vlm_eval/datasets/pascal_voc.py")
print("     - ADE20K in vlm_eval/datasets/ade20k.py")
print("=" * 80)
