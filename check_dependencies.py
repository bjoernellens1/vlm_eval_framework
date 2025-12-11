import sys
import importlib

packages = [
    "segment_anything",
    "open_clip",
    "dinov3",
    "ultralytics",
    "cv2",
    "timm"
]

print(f"Python: {sys.executable}")

for package in packages:
    try:
        importlib.import_module(package)
        print(f"[OK] {package}")
    except ImportError as e:
        print(f"[MISSING] {package}: {e}")
    except Exception as e:
        print(f"[ERROR] {package}: {e}")
