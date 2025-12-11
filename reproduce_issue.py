from vlm_eval.core import EncoderRegistry
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_names = ["concept_fusion", "dino_fusion", "x_fusion", "naradio_fusion"]
models = {}

for name in model_names:
    try:
        print(f"Loading {name}...")
        models[name] = EncoderRegistry.get(name, device=device)
        print(f"Loaded {name}")
    except Exception as e:
        print(f"Failed to load {name}: {e}")
