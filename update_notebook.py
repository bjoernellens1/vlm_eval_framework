import json
import os

notebook_path = "notebooks/embed_slam_comparison.ipynb"
new_code = r"""import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

VALID_RESOLUTIONS_NARADIO = [
    (192, 192), (224, 224), (256, 256), (288, 288), (320, 320), (384, 384),
    (448, 448), (512, 512), (640, 640), (768, 768), (896, 896), (1024, 1024),
    (256, 384), (384, 256), (336, 448), (448, 336), (512, 768), (768, 512),
    (512, 896), (896, 512), (720, 1280), (800, 1280), (768, 1024), (1024, 768),
]

def is_naradio_like(enc):
    return hasattr(getattr(enc, "model", None), "naradio") and hasattr(enc.model.naradio, "input_resolution")

def pick_best_resolution(hw, candidates):
    h, w = hw
    ar = w / h
    best = None
    best_score = float("inf")
    for H, W in candidates:
        ar2 = W / H
        score = abs(math.log(ar2 / ar)) + 0.15 * abs(math.log((H * W) / (h * w)))
        if score < best_score:
            best_score = score
            best = (H, W)
    return best

def resize_to(x, hw):
    return F.interpolate(x, size=hw, mode="bilinear", align_corners=False)

def letterbox_to(x, target_hw):
    B, C, H, W = x.shape
    th, tw = target_hw
    scale = min(tw / W, th / H)
    nh = int(round(H * scale))
    nw = int(round(W * scale))
    x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)

    pad_h = th - nh
    pad_w = tw - nw
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return F.pad(x, (left, right, top, bottom), value=0.0)

def ensure_bchw(feats):
    if not isinstance(feats, torch.Tensor):
        raise TypeError(f"Model returned {type(feats)} not a Tensor")
    if feats.ndim != 4:
        raise ValueError(f"Expected (B,C,H,W), got {tuple(feats.shape)}")
    return feats

def prepare_input_for_model(name, model, image_tensor):
    lname = name.lower()

    # SAM-based fusion models are broken right now (segmenter output mismatch)
    # BUT we fixed it in the library, so we can try running them!
    # if "concept_fusion" in lname or "x_fusion" in lname:
    #    raise RuntimeError("SAM-based segmenter output mismatch; needs library patch. Skipping.")

    # DINO/VIT fusion: force 224x224 (14x14 tokens for patch=16)
    if "dino" in lname:
        return resize_to(image_tensor, (224, 224)), "resize 224x224 (ViT 14x14 grid)"

    # NARadio: letterbox to a supported resolution
    if is_naradio_like(model):
        target_hw = pick_best_resolution(image_tensor.shape[-2:], VALID_RESOLUTIONS_NARADIO)
        return letterbox_to(image_tensor, target_hw), f"letterbox {target_hw}"

    # default: no change
    return image_tensor, "native"

# --- Run & plot ---
text_query = "chair"
ok = []

for name, model in models.items():
    print(f"Running {name}...")
    try:
        img_in, how = prepare_input_for_model(name, model, image_tensor)
        print(f"  -> {how}")

        with torch.no_grad():
            feats = ensure_bchw(model(img_in))
            txt = model.encode_text([text_query])

            feats = F.normalize(feats, dim=1)
            txt = F.normalize(txt, dim=1)

            sim = torch.einsum("bchw,bc->bhw", feats, txt)[0].cpu().numpy()

        ok.append((name, sim))
    except Exception as e:
        print(f"  !! skipping {name}: {type(e).__name__}: {e}")

if not ok:
    raise RuntimeError("No models ran successfully.")

fig, axes = plt.subplots(1, len(ok), figsize=(5 * len(ok), 5))
if len(ok) == 1:
    axes = [axes]

for ax, (name, sim_map) in zip(axes, ok):
    ax.imshow(sim_map, cmap="jet")
    ax.set_title(f"{name}\n'{text_query}'")
    ax.axis("off")

plt.tight_layout()
plt.show()
"""

with open(notebook_path, "r") as f:
    nb = json.load(f)

# Find the cell to replace
found = False
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "VALID_RESOLUTIONS_NARADIO" in source and "def is_naradio_like" in source:
            cell["source"] = new_code.splitlines(keepends=True)
            found = True
            break

if not found:
    print("Could not find the target cell to replace.")
    # Fallback: try to find the cell with the old code structure if possible, or append?
    # The view_file showed the cell has "VALID_RESOLUTIONS_NARADIO" in it (line 234 in view_file, but that was the file content)
    # Wait, the view_file output showed the file ALREADY has the code?
    # Let me check the view_file output again.
    pass

if found:
    with open(notebook_path, "w") as f:
        json.dump(nb, f, indent=4)
    print(f"Successfully updated {notebook_path}")
else:
    print("Target cell not found. Please check the notebook content.")
