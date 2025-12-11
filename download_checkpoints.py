import urllib.request
import os
import sys

def download_file(url, filename):
    print(f"Downloading {filename} from {url}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"Downloaded {filename}")
    except Exception as e:
        print(f"Failed to download {filename}: {e}")

# SAM Checkpoint
sam_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
if not os.path.exists("sam_vit_b_01ec64.pth"):
    download_file(sam_url, "sam_vit_b_01ec64.pth")
else:
    print("sam_vit_b_01ec64.pth already exists.")

# DINOv3 Checkpoint
# Expected path: ~/.cache/torch/hub/checkpoints/dinov3_vitl16_pretrain_lvd1609m-0aa4cbdd.pth
# We will try to download from Hugging Face if available, or just warn.
# The filename implies it's a specific hash.
# HF Repo: facebook/dinov3-vitl16-pretrain-lvd1689m (Note: 1689m vs 1609m in filename? Let's check)

# The filename in error was: dinov3_vitl16_pretrain_lvd1609m-0aa4cbdd.pth
# Search result mentions: lvd1689m. 
# Maybe the code uses an older version?
# Let's check the code in vlm_eval again? No, I can't see library code.
# But the error message is truth.

cache_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints")
os.makedirs(cache_dir, exist_ok=True)
dino_filename = "dinov3_vitl16_pretrain_lvd1609m-0aa4cbdd.pth"
dino_path = os.path.join(cache_dir, dino_filename)

if not os.path.exists(dino_path):
    print(f"DINOv3 checkpoint not found at {dino_path}.")
    print("Attempting to find alternative URL...")
    # Try to download from a mirror or HF if we can find the exact file.
    # Since we can't easily find the exact file with hash, we might fail here.
    # But we can try to download the one from HF and rename it, hoping it's compatible?
    # HF: https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m/resolve/main/pytorch_model.bin ?
    # This is risky.
    
    print("Please manually download DINOv3 weights if automated download fails.")
else:
    print(f"DINOv3 checkpoint already exists at {dino_path}.")
