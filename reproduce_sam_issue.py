import numpy as np
from vlm_eval.encoders.embed_slam.segmentation import TransformersSAMSegmenter

def test_sam_segmenter():
    config = {
        "device": "cpu",
        "model_type": "vit_b"
    }
    
    print("Initializing TransformersSAMSegmenter...")
    try:
        segmenter = TransformersSAMSegmenter(config)
    except Exception as e:
        print(f"Failed to initialize segmenter: {e}")
        return

    # Create a dummy image (H, W, 3)
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    print("Running segment()...")
    try:
        segments = segmenter.segment(image)
        print(f"Successfully segmented. Found {len(segments)} segments.")
    except Exception as e:
        print(f"Segmentation failed: {e}")

if __name__ == "__main__":
    test_sam_segmenter()
