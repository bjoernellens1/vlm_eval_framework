import sys
import traceback

print("Python executable:", sys.executable)
print("Python path:", sys.path)

try:
    print("Attempting to import vlm_eval.encoders.embed_slam.concept_fusion")
    from vlm_eval.encoders.embed_slam import concept_fusion
    print("Success!")
except Exception:
    traceback.print_exc()

try:
    print("\nAttempting to import vlm_eval.encoders.embed_slam.dino_fusion")
    from vlm_eval.encoders.embed_slam import dino_fusion
    print("Success!")
except Exception:
    traceback.print_exc()

try:
    print("\nAttempting to import vlm_eval.encoders.embed_slam.x_fusion")
    from vlm_eval.encoders.embed_slam import x_fusion
    print("Success!")
except Exception:
    traceback.print_exc()

try:
    print("\nAttempting to import vlm_eval.encoders.embed_slam.naradio_fusion")
    from vlm_eval.encoders.embed_slam import naradio_fusion
    print("Success!")
except Exception:
    traceback.print_exc()
