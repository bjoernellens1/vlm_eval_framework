import sys
import traceback

print(f"Python: {sys.executable}")
print(f"Path: {sys.path}")

try:
    import rosbags
    print(f"rosbags: {rosbags.__file__}")
    from rosbags.rosbag1 import Reader
    print("rosbags.rosbag1.Reader imported successfully")
except ImportError:
    print("ImportError caught:")
    traceback.print_exc()
except Exception:
    print("Exception caught:")
    traceback.print_exc()
