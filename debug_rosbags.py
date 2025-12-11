import sys
import traceback

print(f"Python: {sys.executable}")

try:
    print("Importing rosbags.rosbag1...")
    from rosbags.rosbag1 import Reader as Reader1
    print("OK")
    
    print("Importing rosbags.rosbag2...")
    from rosbags.rosbag2 import Reader as Reader2
    print("OK")
    
    print("Importing rosbags.serde...")
    from rosbags.serde import deserialize_cdr, ros1_to_cdr
    print("OK")
    
    print("Importing rosbags.typesys...")
    from rosbags.typesys import get_types_from_msg, register_types
    print("OK")
    
except ImportError:
    print("ImportError caught:")
    traceback.print_exc()
except Exception:
    print("Exception caught:")
    traceback.print_exc()
