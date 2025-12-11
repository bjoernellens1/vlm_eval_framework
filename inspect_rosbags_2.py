import sys
import traceback

print(f"Python: {sys.executable}")

try:
    import rosbags.serde.cdr
    print(f"rosbags.serde.cdr: {dir(rosbags.serde.cdr)}")
except ImportError:
    print("rosbags.serde.cdr not found")

try:
    import rosbags.serde.ros1
    print(f"rosbags.serde.ros1: {dir(rosbags.serde.ros1)}")
except ImportError:
    print("rosbags.serde.ros1 not found")

try:
    from rosbags.serde import deserialize_cdr, ros1_to_cdr
    print("Direct import worked (unexpected)")
except ImportError:
    print("Direct import failed as expected")
