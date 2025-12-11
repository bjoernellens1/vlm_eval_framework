try:
    import rosbags
    print(f"rosbags imported successfully: {rosbags.__file__}")
except ImportError as e:
    print(f"Failed to import rosbags: {e}")

try:
    import cv2
    print(f"cv2 imported successfully: {cv2.__file__}")
except ImportError as e:
    print(f"Failed to import cv2: {e}")
