from vlm_eval.core import EncoderRegistry, DatasetRegistry
from vlm_eval.encoders import *
from vlm_eval.datasets.tum_rosbag import TUMRosbagDataset

print("Successfully imported TUMRosbagDataset")
print("Available encoders:", EncoderRegistry.list_available())

try:
    # Try to instantiate to check if abstract methods are implemented
    # We pass a dummy path, it will fail with FileNotFoundError but that means class is valid
    dataset = TUMRosbagDataset(bag_path="dummy.bag")
except FileNotFoundError:
    print("Caught expected FileNotFoundError, class instantiation attempted.")
except TypeError as e:
    print(f"TypeError during instantiation (likely missing abstract methods): {e}")
except Exception as e:
    print(f"An error occurred: {e}")
