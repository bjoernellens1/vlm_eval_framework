import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from typing import Any, Dict, List, Optional, Union
import os
from pathlib import Path

# Try importing rosbags (preferred for flexibility) or rosbag (legacy)
try:
    from rosbags.rosbag1 import Reader as Reader1
    from rosbags.rosbag2 import Reader as Reader2
    from rosbags.serde import deserialize_cdr, ros1_to_cdr
    from rosbags.typesys import get_types_from_msg, register_types
    ROSBAGS_AVAILABLE = True
except ImportError:
    ROSBAGS_AVAILABLE = False

# Try importing cv_bridge
try:
    from cv_bridge import CvBridge
    CV_BRIDGE_AVAILABLE = True
except ImportError:
    CV_BRIDGE_AVAILABLE = False

from vlm_eval.core import DatasetRegistry

@DatasetRegistry.register("tum_rosbag")
class TUMRosbagDataset(Dataset):
    """Dataset for loading images from ROS bags (e.g. TUM dataset).
    
    Supports both ROS1 (.bag) and ROS2 (.mcap, .db3) files via 'rosbags' library.
    
    Args:
        bag_path: Path to the bag file.
        topics: List of topics to read. Defaults to RGB topics.
        image_size: Resize images to this size (optional).
    """
    def __init__(
        self, 
        bag_path: str, 
        topics: List[str] = ["/camera/rgb/image_color", "/camera/depth/image"],
        image_size: Optional[int] = None
    ):
        if not ROSBAGS_AVAILABLE:
            raise ImportError(
                "rosbags library not found. "
                "Please install it with: pip install rosbags"
            )
            
        self.bag_path = Path(bag_path)
        if not self.bag_path.exists():
            raise FileNotFoundError(f"Bag file not found: {bag_path}")
            
        self.topics = topics
        self.image_size = image_size
        self.bridge = CvBridge() if CV_BRIDGE_AVAILABLE else None
        
        self.messages = []
        
        print(f"Indexing bag file: {bag_path}...")
        
        # Determine if ROS1 or ROS2
        is_ros2 = self.bag_path.suffix in ['.mcap', '.db3'] or self.bag_path.is_dir()
        
        if is_ros2:
            self.reader = Reader2(self.bag_path)
        else:
            self.reader = Reader1(self.bag_path)
            
        self.reader.open()
        
        # Filter connections by topic
        connections = [x for x in self.reader.connections if x.topic in topics]
        
        for conn, timestamp, rawdata in self.reader.messages(connections=connections):
            self.messages.append((conn, timestamp, rawdata))
            
        print(f"Indexed {len(self.messages)} messages.")
        
    def __len__(self) -> int:
        return len(self.messages)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        conn, timestamp, rawdata = self.messages[idx]
        topic = conn.topic
        
        try:
            # Deserialize
            if isinstance(self.reader, Reader1):
                msg = deserialize_cdr(ros1_to_cdr(rawdata, conn.msgtype), conn.msgtype)
            else:
                msg = deserialize_cdr(rawdata, conn.msgtype)
            
            # Check if it's an image
            if hasattr(msg, "encoding") and hasattr(msg, "data"):
                # It's likely an image
                # Convert data to numpy
                
                if self.bridge:
                    # If we have cv_bridge and it works with this msg object (duck typing)
                    # cv_bridge expects a ROS message object. 'rosbags' produces python objects with slots.
                    # cv_bridge might fail if it strictly checks type.
                    # Let's try manual decoding for robustness if cv_bridge fails or is not present.
                    pass
                
                # Manual decoding
                dtype = np.uint8
                n_channels = 3
                
                if "8uc3" in msg.encoding:
                    n_channels = 3
                elif "8uc1" in msg.encoding:
                    n_channels = 1
                elif "16uc1" in msg.encoding:
                    dtype = np.uint16
                    n_channels = 1
                elif "32fc1" in msg.encoding:
                    dtype = np.float32
                    n_channels = 1
                
                img_data = np.frombuffer(msg.data, dtype=dtype)
                
                # Reshape
                try:
                    img_data = img_data.reshape((msg.height, msg.width, n_channels))
                except ValueError:
                    # Sometimes padding or stride issues
                    # Use step if available
                    if hasattr(msg, "step"):
                         img_data = img_data[:msg.height * msg.step]
                         img_data = img_data.reshape((msg.height, msg.width, n_channels))
                
                if n_channels == 3:
                    if "bgr" in msg.encoding.lower():
                        img_data = img_data[..., ::-1] # BGR to RGB
                
                if n_channels == 1:
                    img_data = img_data.squeeze(-1)
                    
                image = Image.fromarray(img_data)
                
                if self.image_size:
                    image = image.resize((self.image_size, self.image_size))
                
                # Convert to tensor
                img_tensor = torch.from_numpy(np.array(image)).float()
                
                if n_channels == 3:
                    img_tensor = img_tensor.permute(2, 0, 1) / 255.0
                else:
                    img_tensor = img_tensor.unsqueeze(0)
                    if dtype == np.uint16:
                        img_tensor = img_tensor / 1000.0 # Depth usually mm to meters
                    elif dtype == np.uint8:
                        img_tensor = img_tensor / 255.0
                
                return {
                    "image": img_tensor,
                    "timestamp": timestamp / 1e9, # ns to sec
                    "topic": topic,
                    "index": idx
                }
            else:
                return {"data": msg, "timestamp": timestamp / 1e9, "topic": topic}
                
        except Exception as e:
            print(f"Error decoding message at index {idx}: {e}")
            return {}

    def close(self):
        self.reader.close()
