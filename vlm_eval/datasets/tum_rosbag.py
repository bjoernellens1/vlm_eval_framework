import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from typing import Any, Dict, List, Optional, Union
import os
from pathlib import Path

# Try importing ROS libraries
try:
    import rosbag
    from cv_bridge import CvBridge
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    rosbag = None
    CvBridge = None

from vlm_eval.core import DatasetRegistry

@DatasetRegistry.register("tum_rosbag")
class TUMRosbagDataset(Dataset):
    """Dataset for loading images from ROS bags (e.g. TUM dataset).
    
    Args:
        bag_path: Path to the .bag file.
        topics: List of topics to read. Defaults to RGB topics.
        image_size: Resize images to this size (optional).
    """
    def __init__(
        self, 
        bag_path: str, 
        topics: List[str] = ["/camera/rgb/image_color", "/camera/depth/image"],
        image_size: Optional[int] = None
    ):
        if not ROS_AVAILABLE:
            raise ImportError(
                "ROS libraries (rosbag, cv_bridge) not found. "
                "Please install them to use TUMRosbagDataset."
            )
            
        self.bag_path = Path(bag_path)
        if not self.bag_path.exists():
            raise FileNotFoundError(f"Bag file not found: {bag_path}")
            
        self.topics = topics
        self.image_size = image_size
        self.bag = rosbag.Bag(str(self.bag_path), "r")
        self.bridge = CvBridge()
        
        print(f"Indexing bag file: {bag_path}...")
        self.messages = []
        self.topic_indices = {t: [] for t in topics}
        
        # Iterate once to index messages
        # We store (offset, topic, timestamp) to avoid keeping all msgs in memory
        # But rosbag.read_messages generator doesn't give offset easily without hacking.
        # Actually, for small bags we can store messages. For large ones, we might need a better strategy.
        # Given "TUM dataset rosbags", they are usually manageable. 
        # But for efficiency, we can just store the list of messages if we read them all?
        # No, read_messages reads them.
        
        # We will just store the messages in memory for now as it's the simplest way 
        # to support random access __getitem__.
        # WARNING: This consumes memory.
        
        count = 0
        for topic, msg, t in self.bag.read_messages(topics=topics):
            self.messages.append((topic, msg, t))
            self.topic_indices[topic].append(count)
            count += 1
            
        print(f"Indexed {len(self.messages)} messages.")
        
    def __len__(self) -> int:
        return len(self.messages)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        topic, msg, t = self.messages[idx]
        
        try:
            # Check if it's an image message
            if hasattr(msg, "encoding"):
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                
                # Handle encoding
                if "bgr" in msg.encoding.lower():
                    cv_image = cv_image[..., ::-1] # BGR to RGB
                elif "rgb" in msg.encoding.lower():
                    pass
                elif "mono" in msg.encoding.lower() or "depth" in topic:
                    # Depth or mono
                    pass
                
                image = Image.fromarray(cv_image)
                
                if self.image_size:
                    image = image.resize((self.image_size, self.image_size))
                
                # Convert to tensor
                img_tensor = torch.from_numpy(np.array(image)).float()
                if len(img_tensor.shape) == 3:
                    img_tensor = img_tensor.permute(2, 0, 1) / 255.0
                else:
                    img_tensor = img_tensor.unsqueeze(0) # Depth/Mono
                
                return {
                    "image": img_tensor,
                    "timestamp": t.to_sec(),
                    "topic": topic,
                    "index": idx
                }
            else:
                return {"data": msg, "timestamp": t.to_sec(), "topic": topic}
                
        except Exception as e:
            print(f"Error decoding message at index {idx}: {e}")
            return {}

    def close(self):
        self.bag.close()
