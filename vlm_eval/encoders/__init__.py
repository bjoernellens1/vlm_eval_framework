"""Encoder implementations."""

from vlm_eval.encoders.simple_cnn import SimpleCNNEncoder
from vlm_eval.encoders.radio import RADIOEncoder
from vlm_eval.encoders.clip import CLIPEncoder
from vlm_eval.encoders.naradio import NARadioEncoder

from vlm_eval.encoders.embed_slam_wrappers import (
    ConceptFusionEncoder,
    DINOFusionEncoder,
    XFusionEncoder,
    NARadioFusionEncoder
)

__all__ = [
    "SimpleCNNEncoder", 
    "RADIOEncoder", 
    "CLIPEncoder", 
    "NARadioEncoder",
    "ConceptFusionEncoder",
    "DINOFusionEncoder",
    "XFusionEncoder",
    "NARadioFusionEncoder"
]
