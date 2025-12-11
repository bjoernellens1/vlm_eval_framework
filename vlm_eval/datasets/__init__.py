"""Dataset implementations."""

from vlm_eval.datasets.dummy import DummyDataset
from vlm_eval.datasets.pascal_voc import PascalVOCDataset
from vlm_eval.datasets.replica import ReplicaDataset
from vlm_eval.datasets.scannet import ScanNetDataset
from vlm_eval.datasets.tum import TUMDataset

__all__ = ["DummyDataset", "PascalVOCDataset", "ReplicaDataset", "ScanNetDataset", "TUMDataset"]
