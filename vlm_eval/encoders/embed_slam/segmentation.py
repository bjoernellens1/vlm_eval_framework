from typing import Any
import numpy.typing as npt

import abc
from dataclasses import dataclass

import os
import cv2
import numpy as np
import torch
from torchvision.ops.boxes import batched_nms, box_area

try:
    from segment_anything.utils.amg import MaskData, batched_mask_to_box, box_xyxy_to_xywh
except ImportError:
    MaskData = None
    batched_mask_to_box = None
    box_xyxy_to_xywh = None


@dataclass
class Segment:
    mask: npt.ArrayLike # binary mask, [w, h, 1]
    bbox: npt.ArrayLike # bounding box, [x, y, w, h]


class Segmenter(abc.ABC):
    @abc.abstractmethod
    def __init__(self, cfg: dict[str, Any]) -> None:
        pass

    @abc.abstractmethod
    def segment(self, image: npt.ArrayLike) -> list[Segment]:
        pass

    def refine(self, image: npt.ArrayLike, point_coords: npt.ArrayLike) -> list[Segment]:
        raise NotImplementedError

    def to(self, device: str) -> None:
        pass


class TransformersSAMSegmenter(Segmenter):
    def __init__(self, config: dict[str, Any]):
        from transformers import pipeline, SamModel, SamProcessor
        self.device = config.get("device", "cpu")
        model_name = "facebook/sam-vit-base"
        if config.get("model_type") == "vit_h":
             model_name = "facebook/sam-vit-huge"
        elif config.get("model_type") == "vit_l":
             model_name = "facebook/sam-vit-large"
        
        self.generator = pipeline("mask-generation", model=model_name, device=self.device)
        self.processor = SamProcessor.from_pretrained(model_name)
        self.model = self.generator.model

    def to(self, device: str) -> None:
        self.model.to(device)
        self.device = device

    def segment(self, image: npt.ArrayLike) -> list[Segment]:
        # image: numpy array (H, W, 3)
        from PIL import Image
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        else:
            image_pil = image
            
        outputs = self.generator(image_pil)
        
        segments = []
        
        # Handle different output formats
        # Case 1: Dict with 'masks' (and optional 'scores') - observed in recent transformers
        if isinstance(outputs, dict) and "masks" in outputs:
            masks = outputs["masks"]
            scores = outputs.get("scores", [1.0] * len(masks))
            
            for mask, score in zip(masks, scores):
                # mask might be a tensor or numpy array
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
                
                # Ensure mask is boolean or binary
                mask_bool = mask > 0
                
                # Calculate bbox
                y_indices, x_indices = np.where(mask_bool)
                if len(y_indices) > 0:
                    x_min, x_max = np.min(x_indices), np.max(x_indices)
                    y_min, y_max = np.min(y_indices), np.max(y_indices)
                    w = x_max - x_min
                    h = y_max - y_min
                    segments.append(Segment(mask_bool, [x_min, y_min, w, h]))
                    
        # Case 2: List of dicts (legacy or different pipeline version)
        elif isinstance(outputs, list):
            for out in outputs:
                if 'mask' in out and 'box' in out:
                    mask = out['mask']
                    box_xyxy = out['box']
                    x, y, x2, y2 = box_xyxy
                    w = x2 - x
                    h = y2 - y
                    segments.append(Segment(mask, [x, y, w, h]))
                elif 'mask' in out:
                     # Fallback if box is missing
                    mask = out['mask']
                    y_indices, x_indices = np.where(mask)
                    if len(y_indices) > 0:
                        x_min, x_max = np.min(x_indices), np.max(x_indices)
                        y_min, y_max = np.min(y_indices), np.max(y_indices)
                        w = x_max - x_min
                        h = y_max - y_min
                        segments.append(Segment(mask, [x_min, y_min, w, h]))
        
        return segments

    def refine(self, image: npt.ArrayLike, point_coords: npt.ArrayLike) -> list[Segment]:
        # point_coords: [N, 2]
        from PIL import Image
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        else:
            image_pil = image
            
        # Prepare inputs
        # Transformers SAM expects input_points as list of lists of lists? 
        # input_points (np.ndarray or List[List[List[float]]], optional) — Input 2D spatial points
        # shape (batch_size, point_batch_size, num_points_per_mask, 2)
        
        # We treat each point as a separate prompt for a separate mask?
        # The original code does:
        # coords_torch[:, None, :] -> (N, 1, 2)
        # So N prompts, each with 1 point.
        
        input_points = [[list(p)] for p in point_coords] # List of [x, y] -> List of [[x, y]]
        # Actually we want batch of points.
        # processor(images=image, input_points=[[[x, y], [x, y]]]) -> 1 image, 2 points for 1 mask?
        # No, we want N masks.
        
        # We can run batch inference if we duplicate the image? Or does processor support one image multiple point sets?
        # Processor supports: images (PIL.Image.Image, np.ndarray, List[PIL.Image.Image], List[np.ndarray])
        # input_points (np.ndarray, List[List[List[float]]])
        
        # If we provide 1 image and N point sets, it might not work as expected for batching.
        # Usually we replicate the image.
        
        n_points = len(point_coords)
        images = [image_pil] * n_points
        input_points = [[[list(p)]] for p in point_coords] # (N, 1, 1, 2) ?
        # Processor expects: (batch_size, num_masks_per_image, num_points_per_mask, 2)
        # If we pass list of images, batch_size = N.
        # input_points should be list of list of list.
        # Outer list: batch (N)
        # Middle list: points per mask (1) - wait, SAM can take multiple points per mask.
        # Inner list: coordinates (2)
        
        # Let's try: input_points = [[[x, y]]] for each image.
        
        inputs = self.processor(images=images, input_points=input_points, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
        )
        # masks: List of tensors. Each tensor (num_masks_per_image, num_generated_masks, H, W)
        # num_generated_masks is usually 3 (multimask_output=True by default).
        
        segments = []
        for i, mask_tensor in enumerate(masks):
            # mask_tensor: (1, 3, H, W)
            # We pick the best one? Or all?
            # Original code:
            # masks, iou_predictions, low_res_masks = self.mask_generator.predictor.predict_torch(...)
            # It returns multiple masks.
            # Then it filters them.
            
            # For simplicity, let's take the one with highest score.
            iou_scores = outputs.iou_scores[i, 0, :] # (3,)
            best_idx = torch.argmax(iou_scores).item()
            
            best_mask = mask_tensor[0, best_idx, :, :].numpy() # (H, W)
            
            # Bbox
            y_indices, x_indices = np.where(best_mask)
            if len(y_indices) > 0:
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                w = x_max - x_min
                h = y_max - y_min
                segments.append(Segment(best_mask, [x_min, y_min, w, h]))
                
        return segments


class SAMSegmenter(Segmenter):
    def __init__(self, config: dict[str, Any]):
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        sam = sam_model_registry[config["model_type"]](checkpoint=config["checkpoint"]).eval()
        sam.to(device=config["device"])
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=8,
            pred_iou_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
        )

    def to(self, device: str) -> None:
        self.mask_generator.predictor.model.to(device=device)

    def segment(self, image: npt.ArrayLike) -> list[Segment]:
        masks = self.mask_generator.generate(image)
        return [Segment(mask["segmentation"], mask["bbox"]) for mask in masks]

    def refine(self, image: npt.ArrayLike, point_coords: npt.ArrayLike) -> list[Segment]:
        self.mask_generator.predictor.set_image(image)

        original_size = image.shape[:2] # (h, w)

        assert point_coords.shape[1] == 2
        point_labels = np.ones((point_coords.shape[0]))

        device = self.mask_generator.predictor.model.device

        point_coords = self.mask_generator.predictor.transform.apply_coords(point_coords, original_size)
        coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=device)
        labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=device)
        masks, iou_predictions, low_res_masks = self.mask_generator.predictor.predict_torch(
            coords_torch[:, None, :],
            labels_torch[:, None],
            None,
            None,
            False,
            return_logits=False,
        )

        data = MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_predictions.flatten(0, 1),
            points=torch.as_tensor(point_coords.repeat(masks.shape[1], axis=0)),
        )
        del masks

        # Threshold masks and calculate boxes
        data["masks"] = data["masks"] > self.mask_generator.predictor.model.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"])

        scores = 1 / box_area(data["boxes"])
        scores = scores.to(data["boxes"].device)
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            scores,
            torch.zeros_like(data["boxes"][:, 0]),
            iou_threshold=self.mask_generator.crop_nms_thresh,
        )
        data.filter(keep_by_nms)

        segments = []
        for masks, boxes in zip(data["masks"], data["boxes"], strict=True):
            bbox = box_xyxy_to_xywh(boxes).tolist()
            if (bbox[2] * bbox[3]) > 0:
                segments.append(Segment(masks.cpu().numpy(), bbox))
        return segments


class FastSAMSegmenter(Segmenter):
    def __init__(self, config: dict[str, Any]) -> None:
        from ultralytics.models.fastsam import FastSAM
        from ultralytics.utils import ops
        self.config = config
        self.fsam = FastSAM(model=os.path.expanduser("~/ultralytics/FastSAM-x.pt"))
        self.scale_masks = ops.scale_masks

    def to(self, device: str) -> None:
        self.fsam.to(device=device)

    def segment(self, image: npt.ArrayLike) -> list[Segment]:
        prediction_result = self.fsam.predict(
            image,
            device=self.config["device"],
            retina_masks=False,
            conf=0.4,
            iou=0.9,
            verbose=False,
        )[0]

        prediction_result.masks.data = self.scale_masks(prediction_result.masks.data.unsqueeze(0), image.shape[:2]).squeeze(0)

        segms = []
        for mask, bbox in zip(prediction_result.masks.data, prediction_result.boxes.xywh, strict=True):
            segms.append(Segment(mask.detach().cpu().numpy(), bbox.detach().cpu().numpy()))
        return segms


class UltralyticsSAMSegmenter(Segmenter):
    def __init__(self, config: dict[str, Any]) -> None:
        from ultralytics.models.sam import SAM
        self.config = config
        self.sam = SAM(model=os.path.expanduser("~/ultralytics/sam_b.pt"))

    def to(self, device: str) -> None:
        self.sam.to(device=device)

    def segment(self, image: npt.ArrayLike) -> list[Segment]:
        prediction_result = self.sam.predict(
            image,
            device=self.config["device"],
            retina_masks=False,
            imgsz=image.shape[1],
            conf=0.4,
            iou=0.9,
            verbose=False,
        )[0]

        segms = []
        for mask, bbox in zip(prediction_result.masks.data, prediction_result.boxes.xywh, strict=True):
            segms.append(Segment(mask.detach().cpu().numpy(), bbox.detach().cpu().numpy()))
        return segms


class SLICSegmenter(Segmenter):
    def __init__(self, config: dict[str, Any] = {}) -> None:
        pass

    def segment(self, image: npt.ArrayLike) -> list[Segment]:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        slic = cv2.ximgproc.createSuperpixelSLIC(
            image_rgb,
            algorithm=cv2.ximgproc.SLIC,
            region_size=200,
            ruler=100)
        slic.iterate(1)
        labels = slic.getLabels()
        num_labels = slic.getNumberOfSuperpixels()

        segms = []
        for label in range(num_labels):
            mask = (labels == label)

            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if not rows.any() or not cols.any():
                continue
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]

            bbox = [x_min, y_min, (x_max-x_min), (y_max-y_min)]

            segms.append(Segment(mask, bbox))
        return segms