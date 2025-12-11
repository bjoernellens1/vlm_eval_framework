from typing import Optional

import torch
import numpy as np
import numpy.typing as npt
from pathlib import Path
from PIL import Image
import time
import copy
import os

from .segmentation import SAMSegmenter, FastSAMSegmenter, UltralyticsSAMSegmenter, SLICSegmenter, TransformersSAMSegmenter
# import open_clip
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import cv2
import random

# deterministic sampling
seed = 1337
random.seed(seed)
np.random.seed(seed)

# https://pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def np_softmax(x):
    exp = np.exp(x)
    return exp / np.sum(exp)

class XFusion:
    def __init__(
        self,
        sam_checkpoint: Optional[Path] = None,
        sam: str = "SAM",
        device: str = "cpu",
        timing: bool = False,
    ):
        self.device = device
        self.timing = timing

        torch.autograd.set_grad_enabled(False)

        if self.timing:
            tinit = time.time()

        if sam == "SAM":
            # self.segmenter = SAMSegmenter({
            #     "model_type": "vit_b",
            #     "checkpoint": sam_checkpoint,
            #     "device": device,
            # })
            self.segmenter = TransformersSAMSegmenter({
                "model_type": "vit_b",
                "device": device,
            })
        elif sam == "FastSAM":
            self.segmenter = FastSAMSegmenter({
                "device": device,
            })
        elif sam == "UltralyticsSAMSegmenter":
            self.segmenter = UltralyticsSAMSegmenter({
                "device": device,
            })
        elif sam == "SLIC":
            self.segmenter = SLICSegmenter()

        self.segmenter.to(device)

        # clip_model_name = "ViT-B-16"
        # self.clip_model, _, self.clip_preprocess = \
        #     open_clip.create_model_and_transforms(clip_model_name, "laion2b_s34b_b88k", precision="fp16")
        # self.clip_tokenizer = open_clip.get_tokenizer(clip_model_name)
        
        model_id = "laion/CLIP-ViT-B-16-laion2B-s34B-b88K"
        self.clip_model = CLIPModel.from_pretrained(model_id).to(device)
        self.clip_processor = CLIPProcessor.from_pretrained(model_id)
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(model_id)
        
        self.clip_model.eval()
        # self.clip_model.to(device)

        if self.timing:
            print(f"initialisation: {time.time() - tinit:.2f} s")


    @torch.no_grad()
    def process(self, image: npt.ArrayLike) -> torch.Tensor:
        segment_strategy = "fillin_refine"

        if self.timing:
            tsegment = time.time()

        if segment_strategy == "original":
            # extract segments once in the original resolution
            masks = self.segmenter.segment(image)
            # filter out segments with no area
            masks = [m for m in masks if (m.bbox[2] * m.bbox[3]) > 0]
            mask_valid = np.dstack([segment.mask for segment in masks]).any(axis = -1)

        elif segment_strategy == "fillin_refine":
            # initial masks
            masks = self.segmenter.segment(image)
            masks = [m for m in masks if (m.bbox[2] * m.bbox[3]) > 0]
            mask_valid = np.dstack([segment.mask for segment in masks]).any(axis = -1)

            # sample coordinates from non-segmented area
            N = 30
            coord_invalid = np.where(np.logical_not(mask_valid))
            coord_invalid = np.array([coord_invalid[1], coord_invalid[0]]).transpose()
            random_indices = random.sample(range(coord_invalid.shape[0]), N)

            masks.extend(self.segmenter.refine(image, coord_invalid[random_indices]))
            mask_valid = np.dstack([segment.mask for segment in masks]).any(axis = -1)

        elif segment_strategy == "fillin_scale":
            # fill-in segments from downscaled segmentation
            scales = [2/3]
            size_orig = image.shape[:2]
            masks = []
            for iscale, scale in enumerate([1] + scales):
                image_s = image.copy()
                if iscale > 0:
                    image_s = cv2.resize(image, dsize=(None, None), fx=scale, fy=scale, interpolation= cv2.INTER_LINEAR)
                # extract segments on scaled images
                masks_s = self.segmenter.segment(image_s)
                # filter out segments with no area
                masks_s = [m for m in masks_s if (m.bbox[2] * m.bbox[3]) > 0]
                if iscale > 0:
                    # upscale results to original resolution
                    for mask in masks_s:
                        mask.mask = cv2.resize(mask.mask.astype(np.uint8), dsize=(size_orig[1], size_orig[0]), fx=1/scale, fy=1/scale, interpolation= cv2.INTER_NEAREST).astype(bool)
                        mask.bbox = np.array(mask.bbox) * 1/scale

                        # add scaled mask if it fills in at least 80% of unknown pixels
                        _x, _y, _w, _h = (int(v) for v in mask.bbox)
                        mask_invalid_roi = np.logical_not(mask_valid[_y : _y + _h, _x : _x + _w])
                        if np.sum(mask_invalid_roi) == 0:
                            continue
                        mask_new_valid_roi = mask.mask[_y : _y + _h, _x : _x + _w]
                        newsegm = np.logical_and(mask_new_valid_roi, mask_invalid_roi)
                        newsegm_ratio = np.sum(newsegm) / np.sum(mask_new_valid_roi)
                        if newsegm_ratio >= 0.8:
                            masks.append(mask)
                else:
                    masks.extend(masks_s)

                mask_valid = np.dstack([segment.mask for segment in masks]).any(axis = -1)

        if self.timing:
            print(f"segmentation: {time.time() - tsegment:.2f} s")

        # local OpenCLIP embeddings per segment
        if self.timing:
            tlocal = time.time()
        mask_rois = torch.empty(len(masks), image.shape[0], image.shape[1], dtype=torch.bool, device=self.device)
        # batch of preprocess segments
        # img_rois = torch.empty(len(masks), image.shape[2], 224, 224, dtype=torch.half, device=self.device)
        img_rois_list = []
        for isegm, segment in enumerate(masks):
            _x, _y, _w, _h = (int(v) for v in segment.bbox)  # xywh bounding box
            mask_rois[isegm] = torch.from_numpy(segment.mask)
            # Note: Image is (H, W, 3). In SAM output, y coords are along height, x along width
            img_roi = copy.deepcopy(image[_y : _y + _h, _x : _x + _w, :])
            mask_roi = segment.mask[_y : _y + _h, _x : _x + _w]

            # blank out invalid data
            img_roi[np.logical_not(mask_roi), :] = [255, 0, 255]

            img_roi = Image.fromarray(img_roi)
            # img_rois[isegm] = self.clip_preprocess(img_roi).unsqueeze(0).to(device=self.device)
            img_rois_list.append(img_roi)

        # feat_per_roi = self.clip_model.encode_image(img_rois, normalize=True).detach()
        if len(img_rois_list) > 0:
            inputs_rois = self.clip_processor(images=img_rois_list, return_tensors="pt", padding=True).to(self.device)
            feat_per_roi = self.clip_model.get_image_features(**inputs_rois).detach()
            feat_per_roi = torch.nn.functional.normalize(feat_per_roi, dim=-1)
        else:
            feat_per_roi = torch.empty(0, self.clip_model.config.projection_dim, device=self.device)

        if self.timing:
            print(f"local: {time.time() - tlocal:.2f} s")

        # embedding fusion
        if self.timing:
            tfusion = time.time()
        outfeat = torch.zeros(image.shape[0], image.shape[1], feat_per_roi.shape[-1], dtype=torch.half, device=self.device)
        for m, f in zip(mask_rois, feat_per_roi, strict=True):
            outfeat[m] = f
        # mask pixels without valid embedding
        outfeat[~mask_valid] = torch.nan
        if self.timing:
            print(f"fusion: {time.time() - tfusion:.2f} s")

        return outfeat

    @torch.no_grad()
    def query(self, map_embeddings: torch.Tensor, query_text: str) -> torch.Tensor:
        # text = self.clip_tokenizer([query_text]).to(device=self.device)
        # textfeat = self.clip_model.encode_text(text)
        inputs_text = self.clip_processor(text=[query_text], return_tensors="pt", padding=True).to(self.device)
        textfeat = self.clip_model.get_text_features(**inputs_text)
        textfeat = torch.nn.functional.normalize(textfeat, dim=-1)

        cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

        map_embeddings = map_embeddings.to(device=self.device)

        # assume that all embeddings are already normalised
        assert torch.allclose(
            torch.linalg.vector_norm(map_embeddings, dim = -1),
            torch.tensor([1], dtype = torch.float32, device=self.device),
            atol = 1e-3,
        )

        similarity = cosine_similarity(map_embeddings, textfeat)
        return similarity
