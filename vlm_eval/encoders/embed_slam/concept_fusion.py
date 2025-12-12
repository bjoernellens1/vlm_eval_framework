
# code adapted from "ConceptFusion: Open-set multimodal 3D mapping"
# - http://doi.org/10.15607/RSS.2023.XIX.066
# - https://github.com/concept-fusion/concept-fusion

import torch
import numpy as np
import numpy.typing as npt
from pathlib import Path
from PIL import Image
import time

from .segmentation import SAMSegmenter, FastSAMSegmenter, UltralyticsSAMSegmenter, SLICSegmenter, TransformersSAMSegmenter
import open_clip
# from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer


class ConceptFusion:
    def __init__(
        self,
        sam_checkpoint: Path,
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
        elif sam == "UltralyticsSAM":
            self.segmenter = UltralyticsSAMSegmenter({
                "device": device,
            })
        elif sam == "SLIC":
            self.segmenter = SLICSegmenter()

        self.segmenter.to(device)

        clip_model_name = "ViT-H-14"
        self.clip_model, _, self.clip_preprocess = \
            open_clip.create_model_and_transforms(clip_model_name, "laion2b_s32b_b79k")
        self.clip_tokenizer = open_clip.get_tokenizer(clip_model_name)
        
        # model_id = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        # self.clip_model = CLIPModel.from_pretrained(model_id).to(device)
        # self.clip_processor = CLIPProcessor.from_pretrained(model_id)
        # self.clip_tokenizer = CLIPTokenizer.from_pretrained(model_id)
        
        self.clip_model.eval()
        self.clip_model.to(device)

        if self.timing:
            print(f"initialisation: {time.time() - tinit:.2f} s")


    @torch.no_grad()
    def process(self, image: npt.ArrayLike) -> torch.Tensor:
        # extract segments
        if self.timing:
            tsegment = time.time()
        masks = self.segmenter.segment(image)
        # filter out segments with no area
        masks = [m for m in masks if (m.bbox[2] * m.bbox[3]) > 0]
        if self.timing:
            print(f"segmentation: {time.time() - tsegment:.2f} s")

        mask_valid = np.dstack([segment.mask for segment in masks]).any(axis = -1)

        # global OpenCLIP embeddings
        if self.timing:
            tglobal = time.time()
        global_feat = None
        with torch.autocast(device_type=self.device):
            _img = self.clip_preprocess(Image.fromarray(image)).unsqueeze(0).to(device=self.device)
            global_feat = self.clip_model.encode_image(_img)
            # inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            # global_feat = self.clip_model.get_image_features(**inputs)
            global_feat /= global_feat.norm(dim=-1, keepdim=True)
        global_feat = global_feat.half().to(device=self.device)
        global_feat = torch.nn.functional.normalize(global_feat, dim=-1)  # --> (1, 1024)
        feat_dim = global_feat.shape[-1]
        cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        if self.timing:
            print(f"global: {time.time() - tglobal:.2f} s")

        # local OpenCLIP embeddings per segment
        if self.timing:
            tlocal = time.time()
        feat_per_roi = []
        roi_nonzero_inds = []
        similarity_scores = []
        for segment in masks:
            _x, _y, _w, _h = (int(v) for v in segment.bbox)  # xywh bounding box
            nonzero_inds = torch.argwhere(torch.from_numpy(segment.mask))
            # Note: Image is (H, W, 3). In SAM output, y coords are along height, x along width
            img_roi = image[_y : _y + _h, _x : _x + _w, :]
            img_roi = Image.fromarray(img_roi)
            img_roi = self.clip_preprocess(img_roi).unsqueeze(0).to(device=self.device)
            roifeat = self.clip_model.encode_image(img_roi)
            # inputs_roi = self.clip_processor(images=img_roi, return_tensors="pt").to(self.device)
            # roifeat = self.clip_model.get_image_features(**inputs_roi)
            
            roifeat = torch.nn.functional.normalize(roifeat, dim=-1).half()
            feat_per_roi.append(roifeat)
            roi_nonzero_inds.append(nonzero_inds)
            _sim = cosine_similarity(global_feat, roifeat)
            similarity_scores.append(_sim)
        if self.timing:
            print(f"local: {time.time() - tlocal:.2f} s")

        # embedding fusion
        if self.timing:
            tfusion = time.time()
        similarity_scores = torch.cat(similarity_scores)
        softmax_scores = torch.nn.functional.softmax(similarity_scores, dim=0)
        outfeat = torch.zeros(image.shape[0], image.shape[1], feat_dim, dtype=torch.half, device=global_feat.device)
        for maskidx in range(len(masks)):
            _weighted_feat = softmax_scores[maskidx] * global_feat + (1 - softmax_scores[maskidx]) * feat_per_roi[maskidx]
            _weighted_feat = torch.nn.functional.normalize(_weighted_feat, dim=-1)
            outfeat[roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]] += _weighted_feat[0].detach().half()
            outfeat[roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]] = torch.nn.functional.normalize(
                outfeat[roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]].float(), dim=-1
            ).half()
        # mask pixels without valid embedding
        outfeat[~mask_valid] = torch.nan
        if self.timing:
            print(f"fusion: {time.time() - tfusion:.2f} s")

        return outfeat

    @torch.no_grad()
    def query(self, map_embeddings: torch.Tensor, query_text: str) -> torch.Tensor:
        text = self.clip_tokenizer([query_text]).to(device=self.device)
        textfeat = self.clip_model.encode_text(text)
        # inputs_text = self.clip_processor(text=[query_text], return_tensors="pt", padding=True).to(self.device)
        # textfeat = self.clip_model.get_text_features(**inputs_text)
        
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
