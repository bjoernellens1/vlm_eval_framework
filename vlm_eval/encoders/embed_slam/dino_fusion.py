from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
import numpy.typing as npt
from PIL import Image
import time
import os

from dinov3.data.transforms import make_classification_eval_transform, make_base_transform
from dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l

# https://pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class DINOFusion:
    def __init__(
        self,
        device: str = "cpu",
        timing: bool = False,
    ):
        self.device = device
        self.timing = timing

        torch.autograd.set_grad_enabled(False)

        if self.timing:
            tinit = time.time()

        # place DINOv3 weights in "$HOME/.cache/torch/hub/checkpoints"
        # https://github.com/facebookresearch/dinov3?tab=readme-ov-file#pretrained-models
        self.dino_model, self.dino_tokenizer = dinov3_vitl16_dinotxt_tet1280d20h24l()
        self.dino_model.eval()
        self.dino_model = self.dino_model.to(device)
        self.dino_image_preprocess = make_base_transform()

        if self.timing:
            print(f"initialisation: {time.time() - tinit:.2f} s")


    @torch.no_grad()
    def process(self, image: npt.ArrayLike) -> torch.Tensor:
        if self.timing:
            start = time.time()

        image_tensor = self.dino_image_preprocess(Image.fromarray(image)).unsqueeze(0).to(device=self.device)
        image_class_tokens, image_patch_tokens, backbone_patch_tokens = \
            self.dino_model.encode_image_with_patch_tokens(image_tensor, normalize=True)

        B, P, D = image_patch_tokens.shape
        assert B == 1
        PATCH_SIZE = 16
        H = int(image.shape[0] / PATCH_SIZE)
        W = int(image.shape[1] / PATCH_SIZE)
        x = image_patch_tokens.movedim(2, 1).unflatten(2, (H, W)).float()  # [B, D, H, W]

        # upsample from patches to full image resolution
        outfeat = F.normalize(F.interpolate(x, size=image.shape[:2], mode="bilinear"), p=2, dim=1).squeeze(0)
        outfeat = outfeat.movedim(0, 2)  # [H, W, D]

        if self.timing:
            print(f"embeddings: {time.time() - start:.2f} s")

        return outfeat

    @torch.no_grad()
    def query(self, map_embeddings: torch.Tensor, query_text: str) -> torch.Tensor:
        tokenized_texts_tensor = self.dino_tokenizer.tokenize([query_text]).to(device=self.device)
        textfeat = self.dino_model.encode_text(tokenized_texts_tensor)[:, 1024:]
        textfeat = F.normalize(textfeat.float(), p=2, dim=1)

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
