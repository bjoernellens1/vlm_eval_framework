import numpy.typing as npt

import torch
from torch.nn import functional as F
import time
import os

# https://pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from .naradio import NARadioEncoder


class NARadioFusion:
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

        # initialise NARadioEncoder with a default resolution of 64x64 and update when the resolution changes.
        # NOTE: "compile=True" (default) causes:
        #   "RuntimeError: CUDA error: CUBLAS_STATUS_NOT_INITIALIZED when calling `cublasCreate(handle)`"
        # in "torch/_inductor/cudagraph_trees.py"
        self.naradio = NARadioEncoder(
            model_version="radio_v2.5-b",
            lang_model="siglip",
            input_resolution=[64, 64],
            compile=False,
        )

        if self.timing:
            print(f"initialisation: {time.time() - tinit:.2f} s")

    @torch.no_grad()
    def process(self, image: npt.ArrayLike) -> torch.Tensor:
        if self.timing:
            start = time.time()

        resolution = [image.shape[1], image.shape[0]]

        # update resolution
        if self.naradio.input_resolution != resolution:
            self.naradio.input_resolution = resolution

        tensor_image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device).float() / 255.0 # [B, D, H, W]

        feat_map = self.naradio.encode_image_to_feat_map(tensor_image)
        lang_aligned_feat_map = self.naradio.align_spatial_features_with_language(feat_map)
        lang_aligned_feat_map = torch.nn.functional.interpolate(lang_aligned_feat_map, resolution, mode="bilinear", antialias=True)
        lang_aligned_feat_map = lang_aligned_feat_map.squeeze(0).permute(1, 2, 0)
        lang_aligned_feat_map = torch.nn.functional.normalize(lang_aligned_feat_map, dim=-1)

        outfeat = lang_aligned_feat_map.movedim(0, 1)  # [H, W, D]

        if self.timing:
            print(f"embeddings: {time.time() - start:.2f} s")

        return outfeat

    @torch.no_grad()
    def query(self, map_embeddings: torch.Tensor, query_text: str) -> torch.Tensor:
      prompt_embeddings = self.naradio.encode_prompts([query_text])
      prompt_embeddings = F.normalize(prompt_embeddings, dim=-1)

      cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

      map_embeddings = map_embeddings.to(device=self.device)

      # assume that all embeddings are already normalised
      assert torch.allclose(
          torch.linalg.vector_norm(map_embeddings, dim = -1),
          torch.tensor([1], dtype = torch.float32, device=self.device),
          atol = 1e-3,
      )

      similarity = cosine_similarity(map_embeddings, prompt_embeddings)
      return similarity
