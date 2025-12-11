"""NARadio encoder from RayFronts."""

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import use_fused_attn
from typing_extensions import override

from vlm_eval.core import BaseEncoder, EncoderRegistry


class GaussKernelAttn(nn.Module):
    """Encompases the NACLIP attention mechanism."""

    def __init__(
        self,
        orig_attn,
        input_resolution: tuple,
        gauss_std: float,
        device,
        chosen_cls_id: int,
        dim: int,
        qk_norm: bool = False,
        num_prefix_tokens: int = 8,
    ) -> None:
        super().__init__()
        num_heads = orig_attn.num_heads
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = use_fused_attn()
        self.input_resolution = input_resolution

        h, w = input_resolution
        n_patches = (w // 16, h // 16)
        window_size = [side * 2 - 1 for side in n_patches]
        window = GaussKernelAttn.gaussian_window(
            *window_size, std=gauss_std, device=device
        )
        self.attn_addition = GaussKernelAttn.get_attention_addition(
            *n_patches, window, num_prefix_tokens
        ).unsqueeze(0)

        self.chosen_cls_id = chosen_cls_id
        self.gauss_std = gauss_std

        self.qkv = orig_attn.qkv
        self.q_norm = orig_attn.q_norm if qk_norm else nn.Identity()
        self.k_norm = orig_attn.k_norm if qk_norm else nn.Identity()
        self.attn_drop = orig_attn.attn_drop
        self.proj = orig_attn.proj
        self.proj_drop = orig_attn.proj_drop
        self.device = device
        self.num_prefix_tokens = num_prefix_tokens

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        B, N, C = x.shape
        x_out = self.custom_attn(x.permute(1, 0, 2))
        x_out = x_out.permute(1, 0, 2)
        return x_out

    @staticmethod
    def gaussian_window(dim1, dim2, std=5.0, device="cuda"):
        constant = 1 / (std * math.sqrt(2))
        start = -(dim1 - 1) / 2.0
        k1 = torch.linspace(
            start=start * constant,
            end=(start + (dim1 - 1)) * constant,
            steps=dim1,
            dtype=torch.float,
            device=device,
        )
        start = -(dim2 - 1) / 2.0
        k2 = torch.linspace(
            start=start * constant,
            end=(start + (dim2 - 1)) * constant,
            steps=dim2,
            dtype=torch.float,
            device=device,
        )
        dist_square_to_mu = (
            torch.stack(torch.meshgrid(k1, k2, indexing="ij")) ** 2
        ).sum(0)

        return torch.exp(-dist_square_to_mu)

    @staticmethod
    def get_attention_addition(dim1, dim2, window, num_prefix_tokens=8):
        d = window.device
        m = torch.einsum(
            "ij,kl->ijkl", torch.eye(dim1, device=d), torch.eye(dim2, device=d)
        )
        m = m.permute((0, 3, 1, 2)).contiguous()
        out = F.conv2d(
            m.view(-1, dim1, dim2).unsqueeze(1),
            window.unsqueeze(0).unsqueeze(1),
            padding="same",
        ).squeeze(1)

        out = out.view(dim1 * dim2, dim1 * dim2)
        if num_prefix_tokens > 0:
            v_adjusted = torch.vstack(
                [torch.zeros((num_prefix_tokens, dim1 * dim2), device=d), out]
            )
            out = torch.hstack(
                [
                    torch.zeros(
                        (dim1 * dim2 + num_prefix_tokens, num_prefix_tokens), device=d
                    ),
                    v_adjusted,
                ]
            )

        return out

    def custom_attn(self, x):
        num_heads = self.num_heads
        num_tokens, bsz, embed_dim = x.size()
        head_dim = embed_dim // num_heads
        scale = head_dim**-0.5

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        # kk.T vs kq.T has the most impact
        attn_weights = torch.bmm(k, k.transpose(1, 2)) * scale

        # Gaussian attention seems to have minimal impact
        attn_weights += self.attn_addition
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = torch.bmm(attn_weights, v)
        attn_output = (
            attn_output.transpose(0, 1).contiguous().view(-1, bsz, embed_dim)
        )
        attn_output = self.proj(attn_output)
        attn_output = self.proj_drop(attn_output)

        return attn_output

    def update_input_resolution(self, input_resolution):
        h, w = input_resolution
        n_patches = (w // 16, h // 16)
        window_size = [side * 2 - 1 for side in n_patches]
        window = GaussKernelAttn.gaussian_window(
            *window_size, std=self.gauss_std, device=self.device
        )
        self.attn_addition = GaussKernelAttn.get_attention_addition(
            *n_patches, window, self.num_prefix_tokens
        ).unsqueeze(0)


@EncoderRegistry.register("naradio")
class NARadioEncoder(BaseEncoder):
    """The RayFronts Encoder based on NACLIP + RADIO models.

    The model modifies the attention of the last layer of RADIO following the
    example of NACLIP improving spatial structure. And uses the Summary CLS
    projection to project the patch-wise tokens to SIGLIP or CLIP language aligned
    feature spaces.
    """

    def __init__(
        self,
        model_version: str = "radio_v2.5-b",
        lang_model: str = "siglip",
        input_size: int = 512,
        gauss_std: float = 7.0,
        return_radio_features: bool = True,
        compile_model: bool = False,
        amp: bool = True,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_version: Choose from "radio_v2.5-x" where x can be b,l, or g.
            lang_model: choose from ["siglip", "clip"]
            input_size: Size of the input images (assumed square).
            gauss_std: Standard deviation of the gaussian kernel.
            return_radio_features: Whether to return radio features which are not
                language aligned or whether to project them to the language aligned
                space directly.
            compile_model: Whether to compile the model or not.
            amp: Whether to use automatic mixed percision or not.
            device: "cpu" or "cuda", set to None to use CUDA if available.
        """
        super().__init__()
        
        self.input_size = input_size
        self.compile_model = compile_model
        self.amp = amp
        self.model_version = model_version
        self.return_radio_features = return_radio_features
        self.lang_model_name = lang_model
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load model
        self.model = torch.hub.load(
            "NVlabs/RADIO",
            "radio_model",
            version=model_version,
            progress=True,
            skip_validation=True,
            adaptor_names=[lang_model],
        )
        self.model.eval()
        self.model = self.model.to(self.device)
        self.model.make_preprocessor_external()
        
        # Steal adaptors from RADIO so it does not auto compute adaptor output.
        self.lang_adaptor = self.model.adaptors[lang_model]
        self.model.adaptors = None
        
        last_block = self.model.model.blocks[-1]
        last_block.attn = GaussKernelAttn(
            last_block.attn,
            (input_size, input_size),
            gauss_std,
            dim=self.model.model.embed_dim,
            chosen_cls_id=self.lang_adaptor.head_idx,
            device=self.device,
            num_prefix_tokens=self.model.num_summary_tokens,
        )

        if self.compile_model:
            self.model.compile(fullgraph=True, options={"triton.cudagraphs": True})
            self.lang_adaptor.compile(
                fullgraph=True, options={"triton.cudagraphs": True}
            )
            
        self._patch_size = self.model.patch_size
        
        # Determine output channels
        # We need to run a dummy pass or check dimensions
        if return_radio_features:
            self._output_channels = self.model.model.embed_dim
        else:
            # If projected, it depends on the language model
            # But wait, the original code says:
            # if not self.return_radio_features:
            #   out = self.lang_adaptor.head_mlp(out)
            # So we need to know the output dim of head_mlp
            # Let's do a dummy run to be safe and sure
            with torch.no_grad():
                dummy = torch.zeros(1, 3, input_size, input_size, device=self.device)
                if amp:
                    with torch.autocast("cuda", dtype=torch.float16):
                        out = self.model(dummy).features
                        if not return_radio_features:
                            out = self.lang_adaptor.head_mlp(out)
                else:
                    out = self.model(dummy).features
                    if not return_radio_features:
                        out = self.lang_adaptor.head_mlp(out)
                self._output_channels = out.shape[-1]

    @property
    def output_channels(self) -> int:
        return self._output_channels

    @property
    def patch_size(self) -> int:
        return self._patch_size

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, H, W)
        Returns:
            (B, C, H', W')
        """
        # Ensure input size matches what was configured, or update resolution
        # The original code supports updating resolution.
        # For now, let's assume input is resized to self.input_size before calling,
        # or we resize it here. BaseEncoder usually expects resizing to happen outside 
        # or we handle it.
        
        B, C, H, W = images.shape
        
        if H != self.input_size or W != self.input_size:
             images = F.interpolate(
                images,
                size=(self.input_size, self.input_size),
                mode='bilinear',
                align_corners=False
            )
        
        # Move images to device if needed
        if images.device != self.device:
            images = images.to(self.device)

        H_, W_ = self.input_size // self.model.patch_size, self.input_size // self.model.patch_size
        
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            out = self.model(images).features
            if not self.return_radio_features:
                out = self.lang_adaptor.head_mlp(out)
        
        # out is (B, N, C) -> (B, C, H, W)
        return out.permute(0, 2, 1).reshape(B, -1, H_, W_)

    def get_config(self) -> Dict[str, Any]:
        return {
            "model_version": self.model_version,
            "lang_model": self.lang_model_name,
            "input_size": self.input_size,
            "return_radio_features": self.return_radio_features,
            "compile_model": self.compile_model,
            "amp": self.amp,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "NARadioEncoder":
        config = {k: v for k, v in config.items() if k != "name"}
        return cls(**config)
