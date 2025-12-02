"""
base_vision.py

Abstract class definition of a Vision Backbone (Visual Featurizer), with full annotations of class methods, utility
functions, and initialization logic.

We also define the generic TimmViTBackbone class here, providing a default interface for loading any TIMM Vision
Transformer model for feature extraction.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Optional, Protocol, Tuple, Union

import timm
import torch
import torch.nn as nn
import torchvision.transforms.functional as TVF
from PIL.Image import Image
from timm.models.vision_transformer import Block, VisionTransformer
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy, transformer_auto_wrap_policy
from torchvision.transforms import Compose, Resize

from prismatic.models.backbones.vision.vat_modeling import ActionVisionFusionPlusCrossAttnTransformer
from prismatic.models.backbones.vision.actionmodule_modeling import ActionModuleTransformer
from prismatic.models.backbones.vision.vat_small_modeling import VATSMALL
from prismatic.models.backbones.vision.vat_vit_modeling import VATVIT
# === Utility Functions for Monkey-Patching ===
def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper



# === Interface for an Image Transform ===
class ImageTransform(Protocol):
    def __call__(self, img: Image, **kwargs: str) -> Union[torch.Tensor, Dict[str, torch.Tensor]]: ...


# === Custom Torchvision Image Transforms ===
@dataclass
class LetterboxPad:
    padding_fill_value: Tuple[int, int, int]

    def __call__(self, image: Image) -> Image:
        """Given a PIL.Image, pad to square by adding a symmetric border around the height/width."""
        (w, h), max_wh = image.size, max(image.size)
        horizontal_pad, vertical_pad = int((max_wh - w) / 2), int((max_wh - h) / 2)
        padding = (horizontal_pad, vertical_pad, horizontal_pad, vertical_pad)
        return TVF.pad(image, padding, fill=self.padding_fill_value, padding_mode="constant")
    



def vatmodel_load_state_dict(model, state_dict, vit_weight_path):
    """
    加载预训练参数到新模型
    
    Args:
        model: 要加载参数的新模型
        state_dict: 旧的参数字典
        vit_weight_path: ViT权重路径
        load_mode: 加载模式
            1: 所有Attention参数都加载旧qkv参数(分割后分配)
            2: 仅qkv_non_act加载旧qkv参数,q_act/kv_act保持初始值
    """
    if vit_weight_path is not None:
        vit_state_dict = torch.load(vit_weight_path, map_location="cpu").get('state_dict', {})
        state_dict = {
            k.replace("module.visual.trunk.", "").replace("visual.trunk.", ""): v 
            for k, v in vit_state_dict.items() 
            if k.startswith("visual.trunk.") or k.startswith("module.visual.trunk.")
        }
        print(f'###########  Loaded state dict from {vit_weight_path}  ###########')
    else:
        print(f'###########  Using siglip state dict from huggingface  ###########')
    
    modified_state_dict = {}
    
    for key, value in state_dict.items():
        if 'attn.qkv' in key:
            parts = key.split('.')
            block_idx = parts[1]
            suffix = parts[-1]
            
            new_key = f"blocks.{block_idx}.attn.qkv_non_act.{suffix}"
            modified_state_dict[new_key] = value
        elif 'attn.proj' in key:
            parts = key.split('.')
            block_idx = parts[1]
            suffix = parts[-1]
            
            if 'weight' in suffix:
                modified_state_dict[f"blocks.{block_idx}.attn.proj_non_act.weight"] = value
                
            elif 'bias' in suffix:
                modified_state_dict[f"blocks.{block_idx}.attn.proj_non_act.bias"] = value
        elif 'norm' in key and any(sub in key for sub in ['weight', 'bias']) and 'attn_pool' not in key:
            if key == 'norm.weight':
                modified_state_dict[f'norm_non_act.weight'] = value
            elif key == 'norm.bias':
                modified_state_dict[f'norm_non_act.bias'] = value
            else:
                parts = key.split('.')
                block_idx = parts[1]
                norm_index = parts[2]
                suffix = parts[-1]
                
                if norm_index == 'norm1':
                    modified_state_dict[f"blocks.{block_idx}.norm1_non_act.{suffix}"] = value
                elif norm_index == 'norm2':
                    modified_state_dict[f"blocks.{block_idx}.norm2_non_act.{suffix}"] = value
        elif '.mlp.' in key and 'attn_pool' not in key:
            modified_state_dict[key.replace('.mlp.', '.mlp_non_act.')] = value
        else:
            modified_state_dict[key] = value
    print(f'###########  Load qkv and MLP for non-act branch only  ###########')
    
    model_state_dict = model.state_dict()
    
    missing_keys = set(model_state_dict.keys()) - set(modified_state_dict.keys())
    
    allowed_missing = ["action_token", "action_pos_embed", "task_embedding", "film_modulator"]
    
  
    allowed_missing = allowed_missing + ["attn.q_act", "attn.kv_act", "mlp_act", "proj_act", "norm1_act", "norm2_act", "norm_act"]
    
    if all(any(allow in key for allow in allowed_missing) for key in missing_keys):
        print(f"###########  Loading with strict=False (ignoring allowed missing keys: {allowed_missing})  ###########")
        model.load_state_dict(modified_state_dict, strict=False)
    else:
        print("###########  Loading with strict=True  ###########")
        model.load_state_dict(modified_state_dict, strict=True)
    
    return model

vit_large_param_names = {
    "img_size": 256,
    "patch_size": 16, 
    "in_chans": 3, 
    "num_classes": 0, 
    "global_pool": "map", 
    "embed_dim": 1536, 
    "depth": 40, 
    "num_heads": 16, 
    "mlp_ratio": 4.0, 
    "qkv_bias": True, 
    "qk_norm": False, 
    "init_values": None, 
    "class_token": False, 
    "no_embed_class": False,
    "reg_tokens": 0, 
    "pre_norm": False, 
    "fc_norm": None,
    "dynamic_img_size": False, 
    "dynamic_img_pad": False, 
    "drop_rate": 0,
    "pos_drop_rate": 0, 
    "patch_drop_rate": 0, 
    "proj_drop_rate": 0,
    "attn_drop_rate": 0, 
    "drop_path_rate": 0, 
    "weight_init": '', 
}
 
dinov2_param_names = {
    "img_size": 224,
    "patch_size": 14, 
    "in_chans": 3, 
    "num_classes": 0, 
    "global_pool": "token", 
    "embed_dim": 1024, 
    "depth": 24, 
    "num_heads": 16, 
    "mlp_ratio": 4.0, 
    "qkv_bias": True, 
    "qk_norm": False, 
    "init_values": 1e-05, 
    "class_token": True, 
    "no_embed_class": True,
    "reg_tokens": 4, 
    "pre_norm": False, 
    "fc_norm": None,
    "dynamic_img_size": False, 
    "dynamic_img_pad": False, 
    "drop_rate": 0,
    "pos_drop_rate": 0, 
    "patch_drop_rate": 0, 
    "proj_drop_rate": 0,
    "attn_drop_rate": 0, 
    "drop_path_rate": 0, 
    "weight_init": '', 

}

# === Abstract Base Class for arbitrary Vision Backbones ===
class VisionBackbone(nn.Module, ABC):
    def __init__(self, vision_backbone_id: str, image_resize_strategy: str, default_image_size: int = 224) -> None:
        super().__init__()
        self.identifier: str = vision_backbone_id
        self.image_resize_strategy: str = image_resize_strategy
        self.default_image_size: int = default_image_size

        # Instance attributes for a Vision Backbone
        self.featurizer: nn.Module = None
        self.image_transform: ImageTransform = None

    def get_image_transform(self) -> ImageTransform:
        return self.image_transform

    @abstractmethod
    def get_fsdp_wrapping_policy(self) -> Callable: ...

    @abstractmethod
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the featurizer given a set of processed images, returning patch/grid features."""
        raise NotImplementedError

    @property
    @abstractmethod
    def default_image_resolution(self) -> Tuple[int, int, int]: ...

    @property
    @abstractmethod
    def embed_dim(self) -> int: ...

    @property
    @abstractmethod
    def num_patches(self) -> int: ...

    @property
    @abstractmethod
    def half_precision_dtype(self) -> torch.dtype: ...


# === Abstract Base Class for Arbitrary TIMM Vision Transformer Backbones ===
class TimmViTBackbone(VisionBackbone, ABC):
    def __init__(
        self,
        vision_backbone_id: str,
        timm_path_or_url: str,
        image_resize_strategy: str,
        default_image_size: int = 224,
        override_act_layer: Optional[str] = None,
        use_actionvision_vit: bool = False,  
        action_dim: int = 7,
        action_chunk: int = 1,
        use_diffusion: bool = False,
        vit_weight_path: Optional[str] = None,
        eval_mode: bool = False,
        use_proprio: bool = False,
        end_lastlayer: int = 2,
        use_film: bool = True,
        taskembedding_add: bool = False,
        baseline: bool = False,
        vat_small_factor: int = 1,
        vat_vit: bool = False,
    ) -> None:
        super().__init__(vision_backbone_id, image_resize_strategy, default_image_size=default_image_size)
        self.timm_path_or_url = timm_path_or_url
        self.override_act_layer = override_act_layer
        self.dtype = torch.bfloat16
        self.baseline = baseline
        
        vision_backbone_params = {}
        if 'giantopt' in self.timm_path_or_url: 
            vision_backbone_params = vit_large_param_names
        elif 'dinov2' in self.timm_path_or_url:
            vision_backbone_params = dinov2_param_names
        if use_actionvision_vit == True:
            # for training mode, initialize featurizer from siglip for loading state dict
            if not eval_mode and vit_weight_path is None:
                
                featurizer: VisionTransformer = timm.create_model(
                        self.timm_path_or_url, 
                        pretrained=True, 
                        num_classes=0, 
                        img_size=self.default_image_size,       
                    )
            
            if baseline == True:
                self.vit = featurizer
                # n is a list including from 0 to len(self.vit.blocks)
                # return a tuple of (out1, out2, ...)
                self.vit.forward = lambda x: self.vit.get_intermediate_layers(
                x, n=list(range(len(self.vit.blocks))), return_prefix_tokens=False
                )
                self.featurizer: VisionTransformer = ActionModuleTransformer(
                        action_dim=action_dim, 
                        action_chunk=action_chunk, 
                        use_diffusion=use_diffusion,
                        use_proprio=use_proprio,
                        use_film=use_film,
                        taskembedding_add=taskembedding_add,
                        **vision_backbone_params
                )
                # raise ValueError("baseline must be False when use_actionvision_vit is True")
            else:
                if vat_small_factor != 1:
                    self.featurizer: VisionTransformer = VATSMALL(
                            action_dim=action_dim, 
                            action_chunk=action_chunk, 
                            use_diffusion=use_diffusion,
                            use_proprio=use_proprio,
                            use_film=use_film,
                            taskembedding_add=taskembedding_add,
                            embed_scaling_factor=vat_small_factor,
                            **vision_backbone_params
                            )
                else:
                    if vat_vit:
                        self.featurizer: VisionTransformer = VATVIT(
                            action_dim=action_dim, 
                            action_chunk=action_chunk, 
                            use_diffusion=use_diffusion,
                            use_proprio=use_proprio,
                            use_film=use_film,
                            taskembedding_add=taskembedding_add,
                            **vision_backbone_params
                            )
                    else:
                        self.featurizer: VisionTransformer = ActionVisionFusionPlusCrossAttnTransformer(
                            action_dim=action_dim, 
                            action_chunk=action_chunk, 
                            use_diffusion=use_diffusion,
                            use_proprio=use_proprio,
                            use_film=use_film,
                            taskembedding_add=taskembedding_add,
                            **vision_backbone_params
                            )
            if not eval_mode and not self.baseline:
                vatmodel_load_state_dict(
                        self.featurizer, 
                        None if vit_weight_path is not None else featurizer.state_dict(),
                        vit_weight_path, 
                    )
        
            self.featurizer.eval()

        else:
            raise ValueError("TimmViTBackbone must be used with use_actionvision_vit=True") 
            

        # Monkey-Patch the `forward()` function of the featurizer to ensure FSDP-compatibility
        #   => Note: By default set `get_intermediate_layers` to return the *SECOND-TO-LAST* layer patches!
        #   => TODO (siddk) Remove after resolution of https://github.com/pytorch/pytorch/issues/109385
        if use_actionvision_vit == True:
            # use unpack tuple to get (out1, prefix1) from outputs: ((out1, prefix1), (out2, prefix2), ...)
            self.featurizer.forward = unpack_tuple(
                partial(self.featurizer.get_intermediate_layers, n={len(self.featurizer.blocks) - end_lastlayer}, 
                        return_prefix_tokens=True,)
            )
        else:
            raise ValueError("TimmViTBackbone must be used with use_actionvision_vit=True") 
        # Validation =>> for now, this class *only* supports TIMM Vision Transformers (but can be extended!)
        assert isinstance(self.featurizer, VisionTransformer), (
            "Featurizer is not a TIMM VisionTransformer; if you would like to support a new visual representation, "
            "file an issue or implement the requisite logic (see `prismatic/models/backbones/vision/base_vision.py`)!"
        )

        # Get Config =>> Note :: Override default image size to ensure correct image transform
        self.data_cfg = timm.data.resolve_model_data_config(self.featurizer)
        self.data_cfg["input_size"] = (3, self.default_image_size, self.default_image_size)

        # Initialize Default Image Transform --> Modified by `self.image_resize_strategy`
        default_image_transform = timm.data.create_transform(**self.data_cfg, is_training=False)

        # Fix =>> SigLIP & IN1K default transforms resize to *larger* than `self.default_image_size` (crops image)!
        if "siglip" in self.timm_path_or_url or "in1k" in self.timm_path_or_url:
            assert isinstance(default_image_transform, Compose), "Unexpected `default_image_transform`!"
            assert isinstance(default_image_transform.transforms[0], Resize)
            default_image_transform = Compose(
                [
                    Resize(self.default_image_size, interpolation=default_image_transform.transforms[0].interpolation),
                    *default_image_transform.transforms[1:],
                ]
            )

        # Switch on `image_resize_strategy`
        if self.image_resize_strategy == "resize-naive":
            assert isinstance(default_image_transform, Compose), "Unexpected `default_image_transform`!"
            assert isinstance(default_image_transform.transforms[0], Resize)

            target_size = (self.default_image_size, self.default_image_size)
            self.image_transform = Compose(
                [
                    Resize(target_size, interpolation=default_image_transform.transforms[0].interpolation),
                    *default_image_transform.transforms[1:],
                ]
            )

        elif self.image_resize_strategy == "resize-crop":
            self.image_transform = default_image_transform

        elif self.image_resize_strategy == "letterbox":
            assert isinstance(default_image_transform, Compose), "Unexpected `default_image_transform`!"
            assert "mean" in self.data_cfg, "TIMM `data_cfg` missing image normalization mean!"

            # Compute Padding Fill Value (rescaled normalization mean if applicable)
            fill = tuple([int(x * 255) for x in self.data_cfg["mean"]])

            # Build New Transform
            self.image_transform = Compose([LetterboxPad(fill), *default_image_transform.transforms])

        else:
            raise ValueError(f"Image Resize Strategy `{self.image_resize_strategy}` is not supported!")

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return a simple FSDP policy that wraps each ViT block and then the _entire_ featurizer."""
        vit_wrap_policy = partial(_module_wrap_policy, module_classes={VisionTransformer})
        transformer_block_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
        return partial(_or_policy, policies=[vit_wrap_policy, transformer_block_policy])

    def forward(self, pixel_values: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Runs transformed image/pixel tensor through vision backbone, returning _all_ patch features."""
        return self.featurizer(pixel_values)

    @property
    def default_image_resolution(self) -> Tuple[int, int, int]:
        return self.data_cfg["input_size"]

    @property
    def embed_dim(self) -> int:
        return self.featurizer.embed_dim

    @property
    def num_patches(self) -> int:
        return self.featurizer.patch_embed.num_patches

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return self.dtype
