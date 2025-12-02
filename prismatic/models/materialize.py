"""
materialize.py

Factory class for initializing Vision Backbones, LLM Backbones, and VLMs from a set registry; provides and exports
individual functions for clear control flow.
"""

from typing import Optional, Tuple
from prismatic.models.backbones.vision import (
    ImageTransform,
    VisionBackbone,
    VisionActionViTBackbone
)

# === Registries =>> Maps ID --> {cls(), kwargs} :: Different Registries for Vision Backbones, LLM Backbones, VLMs ===
# fmt: off

# === Vision Backbone Registry ===
VISION_BACKBONES = {

    # === vision action VIT Backbones ===
    "visionaction-siglip-vit-so400m": {"cls": VisionActionViTBackbone, "kwargs": {"default_image_size": 224}},
    "visionaction-vit-giantopt-patch16-siglip-256": {"cls": VisionActionViTBackbone, "kwargs": {"default_image_size": 256}},
    "visionaction-dinov2": {"cls": VisionActionViTBackbone, "kwargs": {"default_image_size": 224}},

   
}


def get_vision_backbone_and_transform(
    vision_backbone_id: str, image_resize_strategy: str
) -> Tuple[VisionBackbone, ImageTransform]:
    """Instantiate a Vision Backbone, returning both the nn.Module wrapper class and default Image Transform."""
    if vision_backbone_id in VISION_BACKBONES:
        vision_cfg = VISION_BACKBONES[vision_backbone_id]
        vision_backbone: VisionBackbone = vision_cfg["cls"](
            vision_backbone_id, image_resize_strategy, **vision_cfg["kwargs"]
        )
        image_transform = vision_backbone.get_image_transform()
        return vision_backbone, image_transform

    else:
        raise ValueError(f"Vision Backbone `{vision_backbone_id}` is not supported!")

def get_actionvision_backbone_and_transform(
    vision_backbone_id: str, image_resize_strategy: str, action_dim: int, action_chunk: int, 
    use_diffusion: bool, vit_weight_path: None | str = None,  
    eval_mode: bool = False,
    use_proprio: bool = False,
    end_lastlayer: int = 2,
    use_film: bool = True,
    taskembedding_add: bool = False,
    baseline: bool = False,
    vat_small_factor: int = 1,
    vat_vit: bool = False,
) -> Tuple[VisionBackbone, ImageTransform]:
    """Instantiate a Vision Backbone, returning both the nn.Module wrapper class and default Image Transform."""
    if vision_backbone_id in VISION_BACKBONES:
        vision_cfg = VISION_BACKBONES[vision_backbone_id]
        vision_backbone: VisionBackbone = vision_cfg["cls"](
            vision_backbone_id, 
            image_resize_strategy, 
            action_dim=action_dim, 
            action_chunk=action_chunk, 
            use_diffusion=use_diffusion, 
            vit_weight_path=vit_weight_path, 
            eval_mode=eval_mode, use_proprio=use_proprio,
            end_lastlayer=end_lastlayer,
            use_film=use_film,
            taskembedding_add=taskembedding_add,
            baseline=baseline,
            vat_small_factor=vat_small_factor,
            vat_vit=vat_vit,
            **vision_cfg["kwargs"]
        )
        image_transform = vision_backbone.get_image_transform()
        return vision_backbone, image_transform

    else:
        raise ValueError(f"Vision Backbone `{vision_backbone_id}` is not supported!")





