"""
visionaction_vit.py
"""

from typing import Callable, Dict, Optional, Tuple, Type, Union

import torch
from prismatic.models.backbones.vision.base_vision import TimmViTBackbone


# Registry =>> Supported SigLIP Vision Backbones (from TIMM) =>> Note:: Using SigLIP w/ Patch = 14 (but SO400M Arch)
VisionAction_VISION_BACKBONES = {
    "visionaction-siglip-vit-so400m": "vit_so400m_patch14_siglip_224",
    "visionaction-vit-giantopt-patch16-siglip-256": "vit_giantopt_patch16_siglip_256",
    "visionaction-dinov2": "vit_large_patch14_reg4_dinov2.lvd142m"
}


class VisionActionViTBackbone(TimmViTBackbone):
    def __init__(self, vision_backbone_id: str, image_resize_strategy: str, default_image_size: int = 224, 
                 action_dim: int = 7, action_chunk: int = 1, use_diffusion: bool = False, 
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
        super().__init__(
            vision_backbone_id,
            VisionAction_VISION_BACKBONES[vision_backbone_id],
            image_resize_strategy,
            default_image_size=default_image_size,
            use_actionvision_vit=True,
            action_dim=action_dim,
            action_chunk=action_chunk, 
            use_diffusion=use_diffusion,
            vit_weight_path=vit_weight_path,
            eval_mode=eval_mode,
            use_proprio=use_proprio,
            end_lastlayer=end_lastlayer,
            use_film=use_film,
            taskembedding_add=taskembedding_add,
            baseline=baseline,
            vat_small_factor=vat_small_factor,
            vat_vit=vat_vit,
        )

    def forward(self, pixel_values: Union[torch.Tensor, Dict[str, torch.Tensor]],
                diffusion_timestep_embeddings: Optional[torch.Tensor] = None,
                noisy_action_features: Optional[torch.Tensor] = None,
                proprio: Optional[torch.Tensor] = None,
                task_id: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        """Runs transformed image/pixel tensor through vision backbone, returning _all_ patch features."""
        # pixel_values = torch.load('pixel.pth').to(pixel_values.device)
        if not self.baseline:
            # get tensor out1 and prefix1 from tuple(out1, prefix1)
            _, action_token = self.featurizer(pixel_values, diffusion_timestep_embeddings=diffusion_timestep_embeddings, noisy_action_features=noisy_action_features, proprio=proprio, task_id=task_id)
        else:
            # pixel values shape [B, 3*image_num, 224, 224]
         
            batch_size, channel_mul_image_num, h, w = pixel_values.shape
            # channel per image is 3
            image_num = channel_mul_image_num // 3 

     
            pixel_values_reshaped = pixel_values.reshape(batch_size * image_num, 3, h, w)

        
            raw_vision_features = self.vit(pixel_values_reshaped)

         
            processed_features = []
            for feature in raw_vision_features:
           
                processed = feature.reshape(batch_size, image_num * feature.shape[1], -1)
                processed_features.append(processed)
            
            _, action_token = self.featurizer(vision_features=processed_features, task_id=task_id)
        return _, action_token
    