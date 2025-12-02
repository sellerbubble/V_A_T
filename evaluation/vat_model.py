import argparse
import dataclasses
import enum
import logging
import socket


"""
openvla_model.py

Provides functionality to create OpenVLA model and get robot actions.
"""

import argparse
import os.path
import json
import logging
import numpy as np
from typing import Any, Dict, List, Optional
import re

import torch
from PIL import Image

from experiments.robot.openvla_utils import (
    get_action_head_forvit,
    get_noisy_action_projector,
    get_proprio_projector
)

from evaluation.modeling_vat import get_vat_backbone_and_transform

class VATModel:
    def __init__(self, cfg) -> None:
        """
        A class for OpenVLA models that can predict actions given observations and instructions.
        """
        self.cfg = cfg

        # Load model
        # self.vla = get_vla(cfg)

        if cfg.vit_large:
            vit_name =  "visionaction-vit-giantopt-patch16-siglip-256"   
        elif cfg.dino: 
            vit_name =  "visionaction-dinov2"
        else:
            vit_name = "visionaction-siglip-vit-so400m"
            
        # state dict should be loaded from a local file
        result = get_vat_backbone_and_transform(
            vit_name, image_resize_strategy="resize-naive", 
            action_dim=cfg.action_dim_input, action_chunk=cfg.action_chunk, use_diffusion=cfg.use_diffusion,
            use_proprio=cfg.use_proprio,
            end_lastlayer=cfg.end_lastlayer,
            use_film=cfg.use_film,
            taskembedding_add=cfg.taskembedding_add,
            baseline=cfg.baseline,
            vat_small_factor=cfg.vat_small_factor,
            vat_vit=cfg.vat_vit
        )   
        
        if cfg.baseline:
            self.vit, self.vat = result[0]  
            self.image_transform = result[1]
        else:
            self.vat_backbone = result[0]
            self.image_transform = result[1]
            
        match = re.search(r'--(\d+)_ckpt$', cfg.pretrained_checkpoint)
        step_num = match.group(1)
        if cfg.baseline:
            vit_statedict = torch.load(os.path.join(cfg.pretrained_checkpoint, f"vit--{step_num}_checkpoint.pt"))
            vit_statedict = {k.replace("module.vit.", ""): v for k, v in vit_statedict.items()}
            self.vit.load_state_dict(vit_statedict, strict=True)
            self.vit = self.vit.to("cuda:0")
            vat_statedict = torch.load(os.path.join(cfg.pretrained_checkpoint, f"vat--{step_num}_checkpoint.pt"))
            vat_statedict = {k.replace("module.featurizer.", ""): v for k, v in vat_statedict.items()}
            self.vat.load_state_dict(vat_statedict, strict=True)
            self.vat = self.vat.to("cuda:0")
        else:
            state_dict = torch.load(os.path.join(cfg.pretrained_checkpoint, f"vision_backbone--{step_num}_checkpoint.pt"))
            state_dict = {k.replace("module.featurizer.", ""): v for k, v in state_dict.items()}
            # set strict False for skipping task embedding missing
            self.vat_backbone.load_state_dict(state_dict, strict=True)
            self.vat_backbone = self.vat_backbone.to("cuda:0")
        print(f'###########  Loaded model from {cfg.pretrained_checkpoint}')
                
        # Load continuous action head
        self.action_head = None
        if not cfg.baseline:
            embed_dim = self.vat_backbone.embed_dim // cfg.vat_small_factor
        else:
            embed_dim = self.vat.embed_dim
            
        if cfg.use_l1_regression or cfg.use_diffusion:
            self.action_head = get_action_head_forvit(cfg, embed_dim)

        self.proprio_projector = None
        if cfg.use_proprio:
            self.proprio_projector = get_proprio_projector(
                cfg,
                1152,
                proprio_dim=8,  # 8-dimensional proprio for LIBERO
            )

        # Load noisy action projector
        self.noisy_action_projector = None
        if cfg.use_diffusion:
            self.noisy_action_projector = get_noisy_action_projector(cfg, embed_dim)

        #load dataset statistics
        dataset_statistics_path = os.path.join(cfg.pretrained_checkpoint, "dataset_statistics.json")
        if os.path.isfile(dataset_statistics_path):
            with open(dataset_statistics_path, "r") as f:
                norm_stats = json.load(f)
        self.norm_stats = norm_stats
        # Check that the model contains the action un-normalization key

        if "libero" in cfg.task_suite_name:
            unnorm_key = cfg.task_suite_name + "_no_noops"
        else:
            unnorm_key = cfg.unnorm_key
        assert unnorm_key in self.norm_stats, f"Action un-norm key {unnorm_key} not found in `norm_stats`!"
        self.unnorm_key = unnorm_key


    def _unnormalize_actions(self, normalized_actions, unnorm_key=None):
        """Unnormalize actions using dataset statistics"""
        action_norm_stats = self.norm_stats[unnorm_key]["action"]

        # if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
        # mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
        # action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
        # elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        # else:
        #     raise ValueError("Unsupported action/proprio normalization type detected!")
        normalized_actions = normalized_actions.cpu().float()
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low + 1e-8) + action_low,
            normalized_actions,
        )

        return actions

    def run_diffusion_sampling(
        self,
        pixel_values,
        num_actions_chunk,
        action_dim,
        proprio,
        task_id,
    ) -> torch.Tensor:
        """
        Run diffusion sampling (reverse diffusion) to generate actions.

        Args:
            vla (OpenVLAForActionPrediction): Vision-language-action policy.
            action_head (nn.Module): Action head module.
            noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
            proprio_projector (nn.Module): Proprioceptive state projector module.
            batch (dict): Input batch.
            batch_size (int): Batch size.
            num_patches (int): Number of vision patches.
            actions_shape (tuple): Shape of ground-truth actions.
            'cuda:0' (str): Device ID.
            current_action_mask (torch.Tensor): Mask for current action.
            next_actions_mask (torch.Tensor): Mask for next actions.
            use_proprio (bool): Whether to use proprioceptive state as input.
            use_film (bool): Whether to use FiLM for better language following.

        Returns:
            torch.Tensor: Predicted actions.
        """
        # Sample random noisy action, used as the starting point for reverse diffusion
        generator = torch.Generator(device='cuda:0').manual_seed(42)

        noise = torch.randn( 
            size=(1, num_actions_chunk, action_dim),
            device='cuda:0',
            dtype=torch.bfloat16,
            generator=generator
        )  # (B, chunk_len, action_dim)

        # Set diffusion timestep values
        self.action_head.noise_scheduler.set_timesteps(self.action_head.num_diffusion_steps)

        # Reverse diffusion: Iteratively denoise to generate action, conditioned on observation
        curr_noisy_actions = noise
        list_ = []
        for t in self.action_head.noise_scheduler.timesteps:
            list_.append(curr_noisy_actions.cpu().float().numpy().tolist())
            # Get diffusion model's noise prediction (conditioned on VLA latent embedding, current noisy action embedding,
            # and diffusion timestep embedding)
            timesteps = torch.Tensor([t]).repeat(1).to('cuda:0')
            diffusion_timestep_embeddings = (
                self.action_head.time_encoder(timesteps).to(curr_noisy_actions.dtype).to(curr_noisy_actions.device)
            )  # (B, llm_dim)
            diffusion_timestep_embeddings = diffusion_timestep_embeddings.unsqueeze(1)  # (B, 1, llm_dim)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                curr_noisy_actions = curr_noisy_actions.reshape(1, -1).unsqueeze(-1)     # (B, chunk_len * action_dim, 1)
                curr_noisy_action_features = self.noisy_action_projector(curr_noisy_actions)  # (B, chunk_len * action_dim, llm_dim)
                # action token is a tensor of shape (B, action_num, vit_dim(1152 of siglip)) 
                # or (B, 1 + action_num, vit_dim) if use diffusion because add a time step token at the beginning
        
                _, action_token, _, _ = self.vat_backbone(
                    pixel_values.to(torch.bfloat16).to('cuda:0'), diffusion_timestep_embeddings, curr_noisy_action_features, proprio, task_id
                )
                noise_pred = self.action_head.predict_noise(action_token)

            # Compute the action at the previous diffusion timestep: x_t -> x_{t-1}
            # change the shape of curr_noisy_actions from (B, chunk_len * action_dim, 1) to (B, chunk_len, action_dim), same as noise_pred
            curr_noisy_actions = curr_noisy_actions.reshape(1, -1, action_dim)
            curr_noisy_actions = self.action_head.noise_scheduler.step(noise_pred, t, curr_noisy_actions).prev_sample


        return curr_noisy_actions.reshape(1, -1, action_dim)




    def predict_vit_action(
        self,
        unnorm_key: Optional[str] = None,
        pixel_values=None,
        proprio=None,
        task_id=None
    ) -> np.ndarray:
        """Predict actions from input sequence, with options for different prediction methods.

        Args:
            input_ids: Input token ids
            unnorm_key: Key for unnormalization statistics
            proprio: Proprioceptive features
            proprio_projector: Projector for proprioceptive features
            action_head: Optional head for L1 regression or diffusion-based prediction
            noisy_action_projector: Projector for noisy actions in diffusion-based prediction
            use_film: Whether to use FiLM conditioning
            **kwargs: Additional arguments including pixel_values and attention_mask

        Returns:
            Tuple of (unnormalized_actions, action_hidden_states)
        """
        # Use diffusion if provided, otherwise use regression or discrete prediction
        use_diffusion = self.noisy_action_projector is not None and hasattr(self.action_head, "noise_scheduler")

        if use_diffusion:
            # Run diffusion-based prediction
            predicted_actions = self.run_diffusion_sampling(pixel_values=pixel_values,
                                                            num_actions_chunk=self.cfg.action_chunk,
                                                            action_dim=self.cfg.action_dim_input,
                                                            proprio=proprio,
                                                            task_id=task_id)
            
        else:
            # Run regression or discrete token-based prediction
            with torch.autocast("cuda", dtype=torch.bfloat16):
      
                if not self.cfg.baseline:
                    _, action_token = self.vat_backbone(pixel_values.to(torch.bfloat16).to("cuda:0"), proprio=proprio, task_id=task_id)
                else:
                    # pixel values shape [B, 3*image_num, 224, 224]
                 
                    pixel_values = pixel_values.to(torch.bfloat16).to("cuda:0")
                    batch_size, channel_mul_image_num, h, w = pixel_values.shape
                    # channel per image is 3
                    image_num = channel_mul_image_num // 3 

            
                    pixel_values_reshaped = pixel_values.reshape(batch_size * image_num, 3, h, w)
                    raw_vision_features = self.vit(pixel_values_reshaped)

                    processed_features = []
                    for feature in raw_vision_features:
                        processed = feature.reshape(batch_size, image_num * feature.shape[1], -1)
                        processed_features.append(processed)
                    
                    _, action_token = self.vat(vision_features=processed_features, task_id=task_id)
                predicted_actions = self.action_head.predict_action(action_token, self.cfg.action_chunk)
        # Unnormalize predicted actions
        actions = self._unnormalize_actions(predicted_actions, unnorm_key)

        return actions



    def get_vit_action(
        self,
        obs: Dict[str, Any],
        task_id: int
    ) -> List[np.ndarray]:
        """
        Generate action predictions with the VLA policy.

        Args:
            cfg: Configuration object with parameters
            vla: The VLA model
            processor: Model processor for inputs
            obs: Observation dictionary
            task_label: Text description of the task
            action_head: Optional action head for continuous actions
            proprio_projector: Optional proprioception projector
            noisy_action_projector: Optional noisy action projector for diffusion
            use_film: Whether to use FiLM

        Returns:
            List[np.ndarray]: Predicted actions
        """
        with torch.inference_mode():
            
            # for real and libero simulation
            assert 'full_image' in obs, 'No image found in obs!'
            # get primary image from obs
            primary_image = obs['full_image']
            wrist_image = obs['wrist_image']
            if 'right_wrist_image' in obs:
                right_wrist_image = obs['right_wrist_image']
            else:
                right_wrist_image = None
            
            if not isinstance(primary_image, Image.Image):
                primary_image = Image.fromarray(primary_image)
            if not isinstance(wrist_image, Image.Image):
                wrist_image = Image.fromarray(wrist_image)
            if right_wrist_image is not None and not isinstance(right_wrist_image, Image.Image):
                right_wrist_image = Image.fromarray(right_wrist_image)
                
            # Process primary image
            pixel_values = self.image_transform(primary_image)
            pixel_values = pixel_values.unsqueeze(0)

            if self.cfg.use_wrist_image:
                wrist_pixel_values = self.image_transform(wrist_image)
                wrist_pixel_values = wrist_pixel_values.unsqueeze(0)
                if right_wrist_image is not None:
                    right_wrist_pixel_values = self.image_transform(right_wrist_image)
                    right_wrist_pixel_values = right_wrist_pixel_values.unsqueeze(0)
                if not self.cfg.only_use_wrist:
                    pixel_values = torch.cat([pixel_values, wrist_pixel_values], dim=1)
                    if right_wrist_image is not None:
                        pixel_values = torch.cat([pixel_values, right_wrist_pixel_values], dim=1)
                
                else:
                    raise ValueError('only_use_wrist is True, which is not allowed')
            
            proprio = None
            if self.proprio_projector is not None:
                proprio = obs["state"]
                proprio_norm_stats = self.norm_stats[self.unnorm_key]["proprio"]
                from experiments.robot.openvla_utils import normalize_proprio
                obs["state"] = normalize_proprio(proprio, proprio_norm_stats)
                proprio = obs["state"]
                # change numpy array to tensor and add dim 0
                proprio = torch.from_numpy(proprio).float().reshape(1, 1, -1).to('cuda:0')
                proprio = self.proprio_projector(proprio)

            task_id = torch.tensor([task_id]).reshape(1, -1).to('cuda:0')

            # Custom action head for continuous actions
            action = self.predict_vit_action(
                unnorm_key=self.unnorm_key,
                pixel_values=pixel_values,
                proprio=proprio,
                task_id=task_id
            )

        # Extract subset of actions for open loop steps
        return [action[i] for i in range(min(len(action), self.cfg.action_chunk))]

    # def preprocess_image(self, obs: Dict[str, Any]) -> Dict[str, Any]:
    #     """
    #     Preprocess the image in the observation.

    #     Args:
    #         obs: Observation dictionary

    #     Returns:
    #         Dict[str, Any]: Preprocessed observation
    #     """
    #     # Resize image
    #     img_resized = resize_image_for_policy(np.array(obs['full_image']), (224,224))
    #     wrist_img_resized = resize_image_for_policy(np.array(obs['wrist_image']), (224,224))
    #     if 'right_wrist_image' in obs:
    #         right_wrist_img_resized = resize_image_for_policy(np.array(obs['right_wrist_image']), (224,224))
    #     # Apply center crop if specified
        
    #     img_resized = center_crop_image(img_resized)
    #     wrist_img_resized = center_crop_image(wrist_img_resized)
    #     if 'right_wrist_image' in obs:
    #         right_wrist_img_resized = center_crop_image(right_wrist_img_resized)

    #     obs['full_image'] = img_resized
    #     obs['wrist_image'] = wrist_img_resized
    #     if 'right_wrist_image' in obs:
    #         obs['right_wrist_image'] = right_wrist_img_resized

    #     return obs







