# import packages and module here
from evaluation.vat_model import VATModel
import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union
import numpy as np
from experiments.robot.openvla_utils import (
    resize_image_for_policy,
    center_crop_image
)

@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    pretrained_checkpoint: Union[str, Path] = None          # Pretrained checkpoint path

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps: int = 50                    # (When `diffusion==True`) Number of diffusion steps for inference
    num_images_in_input: int = 1                   # Number of images in the VLA input (default: 1)                      # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 25                     # Number of actions to execute open-loop before requerying policy

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on
    task_suite_name: str = ""
    unnorm_key: str = ""
    action_dim_input: int = 14
    action_chunk: int = 25
    use_proprio: bool = False
    use_wrist_image: bool = True
    mix_dataset: bool = False
    vit_large: bool = False
    dino: bool = False
    only_use_wrist: bool = False
    end_lastlayer: int = 2
    save_attention: bool = False
    attention_save_path: str = None
    use_film: bool = True                           # If True, uses FiLM to infuse language inputs into visual features
    taskembedding_add: bool = False
    
def encode_obs(observation):  # Post-Process Observation
    """Prepare observation for policy input."""
    # Get preprocessed images
    full_image = observation["observation"]["head_camera"]["rgb"]
    left_wrist_image = observation["observation"]["left_camera"]["rgb"]
    right_wrist_image = observation["observation"]["right_camera"]["rgb"]
    state = observation["joint_action"]["vector"]
    # instruction = observation["language"]
        

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(full_image, 224)
    left_wrist_img_resized = resize_image_for_policy(left_wrist_image, 224)
    right_wrist_img_resized = resize_image_for_policy(right_wrist_image, 224)
    
    
    img_resized = center_crop_image(img_resized)
    left_wrist_img_resized = center_crop_image(left_wrist_img_resized)
    right_wrist_img_resized = center_crop_image(right_wrist_img_resized)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "wrist_image": left_wrist_img_resized,
        "right_wrist_image": right_wrist_img_resized,
        "state": state,
        # "instruction": instruction,
    }

    return observation
  


def get_model(usr_args):  # from deploy_policy.yml and eval.sh (overrides)
    cfg = GenerateConfig(
        pretrained_checkpoint=usr_args['ckpt_setting'],
        unnorm_key=usr_args['unnorm_key'],
    )
    model = VATModel(cfg)
    # ...
    return model  # return your policy model

def unnormalize_actions(action, model):
    """Unnormalize actions using dataset statistics"""
    action_norm_stats = model.norm_stats[model.unnorm_key]["action"]


    #  ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
    action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])


    action = np.where(
        mask,
        0.5 * (action + 1) * (action_high - action_low + 1e-8) + action_low,
        action,
    )

    return action

def eval(TASK_ENV, model, observation):
    """
    All the function interfaces below are just examples
    You can modify them according to your implementation
    But we strongly recommend keeping the code logic unchanged
    """
    obs = encode_obs(observation)  # Post-Process Observation
    instruction = TASK_ENV.get_instruction()

    actions = model.get_vit_action(obs, task_id=0)
    actions = actions[0]

    for action in actions:  # Execute each step of the action
        # see for https://robotwin-platform.github.io/doc/control-robot.md more details
        # action = unnormalize_actions(action, model)
        # check the default action type, openvlaoft use the default type
        TASK_ENV.take_action(action, action_type='qpos') # joint control: [left_arm_joints + left_gripper + right_arm_joints + right_gripper]
        # TASK_ENV.take_action(action, action_type='ee') # endpose control: [left_end_effector_pose (xyz + quaternion) + left_gripper + right_end_effector_pose + right_gripper]
        # TASK_ENV.take_action(action, action_type='delta_ee') # delta endpose control: [left_end_effector_delta (xyz + quaternion) + left_gripper + right_end_effector_delta + right_gripper]
        


def reset_model(model):  
    # Clean the model cache at the beginning of every evaluation episode, such as the observation window
    pass
