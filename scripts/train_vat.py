"""
finetune.py

Fine-tunes OpenVLA via LoRA.
"""

import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union

import draccus
import torch
import torch.distributed as dist
import torch.nn as nn
import tqdm
from accelerate import PartialState
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
from torch.utils.data import DataLoader
import json
import math
import wandb
import numpy as np
import random
import re

from prismatic.models.action_heads import DiffusionActionHead, L1RegressionActionHead
from prismatic.models.projectors import (
    NoisyActionProjector,
    ProprioProjector,
)

from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.constants import (
    PROPRIO_DIM,
)
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.models import get_actionvision_backbone_and_transform

import sys


import numpy as np
# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class FinetuneConfig:
    
    # Dataset
    data_root_dir: Path = Path("datasets/rlds")      # Directory containing RLDS datasets
    dataset_name: str = "aloha_scoop_x_into_bowl"    # Name of fine-tuning dataset (e.g., `aloha_scoop_x_into_bowl`)
    run_root_dir: Path = Path("runs")                # Path to directory to store logs & checkpoints
    shuffle_buffer_size: int = 100_000               # Dataloader shuffle buffer size (can reduce if OOM errors occur)

    # Algorithm and architecture
    use_l1_regression: bool = True                   # If True, trains continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, trains continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps: int = 50                    # (When `diffusion==True`) Number of diffusion steps for training
    use_proprio: bool = False                        # If True, includes robot proprioceptive state in input

    # Training configuration
    batch_size: int = 8                              # Batch size per device (total batch size = batch_size * num GPUs)
    learning_rate: float = 5e-4                      # Learning rate
    lr_warmup_steps: int = 0                         # Number of steps to warm up learning rate (from 10% to 100%)
    num_steps_before_decay: int = 100_000            # Number of steps before LR decays by 10x
    grad_accumulation_steps: int = 1                 # Number of gradient accumulation steps
    max_steps: int = 200_000                         # Max number of training steps
    use_val_set: bool = False                        # If True, uses validation set and log validation metrics
    val_freq: int = 10_000                           # (When `use_val_set==True`) Validation set logging frequency in steps
    val_time_limit: int = 180                        # (When `use_val_set==True`) Time limit for computing validation metrics
    save_freq: int = 10_000                          # Checkpoint saving frequency in steps
    save_latest_checkpoint_only: bool = False        # If True, saves only 1 checkpoint, overwriting latest checkpoint
                                                     #   (If False, saves all checkpoints)
    resume: bool = False                             # If True, resumes from checkpoint
    resume_step: Optional[int] = None                # (When `resume==True`) Step number that we are resuming from
    image_aug: bool = True                           # If True, trains with image augmentations (HIGHLY RECOMMENDED)
    diffusion_sample_freq: int = 50                  # (When `use_diffusion==True`) Frequency for sampling in steps


    # Logging
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    run_id_override: Optional[str] = None            # Optional string to override the run ID with
    wandb_log_freq: int = 10                         # WandB logging frequency in steps
    # fmt: on

    action_dim_input: int = 1
    action_chunk: int = 1
    visionbackbone_requiregrad: bool = False

    vit_weight_path: Optional[str] = None

    vit_trainable_layer: Union[List[int], None] = None
    
    epochs: Optional[int] = None
    save_freq_epochs: Optional[int] = None

    only_train_actionmodule: bool = False
    use_cosinelr: bool = False
 
    use_wrist_image: bool = False
 
    no_training: bool = False
    vit_large: bool = False
    dino: bool = False
    pretrained_checkpoint: Optional[str] = None
 
    only_use_wrist: bool = False
    end_lastlayer: int = 2
    num_images_in_input: int = 3                     # Number of images in the VLA input (default: 1)
    use_film: bool = True                           # If True, uses FiLM to infuse task embedding into visual features
    taskembedding_add: bool = False
    baseline: bool = False
    vat_small_factor: int = 1
    vat_vit: bool = False
    
def remove_ddp_in_checkpoint(state_dict) -> dict:
    """
    Removes the 'module.' prefix from parameter names in a PyTorch model state dictionary that was saved using
    DistributedDataParallel (DDP).

    When a model is trained using PyTorch's DistributedDataParallel, the saved state dictionary contains parameters
    prefixed with 'module.'. This function removes these prefixes to make the state dictionary compatible when
    loading into models that are not yet wrapped in DDP.

    Args:
        state_dict (dict): PyTorch model state dictionary.

    Returns:
        dict: A new state dictionary with the same contents but with 'module.' prefixes removed from parameter names.
              Parameters without the 'module.' prefix remain unchanged.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k[:7] == "module.":
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def get_run_id(cfg) -> str:
    """
    Generates or retrieves an identifier string for an experiment run.

    Args:
        cfg (FinetuneConfig): Training configuration.

    Returns:
        str: Experiment run ID.
    """
    if cfg.run_id_override is not None:
        # Override the run ID with the user-provided ID
        run_id = cfg.run_id_override
   
    else:
        run_id = (
            f"{cfg.dataset_name}"
            f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
            f"+lr-{cfg.learning_rate}"
        )
        if cfg.image_aug:
            run_id += "--image_aug"
        if cfg.run_id_note is not None:
            run_id += f"--{cfg.run_id_note}"
    return run_id


def load_checkpoint(module_name: str, path: str, step: int, device: str = "cpu") -> dict:
    """
    Loads a checkpoint for a given module.

    Args:
        module_name (str): Name of model component to load checkpoint for.
        path (str): Path to checkpoint directory.
        step (int): Gradient step number of saved checkpoint.
        device (str): String specifying how to remap storage locations (default = "cpu").

    Returns:
        dict: PyTorch model state dictionary.
    """
    checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    return remove_ddp_in_checkpoint(state_dict)


def wrap_ddp(module: nn.Module, device_id: int, find_unused: bool = False) -> DDP:
    """
    Wrap a module with DistributedDataParallel.

    Args:
        module (nn.Module): PyTorch module.
        device_id (str): Device ID.
        find_unused (bool): Whether to detect parameters without gradients in distributed training.

    Returns:
        DistributedDataParallel: PyTorch module wrapped with DDP.
    """
    return DDP(module, device_ids=[device_id], find_unused_parameters=find_unused, gradient_as_bucket_view=True)





def count_parameters(module: nn.Module, name: str) -> None:
    """
    Counts and prints the number of trainable parameters in a module.

    Args:
        module (nn.Module): PyTorch module.
        module_name (str): Name of model component.

    Returns:
        None.
    """
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"# trainable params in {name}: {num_params}")


def init_module(
    module_class: Type[nn.Module],
    module_name: str,
    cfg: FinetuneConfig,
    device_id: int,
    module_args: dict,
    to_bf16: bool = False,
    find_unused_params: bool = False,
) -> DDP:
    """
    Initializes a module, optionally loads checkpoint, moves to device, and wraps with DDP.

    Args:
        module_class (Type[nn.Module]): Class of PyTorch module to initialize.
        module_name (str): Name of model component to load checkpoint for.
        cfg (FinetuneConfig): Training configuration.
        device_id (str): Device ID.
        module_args (dict): Args for initializing the module.
        to_bf16 (bool): Whether to convert to torch.bfloat16 data type.
        find_unused_params (bool): Whether to detect parameters without gradients in distributed training.

    Returns:
        DistributedDataParallel: PyTorch module wrapped with DDP.
    """

    
    module = module_class(**module_args)
    count_parameters(module, module_name)

    # if cfg.resume:
    #     state_dict = load_checkpoint(module_name, cfg.pretrained_checkpoint, cfg.resume_step)
    #     module.load_state_dict(state_dict)

    if to_bf16:
        module = module.to(torch.bfloat16)
    module = module.to(device_id)

    return wrap_ddp(module, device_id, find_unused_params)


def run_forward_pass(
    cfg,
    vision_backbone,
    action_head,
    noisy_action_projector,
    proprio_projector,
    batch,
    device_id,
    use_l1_regression,
    use_diffusion,
    use_proprio,
    num_patches,
    compute_diffusion_l1=False,
    num_diffusion_steps=None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute model forward pass and metrics for both training and validation.
    Returns:
        tuple: (loss, metrics_dict)
            loss: The loss tensor with gradient for backpropagation.
            metrics_dict: Dictionary of computed metrics (detached values for logging).
    """
    metrics = {}

    # Get ground-truth action labels
    ground_truth_actions = batch["actions"].to(device_id).to(torch.bfloat16)

    # [Only for diffusion] Sample noisy actions used as input for noise predictor network
    if use_diffusion:
        noisy_dict = action_head.module.sample_noisy_actions(ground_truth_actions)
        noise, noisy_actions, diffusion_timestep_embeddings = (
            noisy_dict["noise"],
            noisy_dict["noisy_actions"],
            noisy_dict["diffusion_timestep_embeddings"],
        )
        # Reshape noisy actions into individual action tokens
        # noisy_actions: (B, chunk_len, action_dim) -> (B, chunk_len * action_dim, 1)
        B = noisy_actions.shape[0]
        noisy_actions = noisy_actions.reshape(B, -1).unsqueeze(-1)

        # Project noisy action tokens into language model embedding space
        with torch.autocast("cuda", dtype=torch.bfloat16):
            noisy_action_features = noisy_action_projector(noisy_actions)  # (B, chunk_len * action_dim, llm_dim)
    else:
        noise, noisy_actions, diffusion_timestep_embeddings, noisy_action_features = None, None, None, None

    # Get proprioceptive state and task id, note that collator already set None if not use proprio or task id, here set None explicitly for clarity 
    if use_proprio:
        proprio = batch["proprio"]
        proprio = proprio_projector(proprio)
    else:
        proprio = None
    
 
    task_id = batch['task_id'].to(device_id)
   

    # VLA forward pass
    with torch.autocast("cuda", dtype=torch.bfloat16):
        # action token is a tensor of shape (B, action_num, vit_dim(1152 of siglip)) 
        # or (B, 1 + action_num, vit_dim) if use diffusion because add a time step token at the beginning
        
        # torch.save(task_id, "task_id.pt")
        # batch["pixel_values"] = torch.load("dualpv.pt")
        # task_id = torch.load("dualtask.pt")
        _, action_token = vision_backbone(
            batch["pixel_values"].to(torch.bfloat16).to(device_id), diffusion_timestep_embeddings, noisy_action_features, proprio, task_id
        )

    # Compute metrics for discrete action representation (next-token prediction)
    if not (use_l1_regression or use_diffusion):
        pass
    # Compute metrics for continuous action representations (L1 regression | diffusion)
    else:
        if use_l1_regression:
            # Predict action
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predicted_actions = action_head.module.predict_action(action_token)

            # Get full L1 loss
            loss = torch.nn.L1Loss()(ground_truth_actions, predicted_actions)
     

        if use_diffusion:
            # Predict noise
            with torch.autocast("cuda", dtype=torch.bfloat16):
                noise_pred = action_head.module.predict_noise(action_token)
            # Get diffusion noise prediction MSE loss
            noise_pred = noise_pred.reshape(noise.shape)
            loss = nn.functional.mse_loss(noise_pred, noise, reduction="mean")

            # Only sample actions and compute L1 losses if specified
            batch_size = batch["actions"].shape[0]
            if compute_diffusion_l1:
                with torch.no_grad():
                    predicted_actions = run_diffusion_sampling(
                        cfg=cfg,
                        vision_backbone=vision_backbone,
                        action_head=action_head,
                        noisy_action_projector=noisy_action_projector,
                        proprio_projector=proprio_projector,
                        batch=batch,
                        batch_size=batch_size,
                        num_patches=num_patches,
                        actions_shape=ground_truth_actions.shape,
                        device_id=device_id,
                        proprio=proprio,
                        task_id=task_id
                    )
           

        metrics.update(
            {
                "loss_value": loss.item(),  # Detached value for logging
            }
        )
        

        # Get detailed L1 losses for logging
        should_log_l1_loss = not use_diffusion or (use_diffusion and compute_diffusion_l1)
        if should_log_l1_loss:
            ground_truth_curr_action = ground_truth_actions[:, 0]
            predicted_curr_action = predicted_actions[:, 0]
            ground_truth_next_actions = ground_truth_actions[:, 1:]
            predicted_next_actions = predicted_actions[:, 1:]
            curr_action_l1_loss = torch.nn.L1Loss()(ground_truth_curr_action, predicted_curr_action)
            next_actions_l1_loss = torch.nn.L1Loss()(ground_truth_next_actions, predicted_next_actions)
            metrics.update(
                {
                    "curr_action_l1_loss": curr_action_l1_loss.item(),
                    "next_actions_l1_loss": next_actions_l1_loss.item(),
                }
            )

            all_ground_truth = ground_truth_actions
            all_predicted = predicted_actions
            
            absolute_errors = torch.abs(all_ground_truth - all_predicted)
            per_dim_l1_loss = absolute_errors.mean(dim=[0, 1])
            
            for dim in range(per_dim_l1_loss.shape[0]):
                metrics.update({f"dim{dim}_l1_loss": per_dim_l1_loss[dim].item()})

    # Return both the loss tensor (with gradients) and the metrics dictionary (with detached values)
    return loss, metrics


def run_diffusion_sampling(
    cfg,
    vision_backbone,
    action_head,
    noisy_action_projector,
    proprio_projector,
    batch,
    batch_size,
    num_patches,
    actions_shape,
    device_id,
    proprio,
    task_id
) -> torch.Tensor:
    """
    Run diffusion sampling (reverse diffusion) to generate actions.

    Returns:
        torch.Tensor: Predicted actions.
    """
    # Sample random noisy action, used as the starting point for reverse diffusion
    generator = torch.Generator(device=device_id).manual_seed(42)

    noise = torch.randn(
        size=(batch_size, cfg.action_chunk, cfg.action_dim_input),
        device=device_id,
        dtype=torch.bfloat16,
        generator=generator,
    )  # (B, chunk_len, action_dim)

    # Set diffusion timestep values
    action_head.module.noise_scheduler.set_timesteps(action_head.module.num_diffusion_steps)

    # Reverse diffusion: Iteratively denoise to generate action, conditioned on observation
    curr_noisy_actions = noise
    list_ = []
    for t in action_head.module.noise_scheduler.timesteps:
        list_.append(curr_noisy_actions.cpu().float().numpy().tolist())
        # Get diffusion model's noise prediction (conditioned on VLA latent embedding, current noisy action embedding,
        # and diffusion timestep embedding)
        timesteps = torch.Tensor([t]).repeat(batch_size).to(device_id)
        diffusion_timestep_embeddings = (
            action_head.module.time_encoder(timesteps).to(curr_noisy_actions.dtype).to(curr_noisy_actions.device)
        )  # (B, llm_dim)
        diffusion_timestep_embeddings = diffusion_timestep_embeddings.unsqueeze(1)  # (B, 1, llm_dim)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            B = curr_noisy_actions.shape[0]
            curr_noisy_actions = curr_noisy_actions.reshape(B, -1).unsqueeze(-1)     # (B, chunk_len * action_dim, 1)
            curr_noisy_action_features = noisy_action_projector(curr_noisy_actions)  # (B, chunk_len * action_dim, llm_dim)
            # action token is a tensor of shape (B, action_num, vit_dim(1152 of siglip)) 
            # or (B, 1 + action_num, vit_dim) if use diffusion because add a time step token at the beginning
    
            _, action_token = vision_backbone(
                batch["pixel_values"].to(torch.bfloat16).to(device_id), diffusion_timestep_embeddings, curr_noisy_action_features, proprio, task_id
            )

            noise_pred = action_head.module.predict_noise(action_token)

        # Compute the action at the previous diffusion timestep: x_t -> x_{t-1}
        # change the shape of curr_noisy_actions from (B, chunk_len * action_dim, 1) to (B, chunk_len, action_dim), same as noise_pred
        curr_noisy_actions = curr_noisy_actions.reshape(B, -1, cfg.action_dim)
        curr_noisy_actions = action_head.module.noise_scheduler.step(noise_pred, t, curr_noisy_actions).prev_sample
    

    return curr_noisy_actions.reshape(actions_shape)


def compute_smoothened_metrics(metrics_deques) -> dict:
    """
    Compute smoothened metrics from recent deques.

    Args:
        metrics_deques (dict): Dictionary of deques containing recent metrics.

    Returns:
        dict: Dictionary of smoothened metrics.
    """
    smoothened_metrics = {}
    for name, deque in metrics_deques.items():
        if deque and len(deque) > 0:
            smoothened_metrics[name] = sum(deque) / len(deque)
    return smoothened_metrics


def log_metrics_to_wandb(metrics, prefix, step, wandb_entity) -> None:
    """
    Log metrics to Weights & Biases.

    Args:
        metrics (dict): Dictionary of metrics to log
        prefix (str): Prefix for metric names
        step (int): Training step
        wandb_entity (str): W&B entity instance

    Returns:
        None.
    """
    log_dict = {}
    for name, value in metrics.items():
        # Map loss_value to Loss for better readability in W&B
        if name == "loss_value":
            log_dict[f"{prefix}/Loss"] = value
        # Keep other metrics as is
        else:
            log_dict[f"{prefix}/{name.replace('_', ' ').title()}"] = value
    wandb_entity.log(log_dict, step=step)


def save_training_checkpoint(
    cfg,
    run_dir,
    log_step,
    vision_backbone,
    proprio_projector,
    noisy_action_projector,
    action_head,
    train_dataset,
    distributed_state,
) -> None:
    """
    Save all training checkpoints including model components, LoRA adapter, and dataset statistics.

    Returns:
        None.
    """
    # Determine checkpoint paths and naming
    if cfg.save_latest_checkpoint_only:
        checkpoint_dir = run_dir
        checkpoint_name_suffix = "latest_checkpoint.pt"
    else:
        checkpoint_dir = Path(str(run_dir) + f"--{log_step}_ckpt")
        checkpoint_name_suffix = f"{log_step}_checkpoint.pt"


    # Create directories and save dataset statistics (main process only)
    if distributed_state.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        save_dataset_statistics(train_dataset.dataset_statistics, checkpoint_dir)
        print(f"Saving Model Checkpoint for Step {log_step}")

    # Wait for directories to be created
    dist.barrier()

    # Save model components (main process only)
    if distributed_state.is_main_process:
        # Save LoRA adapter
        
        
        if cfg.baseline:
            torch.save(vision_backbone.module.vit.state_dict(), checkpoint_dir / f"vit--{checkpoint_name_suffix}")
            torch.save(vision_backbone.module.featurizer.state_dict(), checkpoint_dir / f"vat--{checkpoint_name_suffix}")
        else:
            torch.save(vision_backbone.module.featurizer.state_dict(), checkpoint_dir / f"vision_backbone--{checkpoint_name_suffix}")
        # Save other components
        if cfg.use_proprio and proprio_projector is not None:
            torch.save(proprio_projector.module.state_dict(), checkpoint_dir / f"proprio_projector--{checkpoint_name_suffix}")

        if cfg.use_diffusion and noisy_action_projector is not None:
            torch.save(
                noisy_action_projector.module.state_dict(), checkpoint_dir / f"noisy_action_projector--{checkpoint_name_suffix}"
            )

        if (cfg.use_l1_regression or cfg.use_diffusion) and action_head is not None:
            torch.save(action_head.module.state_dict(), checkpoint_dir / f"action_head--{checkpoint_name_suffix}")

    # Wait for model components to be saved
    dist.barrier()


def run_validation(
    vision_backbone,
    action_head,
    noisy_action_projector,
    proprio_projector,
    val_dataloader,
    device_id,
    cfg,
    num_patches,
    log_step,
    distributed_state,
    val_time_limit,
) -> None:
    """
    Compute validation set metrics for logging.

    Returns:
        None.
    """
    val_start_time = time.time()
    vision_backbone.eval()
    val_batches_count = 0

    # List to store validation metrics
    all_val_metrics = []

    with torch.no_grad():
        for batch in val_dataloader:
            # Always compute L1 loss for validation, even for diffusion
            _, metrics = run_forward_pass(
                cfg,
                vision_backbone=vision_backbone,
                action_head=action_head,
                noisy_action_projector=noisy_action_projector,
                proprio_projector=proprio_projector,
                batch=batch,
                device_id=device_id,
                use_l1_regression=cfg.use_l1_regression,
                use_diffusion=cfg.use_diffusion,
                use_proprio=cfg.use_proprio,
                num_patches=num_patches,
                compute_diffusion_l1=True,
                num_diffusion_steps=cfg.num_diffusion_steps if cfg.use_diffusion else None,
                use_task_id=cfg.use_task_id,
            )

            # Add the loss value to the metrics
            metrics["loss"] = metrics["loss_value"]
            all_val_metrics.append(metrics)
            val_batches_count += 1

            # Cut testing on validation set short if it exceeds time limit
            if time.time() - val_start_time > val_time_limit:
                break

    # Compute average validation metrics
    avg_val_metrics = {}
    for metric_name in all_val_metrics[0].keys():
        values = [metrics[metric_name] for metrics in all_val_metrics if metric_name in metrics]
        if values:
            avg_val_metrics[metric_name] = sum(values) / len(values)

    # Add batch count to metrics
    avg_val_metrics["val_batches_count"] = val_batches_count

    # Log validation metrics to W&B
    if distributed_state.is_main_process:
        log_metrics_to_wandb(avg_val_metrics, "VLA Val", log_step, wandb)

def set_blocks_requires_grad(featurizer, trainable_indices):
    featurizer.requires_grad_(False)
    featurizer.action_token.requires_grad_(True)
    featurizer.action_pos_embed.requires_grad_(True)
    # featurizer.
    for i, block in enumerate(featurizer.blocks):
        if i in trainable_indices or i - len(featurizer.blocks) in trainable_indices:
            block.requires_grad_(True)

def set_blocks_requires_grad_for_actionmodule(featurizer):
    featurizer.requires_grad_(False)
    # set params out of blocks to True first
    featurizer.action_token.requires_grad_(True)
    # featurizer.action_pos_embed.requires_grad_(True)
    featurizer.action_pos_embed_generator.requires_grad_(True)
    featurizer.task_embedding.requires_grad_(True)
    for i, block in enumerate(featurizer.blocks):
        for name, param in block.named_parameters():
            base_condition = 'q_act' in name or 'kv_act' in name or 'proj_act' in name or 'norm_act' in name
            
            plus_condition = base_condition or ('mlp_act' in name) or ('norm1_act' in name) or ('norm2_act' in name)
            
            if plus_condition:
                param.requires_grad_(True)

    def count_parameters_by_grad(model):
        """统计模型中 requires_grad=True/False 的参数量"""
        total_params = 0
        trainable_params = 0
        frozen_params = 0
        
        for param in model.parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
            else:
                frozen_params += param.numel()
        
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "frozen_params": frozen_params,
            "trainable_ratio": trainable_params / total_params if total_params > 0 else 0.0
        }

    stats = count_parameters_by_grad(featurizer)
    print(f' #################  only train fusion params  #################')
    print(f"Parameter Summary:")
    print(f"  Total params: {stats['total_params']:,}")
    print(f"  Trainable params: {stats['trainable_params']:,} ({stats['trainable_ratio']:.1%})")
    print(f"  Frozen params: {stats['frozen_params']:,}")
    
def set_seed(seed=42):
    torch.manual_seed(seed)
    
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
def find_checkpoint_file(pretrained_checkpoint: str, file_pattern: str) -> str:
    """
    Find a specific checkpoint file matching a pattern.

    Args:
        pretrained_checkpoint: Path to the checkpoint directory
        file_pattern: String pattern to match in filenames

    Returns:
        str: Path to the matching checkpoint file

    Raises:
        AssertionError: If no files or multiple files match the pattern
    """
    assert os.path.isdir(pretrained_checkpoint), f"Checkpoint path must be a directory: {pretrained_checkpoint}"

    checkpoint_files = []
    for filename in os.listdir(pretrained_checkpoint):
        if file_pattern in filename and "checkpoint" in filename:
            full_path = os.path.join(pretrained_checkpoint, filename)
            checkpoint_files.append(full_path)

    assert len(checkpoint_files) == 1, (
        f"Expected exactly 1 {file_pattern} checkpoint but found {len(checkpoint_files)} in directory: {pretrained_checkpoint}"
    )

    return checkpoint_files[0]

def load_component_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """
    Load a component's state dict from checkpoint and handle DDP prefix if present.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Dict: The processed state dictionary for loading
    """
    state_dict = torch.load(checkpoint_path, weights_only=True)

    # If the component was trained with DDP, elements in the state dict have prefix "module." which we must remove
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    return new_state_dict

@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    """
    Fine-tunes base VLA on demonstration dataset via LoRA.

    Allows toggling different action representations (discrete vs. continuous), different learning objectives
    (next-token prediction vs. L1 regression vs. diffusion), FiLM. Also allows for additional model inputs,
    such as additional camera images and robot proprioceptive state. Assumes parallel action generation with
    action chunking.

    Args:
        cfg (FinetuneConfig): Training configuration.

    Returns:
        None.
    """
    set_seed(42)
   
    assert not (cfg.use_l1_regression and cfg.use_diffusion), (
        "Cannot do both L1 regression and diffusion. Please pick one of them!"
    )


    # Get experiment run ID
    run_id = get_run_id(cfg)

    # Create experiment run directory
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)

    # save cfg information to run_root_dir as json file
    import yaml
    from dataclasses import asdict

    def save_to_yaml(cfg, file_path: Path):
        cfg_dict = asdict(cfg)
        with open(file_path, "w") as f:
            yaml.dump(cfg_dict, f, default_flow_style=False)

    save_to_yaml(cfg, Path(cfg.run_root_dir / "config.yaml"))

    # GPU setup
    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    # Initialize wandb logging
    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{run_id}")

    if cfg.vit_large:
        vit_name =  "visionaction-vit-giantopt-patch16-siglip-256"   
    elif cfg.dino: 
        vit_name =  "visionaction-dinov2"
    else:
        vit_name = "visionaction-siglip-vit-so400m"
        
    vision_backbone, image_transform = get_actionvision_backbone_and_transform(
        vit_name, image_resize_strategy="resize-naive", 
        action_dim=cfg.action_dim_input, action_chunk=cfg.action_chunk, use_diffusion=cfg.use_diffusion,
        vit_weight_path=cfg.vit_weight_path,
        use_proprio=cfg.use_proprio,
        end_lastlayer=cfg.end_lastlayer,
        use_film=cfg.use_film,
        taskembedding_add=cfg.taskembedding_add,
        baseline=cfg.baseline,
        vat_small_factor=cfg.vat_small_factor,
        vat_vit=cfg.vat_vit,
    )   
    vision_backbone = vision_backbone.to(device_id)
    if cfg.vit_trainable_layer != None:
        set_blocks_requires_grad(vision_backbone.featurizer, cfg.vit_trainable_layer)
    elif cfg.only_train_actionmodule:
        set_blocks_requires_grad_for_actionmodule(vision_backbone.featurizer)
    else:
        print(f'##########  train all params  ##########')
        vision_backbone.requires_grad_(cfg.visionbackbone_requiregrad)
    

    # Wrap VLA with DDP
    
    vision_backbone = wrap_ddp(vision_backbone, device_id, find_unused=True)
   

    # If applicable, instantiate proprio projector
    if cfg.use_proprio:
        proprio_projector = init_module(
            ProprioProjector,
            "proprio_projector",
            cfg,
            device_id,
            {"llm_dim": 1152, "proprio_dim": PROPRIO_DIM},
        )

    # If applicable, instantiate continuous action head for L1 regression
    if cfg.use_l1_regression:
        action_head = init_module(
            L1RegressionActionHead,
            "action_head",
            cfg,
            device_id,
            {"input_dim": vision_backbone.module.embed_dim, "hidden_dim": vision_backbone.module.embed_dim, "action_dim": cfg.action_dim_input},
            to_bf16=False,
        
        )

    # If applicable, instantiate diffusion action head and noisy action projector
    if cfg.use_diffusion:
        action_head = init_module(
            DiffusionActionHead,
            "action_head",
            cfg,
            device_id,
            {
                "input_dim": vision_backbone.module.embed_dim,
                "hidden_dim": vision_backbone.module.embed_dim,
                "action_dim": cfg.action_dim_input,
                "num_diffusion_steps": cfg.num_diffusion_steps,
            },
            to_bf16=True,
        )
        noisy_action_projector = init_module(
            NoisyActionProjector, "noisy_action_projector", cfg, device_id, {"llm_dim": vision_backbone.module.embed_dim}
        )

    if cfg.resume:
        match = re.search(r'--(\d+)_ckpt$', cfg.pretrained_checkpoint)
        step_num = match.group(1)
        state_dict = torch.load(os.path.join(cfg.pretrained_checkpoint, f"vision_backbone--{step_num}_checkpoint.pt"))
        state_dict = {k.replace("module.featurizer.", ""): v for k, v in state_dict.items()}
        # set strict False for skipping task embedding missing
        vision_backbone.module.featurizer.load_state_dict(state_dict, strict=True)

        action_head_checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, "action_head")
        state_dict = load_component_state_dict(action_head_checkpoint_path)
        action_head.module.load_state_dict(state_dict)

        if cfg.use_proprio:
            proprio_projector_checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, "proprio_projector")
            state_dict = load_component_state_dict(proprio_projector_checkpoint_path)
            proprio_projector.module.load_state_dict(state_dict)


    # Get number of vision patches
    # NUM_PATCHES = vision_backbone.module.featurizer.patch_embed.num_patches * cfg.num_images_in_input
    NUM_PATCHES = 256  * cfg.num_images_in_input
    # If we have proprio inputs, a single proprio embedding is appended to the end of the vision patch embeddings
    if cfg.use_proprio:
        NUM_PATCHES += 1
    # For diffusion, a single diffusion timestep embedding is appended to the end of the vision patch embeddings
    if cfg.use_diffusion:
        NUM_PATCHES += 1

    # Instantiate optimizer
    trainable_params = [param for param in vision_backbone.parameters() if param.requires_grad]
    if cfg.use_l1_regression or cfg.use_diffusion:
        trainable_params += [param for param in action_head.parameters() if param.requires_grad]
    if cfg.use_diffusion:
        trainable_params += [param for param in noisy_action_projector.parameters() if param.requires_grad]
    if cfg.use_proprio:
        trainable_params += [param for param in proprio_projector.parameters() if param.requires_grad]
    print(f"# total trainable params: {sum(p.numel() for p in trainable_params)}")
    
    if cfg.no_training:
        # add a random tensor to trainable_params
        trainable_params = [torch.randn(1, requires_grad=True)]

    # We assume that the model takes as input one third-person camera image and 1 or 2 optional wrist camera image(s)
    use_wrist_image = cfg.use_wrist_image

    # Create training and optional validation datasets
    batch_transform = RLDSBatchTransform(
        image_transform=image_transform,
        use_wrist_image=use_wrist_image,
        use_proprio=cfg.use_proprio,
        task_dict_name=cfg.dataset_name
    )
    train_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple([224, 224]),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )
    if cfg.use_val_set:
        val_dataset = RLDSDataset(
            cfg.data_root_dir,
            cfg.dataset_name,
            batch_transform,
            resize_resolution=tuple([224, 224]),
            shuffle_buffer_size=cfg.shuffle_buffer_size // 10,
            image_aug=cfg.image_aug,
            train=False,
        )

    # [Important] Save dataset statistics so that we can unnormalize actions during inference
    if distributed_state.is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)

    # Create collator and dataloader
    collator = PaddedCollatorForActionPrediction(
         only_use_wrist=cfg.only_use_wrist
    )

    local_batch_size = cfg.batch_size // torch.distributed.get_world_size()
    
    dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important: Set to 0 if using RLDS, which uses its own parallelism
    )
    if cfg.use_val_set:
        val_batch_size = local_batch_size
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,  # Important: Set to 0 if using RLDS, which uses its own parallelism
        )

        # replace max_steps with epochs  
    if cfg.epochs != None:
        cfg.max_steps = cfg.epochs * len(dataloader) // torch.distributed.get_world_size()
    if cfg.save_freq_epochs!= None:
        cfg.save_freq = cfg.save_freq_epochs * len(dataloader) // torch.distributed.get_world_size()

    if cfg.use_cosinelr:
        min_lr = 1e-7
        max_lr = cfg.learning_rate
        optimizer = AdamW(trainable_params, lr=max_lr)
        # Record original learning rate
        num_training_steps = cfg.max_steps
        num_warmup_steps = int(num_training_steps * 0.1)

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return current_step / max(1, num_warmup_steps)
            
            progress = (current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
            decay_factor = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
            
            return min_lr/max_lr + (1 - min_lr/max_lr) * decay_factor

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

        # Record original learning rate
        original_lr = optimizer.param_groups[0]["lr"]
    # Create learning rate scheduler
        scheduler = MultiStepLR(
            optimizer,
            milestones=[cfg.num_steps_before_decay],  # Number of steps after which LR will change
            gamma=0.1,  # Multiplicative factor of learning rate decay
        )

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_metrics = {
        "loss_value": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "dim0_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "dim1_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "dim2_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "dim3_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "dim4_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "dim5_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "dim6_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),

    }

    
    # Start training
    with tqdm.tqdm(total=cfg.max_steps, leave=True) as progress:
        vision_backbone.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            # Compute training metrics and loss
            compute_diffusion_l1 = cfg.use_diffusion and batch_idx % cfg.diffusion_sample_freq == 0
            loss, metrics = run_forward_pass(
                cfg=cfg,
                vision_backbone=vision_backbone,
                action_head=action_head,
                noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                proprio_projector=proprio_projector if cfg.use_proprio else None, 
                batch=batch,
                device_id=device_id,
                use_l1_regression=cfg.use_l1_regression,
                use_diffusion=cfg.use_diffusion,
                use_proprio=cfg.use_proprio,
                num_patches=NUM_PATCHES,
                compute_diffusion_l1=compute_diffusion_l1,
                num_diffusion_steps=cfg.num_diffusion_steps if cfg.use_diffusion else None,
                
            )

            # Normalize loss to account for gradient accumulation
            normalized_loss = loss / cfg.grad_accumulation_steps

            # Backward pass
            normalized_loss.backward()

            # Store recent train metrics
            for metric_name, value in metrics.items():
                if metric_name in recent_metrics:
                    recent_metrics[metric_name].append(value)

            # Compute gradient step index
            gradient_step_idx = batch_idx  // cfg.grad_accumulation_steps

            # Compute smoothened train metrics
            smoothened_metrics = compute_smoothened_metrics(recent_metrics)

            # Push Metrics to W&B (every wandb_log_freq gradient steps)
            log_step = gradient_step_idx + 1 if not cfg.resume else int(step_num) + gradient_step_idx
            if distributed_state.is_main_process and log_step % cfg.wandb_log_freq == 0:
                log_metrics_to_wandb(smoothened_metrics, "VLA Train", log_step, wandb)

            # [If applicable] Linearly warm up learning rate from 10% to 100% of original
            if cfg.lr_warmup_steps > 0:
                lr_progress = min((gradient_step_idx + 1) / cfg.lr_warmup_steps, 1.0)  # Cap at 1.0
                current_lr = original_lr * (0.1 + 0.9 * lr_progress)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr

            if distributed_state.is_main_process and gradient_step_idx % cfg.wandb_log_freq == 0:
                # Log the learning rate
                # Make sure to do this AFTER any learning rate modifications (e.g., warmup/decay)
                wandb.log(
                    {
                        "VLA Train/Learning Rate": scheduler.get_last_lr()[0],
                    },
                    step=log_step,
                )

            # Optimizer and LR scheduler step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress.update()

            # Save model checkpoint: either keep latest checkpoint only or all checkpoints
            if log_step % cfg.save_freq == 0:
                save_training_checkpoint(
                    cfg=cfg,
                    run_dir=run_dir,
                    log_step=log_step,
                    vision_backbone=vision_backbone,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                    action_head=action_head if (cfg.use_l1_regression or cfg.use_diffusion) else None,
                    train_dataset=train_dataset,
                    distributed_state=distributed_state,
                )

            # Test model on validation set
            if cfg.use_val_set and log_step > 0 and log_step % cfg.val_freq == 0:
                run_validation(
                    vision_backbone=vision_backbone,
                    action_head=action_head,
                    noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    val_dataloader=val_dataloader,
                    device_id=device_id,
                    cfg=cfg,
                    num_patches=NUM_PATCHES,
                    log_step=log_step,
                    distributed_state=distributed_state,
                    val_time_limit=cfg.val_time_limit,
                )
                # Set model back to training mode after validation
                vision_backbone.train()

            # Stop training when max_steps is reached
            if log_step == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    finetune()
