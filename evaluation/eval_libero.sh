#!/bin/bash
export PYTHONPATH=./vat:./vat/LIBERO:$PYTHONPATH
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0
python evaluation/run_libero_eval.py \
  --pretrained_checkpoint ./ckpt/ \
  --action_dim_input 7 \
  --task_suite_name "libero_10" \
  --use_diffusion False \
  --use_l1_regression True \
  --action_chunk 8 \
  --local_log_dir "./logs" \
  --use_wandb False \
  --use_proprio False \
  --use_wrist_image True \
  --center_crop True \
  --vit_large False \
  --dino False \
  --num_trials_per_task 50 \
  --only_use_wrist False \
  --use_film True \
  --taskembedding_add False \
  --baseline False \
  --vat_small_factor 1 \
  --vat_vit False 



