export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --standalone --nnodes 1 --nproc-per-node 4 vat/scripts/train_vat.py \
  --data_root_dir ./libero_rlds \
  --dataset_name libero_10_no_noops \
  --run_root_dir ./ckpt/ \
  --use_l1_regression True \
  --use_diffusion False \
  --batch_size 128 \
  --learning_rate 2e-5 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --wandb_entity "" \
  --wandb_project "" \
  --use_lora False \
  --action_dim_input 7 \
  --action_chunk 8 \
  --visionbackbone_requiregrad True \
  --shuffle_buffer_size 5000 \
  --run_id_note train_vat \
  --epochs 100 \
  --save_freq_epochs 20 \
  --only_train_actionmodule False \
  --use_proprio False \
  --use_cosinelr True \
  --use_wrist_image True \
  --vit_large False \
  --dino False \
  --use_film True \
  --taskembedding_add False \
  --baseline False \
  --vat_small_factor 1 \
  --vat_vit False 
 
  
  
