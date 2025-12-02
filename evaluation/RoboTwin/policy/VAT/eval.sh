#!/bin/bash

export PYTHONPATH=/root/code/vat:/root/code/vat/RoboTwin:$PYTHONPATH
policy_name=VAT # [TODO] 
task_name="press_stapler"
task_config="demo_clean"
ckpt_setting="/limx/tos/users/wenhao/correct_openvlaoft_ckpt/robotwin_chunk25/press_stapler/openvla-7b+robotwin_press_stapler+b64+lr-2e-05--image_aug--bread--9675_ckpt"
unnorm_key="robotwin_press_stapler"

seed=7
gpu_id=1
# [TODO] add parameters here

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../.. # move to root

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --unnorm_key ${unnorm_key} \
    # [TODO] add parameters here
