#!/bin/bash

export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_2,mlx5_3

# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000  # timeout set to ~3 hours
export NCCL_SOCKET_TIMEOUT_MS=360000

###########################################################################################
# === Please modify the following paths according to your environment ===

# Framework name must match the registry key in QwenEBT.py
Framework_name=QwenEBT 

# Using Qwen2.5-VL-3B as defined in your EBT yaml
base_vlm=playground/Pretrained_models/Qwen2.5-VL-3B-Instruct 

# Path to your EBT-specific config
config_yaml=./starVLA/config/training/starvla_cotrain_oxe_ebt.yaml 

# Dataset paths (Aligned with the EBT YAML defaults)
oxe_data_root=playground/Datasets/OXE_LEROBOT_DATASET
data_mix=bridge_rt_1

run_root_dir=./results/Checkpoints
run_id=qwen_ebt_lerobot_run
freeze_module_list=''

# === End of environment variable configuration ===
###########################################################################################

export WANDB_MODE=online # or disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# Copy this script to the output dir for reproducibility
cp $0 ${output_dir}/

echo "Starting training for Framework: ${Framework_name}"
echo "Config: ${config_yaml}"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${oxe_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 100 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_EBT \
  --wandb_entity jinhuiye