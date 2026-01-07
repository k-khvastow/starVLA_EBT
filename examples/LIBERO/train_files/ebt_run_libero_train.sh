#!/bin/bash

# =========================================================================
# 1. Path Fix: Automatically jump to Project Root
# =========================================================================
SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"

echo "Current location: $(pwd)"
echo "Switching to Project Root: $PROJECT_ROOT"
cd "$PROJECT_ROOT" || { echo "Failed to navigate to project root"; exit 1; }

# Fix Model Path
base_vlm_relative="playground/Pretrained_models/Qwen2.5-VL-3B-Instruct"
base_vlm="$(realpath "$base_vlm_relative")"

if [ ! -d "$base_vlm" ]; then
    echo "Error: Model directory not found at $base_vlm"
    exit 1
fi

# =========================================================================
# 2. Environment Configuration
# =========================================================================

# --- CRITICAL FIXES ---
# 1. Disable the experimental non-blocking communicator (CAUSED THE TIMEOUT)
unset TORCH_NCCL_USE_COMM_NONBLOCKING

# 2. Force NCCL to use local loopback (Safety for single-node)
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# 3. Explicit Master Port to avoid collisions (29500 is often taken)
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29505
# ----------------------

export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000 
export NCCL_DEBUG=INFO
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Framework & Base Config
Framework_name=QwenEBT
config_yaml=./starVLA/config/training/starvla_cotrain_oxe_ebt.yaml 

# Dataset Configuration (LIBERO)
libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_all
action_type=delta_qpos 

# Output Settings
run_root_dir=./results/Checkpoints
run_id=qwen_ebt_libero_all
freeze_module_list='qwen_vl_interface'

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp "$SCRIPT_PATH" "${output_dir}/"

# =========================================================================
# 3. Launch Training
# =========================================================================
echo "Launching QwenEBT training..."
echo "Config: ${config_yaml}"
echo "Master: $MASTER_ADDR:$MASTER_PORT"

# We pass main_process_ip/port explicitly to ensure Accelerate respects our manual settings
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 1 \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.dataset_py lerobot_datasets \
  --datasets.vla_data.data_root_dir ${libero_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.action_type ${action_type} \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 100 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project RecurrentVLA_LIBERO_Debug \
  --wandb_entity kirill-khvastow-technical-university-of-munich

