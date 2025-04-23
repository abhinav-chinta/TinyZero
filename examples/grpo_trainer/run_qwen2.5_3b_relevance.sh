#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

set -x # Print commands for debugging

# --- User Configuration ---

# 1. Data Paths (Using $HOME for better script portability)
TRAIN_FILE="$HOME/repos/grpo_training/TinyZero/data/relevance_rerankers_verl/train.parquet"
VAL_FILE="$HOME/repos/grpo_training/TinyZero/data/relevance_rerankers_verl/test.parquet"

# 2. Model Selection
MODEL_PATH="Qwen/Qwen2.5-3B" # Base model as requested

# 3. GPU Configuration
N_GPUS_PER_NODE=4 # You have 4 GPUs
ROLLOUT_TP_SIZE=4 # Match this to your GPU count

# 4. Experiment Naming
PROJECT_NAME="verl_relevance_reranker"
EXPERIMENT_NAME="grpo-qwen2.5-3b-base-relevance-run"

GRPO_N_ROLLOUTS=4 # Number of responses for GRPO

TOTAL_EPOCHS=10 # Increase epochs slightly due to smaller batches
TEST_FREQ=2
SAVE_FREQ=-1 # Disable saving by default, enable if needed

# --- Environment Settings ---
export VLLM_ATTENTION_BACKEND=XFORMERS # Often helps with compatibility/performance


# --- Build Command ---
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.train_batch_size=512 \
    data.val_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=$TOTAL_EPOCHS \
    "$@" # Allow passing extra arguments from command line