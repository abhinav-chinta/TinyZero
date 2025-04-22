
#!/bin/bash

# Script to run GRPO training for relevance reranking on Qwen/Qwen2.5-3B (Base)
# Target Hardware: 4x RTX 3090 (24GB VRAM)

set -x # Print commands for debugging

# --- User Configuration ---

# 1. Data Paths (Using $HOME for better script portability)
TRAIN_FILE="$HOME/repos/grpo_training/TinyZero/data/relevance_rerankers_verl/train.parquet"
VAL_FILE="$HOME/repos/grpo_training/TinyZero/data/relevance_rerankers_verl/test.parquet"

# 2. Model Selection
MODEL_PATH="Qwen/Qwen2.5-3B" # Base model as requested

# 3. GPU Configuration
N_GPUS_PER_NODE=4 # You have 4 GPUs
NNODES=1 # Running on a single machine
ROLLOUT_TP_SIZE=4 # Match this to your GPU count

# 4. Experiment Naming
PROJECT_NAME="verl_relevance_reranker"
EXPERIMENT_NAME="grpo-qwen2.5-3b-base-relevance-run"

# 5. Training Parameters (Adjusted for 4x 3090)
TRAIN_BATCH_SIZE=64 # Total samples per PPO iteration
VAL_BATCH_SIZE=64 # Validation batch size
PPO_MINI_BATCH_SIZE=4 # Reduced to prevent deadlock (must be divisible by N_GPUS)
PPO_MICRO_BATCH_SIZE=1 # CRITICAL: Micro-batch size
ROLLOUT_LOGPROB_MICRO_BATCH_SIZE=1 # Keep small for memory
CRITIC_MICRO_BATCH_SIZE=1 # Keep small for memory

MAX_PROMPT_LENGTH=2048 # From your data inspection
MAX_RESPONSE_LENGTH=256 # Reduced - your expected output seems short

ACTOR_LR=1e-6
CRITIC_LR=1e-5
KL_COEF=0.001
GRPO_N_ROLLOUTS=5 # Number of responses for GRPO

TOTAL_EPOCHS=10 # Increase epochs slightly due to smaller batches
TEST_FREQ=2
SAVE_FREQ=-1 # Disable saving by default, enable if needed

# --- Environment Settings ---
export VLLM_ATTENTION_BACKEND=XFORMERS # Often helps with compatibility/performance
# export TOKENIZERS_PARALLELISM=false # Recommended by HF to avoid issues

# --- Build Command ---
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.prompt_key=prompt \
    actor_rollout_ref.model.path=$MODEL_PATH \
    +actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr=$ACTOR_LR \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size=$PPO_MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=$ROLLOUT_LOGPROB_MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=$GRPO_N_ROLLOUTS \
    actor_rollout_ref.ref.log_prob_micro_batch_size=$ROLLOUT_LOGPROB_MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    critic.model.path=$MODEL_PATH \
    +critic.model.trust_remote_code=True \
    critic.optim.lr=$CRITIC_LR \
    critic.model.use_remove_padding=True \
    critic.ppo_micro_batch_size=$CRITIC_MICRO_BATCH_SIZE \
    critic.model.enable_gradient_checkpointing=True \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.grad_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    reward_model.enable=False \
    algorithm.kl_ctrl.kl_coef=$KL_COEF \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=$TOTAL_EPOCHS \
    +trainer.val_before_train=False \
    "$@" # Allow passing extra arguments from command line