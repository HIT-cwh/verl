#!/usr/bin/env bash
set -x

/mnt/shared-storage-user/caoweihan/miniconda3/envs/verl/bin/ray stop --force

cd /mnt/shared-storage-user/caoweihan/projects/verl

export RAY_MASTER_PORT=6379
export RAY_DASHBOARD_PORT=8265
export TRITON_CACHE_DIR="/tmp/triton"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export PYTHONUNBUFFERED=1

export ROUTER_N_GROUPS=8

if [ "$NODE_RANK" -eq 0 ]; then
    /mnt/shared-storage-user/caoweihan/miniconda3/envs/verl/bin/ray start --head --dashboard-host=0.0.0.0 --dashboard-port=$RAY_DASHBOARD_PORT
    sleep 30
else
    /mnt/shared-storage-user/caoweihan/miniconda3/envs/verl/bin/ray start --address="$MASTER_ADDR:$RAY_MASTER_PORT"  --block
fi
sleep 20

project_name='DAPO'
exp_name='rl_qwen3_30b_base_wo_sft_wo_grouped_router_dapo_wo_ds_grouped'

NNODES=8
NPUS_PER_NODE=8

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 20))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0
loss_agg_mode="token-mean"
ppo_mini_batch_size=32

enable_filter_groups=False
train_prompt_bsz=512
n_resp_per_prompt=16

# RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
# WORKING_DIR=${WORKING_DIR:-"${PWD}"}
# RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
# RAY_ADDRESS="http://10.102.242.79:8265"
RUNTIME_ENV="/mnt/shared-storage-user/caoweihan/projects/verl/recipe/dapo/runtime_env.yaml"

# Paths
# RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
# MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen3-30B-A3B-Base"}
# CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
# TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo-math-17k.parquet"}
# TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/aime-2024.parquet"}
MODEL_PATH="/mnt/shared-storage-user/large-model-center-share-weights/hf_hub/models--Qwen--Qwen3-30B-A3B-Base/snapshots/89e5e822ba31507f5f79dc3422c7c5345c422737"
CKPTS_DIR="/mnt/shared-storage-user/caoweihan/projects/verl/ckpts/DAPO/${exp_name}"
TRAIN_FILE="/mnt/shared-storage-user/llmrazor-share/data/dapo_math/dapo-math-17k.parquet"
TEST_FILE="/mnt/shared-storage-user/llmrazor-share/data/dapo_math/aime-2024.parquet"

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

# Performance Related Parameter
sp_size=1 # For load-balance. For smaller cluster this can be set to as less as 2.
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / 1))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / 1))
offload=False
recompute=True
max_num_seqs=128
gen_tp=2

# export PYTHONPATH=.:/mnt/shared-storage-user/caoweihan/packages/:${PYTHONPATH:-}

RAY_ADDRESS="http://127.0.0.1:$RAY_DASHBOARD_PORT"
/mnt/shared-storage-user/caoweihan/miniconda3/envs/verl/bin/ray job submit --runtime-env="${RUNTIME_ENV}" \
    -- /mnt/shared-storage-user/caoweihan/miniconda3/envs/verl/bin/python -m recipe.dapo.main_dapo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.max_num_seqs=${max_num_seqs} \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length))  \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.model.enable_gradient_checkpointing=${recompute} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.ref.fsdp_config.forward_prefetch=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.logger=['console'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${NPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.test_freq=5 \
    trainer.save_freq=10 \
    trainer.total_epochs=1 \
    trainer.rollout_data_dir="${CKPTS_DIR}/rollouts" \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.use_torch_compile=False 2>&1|tee "${exp_name}.txt"
   
