#!/bin/bash
# Qwen3.5-35B-A3B FP8 LoRA GRPO training — Together AI cluster (8xH200).
# BF16 checkpoint + TransformerEngine FP8 training (e4m3 blockwise) + LoRA rank=32.
# TP=2, EP=8, CPU optimizer offload, offload-train for colocated LoRA.
#
# Together AI pod:  kubectl exec -it mouad-qwen122b -n trainers -- bash
#   MODEL_DIR=/root/Qwen3.5-35B-A3B bash scripts/low_precision/run-qwen3.5-35B-A3B-fp8-lora-together.sh

export FLASHINFER_DISABLE_VERSION_CHECK=1
export GPUS_PER_NODE=8
export PYTHONBUFFERED=16

# ── Configurable paths (override via env vars) ────────────────────────
MODEL_DIR=${MODEL_DIR:-/root/Qwen3.5-35B-A3B}
TRAIN_DATA=${TRAIN_DATA:-/root/datasets/dapo-math-17k/dapo-math-17k.jsonl}
EVAL_DATA=${EVAL_DATA:-/root/datasets/aime-2024/aime-2024.jsonl}
MEGATRON_LM_DIR=${MEGATRON_LM_DIR:-/root/Megatron-LM}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/root/checkpoints/qwen3.5-35B-A3B-fp8-lora}
RAY_PORT=${RAY_PORT:-6381}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8266}

# ── Download model if not present ─────────────────────────────────────
if [ ! -d "${MODEL_DIR}" ] || [ -z "$(ls -A ${MODEL_DIR} 2>/dev/null)" ]; then
    echo "Downloading Qwen/Qwen3.5-35B-A3B to ${MODEL_DIR}..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.5-35B-A3B', local_dir='${MODEL_DIR}')
print('Model download complete.')
"
fi

# ── Download datasets if not present ──────────────────────────────────
if [ ! -f "${TRAIN_DATA}" ]; then
    TRAIN_DIR=$(dirname "${TRAIN_DATA}")
    echo "Downloading dapo-math-17k to ${TRAIN_DIR}..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('zhuzilin/dapo-math-17k', repo_type='dataset', local_dir='${TRAIN_DIR}')
print('Train dataset download complete.')
"
fi

if [ ! -f "${EVAL_DATA}" ]; then
    EVAL_DIR=$(dirname "${EVAL_DATA}")
    echo "Downloading aime-2024 to ${EVAL_DIR}..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('zhuzilin/aime-2024', repo_type='dataset', local_dir='${EVAL_DIR}')
print('Eval dataset download complete.')
"
fi

# ── Clean up stale processes ──────────────────────────────────────────
pkill -9 sglang 2>/dev/null || true
sleep 3
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
pkill -9 -f "train.py\|train_async.py" 2>/dev/null || true
sleep 3

set -ex

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../models/qwen3.5-35B-A3B.sh"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}
   --megatron-to-hf-mode bridge
   --save ${CHECKPOINT_DIR}
   --save-interval 50
)

LORA_ARGS=(
   --lora-rank 32
   --lora-alpha 32
   --lora-dropout 0.0
   --target-modules "all-linear"
   --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
   --prompt-data ${TRAIN_DATA}
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 200
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   --rollout-max-response-len 12288
   --system-prompt "Think concisely and efficiently. Provide your reasoning, then put your final answer within \\boxed{}."
   --rollout-temperature 1

   --global-batch-size 128
)

EVAL_ARGS=(
   --eval-interval 50
   --eval-prompt-data aime ${EVAL_DATA}
   --eval-input-key prompt
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 4096
   --eval-top-k 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --qkv-format bshd
   --micro-batch-size 1

   # FP8 training via TransformerEngine
   --transformer-impl transformer_engine
   --bf16
   --fp8-format e4m3
   --fp8-recipe blockwise
   # --fp8-param-gather  # incompatible with CPU Adam
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.01
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 5e-5
   --clip-grad 1.0
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

MLFLOW_ARGS=(
   --use-mlflow
   --mlflow-experiment-name qwen3.5-35B-A3B-fp8-lora
   --mlflow-run-name fp8-e4m3-lora-r32-together
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.7
   --sglang-ep-size 8
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
   --sglang-max-running-requests 512
   --offload-train
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --moe-token-dispatcher-type alltoall
)

# ── Ray setup (port 6381 to avoid SkyPilot conflict on k8s) ──────────
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"

ray start --head \
    --node-ip-address ${MASTER_ADDR} \
    --num-gpus $GPUS_PER_NODE \
    --port=${RAY_PORT} \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=${RAY_DASHBOARD_PORT} \
    --disable-usage-stats

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_LM_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"NVTE_FP8_BLOCK_SCALING_FP32_SCALES\": \"1\",
    \"SGLANG_DISABLE_CUDNN_CHECK\": \"1\",
    \"no_proxy\": \"${no_proxy}\"
  }
}"

ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node $GPUS_PER_NODE \
   --colocate \
   --calculate-per-token-loss \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${LORA_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${MLFLOW_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${ROLLOUT_ARGS[@]}
