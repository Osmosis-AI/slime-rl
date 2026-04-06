#!/bin/bash
# Qwen3.5-122B-A10B LoRA GRPO on 2x8xH200 (multi-node)
# Run on the HEAD node only. Worker must have Ray joined first.
#
# Setup:
#   HEAD:   MASTER_ADDR=<head-ip> bash examples/lora/run-qwen3.5-122B-A10B-megatron-lora-2node.sh
#   WORKER: ray start --address=<head-ip>:6381 --num-gpus 8
#
# K8s 2-pod setup:
#   Pod 1 (head):   kubectl exec mouad-qwen122b -- env MASTER_ADDR=<pod1-ip> bash ...
#   Pod 2 (worker): kubectl exec mouad-qwen122b-worker -- ray start --address=<pod1-ip>:6381 --num-gpus 8

export FLASHINFER_DISABLE_VERSION_CHECK=1
export GPUS_PER_NODE=8
export PYTHONBUFFERED=16
NUM_NODES=2

# ── Configurable paths ──────────────────────────────────────────────────
MODEL_DIR=${MODEL_DIR:-/root/Qwen3.5-122B-A10B}
TRAIN_DATA=${TRAIN_DATA:-/root/datasets/dapo-math-17k/dapo-math-17k.jsonl}
EVAL_DATA=${EVAL_DATA:-/root/datasets/aime-2024/aime-2024.jsonl}
MEGATRON_LM_DIR=${MEGATRON_LM_DIR:-/root/Megatron-LM}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/root/checkpoints}
RAY_PORT=${RAY_PORT:-6381}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8266}

# ── Clean up stale processes ─────────────────────────────────────────────
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
source "${SCRIPT_DIR}/../../scripts/models/qwen3.5-122B-A10B.sh"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}
   --megatron-to-hf-mode bridge
   --save ${CHECKPOINT_DIR}
   --save-interval 20
)

LORA_ARGS=(
   --lora-rank 64
   --lora-alpha 64
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
   --num-rollout 531
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 16384
   --system-prompt "Think concisely and efficiently. Provide your reasoning, then put your final answer within \\boxed{}."
   --rollout-temperature 1

   --global-batch-size 256
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime ${EVAL_DATA}
   --eval-input-key prompt
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 16384
   --eval-top-k 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 16
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --qkv-format bshd
   --micro-batch-size 1

   --log-probs-chunk-size 4096
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
   --lr 2e-5
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
   --mlflow-experiment-name slime-lora-megatron
   --mlflow-run-name qwen3.5-122B-A10B-dapo-lora-2node
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.7
   --sglang-ep-size 8
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)

   --sglang-max-running-requests 128
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

# ── Ray setup (head node only — worker must join before this runs) ─────
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"

ray start --head \
    --node-ip-address ${MASTER_ADDR} \
    --num-gpus $GPUS_PER_NODE \
    --port=${RAY_PORT} \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=${RAY_DASHBOARD_PORT} \
    --disable-usage-stats

# Wait for worker to join
echo "Waiting for 2 nodes in Ray cluster..."
for i in $(seq 1 60); do
    NODE_COUNT=$(ray status 2>/dev/null | grep -c "node_" || echo "0")
    RAY_NODES=$(python3 -c "import ray; ray.init(address='auto'); print(len(ray.nodes()))" 2>/dev/null || echo "0")
    if [ "$RAY_NODES" -ge "$NUM_NODES" ]; then
        echo "All $NUM_NODES nodes joined!"
        break
    fi
    echo "  Waiting... ($RAY_NODES/$NUM_NODES nodes, attempt $i/60)"
    sleep 10
done

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_LM_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"SGLANG_DISABLE_CUDNN_CHECK\": \"1\",
    \"no_proxy\": \"${no_proxy}\"
  }
}"

ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes ${NUM_NODES} \
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
