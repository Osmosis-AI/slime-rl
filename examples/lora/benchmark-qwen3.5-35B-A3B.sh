#!/bin/bash
# LoRA vs Full Fine-Tuning Benchmark for Qwen3.5-35B-A3B
# Runs both configs sequentially (20 rollout steps each), logging to MLflow.
#
# Usage:
#   bash examples/lora/benchmark-qwen3.5-35B-A3B.sh
#
# After completion, generate the comparison report:
#   python tools/benchmark_report.py \
#       --experiment slime-lora-benchmark \
#       --lora-run lora-r32-benchmark \
#       --baseline-run full-ft-benchmark

export FLASHINFER_DISABLE_VERSION_CHECK=1
export GPUS_PER_NODE=8
export PYTHONBUFFERED=16

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../scripts/models/qwen3.5-35B-A3B.sh"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi

EXPERIMENT_NAME="slime-lora-benchmark"
NUM_ROLLOUT=20

# Shared args across both runs
SHARED_ARGS=(
   --actor-num-nodes 1
   --actor-num-gpus-per-node $GPUS_PER_NODE
   --colocate
   --calculate-per-token-loss
   --use-slime-router

   --hf-checkpoint /root/Qwen3.5-35B-A3B/
   --megatron-to-hf-mode bridge

   --prompt-data /root/gsm8k/train.parquet
   --input-key messages
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type math
   --num-rollout $NUM_ROLLOUT
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   --rollout-max-response-len 1024
   --rollout-temperature 1
   --global-batch-size 128

   --eval-interval 10
   --eval-prompt-data aime /root/aime-2024/aime-2024.jsonl
   --eval-input-key prompt
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 1024
   --eval-top-k 1

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

   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28

   --optimizer adam
   --lr 1e-5
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer

   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.7
   --sglang-ep-size 8
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
   --sglang-max-running-requests 512

   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --moe-token-dispatcher-type alltoall

   --use-mlflow
   --mlflow-experiment-name $EXPERIMENT_NAME
)

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"SGLANG_DISABLE_CUDNN_CHECK\": \"1\",
    \"no_proxy\": \"127.0.0.1,${MASTER_ADDR:-127.0.0.1}\"
  }
}"

cleanup() {
    echo "==> Cleaning up processes..."
    pkill -9 sglang 2>/dev/null || true
    sleep 3
    ray stop --force 2>/dev/null || true
    pkill -9 ray 2>/dev/null || true
    pkill -9 python 2>/dev/null || true
    sleep 3
    pkill -9 ray 2>/dev/null || true
    pkill -9 python 2>/dev/null || true
}

start_ray() {
    export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
    export no_proxy="127.0.0.1,${MASTER_ADDR}"
    ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus $GPUS_PER_NODE \
        --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
}

# ============================================================
# Phase 1: LoRA (rank=32)
# ============================================================
echo ""
echo "============================================================"
echo "  Phase 1/2: LoRA r=32 ($NUM_ROLLOUT rollout steps)"
echo "============================================================"
echo ""

cleanup
start_ray

set -x
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   ${MODEL_ARGS[@]} \
   ${SHARED_ARGS[@]} \
   --mlflow-run-name lora-r32-benchmark \
   --offload-train \
   --lr 1e-4 \
   --lora-rank 32 \
   --lora-alpha 32 \
   --lora-dropout 0.0 \
   --target-modules "all-linear"
set +x

LORA_EXIT=$?
echo "==> LoRA run exited with code $LORA_EXIT"

# ============================================================
# Phase 2: Full Fine-Tuning (baseline)
# ============================================================
echo ""
echo "============================================================"
echo "  Phase 2/2: Full Fine-Tuning ($NUM_ROLLOUT rollout steps)"
echo "============================================================"
echo ""

cleanup
start_ray

set -x
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   ${MODEL_ARGS[@]} \
   ${SHARED_ARGS[@]} \
   --mlflow-run-name full-ft-benchmark \
   --moe-token-dispatcher-type flex
set +x

BASELINE_EXIT=$?
echo "==> Baseline run exited with code $BASELINE_EXIT"

cleanup

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "  Benchmark complete!"
echo "  LoRA exit=$LORA_EXIT, Baseline exit=$BASELINE_EXIT"
echo ""
echo "  Generate report:"
echo "    python tools/benchmark_report.py \\"
echo "        --experiment $EXPERIMENT_NAME \\"
echo "        --lora-run lora-r32-benchmark \\"
echo "        --baseline-run full-ft-benchmark"
echo "============================================================"
