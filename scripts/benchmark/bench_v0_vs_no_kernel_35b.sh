#!/bin/bash
# Benchmark: V0 fused MoE backward vs standard autograd on Qwen3.5-35B-A3B
#
# Based on slime-rl/main production config (scripts/run-qwen3.5-35B-A3B.sh)
# but with EP=1 (required by fused MoE backward) and 10 steps only.
#
# Runs sequentially:
#   1. FFT + fused MoE backward (V0 Triton kernels)
#   2. FFT + standard autograd (no fused kernel)
#
# Pod: mouad-fused-kernels (8xH200)
#
# Usage:
#   bash scripts/benchmark/bench_v0_vs_no_kernel_35b.sh
#
# Compare: grep "actor_train_time" /tmp/bench_*.log

# Clean up
pkill -9 sglang 2>/dev/null || true
sleep 3
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
pkill -9 python 2>/dev/null || true
sleep 3
pkill -9 ray 2>/dev/null || true
pkill -9 python 2>/dev/null || true

set -ex

export PYTHONBUFFERED=16
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../models/qwen3.5-35B-A3B.sh"

CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3.5-35B-A3B/
   --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
   --prompt-data /root/gsm8k/train.parquet
   --input-key messages
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type math
   --num-rollout 100
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   --rollout-max-response-len 1024
   --rollout-temperature 1
   --global-batch-size 128
)

PERF_ARGS=(
   # Production-matching parallelism: TP=2, EP=8
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --micro-batch-size 1
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
   --max-training-steps 10
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"SGLANG_DISABLE_CUDNN_CHECK\": \"1\",
    \"no_proxy\": \"${no_proxy}\"
  }
}"

run_benchmark() {
    local label="$1"
    shift
    local logfile="/tmp/bench_${label}.log"

    echo ""
    echo "========================================"
    echo "  Run: ${label}"
    echo "========================================"

    ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 \
        --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

    ray job submit --address="http://127.0.0.1:8265" \
        --runtime-env-json="${RUNTIME_ENV_JSON}" \
        -- python3 train.py \
        --actor-num-nodes 1 \
        --actor-num-gpus-per-node 8 \
        --colocate \
        --calculate-per-token-loss \
        ${MODEL_ARGS[@]} \
        ${CKPT_ARGS[@]} \
        ${ROLLOUT_ARGS[@]} \
        ${OPTIMIZER_ARGS[@]} \
        ${GRPO_ARGS[@]} \
        ${PERF_ARGS[@]} \
        ${SGLANG_ARGS[@]} \
        ${MISC_ARGS[@]} \
        "$@" \
        2>&1 | tee "$logfile"

    ray stop --force
    sleep 5
}

# === Run 1: FFT + fused MoE backward (V0 Triton kernels) ===
run_benchmark "fft_fused" --use-fused-moe-backward

# === Run 2: FFT + standard autograd (no fused kernel) ===
run_benchmark "fft_no_fused"

# === Summary ===
echo ""
echo "========================================"
echo "  Results Summary (FFT, TP=2, EP=8)"
echo "========================================"
echo ""
echo "--- FFT + V0 Fused MoE Backward ---"
grep -E "actor_train_time|tflops|peak_gb" /tmp/bench_fft_fused.log | tail -10
echo ""
echo "--- FFT + Standard Autograd ---"
grep -E "actor_train_time|tflops|peak_gb" /tmp/bench_fft_no_fused.log | tail -10
