#!/bin/bash
# Benchmark: baseline vs phase-1 chunked TP logprob path on Qwen3.5-35B-A3B.
#
# This is the e2e A/B harness for the phase-1 hidden-state logprob implementation.
# It keeps the production LoRA recipe and compares:
#   1. baseline full-logits path
#   2. chunked hidden-state TP logprob path
#
# Usage:
#   bash scripts/benchmark/bench_chunked_tp_logprob_35b.sh
#   NUM_ROLLOUT=20 MAX_TRAINING_STEPS=5 bash scripts/benchmark/bench_chunked_tp_logprob_35b.sh
#   RUN_FUSED_SELECTED_VARIANT=1 bash scripts/benchmark/bench_chunked_tp_logprob_35b.sh
#
# Compare:
#   grep "log_probs_time" /tmp/bench_chunked_tp_*.log
#   grep "actor_train_time" /tmp/bench_chunked_tp_*.log
#   grep "peak_gb" /tmp/bench_chunked_tp_*.log

set -euo pipefail
set -x

NUM_ROLLOUT="${NUM_ROLLOUT:-40}"
MAX_TRAINING_STEPS="${MAX_TRAINING_STEPS:-6}"
CHUNK_SIZES="${CHUNK_SIZES:-512}"       # space-separated list, e.g. "512 256 128"
PROFILE_TARGET="${PROFILE_TARGET:-}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
RUN_FUSED_SELECTED_VARIANT="${RUN_FUSED_SELECTED_VARIANT:-0}"

cleanup() {
    pkill -9 sglang 2>/dev/null || true
    sleep 3
    ray stop --force 2>/dev/null || true
    pkill -9 ray 2>/dev/null || true
    pkill -9 python 2>/dev/null || true
    sleep 3
    pkill -9 ray 2>/dev/null || true
    pkill -9 python 2>/dev/null || true
}

cleanup

if [ ! -f /root/Qwen3.5-35B-A3B/config.json ]; then
    echo "Downloading Qwen3.5-35B-A3B..."
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-35B-A3B', local_dir='/root/Qwen3.5-35B-A3B')"
fi

if [ ! -f /root/gsm8k/train.parquet ] || ! python3 -c "import pandas as pd; df = pd.read_parquet('/root/gsm8k/train.parquet'); assert 'messages' in df.columns" 2>/dev/null; then
    echo "Preparing gsm8k (chat format)..."
    rm -f /root/gsm8k/train.parquet /root/gsm8k/test.parquet
    python3 examples/lora/prep_gsm8k.py
fi

export PYTHONBUFFERED=16
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "${NVLINK_COUNT}" -gt 0 ]; then
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

LORA_ARGS=(
   --lora-rank 32
   --lora-alpha 32
   --lora-dropout 0.0
   --target-modules "all-linear"
)

ROLLOUT_ARGS=(
   --prompt-data /root/gsm8k/train.parquet
   --input-key messages
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type math
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   --rollout-max-response-len 1024
   --rollout-temperature 1
   --global-batch-size 128
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
   --micro-batch-size 1
   --qkv-format bshd
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
   --sglang-mem-fraction-static 0.4
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
   --max-training-steps "${MAX_TRAINING_STEPS}"
)

PROFILE_ARGS=()
if [ -n "${PROFILE_TARGET}" ]; then
    PROFILE_ARGS+=(--use-pytorch-profiler --profile-step-start 1 --profile-step-end 2 --profile-target "${PROFILE_TARGET}")
fi

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
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

run_variant() {
    local label="$1"
    shift
    local logfile="/tmp/bench_chunked_tp_${label}.log"

    echo ""
    echo "========================================"
    echo "  Run: ${label}"
    echo "========================================"

    cleanup
    ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${GPUS_PER_NODE}" \
        --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

    ray job submit --address="http://127.0.0.1:8265" \
        --runtime-env-json="${RUNTIME_ENV_JSON}" \
        -- python3 train.py \
        --actor-num-nodes 1 \
        --actor-num-gpus-per-node "${GPUS_PER_NODE}" \
        --colocate \
        --calculate-per-token-loss \
        "${MODEL_ARGS[@]}" \
        "${CKPT_ARGS[@]}" \
        "${LORA_ARGS[@]}" \
        "${ROLLOUT_ARGS[@]}" \
        "${OPTIMIZER_ARGS[@]}" \
        "${GRPO_ARGS[@]}" \
        "${PERF_ARGS[@]}" \
        "${SGLANG_ARGS[@]}" \
        "${MISC_ARGS[@]}" \
        "${PROFILE_ARGS[@]}" \
        "$@" \
        2>&1 | tee "${logfile}"

    ray stop --force
    sleep 5
}

if [ "${SKIP_BASELINE:-0}" != "1" ]; then
    run_variant "baseline"
fi

for cs in ${CHUNK_SIZES}; do
    run_variant "chunked_seq${cs}" \
        --use-chunked-tp-logprob-loss \
        --chunked-tp-logprob-seq-chunk-size "${cs}"
    if [ "${RUN_FUSED_SELECTED_VARIANT}" = "1" ]; then
        run_variant "fused_seq${cs}" \
            --use-chunked-tp-logprob-loss \
            --chunked-tp-logprob-seq-chunk-size "${cs}" \
            --use-fused-selected-tp-logprob
    fi
done

echo ""
echo "========================================"
echo "  Results Summary"
echo "========================================"

ALL_LABELS="baseline"
for cs in ${CHUNK_SIZES}; do
    ALL_LABELS="${ALL_LABELS} chunked_seq${cs}"
    if [ "${RUN_FUSED_SELECTED_VARIANT}" = "1" ]; then
        ALL_LABELS="${ALL_LABELS} fused_seq${cs}"
    fi
done

for label in ${ALL_LABELS}; do
    echo ""
    echo "--- ${label} ---"
    grep -E "actor_train_time|log_probs_time|train_wait_time|sleep_time|wake_up_time|update_weights_time|tflops|peak_gb|tok_per_s" "/tmp/bench_chunked_tp_${label}.log" || true
done
