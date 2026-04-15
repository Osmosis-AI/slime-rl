#!/bin/bash

# Qwen/Qwen3.5-122B-A10B LoRA GRPO on 4x8xH200 via Ray.
# Run on the head node only.
#
# Assumptions:
# - the cluster exposes MLP_WORKER_0_HOST and MLP_SOCKET_IFNAME
# - /root/mpi_rack_hostfile lists the node IPs reachable over InfiniBand
# - model/data paths are available locally on every node

export FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTHONBUFFERED=16

set -ex

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
NUM_NODES="${NUM_NODES:-4}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
WORKING_DIR="${WORKING_DIR:-${ROOT_DIR}}"

if [ -z "${MLP_WORKER_0_HOST}" ]; then
  echo "MLP_WORKER_0_HOST is not set."
  exit 1
fi

if [ -z "${MLP_SOCKET_IFNAME}" ]; then
  echo "MLP_SOCKET_IFNAME is not set."
  exit 1
fi

if [ ! -f /root/mpi_rack_hostfile ]; then
  echo "/root/mpi_rack_hostfile does not exist."
  exit 1
fi

DATA_DIR=${DATA_DIR:-/data}
MODEL_DIR=${MODEL_DIR:-${DATA_DIR}/Qwen3.5-122B-A10B}
TRAIN_DATA=${TRAIN_DATA:-${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl}
EVAL_DATA=${EVAL_DATA:-${DATA_DIR}/aime-2024/aime-2024.jsonl}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${DATA_DIR}/checkpoints/qwen3.5-122B-A10B-lora-4node}
MEGATRON_LM_DIR=${MEGATRON_LM_DIR:-/root/Megatron-LM}
RAY_PORT=${RAY_PORT:-6381}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8266}

MASTER_ADDR=${MASTER_ADDR:-${MLP_WORKER_0_HOST}}
export MASTER_ADDR
export no_proxy="localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}"

mapfile -t CLUSTER_HOSTS < <(awk '{print $1}' /root/mpi_rack_hostfile | awk 'NF' | head -n "${NUM_NODES}")
if [ "${#CLUSTER_HOSTS[@]}" -lt "${NUM_NODES}" ]; then
  echo "Expected at least ${NUM_NODES} hosts in /root/mpi_rack_hostfile."
  exit 1
fi

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3.5-122B-A10B.sh"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}
   --save ${CHECKPOINT_DIR}
   --save-interval 20
)

LORA_ARGS=(
   --lora-rank 64
   --lora-alpha 64
   --lora-dropout 0.0
   --target-modules "q_proj,k_proj,v_proj,o_proj"
   --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
   --prompt-data ${TRAIN_DATA}
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 100
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
   --context-parallel-size ${CONTEXT_PARALLEL_SIZE}
   --expert-model-parallel-size 16
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --qkv-format bshd
   --micro-batch-size 1

   --log-probs-chunk-size 4096
)

ALGO_ARGS=(
   --advantage-estimator gspo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 4e-4
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 2e-5
   --clip-grad 1.0
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

MLFLOW_ARGS=(
   --use-mlflow
   --mlflow-experiment-name slime-lora-megatron
   --mlflow-run-name qwen3.5-122B-A10B-dapo-lora-4node
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

pkill -9 sglang 2>/dev/null || true
sleep 3
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
pkill -9 -f "train.py\\|train_async.py" 2>/dev/null || true
sleep 3

ray start --head \
    --node-ip-address "${MASTER_ADDR}" \
    --num-gpus "${GPUS_PER_NODE}" \
    --port="${RAY_PORT}" \
    --dashboard-host=0.0.0.0 \
    --dashboard-port="${RAY_DASHBOARD_PORT}" \
    --disable-usage-stats

STARTED_NODES=1
for WORKER_IP in "${CLUSTER_HOSTS[@]}"; do
  if [[ "${WORKER_IP}" == "${MASTER_ADDR}" ]]; then
    continue
  fi

  echo "Starting Ray worker on ${WORKER_IP}"
  ssh root@"${WORKER_IP}" \
    "pkill -9 sglang 2>/dev/null || true; \
     ray stop --force 2>/dev/null || true; \
     pkill -9 ray 2>/dev/null || true; \
     pkill -9 -f 'train.py\\|train_async.py' 2>/dev/null || true; \
     ray start --address=${MASTER_ADDR}:${RAY_PORT} --num-gpus ${GPUS_PER_NODE} --node-ip-address ${WORKER_IP} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=${RAY_DASHBOARD_PORT}" &
  STARTED_NODES=$((STARTED_NODES + 1))
  if [ "${STARTED_NODES}" -ge "${NUM_NODES}" ]; then
    break
  fi
done
wait

echo "Waiting for ${NUM_NODES} nodes in Ray cluster..."
for i in $(seq 1 60); do
    RAY_NODES=$(python3 -c "import ray; ray.init(address='auto'); print(len(ray.nodes()))" 2>/dev/null || echo "0")
    if [ "${RAY_NODES}" -ge "${NUM_NODES}" ]; then
        echo "All ${NUM_NODES} nodes joined."
        break
    fi
    echo "  Waiting... (${RAY_NODES}/${NUM_NODES} nodes, attempt ${i}/60)"
    sleep 10
done

if [ "${RAY_NODES}" -lt "${NUM_NODES}" ]; then
    echo "Ray cluster only reached ${RAY_NODES}/${NUM_NODES} nodes."
    exit 1
fi

RUNTIME_ENV_JSON=$(cat <<EOF
{
  "env_vars": {
    "no_proxy": "${no_proxy}",
    "GLOO_SOCKET_IFNAME": "${MLP_SOCKET_IFNAME}",
    "TP_SOCKET_IFNAME": "${MLP_SOCKET_IFNAME}",
    "MASTER_ADDR": "${MASTER_ADDR}",
    "PYTHONPATH": "${MEGATRON_LM_DIR}",
    "NCCL_CUMEM_ENABLE": "0",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NVTE_BWD_LAYERNORM_SM_MARGIN": "20",
    "NCCL_IB_HCA": "mlx5",
    "NCCL_IB_TC": "160",
    "NCCL_PXN_DISABLE": "0",
    "NCCL_IB_GID_INDEX": "3",
    "NCCL_NET_GDR_LEVEL": "4",
    "NCCL_IB_RETRY_CNT": "7",
    "NCCL_IB_TIMEOUT": "32",
    "NCCL_IB_QPS_PER_CONNECTION": "8",
    "NCCL_P2P_LEVEL": "NVL",
    "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}",
    "NCCL_MIN_CTAS": "4",
    "OMPI_MCA_pml": "ob1",
    "OMPI_MCA_btl": "^openib",
    "OMPI_MCA_routed": "direct",
    "OMPI_MCA_routed_radix": "1024",
    "OMPI_MCA_plm_rsh_no_tree_spawn": "1",
    "OMPI_MCA_oob_tcp_if_include": "${MLP_SOCKET_IFNAME}",
    "OMPI_MCA_btl_tcp_if_include": "${MLP_SOCKET_IFNAME}"
  }
}
EOF
)

ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
   --working-dir="${WORKING_DIR}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes ${NUM_NODES} \
   --actor-num-gpus-per-node ${GPUS_PER_NODE} \
   --colocate \
   --calculate-per-token-loss \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${LORA_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${ALGO_ARGS[@]} \
   ${MLFLOW_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${ROLLOUT_ARGS[@]}
