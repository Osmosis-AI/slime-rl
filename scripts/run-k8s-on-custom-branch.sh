#!/usr/bin/env bash
#
# Submit a single-node Qwen3.5-35B-A3B LoRA test training job to k8s.
#
# Usage:
#   bash scripts/run-k8s-on-custom-branch.sh --repo https://github.com/Osmosis-AI/slime-rl [--branch dev] [--dry-run]
#
# Prerequisites:
#   - kubectl configured for the target cluster
#   - Weka PVC "weka-data" in namespace "trainers" with:
#       /data/test-lora-gsm8k/Qwen3.5-35B-A3B  (model weights, symlink ok)
#       /data/test-lora-gsm8k/gsm8k/{train,test}.parquet
#   - Docker image: osmosisdocker/limes:dev (or override with IMAGE=...)
#
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"

if [[ -f "${ENV_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
    set +a
fi

REQUIRED_ENV_VARS=(
    MLFLOW_URI
    MLFLOW_USER
    MLFLOW_PASS
)

MISSING_ENV_VARS=()
for env_var in "${REQUIRED_ENV_VARS[@]}"; do
    if [[ -z "${!env_var:-}" ]]; then
        MISSING_ENV_VARS+=("${env_var}")
    fi
done

if (( ${#MISSING_ENV_VARS[@]} > 0 )); then
    echo "Warning: missing required environment variables in ${ENV_FILE}:" >&2
    printf '  - %s\n' "${MISSING_ENV_VARS[@]}" >&2
    echo "Stopping before submission so we do not waste cluster time." >&2
    exit 1
fi

NAMESPACE="trainers"
IMAGE="${IMAGE:-osmosisdocker/limes:dev}"
JOB_NAME="local-testrun-lora-$(date +%Y%m%d-%H%M%S)"
DRY_RUN=""
PRIORITY_CLASS="${PRIORITY_CLASS:-skypilot-high-priority}"
RAY_PORT="${RAY_PORT:-6381}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8266}"
SLIME_REPO_URL="${SLIME_REPO_URL:-https://github.com/Osmosis-AI/slime-rl}"
SLIME_REPO_BRANCH="${SLIME_REPO_BRANCH:-}"
SLIME_RUNTIME_DIR="${SLIME_RUNTIME_DIR:-/root/slime-rl-runtime}"

usage() {
    cat <<EOF
Usage:
  bash scripts/run-k8s-on-custom-branch.sh --repo <git-repo-url> [--branch <branch>] [--dry-run]

Examples:
  bash scripts/run-k8s-on-custom-branch.sh --repo https://github.com/Osmosis-AI/slime-rl
  bash scripts/run-k8s-on-custom-branch.sh --repo https://github.com/Osmosis-AI/slime-rl --branch dev
  bash scripts/run-k8s-on-custom-branch.sh --repo https://github.com/Osmosis-AI/slime-rl/tree/dev

Environment overrides:
  SLIME_REPO_URL     Git repository URL to clone inside the pod
  SLIME_REPO_BRANCH  Branch to clone inside the pod (default: dev)
  SLIME_RUNTIME_DIR  Clone destination inside the pod (default: /root/slime-rl-runtime)
  CHECKPOINT_DIR     Checkpoint root on the mounted Weka volume
  SAVE_INTERVAL      Save cadence in rollouts (default: 1)
EOF
}

normalize_repo_spec() {
    SLIME_REPO_URL="${SLIME_REPO_URL%/}"

    if [[ "${SLIME_REPO_URL}" =~ ^https://github\.com/([^/]+)/([^/]+)/tree/(.+)$ ]]; then
        local owner="${BASH_REMATCH[1]}"
        local repo="${BASH_REMATCH[2]}"
        local branch_from_url="${BASH_REMATCH[3]}"
        SLIME_REPO_URL="https://github.com/${owner}/${repo}.git"
        if [[ -z "${SLIME_REPO_BRANCH}" ]]; then
            SLIME_REPO_BRANCH="${branch_from_url}"
        fi
    elif [[ "${SLIME_REPO_URL}" =~ ^https://github\.com/[^/]+/[^/]+$ ]]; then
        SLIME_REPO_URL="${SLIME_REPO_URL}.git"
    fi

    if [[ -z "${SLIME_REPO_BRANCH}" ]]; then
        SLIME_REPO_BRANCH="dev"
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo)
            [[ $# -ge 2 ]] || { echo "--repo requires a value" >&2; exit 1; }
            SLIME_REPO_URL="$2"
            shift 2
            ;;
        --branch)
            [[ $# -ge 2 ]] || { echo "--branch requires a value" >&2; exit 1; }
            SLIME_REPO_BRANCH="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="--dry-run=client"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

normalize_repo_spec

if [[ -n "${DRY_RUN}" ]]; then
    echo "==> Dry-run mode (will not submit)"
fi

# This assumes the weka volume already has the necessary data in place. Adjust paths as needed.
DATA_ROOT="/data/test-lora-gsm8k"
MODEL_PATH="${DATA_ROOT}/Qwen3.5-35B-A3B"
TRAIN_DATA="${DATA_ROOT}/gsm8k/train.parquet"
TEST_DATA="${DATA_ROOT}/gsm8k/test.parquet"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${DATA_ROOT}/checkpoints/${JOB_NAME}}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
NUM_ROLLOUT="${NUM_ROLLOUT:-100}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-10240}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-10240}"

MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-slime-test-lora}"
MLFLOW_RUN_NAME="${MLFLOW_RUN_NAME:-${JOB_NAME}}"
MLFLOW_EXPERIMENT_ID=""
MLFLOW_RUN_ID=""
MLFLOW_RUN_URL=""
STOP_JOB_CMD="kubectl delete job -n ${NAMESPACE} ${JOB_NAME}"

create_mlflow_run() {
    MLFLOW_URI="${MLFLOW_URI}" \
    MLFLOW_USER="${MLFLOW_USER}" \
    MLFLOW_PASS="${MLFLOW_PASS}" \
    MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME}" \
    MLFLOW_RUN_NAME="${MLFLOW_RUN_NAME}" \
    JOB_NAME="${JOB_NAME}" \
    NAMESPACE="${NAMESPACE}" \
    SLIME_REPO_URL="${SLIME_REPO_URL}" \
    SLIME_REPO_BRANCH="${SLIME_REPO_BRANCH}" \
    python3 - <<'PY'
import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request

base = os.environ["MLFLOW_URI"].rstrip("/")
user = os.environ["MLFLOW_USER"]
password = os.environ["MLFLOW_PASS"]
experiment_name = os.environ["MLFLOW_EXPERIMENT_NAME"]
run_name = os.environ["MLFLOW_RUN_NAME"]
job_name = os.environ["JOB_NAME"]
namespace = os.environ["NAMESPACE"]
repo_url = os.environ["SLIME_REPO_URL"]
repo_branch = os.environ["SLIME_REPO_BRANCH"]

auth = base64.b64encode(f"{user}:{password}".encode()).decode()
headers = {
    "Authorization": f"Basic {auth}",
    "Content-Type": "application/json",
}


def request(url: str, *, method: str = "GET", payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)


def get_or_create_experiment_id() -> str:
    url = (
        f"{base}/api/2.0/mlflow/experiments/get-by-name"
        f"?experiment_name={urllib.parse.quote(experiment_name, safe='')}"
    )
    try:
        return request(url)["experiment"]["experiment_id"]
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        return request(
            f"{base}/api/2.0/mlflow/experiments/create",
            method="POST",
            payload={"name": experiment_name},
        )["experiment_id"]


experiment_id = get_or_create_experiment_id()
run = request(
    f"{base}/api/2.0/mlflow/runs/create",
    method="POST",
    payload={
        "experiment_id": experiment_id,
        "run_name": run_name,
        "tags": [
            {"key": "mlflow.runName", "value": run_name},
            {"key": "job_name", "value": job_name},
            {"key": "k8s.namespace", "value": namespace},
            {"key": "git.repo_url", "value": repo_url},
            {"key": "git.branch", "value": repo_branch},
        ],
    },
)

print(experiment_id)
print(run["run"]["info"]["run_id"])
PY
}

delete_mlflow_run() {
    [[ -n "${MLFLOW_RUN_ID}" ]] || return 0
    MLFLOW_URI="${MLFLOW_URI}" \
    MLFLOW_USER="${MLFLOW_USER}" \
    MLFLOW_PASS="${MLFLOW_PASS}" \
    MLFLOW_RUN_ID="${MLFLOW_RUN_ID}" \
    python3 - <<'PY'
import base64
import json
import os
import urllib.request

base = os.environ["MLFLOW_URI"].rstrip("/")
user = os.environ["MLFLOW_USER"]
password = os.environ["MLFLOW_PASS"]
run_id = os.environ["MLFLOW_RUN_ID"]

auth = base64.b64encode(f"{user}:{password}".encode()).decode()
headers = {
    "Authorization": f"Basic {auth}",
    "Content-Type": "application/json",
}
payload = json.dumps({"run_id": run_id}).encode()
req = urllib.request.Request(
    f"{base}/api/2.0/mlflow/runs/delete",
    data=payload,
    headers=headers,
    method="POST",
)
with urllib.request.urlopen(req):
    pass
PY
}

if [[ -z "${DRY_RUN}" ]]; then
    mapfile -t _MLFLOW_VALUES < <(create_mlflow_run)
    MLFLOW_EXPERIMENT_ID="${_MLFLOW_VALUES[0]}"
    MLFLOW_RUN_ID="${_MLFLOW_VALUES[1]}"
    if [[ -z "${MLFLOW_EXPERIMENT_ID}" || -z "${MLFLOW_RUN_ID}" ]]; then
        echo "Failed to pre-create MLflow run" >&2
        exit 1
    fi
    MLFLOW_RUN_URL="${MLFLOW_URI%/}/#/experiments/${MLFLOW_EXPERIMENT_ID}/runs/${MLFLOW_RUN_ID}"
fi

echo "==> Submitting job: ${JOB_NAME}"
echo "    namespace:  ${NAMESPACE}"
echo "    image:      ${IMAGE}"
echo "    repo:       ${SLIME_REPO_URL}"
echo "    branch:     ${SLIME_REPO_BRANCH}"
echo "    model:      ${MODEL_PATH}"
echo "    train data: ${TRAIN_DATA}"
echo "    save dir:   ${CHECKPOINT_DIR}"
echo "    save every: ${SAVE_INTERVAL} rollouts"
echo "    num rollout:${NUM_ROLLOUT}"
echo "    rollout bs: ${ROLLOUT_BATCH_SIZE}"
echo "    samples/pr: ${N_SAMPLES_PER_PROMPT}"
echo "    global bs:  ${GLOBAL_BATCH_SIZE}"
echo "    resp len:   ${ROLLOUT_MAX_RESPONSE_LEN}"
echo "    eval every: ${EVAL_INTERVAL}"
echo "    mlflow exp: ${MLFLOW_EXPERIMENT_NAME}"
echo "    mlflow run: ${MLFLOW_RUN_NAME}"
echo "    stop job:   ${STOP_JOB_CMD}"
if [[ -n "${MLFLOW_RUN_URL}" ]]; then
    echo "    mlflow url: ${MLFLOW_RUN_URL}"
elif [[ -n "${DRY_RUN}" ]]; then
    echo "    mlflow url: exact URL is created on real submission"
fi

TMPFILE=$(mktemp /tmp/k8s-test-job.XXXXXX.yaml)
trap "rm -f ${TMPFILE}" EXIT

cat > "${TMPFILE}" <<ENDOFYAML
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_NAME}
  namespace: ${NAMESPACE}
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 259200
  template:
    spec:
      restartPolicy: Never
      hostNetwork: true
      priorityClassName: ${PRIORITY_CLASS}
      containers:
      - name: trainer
        image: ${IMAGE}
        imagePullPolicy: Always
        command: ["/bin/bash", "-c"]
        args:
        - |
          set -ex

          export FLASHINFER_DISABLE_VERSION_CHECK=1
          export GPUS_PER_NODE=8
          export PYTHONPATH="${SLIME_RUNTIME_DIR}:/root/Megatron-LM/"
          export CUDA_DEVICE_MAX_CONNECTIONS=1
          export NCCL_NVLS_ENABLE=1
          export SGLANG_DISABLE_CUDNN_CHECK=1
          export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true

          pkill -9 sglang || true; sleep 2
          ray stop --force || true
          pkill -9 ray || true; pkill -9 python || true; sleep 2

          command -v git >/dev/null 2>&1
          rm -rf "${SLIME_RUNTIME_DIR}"
          git clone --depth 1 --single-branch --branch "${SLIME_REPO_BRANCH}" "${SLIME_REPO_URL}" "${SLIME_RUNTIME_DIR}"
          cd "${SLIME_RUNTIME_DIR}"
          GIT_COMMIT=\$(git rev-parse HEAD)
          echo "Using slime-rl repo ${SLIME_REPO_URL} branch ${SLIME_REPO_BRANCH} commit \${GIT_COMMIT}"

          source "${SLIME_RUNTIME_DIR}/scripts/models/qwen3.5-35B-A3B.sh"

          export MASTER_ADDR=127.0.0.1
          export no_proxy="127.0.0.1"
          ray start --head --node-ip-address 127.0.0.1 \
              --num-gpus \$GPUS_PER_NODE \
              --port=${RAY_PORT} \
              --disable-usage-stats \
              --dashboard-host=0.0.0.0 --dashboard-port=${RAY_DASHBOARD_PORT}

          RUNTIME_ENV_JSON='{
            "env_vars": {
              "PYTHONPATH": "${SLIME_RUNTIME_DIR}:/root/Megatron-LM/",
              "CUDA_DEVICE_MAX_CONNECTIONS": "1",
              "NCCL_NVLS_ENABLE": "1",
              "SGLANG_DISABLE_CUDNN_CHECK": "1",
              "no_proxy": "127.0.0.1",
              "MLFLOW_TRACKING_URI": "${MLFLOW_URI}",
              "MLFLOW_TRACKING_USERNAME": "${MLFLOW_USER}",
              "MLFLOW_TRACKING_PASSWORD": "${MLFLOW_PASS}",
              "MLFLOW_RUN_ID": "${MLFLOW_RUN_ID}",
              "MLFLOW_RUN_URL": "${MLFLOW_RUN_URL}",
              "SLIME_REPO_URL": "${SLIME_REPO_URL}",
              "SLIME_REPO_BRANCH": "${SLIME_REPO_BRANCH}"
            }
          }'

          ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
            --runtime-env-json="\${RUNTIME_ENV_JSON}" \
            -- python3 ${SLIME_RUNTIME_DIR}/train.py \
            --actor-num-nodes 1 \
            --actor-num-gpus-per-node \$GPUS_PER_NODE \
            --colocate \
            --calculate-per-token-loss \
            \${MODEL_ARGS[@]} \
            --hf-checkpoint ${MODEL_PATH} \
            --megatron-to-hf-mode bridge \
            --lora-rank 32 \
            --lora-alpha 32 \
            --lora-dropout 0.0 \
            --target-modules "all-linear" \
            --save ${CHECKPOINT_DIR} \
            --save-interval ${SAVE_INTERVAL} \
            --prompt-data ${TRAIN_DATA} \
            --input-key messages \
            --label-key label \
            --apply-chat-template \
            --rollout-shuffle \
            --rm-type math \
            --num-rollout ${NUM_ROLLOUT} \
            --rollout-batch-size ${ROLLOUT_BATCH_SIZE} \
            --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT} \
            --rollout-max-response-len ${ROLLOUT_MAX_RESPONSE_LEN} \
            --rollout-temperature 1 \
            --global-batch-size ${GLOBAL_BATCH_SIZE} \
            --eval-interval ${EVAL_INTERVAL} \
            --eval-prompt-data gsm8k ${TEST_DATA} \
            --n-samples-per-eval-prompt 1 \
            --eval-max-response-len ${EVAL_MAX_RESPONSE_LEN} \
            --eval-top-k 1 \
            --skip-eval-before-train \
            --tensor-model-parallel-size 2 \
            --sequence-parallel \
            --pipeline-model-parallel-size 1 \
            --context-parallel-size 1 \
            --expert-model-parallel-size 8 \
            --expert-tensor-parallel-size 1 \
            --recompute-granularity full \
            --recompute-method uniform \
            --recompute-num-layers 1 \
            --qkv-format bshd \
            --micro-batch-size 1 \
            --advantage-estimator gspo \
            --kl-loss-coef 0.00 \
            --kl-loss-type low_var_kl \
            --kl-coef 0.00 \
            --entropy-coef 0.00 \
            --eps-clip 4e-4 \
            --optimizer adam \
            --lr 5e-5 \
            --clip-grad 1.0 \
            --lr-decay-style constant \
            --weight-decay 0.1 \
            --adam-beta1 0.9 \
            --adam-beta2 0.98 \
            --optimizer-cpu-offload \
            --overlap-cpu-optimizer-d2h-h2d \
            --use-precision-aware-optimizer \
            --use-mlflow \
            --mlflow-experiment-name ${MLFLOW_EXPERIMENT_NAME} \
            --mlflow-run-name ${MLFLOW_RUN_NAME} \
            --rollout-num-gpus-per-engine 8 \
            --sglang-mem-fraction-static 0.7 \
            --sglang-ep-size 8 \
            --sglang-cuda-graph-bs 1 2 4 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 136 144 152 160 168 176 184 192 200 208 216 224 232 240 248 256 \
            --sglang-max-running-requests 64 \
            --offload-train \
            --attention-dropout 0.0 \
            --hidden-dropout 0.0 \
            --accumulate-allreduce-grads-in-fp32 \
            --attention-softmax-in-fp32 \
            --attention-backend flash \
            --moe-token-dispatcher-type alltoall
        securityContext:
          privileged: true
        resources:
          requests:
            cpu: "96"
            memory: "800Gi"
            nvidia.com/gpu: "8"
          limits:
            nvidia.com/gpu: "8"
        env:
        - name: MLFLOW_TRACKING_URI
          value: "${MLFLOW_URI}"
        - name: MLFLOW_TRACKING_USERNAME
          value: "${MLFLOW_USER}"
        - name: MLFLOW_TRACKING_PASSWORD
          value: "${MLFLOW_PASS}"
        - name: MLFLOW_RUN_ID
          value: "${MLFLOW_RUN_ID}"
        - name: MLFLOW_RUN_URL
          value: "${MLFLOW_RUN_URL}"
        - name: HF_HOME
          value: "/data/hf_cache"
        volumeMounts:
        - name: weka-data
          mountPath: /data
        - name: dshm
          mountPath: /dev/shm
      volumes:
      - name: weka-data
        persistentVolumeClaim:
          claimName: weka-data
      - name: dshm
        emptyDir:
          medium: Memory
          sizeLimit: 256Gi
ENDOFYAML

if ! kubectl apply ${DRY_RUN} -n "${NAMESPACE}" -f "${TMPFILE}"; then
    if [[ -z "${DRY_RUN}" ]]; then
        delete_mlflow_run || true
    fi
    exit 1
fi

if [[ -z "${DRY_RUN}" ]]; then
    echo ""
    echo "==> Job submitted. Monitor with:"
    echo "    kubectl get pods -n ${NAMESPACE} -l job-name=${JOB_NAME}"
    echo "    kubectl logs -n ${NAMESPACE} -l job-name=${JOB_NAME} -f"
    echo "    ${STOP_JOB_CMD}"
    echo "    ${MLFLOW_RUN_URL}"
fi
