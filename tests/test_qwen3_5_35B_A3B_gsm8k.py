import os

import slime.utils.external_utils.command_utils as U


ENABLE_EVAL = bool(int(os.environ.get("SLIME_TEST_ENABLE_EVAL", "1")))
TIGHT_HOST_MEMORY = bool(int(os.environ.get("SLIME_TEST_TIGHT_HOST_MEMORY", "1")))
USE_DEEPEP = bool(int(os.environ.get("SLIME_TEST_USE_DEEPEP", "0")))
USE_FP8_ROLLOUT = bool(int(os.environ.get("SLIME_TEST_USE_FP8_ROLLOUT", "0")))

# MODEL_NAME = "Qwen3.5-35B-A3B"
# MODEL_TYPE = "qwen3.5-35B-A3B"
MODEL_NAME = "Qwen3.5-4B"
MODEL_TYPE = "qwen3.5-4B"
NUM_GPUS = 8


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    if USE_FP8_ROLLOUT:
        U.exec_command(f"hf download Qwen/{MODEL_NAME}-FP8 --local-dir /root/models/{MODEL_NAME}-FP8")
    U.hf_download_dataset("zhuzilin/gsm8k")


def execute():
    if USE_FP8_ROLLOUT:
        ckpt_args = (
            f"--hf-checkpoint /root/models/{MODEL_NAME}-FP8 "
            "--megatron-to-hf-mode bridge "
        )
    else:
        ckpt_args = (
            f"--hf-checkpoint /root/models/{MODEL_NAME} "
            "--megatron-to-hf-mode bridge "
        )

    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 1 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 2 "
        "--rollout-max-response-len 512 "
        "--rollout-temperature 1 "
        "--global-batch-size 8 "
        "--balance-data "
    )

    eval_args = (
        f"{'--eval-interval 20 ' if ENABLE_EVAL else ''}"
        "--eval-prompt-data gsm8k /root/datasets/gsm8k/test.parquet "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-response-len 4096 "
        "--eval-top-p 1 "
    )

    perf_args = (
        "--tensor-model-parallel-size 2 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 8 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 20480 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 8 "
        "--sglang-mem-fraction-static 0.5 "
        "--sglang-ep-size 8 "
        "--sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256) "
        "--sglang-speculative-algorithm EAGLE "
        "--sglang-speculative-num-steps 2 "
        "--sglang-speculative-eagle-topk 1 "
        "--sglang-speculative-num-draft-tokens 3 "
        "--sglang-max-running-requests 512 "
    )

    if USE_DEEPEP:
        sglang_args += "--sglang-moe-a2a-backend deepep --sglang-deepep-mode auto "

    ci_args = "--ci-test "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 8 "
        "--colocate "
        "--moe-token-dispatcher-type flex "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{ci_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
