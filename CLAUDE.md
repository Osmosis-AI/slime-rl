# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What Is This

Fork of [THUDM/slime](https://github.com/THUDM/slime) — an RL post-training framework combining Megatron-LM (training) with SGLang (rollout inference). Our fork adds **LoRA support via Megatron-Bridge** and **FP8 training** for MoE models (Qwen3.5 family).

Maintained at two remotes:
- `origin` → `mouad-hpc/slime` (personal)
- `slime-rl` → `Osmosis-AI/slime-rl` (team, branch `dev`)

## Build and Run

```bash
# Install (editable, with Megatron on PYTHONPATH)
pip install -e .
export PYTHONPATH=/path/to/Megatron-LM:$PYTHONPATH

# Run training (always via Ray job on GPU nodes)
ray job submit --address="http://127.0.0.1:8266" \
  --runtime-env-json='{"env_vars": {"PYTHONPATH": "/root/Megatron-LM"}}' \
  -- python3 train.py <args>

# Lint
ruff check slime/
black --check --diff slime/  # line_length=119

# Tests (require GPU cluster, not runnable locally)
pytest tests/ -m unit
pytest tests/test_qwen2.5_0.5B_short.py  # single integration test
```

## Architecture Overview

```
train.py → Ray job
  ├── Actor (Megatron-LM): GRPO/PPO training
  │   ├── model_provider.py  — builds GPTModel via Megatron-Bridge or native Megatron
  │   ├── bridge_lora_helpers.py — LoRA injection via Bridge pre-wrap hooks
  │   ├── lora_utils.py — LoRA adapter creation, merge/unmerge, checkpoint
  │   ├── model.py — setup_model_and_optimizer(), forward/backward, save
  │   └── actor.py — MegatronTrainRayActor (init, train loop, weight sync)
  │
  ├── Rollout Engine (SGLang): generates completions with colocated GPU sharing
  │   ├── sglang_rollout.py — orchestrates rollout generation
  │   ├── sglang_engine.py — wraps SGLang server lifecycle
  │   └── update_weight/ — syncs model weights (or LoRA adapters) to SGLang
  │
  ├── Reward Models: rm_hub/ (deepscaler, f1, gpqa, ifbench, math)
  │
  └── Data Buffer: rollout/ (data_source.py, filter_hub/, generate_hub/)
```

### Training Flow

1. `train.py` → `parse_args()` → `create_placement_groups()` (Ray GPU bundles)
2. `create_rollout_manager()` → starts SGLang engines on assigned GPUs
3. `create_training_models()` → `MegatronTrainRayActor.init()` → `initialize_model_and_optimizer()`
4. Main loop: rollout (SGLang generates) → train (Megatron GRPO/PPO step) → `update_weights()` (sync to SGLang)
5. With `--colocate --offload-train`: GPUs shared between train and inference via `TorchMemorySaver` pause/resume

### Two Model Loading Paths

**Bridge path** (`--megatron-to-hf-mode bridge --hf-checkpoint <path>`): Loads HF weights directly via Megatron-Bridge. Supports LoRA. Does NOT automatically propagate FP8/parallelism config — must be set explicitly on the provider before `finalize()`.

**Native path** (`--load <megatron_checkpoint>`): Uses `core_transformer_config_from_args()` which maps all args automatically. No Bridge needed but requires pre-converted Megatron checkpoints.

### Weight Sync to SGLang

- **Full model**: `UpdateWeightFromDistributed` — NCCL broadcast from training ranks to SGLang workers
- **LoRA**: `UpdateWeightFromTensor` — serializes LoRA adapter weights, sends via Ray IPC, calls `load_lora_adapter_from_tensors`

### Parallelism

TP (tensor) + EP (expert) + PP (pipeline) + SP (sequence) + CP (context). Configured via CLI args. MoE models typically use TP=2, EP=8 on 8 GPUs.

## Key Files for Our Fork

| File | Role |
|------|------|
| `slime/backends/megatron_utils/bridge_lora_helpers.py` | Bridge + LoRA setup, Qwen3.5 bridge registration |
| `slime/backends/megatron_utils/lora_utils.py` | LoRA adapter creation, merge/unmerge, checkpoint |
| `slime/backends/megatron_utils/model_provider.py` | Model provider with Bridge path + native path |
| `slime/utils/arguments.py` | All CLI arguments including LoRA and FP8 flags |
| `scripts/models/*.sh` | Model configs (set `MODEL_ARGS` array) |
| `examples/lora/*.sh` | LoRA training scripts for various models |

## Example Scripts

```bash
# BF16 LoRA — Qwen3.5-122B-A10B (single node 8xH200)
bash examples/lora/run-qwen3.5-122B-A10B-megatron-lora.sh

# BF16 LoRA — Qwen3.5-35B-A3B
bash examples/lora/run-qwen3.5-35B-A3B-megatron-lora.sh
```

Key CLI flags:
- `--colocate --offload-train`: share GPUs between train and rollout
- `--lora-rank 64 --lora-alpha 64 --target-modules "all-linear"`: LoRA config
- `--megatron-to-hf-mode bridge --hf-checkpoint <path>`: Bridge mode (HF weights)
- `--log-probs-chunk-size 4096`: prevents OOM from large vocab logits
- `--transformer-impl transformer_engine --bf16`: TransformerEngine backend

## FP8 Benchmark Results

FP8 (`--fp8-format e4m3 --fp8-recipe blockwise`) was tested on both 35B and 122B MoE models. **FP8 does not help MoE models** — expert GEMMs are too small and memory-bandwidth-bound, not compute-bound. FP8 overhead (scaling factors, amax history) outweighs any compute benefit.

- **35B (FFN=512)**: FP8 is slower (+2s) and uses more memory (+13GB) vs BF16
- **122B (FFN=1024)**: BF16 gets 39-41 tflops vs FP8's 23-37 tflops. BF16 wins on every metric.

FP8 would only help dense models or MoE with much larger expert dimensions.

## Docker

Production container: `slime-osmosis` (based on `radixark/miles:dev` image)
- Megatron-Bridge: `Osmosis-AI/Megatron-Bridge@merged-qwen35-lora`
- SGLang LoRA patches: `slime/docker/patch/sglang-lora-*.patch`

## Code Style

- `black` with `line_length=119`
- `isort` with black-compatible profile
- `ruff` for linting (E, F, B, UP rules)

## Key Gotchas

- `--colocate` silently forces `offload_rollout=True`
- `--offload-train` + full FT + `--colocate` = OOM (LoRA only works here)
- deepscaler reward returns 0 if no `</think>` tag or no `\boxed{}`
- OOM root cause for large vocab: fp32 logits = `seq_len * vocab/TP * 4`
- Stale `__pycache__` causes phantom errors after `git reset --hard`
- 4096 max_response_len too short for Qwen3.5 thinking — use 12K+
- Bridge path does NOT auto-propagate FP8 or parallelism config from args — set manually on provider before `finalize()`
- 122B MoE does NOT fit on single-node 8xH200 with colocated train+inference — needs multi-node

## Tools

- Use Playwright MCP for browser automation and web interaction tasks
- Always use a subagent to review/verify your changes before committing or applying them
