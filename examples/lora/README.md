# LoRA Training with Slime

LoRA (Low-Rank Adaptation) for GRPO post-training on MoE and dense models using the Megatron backend with Bridge weight conversion.

## Quick Start

### 1. Build the Docker image

```bash
docker build -f docker/Dockerfile.osmosis -t slimerl/slime:osmosis .
```

### 2. Launch the container

```bash
docker run -d --name slime-lora --gpus all --ipc=host --network=host \
  -v /path/to/your/model:/root/model \
  slimerl/slime:osmosis sleep infinity
```

`--ipc=host` is required for CUDA IPC between colocated training and inference processes.

### 3. Prepare training data

```bash
docker exec slime-lora python examples/lora/prep_gsm8k.py
```

This downloads GSM8K from HuggingFace and converts it to the chat-format parquet that slime expects (columns: `messages`, `label`).

### 4. Run LoRA training

```bash
docker exec -it slime-lora bash
cd /root/slime
bash examples/lora/run-qwen3.5-35B-A3B-megatron-lora.sh
```

## Available Run Scripts

| Script | Model | GPUs | Notes |
|--------|-------|------|-------|
| `run-qwen3.5-35B-A3B-megatron-lora.sh` | Qwen3.5-35B-A3B (MoE) | 8 | TP=2, EP=8, `--offload-train` |
| `run-qwen3.5-35B-A3B-megatron-baseline.sh` | Qwen3.5-35B-A3B (MoE) | 8 | Full FT baseline, `--moe-token-dispatcher-type flex` |
| `run-qwen3.5-27B-megatron-lora.sh` | Qwen3.5-27B (dense) | 4 | TP=4, no MoE flags |
| `run-qwen3.5-4B-megatron-lora.sh` | Qwen3.5-4B | 2 | Small model for quick testing |
| `run-qwen2.5-0.5B-megatron-lora.sh` | Qwen2.5-0.5B | 1 | Smallest, good for CI/smoke tests |
| `benchmark-qwen3.5-35B-A3B.sh` | Qwen3.5-35B-A3B (MoE) | 8 | Runs LoRA + Full FT sequentially, logs to MLflow |

## LoRA-Specific Arguments

Add these to any existing slime run script to enable LoRA:

```bash
--lora-rank 32          # rank of the low-rank matrices (0 = disabled)
--lora-alpha 32         # scaling factor (typically equal to rank)
--lora-dropout 0.0      # dropout on LoRA layers
--target-modules "all-linear"  # which layers to wrap ("all-linear" or comma-separated)
```

Optional:
```bash
--exclude-modules "..."        # comma-separated modules to skip
--lora-adapter-path /path/...  # resume from a saved adapter checkpoint
--offload-train                # offload base weights to CPU during training (recommended for colocated runs)
```

## Key Differences: LoRA vs Full Fine-Tuning

| | LoRA | Full FT |
|---|---|---|
| Trainable params | ~0.5-1% | 100% |
| GPU memory per device | ~10 GB | ~96 GB (for 35B MoE) |
| `--offload-train` | Yes (recommended) | No (causes OOM with `--colocate`) |
| `--moe-token-dispatcher-type` | `alltoall` | `flex` (required for full FT) |
| Learning rate | 5e-5 | 1e-5 |
| Stability | Resistant to collapse | Needs KL regularization |

## Monitoring with MLflow

Run scripts with `--use-mlflow` log metrics automatically. To view:

```bash
# Inside the container
cd /root/slime
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
```

Then open `http://<host-ip>:5000` or use an SSH tunnel:
```bash
ssh -N -L 5000:localhost:5000 user@host
# then open http://localhost:5000
```

## Benchmarking LoRA vs Full FT

```bash
bash examples/lora/benchmark-qwen3.5-35B-A3B.sh
```

This runs both LoRA and Full FT sequentially and logs to the same MLflow experiment for comparison. After completion, generate a report:

```bash
python tools/benchmark_report.py \
    --experiment slime-osmosis-benchmark \
    --lora-run lora-r32-benchmark \
    --baseline-run full-ft-benchmark
```

## Gotchas

- **`--offload-train` + Full FT + `--colocate` = OOM**: `torch_memory_saver` cannot resume ~96 GB of base weights alongside SGLang. Only use `--offload-train` with LoRA.
- **Speculative decoding**: SGLang LoRA only supports NGRAM, not EAGLE/MTP.
- **MoE LoRA targets non-expert layers only**: expert weights are not wrapped with LoRA adapters.
- **`--ipc=host` is required**: without it, CUDA IPC tensors between training and inference fail silently.
- **New containers need data**: GSM8K must be re-downloaded — run `prep_gsm8k.py` after each fresh container.
