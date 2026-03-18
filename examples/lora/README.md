# LoRA

LoRA + GRPO on MoE/dense models via the Megatron backend with Bridge weight sync.

## Docker

Build from `docker/Dockerfile.osmosis`:

```bash
docker build -f docker/Dockerfile.osmosis -t slimerl/slime:osmosis .
```

What's in the image:

| Component | Source | What it does |
|-----------|--------|--------------|
| SGLang v0.5.9 | `slimerl/sglang:v0.5.9` base | Rollout inference |
| Megatron-LM | `Osmosis-AI/Megatron-LM` | Training backend |
| Megatron-Bridge | `Osmosis-AI/Megatron-Bridge` | Megatron<->HF weight conversion for LoRA sync |
| torch_memory_saver | fzyzcjy/torch_memory_saver | CPU offload of base weights |
| flash-attn 2.8.3 + FA3 | Dao-AILab/flash-attention | Attention kernels |
| TransformerEngine 2.12 | NVIDIA | Fused ops |
| MLflow | PyPI | Experiment tracking |

The Dockerfile also creates a `miles -> slime` symlink in site-packages since Megatron-LM still has hardcoded `from miles.xxx` imports.

### SGLang LoRA patches

SGLang v0.5.9 has core LoRA support. Patch status for MoE:

**Merged in v0.5.9:**
- #17692 — LoRA weight loading
- #19710 — adapter management

**Auto-applied by Dockerfile (sed):**
- Composite config — `hf_text_config` instead of `hf_config` for `LoRAManager`. Qwen3.5 MoE puts `num_hidden_layers`/`hidden_size`/`vocab_size` inside `text_config`, not at the top level.

**Not applied (open upstream, saved in `docker/patch/`):**
- `sglang-lora-14105-moe-layers.patch` — CUDA kernels for expert-layer LoRA
- `sglang-lora-18511-tp-embedding.patch` — TP embedding LoRA (TP>1)
- `sglang-lora-19711-moe-triton-kernel.patch` — Triton MoE LoRA kernel (alternative to 14105)

These are only needed if you want LoRA on the expert layers themselves. Without them, LoRA targets attention + shared expert + gates, which is the default.

## Setup

```bash
# launch
docker run -d --name slime-lora --gpus all --ipc=host --network=host \
  -v /path/to/Qwen3.5-35B-A3B:/root/Qwen3.5-35B-A3B \
  slimerl/slime:osmosis sleep infinity

# prep GSM8K (downloads from HF, outputs chat-format parquet)
docker exec slime-lora python examples/lora/prep_gsm8k.py

# run
docker exec -it slime-lora bash -c "cd /root/slime && bash examples/lora/run-qwen3.5-35B-A3B-megatron-lora.sh"
```

`--ipc=host` is required — CUDA IPC between colocated training/inference fails without it.

## Run scripts

| Script | Model | GPUs | Notes |
|--------|-------|------|-------|
| `run-qwen3.5-35B-A3B-megatron-lora.sh` | Qwen3.5-35B-A3B (MoE) | 8 | TP=2, EP=8, `--offload-train` |
| `run-qwen3.5-35B-A3B-megatron-baseline.sh` | Qwen3.5-35B-A3B (MoE) | 8 | Full FT, `--moe-token-dispatcher-type flex` |
| `run-qwen3.5-27B-megatron-lora.sh` | Qwen3.5-27B (dense) | 4 | TP=4 |
| `run-qwen3.5-4B-megatron-lora.sh` | Qwen3.5-4B | 2 | Quick test |
| `run-qwen2.5-0.5B-megatron-lora.sh` | Qwen2.5-0.5B | 1 | Smoke test |
| `benchmark-qwen3.5-35B-A3B.sh` | Qwen3.5-35B-A3B (MoE) | 8 | LoRA + Full FT back-to-back, MLflow |

## LoRA args

Add to any run script to enable LoRA:

```bash
--lora-rank 32                 # 0 = disabled
--lora-alpha 32                # scaling, usually same as rank
--lora-dropout 0.0
--target-modules "all-linear"  # or comma-separated layer names
```

Optional:
```bash
--exclude-modules "..."        # comma-separated, skip these layers
--lora-adapter-path /path/...  # resume from saved adapter
--offload-train                # CPU-offload base weights (use with --colocate)
```

## LoRA vs Full FT

| | LoRA | Full FT |
|---|---|---|
| Trainable params | ~0.5-1% | 100% |
| GPU mem / device | ~10 GB | ~96 GB (35B MoE) |
| `--offload-train` | yes (recommended) | **no** (OOM with `--colocate`) |
| `--moe-token-dispatcher-type` | `alltoall` | `flex` |
| LR | 5e-5 | 1e-5 |

## MLflow

Scripts with `--use-mlflow` log automatically. Start the UI:

```bash
# inside container
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
```

SSH tunnel: `ssh -N -L 5000:localhost:5000 user@host`, then `http://localhost:5000`.

## Benchmark

```bash
bash examples/lora/benchmark-qwen3.5-35B-A3B.sh
```

Runs LoRA then Full FT, both log to the same MLflow experiment. Generate report:

```bash
python tools/benchmark_report.py \
    --experiment slime-osmosis-benchmark \
    --lora-run lora-r32-benchmark \
    --baseline-run full-ft-benchmark
```

## Gotchas

- `--offload-train` + Full FT + `--colocate` = OOM. `torch_memory_saver` can't resume ~96GB base weights alongside SGLang. LoRA only.
- SGLang LoRA only supports NGRAM speculative decoding, not EAGLE/MTP.
- MoE LoRA wraps non-expert layers only (attention, shared expert, gates). Expert-layer LoRA needs the optional patches above.
- `--ipc=host` is mandatory. Without it CUDA IPC silently fails.
- New containers don't have data — run `prep_gsm8k.py` after every fresh container.
