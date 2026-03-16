# LoRA Fine-Tuning with Megatron Backend

LoRA (Low-Rank Adaptation) support for dense and MoE models via the Megatron training backend, ported from miles PRs #409/#684 (pending SGLang #20371)

## Architecture & Dependency Tree

```
limes
├── Megatron-LM (Osmosis-AI/Megatron-LM@miles-main)
│   └── Has `from miles.xxx` imports → resolved via symlink to slime (later limes)
├── Megatron-Bridge (Osmosis-AI/Megatron-Bridge@merged-qwen35-lora)
│   ├── HF ↔ Megatron weight conversion
│   └── export_adapter_weights() — LoRA weight gathering for SGLang sync
├── SGLang v0.5.9 (slimerl/sglang:v0.5.9)
│   ├── Inference engine with LoRA support (load_lora_adapter_from_tensors)
│   ├── Memory saver (release/resume_memory_occupation)
│   └── Patched: hf_text_config for composite models
├── torch_memory_saver
│   └── CPU backup/restore of GPU training tensors during offload (TEMP UNTIL SGLANG ISSUE IS RESOLVED)
└── transformers >= 5.2.0
    └── Native qwen3_5_moe model type
```

## LoRA Weight Sync Flow

During training, LoRA adapter weights are synced from Megatron training GPUs to SGLang inference engines:

1. **Train**: Megatron trains LoRA adapters (A, B matrices) on training GPUs
2. **Export**: `export_adapter_weights()` (Megatron-Bridge) gathers LoRA params across TP ranks
3. **Serialize**: CUDA IPC serialization to SGLang workers (tensors kept alive until next `update_weights`)
4. **Load**: SGLang calls `load_lora_adapter_from_tensors` to hot-load the adapter
5. **Offload safety**: Base model weights preserved via `enable_weights_cpu_backup` during offload cycles

The full-model weight sync path (`update_weights_from_tensor`) is skipped for LoRA only adapter weights are transferred.

## Key Fixes

Bugs discovered and fixed during testing on Qwen3.5-27B (dense) and Qwen3.5-35B-A3B (MoE):

| Bug | Root Cause | Fix | File |
|---|---|---|---|
| Corrupted rollouts after 1st iteration | `enable_weights_cpu_backup` missing with base weights destroyed on `release_memory_occupation` | Add `"enable_weights_cpu_backup": args.offload_rollout` | `sglang_engine.py` |
| CUDA IPC race condition | IPC tensors freed before SGLang TP workers could deserialize them | Store as `self._prev_ipc_tensors`, free at next `update_weights` | `update_weight_from_tensor.py` |
| Router marks workers DEAD during training | Health checks run while workers are under GPU memory pressure → false positives | Start health checks paused, never resume| `router.py` |
| SGLang LoRA crash on MoE models | `LoRAManager` receives composite `hf_config` → missing `num_hidden_layers` | `sed` patch: `hf_config` → `hf_text_config` | Dockerfile |
| EAGLE spec decoding + LoRA crash | SGLang only supports NGRAM spec decoding with LoRA | Documented constraint on SGLang repo, use NGRAM only | `sglang_engine.py` |

## Usage

### Docker Build

```bash
docker build -f docker/Dockerfile.qwen35-lora -t slimerl/slime:qwen35-lora .
```

### Run Scripts

| Script | Model | Config |
|---|---|---|
| `examples/lora/run-qwen3.5-27B-megatron-lora.sh` | Qwen3.5-27B (dense) | TP=4, LoRA rank=32 |
| `examples/lora/run-qwen3.5-35B-A3B-megatron-lora.sh` | Qwen3.5-35B-A3B (MoE) | TP=2, EP=8, 256 experts |

### Key Arguments

```bash
--lora-rank 32              
--lora-alpha 16             
--lora-dropout 0.0          
--lora-type lora           
--target-modules all-linear  
--exclude-modules ""        
--lora-adapter-path /path   
```

### MLflow Tracking

Experiments are logged to MLflow when `--tracking mlflow` is set:

```bash
python3 -m mlflow ui --host 0.0.0.0 --port 5000

ssh -L 5000:<container-ip>:5000 -N user@host
```

## CURRENT Known Limitations

- **MoE LoRA targets non-expert layers only**: SGLang doesn't yet support LoRA on expert FFN layers. Adapters target attention (QKV, O) and shared expert layers.
- **Speculative decoding**: Only NGRAM is compatible with LoRA. EAGLE/MTP spec decoding will crash.
- **Router health checks disabled**: Router operates as a pure load balancer with no dead worker detection. `RolloutHealthMonitor` handles fault tolerance separately at the Ray level.
- **Distributed LoRA sync**: Not yet implemented. LoRA weight sync only works with colocated engines (`--colocate`).

## File Reference

| File | Purpose |
|---|---|
| `slime/backends/megatron_utils/lora_utils.py` | LoRA core: adapter creation, checkpoint save/load, module name mapping |
| `slime/backends/megatron_utils/bridge_lora_helpers.py` | Megatron-Bridge integration, Qwen3.5 bridge registration |
| `slime/backends/megatron_utils/model.py` | Model init with LoRA branch |
| `slime/backends/megatron_utils/actor.py` | Training actor with LoRA offload handling |
| `slime/backends/megatron_utils/checkpoint.py` | LoRA checkpoint save/load wiring |
| `slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py` | LoRA weight sync via CUDA IPC |
| `slime/backends/sglang_utils/sglang_engine.py` | SGLang engine with LoRA + offload support |
| `slime/utils/arguments.py` | LoRA CLI arguments |
| `docker/Dockerfile.qwen35-lora` | Full build with all dependencies |
