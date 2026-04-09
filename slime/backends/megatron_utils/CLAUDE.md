# Megatron Utils — Backend Internals

## Fused MoE Backward Kernels

### What
Custom Triton backward kernels for MoE expert layers, ported from Miles' FSDP backend into slime's Megatron backend. Replaces standard PyTorch autograd with fused kernels for the expert forward/backward pass.

### Files

| File | Role |
|------|------|
| `kernels/fused_moe_triton_backward_kernels.py` | 3 Triton JIT kernels: `backward_input`, `backward_weight`, `backward_topk_weights` |
| `kernels/fused_experts.py` | 4 `torch.autograd.Function` wrappers: GateUpProj, SiluAndMul, DownProj, MoeSumReduce |
| `kernels/fused_moe_integration.py` | Monkey-patches grouped expert modules via `patch_model_for_fused_moe()` |
| `kernels/__init__.py` | Empty — required for relative imports in the package |
| `bridge_lora_helpers.py` | Wires the fused MoE hook via `register_pre_wrap_hook` (after LoRA hook) |

### Pipeline

```
hidden_states
    |
    v
GateUpProjFunction    -- input @ w1.T (SGLang forward kernel, custom Triton backward)
    |                    backward: backward_input_kernel + backward_weight_kernel
    v
SiluAndMulFunction    -- SiLU(gate) * up (SGLang forward, manual PyTorch backward)
    v
DownProjFunction      -- input @ w2.T * topk_weights (SGLang forward, custom Triton backward)
    |                    backward: backward_input + backward_weight + backward_topk_weights
    v
MoeSumReduceFunction  -- sum over top-k experts (SGLang forward, expand backward)
    v
output_hidden_states
```

### Three Triton Backward Kernels

1. **`backward_input_kernel`** — grad w.r.t. hidden states
   - `grad_input[token] = sum(grad_output[token,expert] @ weight[expert])`
   - Uses `tl.atomic_add` for token accumulation (multiple experts -> same token)

2. **`backward_weight_kernel`** — grad w.r.t. expert weights
   - `grad_weight[expert] = input.T @ grad_output` (per expert)
   - Uses `tl.atomic_add` for expert accumulation (multiple token blocks -> same expert)
   - **Dominant bottleneck: 44% of total CUDA time**

3. **`backward_topk_weights_kernel`** — grad w.r.t. routing weights
   - Recomputes forward output in-kernel (`input @ weight.T`) to avoid storing it
   - Only runs for DownProj (`mul_routed_weight=True`)

### Activation and Guards

- CLI flag: `--use-fused-moe-backward`
- Guards in `slime/utils/arguments.py`: requires Bridge mode, BF16, EP=1, expert-TP=1
- Hook order in `bridge_lora_helpers.py`: LoRA hook first, then fused MoE hook
- LoRA on expert weights: currently raises RuntimeError (see LoRA section below)

### SGLang Forward Kernel Dependency

The forward pass uses SGLang's `invoke_fused_moe_kernel` and `moe_align_block_size`. Version-specific kwargs:
- GateUpProj passes `c_sorted=False, filter_expert=True`
- DownProj passes `a_use_tma=False, b_use_tma=False`

Confirmed working on SGLang 0.5.9 (`osmosisdocker/limes:latest`).

### Block Config

Static, no autotuning: `BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32, GROUP_SIZE_M=8`. No `num_warps` or `num_stages` set (Triton defaults: 4 warps, 2 stages). Chunk size: 64K tokens per iteration.

### Memory Characteristics

All three kernels are **memory-bandwidth-bound** at both model dimensions — expert GEMMs are tiny. The `tl.atomic_add` contention dominates, not compute.

| Model | H | FFN | E | topk | Layers | Expert GEMM size |
|-------|---|-----|---|------|--------|-----------------|
| Qwen3.5-35B-A3B | 2048 | 512 | 256 | 8 | 40 | 512x2048 |
| Qwen3.5-122B-A10B | 3072 | 1024 | 256 | 8 | 48 | 1024x3072 |

### Integration Pattern

`patch_model_for_fused_moe()` walks the model tree, finds modules with `linear_fc1` and `linear_fc2` attributes containing `weight<N>` parameters (grouped experts), and replaces their `forward` method. Parameter names are preserved — no renames, no new wrappers holding parameters. This keeps Bridge naming, LoRA injection, HF export, and weight sync contracts intact.

### Key Invariants

- `mlp.experts.linear_fc1` / `mlp.experts.linear_fc2` names must stay visible in `named_parameters()`
- Per-expert `weight<N>` suffixes must remain on the grouped linear modules
- Expert weights are stacked at runtime via `torch.stack` for the fused kernel, not pre-stacked
- The patched forward consumes Megatron's fused expert tensors directly (not HF `ModuleList`)

---

## Profiler Results (H200, Qwen3.5-35B shapes)

Config: T=4096, H=2048, FFN=512, E=64, topk=8. Total fwd+bwd per MoE layer: **9.4ms**.

| Kernel | CUDA time | % | Calls | Bottleneck |
|--------|----------|---|-------|-----------|
| `backward_weight` | 4.17ms | 44% | 2 | `tl.atomic_add` — 8 M-blocks per expert contend |
| `backward_input` | 2.49ms | 26% | 2 | `tl.atomic_add` — multiple experts write same token grad |
| `fused_moe_kernel` (fwd) | 0.76ms | 8% | 2 | SGLang, already optimized |
| `backward_topk_weights` | 0.44ms | 5% | 1 | Recomputes forward, OK |
| SiluAndMul backward | 0.37ms | 4% | 1 | PyTorch elementwise, OK |
| `aten::zero_` (grad init) | 0.40ms | 4% | 13 | Wasteful — buffer reuse eliminates |
| Other (mul, add, copy) | 0.73ms | 8% | — | Framework overhead |

Per training step: ~376ms (35B, 40 layers) or ~451ms (122B, 48 layers) for MoE backward alone.

### Atomic Contention Analysis (backward_weight)

For Qwen3.5-35B GateUpProj backward:
- Weight shape: `(E=256, FFN*2=1024, H=2048)`
- Grid: `ceil(T*topk/64) * ceil(1024/64)` = 512 M-blocks * 16 N-blocks = 8192 blocks
- Tokens per expert: `T*topk / E` = `4096*8 / 256` = 128 = 2 M-blocks per expert
- Each does K/BLOCK_K = 2048/32 = 64 atomic writes per K-loop
- Total atomics: 8192 * 64 = 524,288
- Contention: 2 blocks racing per expert (moderate at E=256; worse at smaller E)

---

## Optimization Roadmap: 4 Benchmark Variants

### V0 — Baseline (current miles port)
The current implementation, byte-identical to miles' FSDP kernels. Static block config, atomic writes every K-iteration, fresh zero-filled grad buffers each call, SiLU backward as separate PyTorch op.

### V1 — Expert-Parallel Grid + Buffer Reuse
**Target: backward_weight (44%) + grad init (4%)**

Replace the current grid `(num_M_blocks * num_N_blocks,)` where multiple M-blocks contend on the same expert with an expert-parallel grid `(num_experts * num_N_blocks,)` where each block owns ALL tokens for one expert.

| | V0 (current) | V1 (expert-parallel) |
|--|--------------|---------------------|
| Grid size | 8192 blocks | 256 experts × 16 N-blocks = 4096 |
| Atomics per block | 64 (one per K-iter) | **0** (sole owner, use `tl.store`) |
| Contention | 2 blocks/expert | None |

Each block iterates over all M-blocks for its expert sequentially, accumulating in registers. Then one `tl.store` instead of `tl.atomic_add`. Also pre-allocate grad buffers once and reuse across calls instead of `torch.zeros_like` every backward.

### V2 — V1 + Fused SiLU Backward
**Target: SiluAndMul backward (4%) + one tensor read/write elimination**

Fuse the SiLU backward computation into the GateUpProj backward kernel. Currently SiluAndMulFunction.backward reads `intermediate_cache1`, computes `dsilu * x2`, writes `grad_input`. Then GateUpProj backward reads this as `grad_output`. Fusing eliminates the intermediate tensor write+read.

Adds `~20 lines` of SiLU derivative logic inside `backward_input_kernel` and `backward_weight_kernel` (read `intermediate_cache1` directly, apply SiLU derivative before the matmul).

### V3 — V2 + Autotune (block sizes, num_warps, num_stages)
**Target: all kernels**

Replace static `DEFAULT_CONFIG` with `@triton.autotune` on all 3 backward kernels. Configs to sweep:

```python
@triton.autotune(
    configs=[
        # (M, N, K, GROUP_M, num_warps, num_stages)
        triton.Config({"BLOCK_SIZE_M": 32,  "BLOCK_SIZE_N": 64,  "BLOCK_SIZE_K": 64,  "GROUP_SIZE_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_M": 64,  "BLOCK_SIZE_N": 64,  "BLOCK_SIZE_K": 32,  "GROUP_SIZE_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_M": 64,  "BLOCK_SIZE_N": 64,  "BLOCK_SIZE_K": 64,  "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE_M": 64,  "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,  "GROUP_SIZE_M": 8}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64,  "BLOCK_SIZE_K": 32,  "GROUP_SIZE_M": 8}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,  "GROUP_SIZE_M": 8}, num_warps=8, num_stages=4),
        # High-warp configs for memory-bound kernels
        triton.Config({"BLOCK_SIZE_M": 64,  "BLOCK_SIZE_N": 64,  "BLOCK_SIZE_K": 32,  "GROUP_SIZE_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 64,  "BLOCK_SIZE_N": 64,  "BLOCK_SIZE_K": 32,  "GROUP_SIZE_M": 8}, num_warps=16, num_stages=2),
    ],
    key=["N", "K"],
)
```

Key tuning parameters:
- **`num_warps`**: More warps = more outstanding memory requests = better latency hiding. Default 4, try 8/16 for memory-bound kernels.
- **`num_stages`**: Software pipelining depth for loads. More stages = more prefetching overlap. Default 2, try 3-4 on H200's larger SMEM.
- **Block sizes**: Larger blocks reduce atomic frequency but increase register pressure. Trade-off depends on expert GEMM dimensions.

---

## Benchmark Matrix

### Kernel-Level Microbenchmark (`tools/profile_fused_moe.py`)

| Variant | 35B shapes | 122B shapes |
|---------|-----------|-------------|
| | H=2048, FFN=512, E=256, K=8 | H=3072, FFN=1024, E=256, K=8 |
| V0: Baseline | ✓ | ✓ |
| V1: Expert-parallel + buffer reuse | ✓ | ✓ |
| V2: V1 + fused SiLU bwd | ✓ | ✓ |
| V3: V2 + autotune | ✓ | ✓ |

Token counts to sweep: 512, 1024, 2048, 4096, 8192.

### Full Training Benchmark (Ray job, 8xH200)

| Config | LoRA (attn-only) | LoRA (all-linear) | Non-LoRA |
|--------|-----------------|-------------------|----------|
| 35B + V0 | fused MoE ON | fused MoE OFF (fallback) | fused MoE ON |
| 35B + V3 | fused MoE ON | fused MoE OFF (fallback) | fused MoE ON |
| 122B + V0 | fused MoE ON | N/A (multi-node) | fused MoE ON |
| 122B + V3 | fused MoE ON | N/A (multi-node) | fused MoE ON |

Metrics: `actor_train_time`, `tflops`, `peak_gb` per step.

---

## LoRA Compatibility

### Current State
- `_has_expert_lora_params()` in `fused_moe_integration.py` checks for `lora_` or `.adapter.` params on expert modules
- If found: **raises RuntimeError** (line 86-88) — crashes, does not skip
- `--target-modules "all-linear"` INCLUDES expert weights → crashes with `--use-fused-moe-backward`

### Workaround for Benchmarking
Use `--target-modules` that excludes experts:
```bash
--target-modules "qkv_proj,o_proj,gate_proj,up_proj,down_proj"  # attention + dense MLP only
```
This keeps LoRA on all non-expert linears while allowing fused MoE on expert layers.

### Proper Fix (TODO)
Change `_has_expert_lora_params` from raising to graceful handling:
1. **Option A (skip)**: Log warning, don't patch that module, fall back to standard autograd for LoRA'd experts
2. **Option B (merge)**: Merge LoRA weights into base before stacking: `W_merged = W_base + lora_B @ lora_A * scaling`. Stack merged weights. Gradients flow through LoRA params via the merge op. This is the correct long-term fix — enables fused MoE backward even with LoRA on expert weights.

---

## Existing Kernels

| File | What |
|------|------|
| `kernels/fp8_kernel.py` | Blockwise FP8 cast (Triton, E4M3, 128x128 blocks) |
| `kernels/int4_qat/` | INT4 fake quant CUDA kernel |
| `megatron_to_hf/processors/quantizer_fp8.py` | FP8 weight quantization for inference export |
| `megatron_to_hf/processors/quantizer_mxfp8.py` | MXFP8 group quantization via SGLang |
