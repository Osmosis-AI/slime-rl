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
   - Primary optimization target: atomic contention when many experts route to same token

2. **`backward_weight_kernel`** — grad w.r.t. expert weights
   - `grad_weight[expert] = input.T @ grad_output` (per expert)
   - Uses `tl.atomic_add` for expert accumulation (multiple token blocks -> same expert)

3. **`backward_topk_weights_kernel`** — grad w.r.t. routing weights
   - Recomputes forward output in-kernel (`input @ weight.T`) to avoid storing it
   - Only runs for DownProj (`mul_routed_weight=True`)

### Activation and Guards

- CLI flag: `--use-fused-moe-backward`
- Guards in `slime/utils/arguments.py`: requires Bridge mode, BF16, EP=1, expert-TP=1
- Hook order in `bridge_lora_helpers.py`: LoRA hook first, then fused MoE hook
- LoRA on expert weights is explicitly rejected (`_has_expert_lora_params` check)

### SGLang Forward Kernel Dependency

The forward pass uses SGLang's `invoke_fused_moe_kernel` and `moe_align_block_size`. Version-specific kwargs:
- GateUpProj passes `c_sorted=False, filter_expert=True`
- DownProj passes `a_use_tma=False, b_use_tma=False`

These may not exist in all SGLang versions. If the installed version doesn't accept them, you'll get a `TypeError` at runtime.

### Block Config

Static, no autotuning: `BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32, GROUP_SIZE_M=8`. Chunk size: 64K tokens per iteration.

### Memory Characteristics (Qwen3.5-35B-A3B: H=2048, FFN=512, E=64, topk=8)

All three kernels are **memory-bandwidth-bound** at these dimensions — expert GEMMs are tiny (512x2048). The `tl.atomic_add` contention dominates, not compute. FP8 would not help (overhead > benefit for small matmuls).

### Integration Pattern

`patch_model_for_fused_moe()` walks the model tree, finds modules with `linear_fc1` and `linear_fc2` attributes containing `weight<N>` parameters (grouped experts), and replaces their `forward` method. Parameter names are preserved — no renames, no new wrappers holding parameters. This keeps Bridge naming, LoRA injection, HF export, and weight sync contracts intact.

### Key Invariants

- `mlp.experts.linear_fc1` / `mlp.experts.linear_fc2` names must stay visible in `named_parameters()`
- Per-expert `weight<N>` suffixes must remain on the grouped linear modules
- Expert weights are stacked at runtime via `torch.stack` for the fused kernel, not pre-stacked
- The patched forward consumes Megatron's fused expert tensors directly (not HF `ModuleList`)

## Existing Kernels

| File | What |
|------|------|
| `kernels/fp8_kernel.py` | Blockwise FP8 cast (Triton, E4M3, 128x128 blocks) |
| `kernels/int4_qat/` | INT4 fake quant CUDA kernel |
| `megatron_to_hf/processors/quantizer_fp8.py` | FP8 weight quantization for inference export |
| `megatron_to_hf/processors/quantizer_mxfp8.py` | MXFP8 group quantization via SGLang |
