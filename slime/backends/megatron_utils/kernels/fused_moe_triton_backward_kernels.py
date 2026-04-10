from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl


def build_expert_block_mapping(
    expert_ids: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build CSR-style index mapping each expert to its M-blocks.

    Args:
        expert_ids: Expert ID for each M-block, from moe_align_block_size.
                    Shape (num_m_blocks,). Padding blocks have value -1.
        num_experts: Total number of experts.

    Returns:
        sorted_block_ids: M-block indices sorted by expert. Shape (num_valid_blocks,), int32.
        expert_offsets: CSR offsets. Expert e owns blocks at
                        sorted_block_ids[expert_offsets[e]:expert_offsets[e+1]].
                        Shape (num_experts + 1,), int32.
    """
    valid_mask = expert_ids >= 0
    valid_experts = expert_ids[valid_mask]
    valid_indices = torch.arange(expert_ids.shape[0], device=expert_ids.device)[valid_mask]

    sort_order = torch.argsort(valid_experts.int(), stable=True)
    sorted_block_ids = valid_indices[sort_order].to(torch.int32)

    counts = torch.bincount(valid_experts.int(), minlength=num_experts)
    expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=expert_ids.device)
    torch.cumsum(counts.int(), dim=0, out=expert_offsets[1:])

    return sorted_block_ids, expert_offsets


@triton.jit
def fused_moe_backward_input_kernel(
    grad_output_ptr,
    weight_ptr,
    grad_input_ptr,
    grad_topk_weights_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K,
    EM,
    num_valid_tokens,
    stride_gom,
    stride_gon,
    stride_we,
    stride_wn,
    stride_wk,
    stride_gim,
    stride_gik,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M < num_tokens_post_padded:
        offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
        offs_token = offs_token.to(tl.int64)
        token_mask = offs_token < num_valid_tokens

        off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
        if off_experts != -1:
            offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
            offs_k = tl.arange(0, BLOCK_SIZE_K)

            grad_output_ptrs = grad_output_ptr + (offs_token[:, None] * stride_gom + offs_n[None, :] * stride_gon)
            grad_out = tl.load(
                grad_output_ptrs,
                mask=token_mask[:, None] & (offs_n[None, :] < N),
                other=0.0,
            )

            if MUL_ROUTED_WEIGHT:
                moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
                grad_out = grad_out * moe_weight[:, None]

            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                curr_offs_k = k * BLOCK_SIZE_K + offs_k
                weight_ptrs = (
                    weight_ptr
                    + off_experts * stride_we
                    + offs_n[:, None] * stride_wn
                    + curr_offs_k[None, :] * stride_wk
                )
                w = tl.load(
                    weight_ptrs,
                    mask=(offs_n[:, None] < N) & (curr_offs_k[None, :] < K),
                    other=0.0,
                )
                contribution = tl.dot(grad_out, w)

                grad_input_ptrs = grad_input_ptr + (
                    (offs_token[:, None] // top_k) * stride_gim + curr_offs_k[None, :] * stride_gik
                )
                grad_input_mask = token_mask[:, None] & (curr_offs_k[None, :] < K)
                tl.atomic_add(grad_input_ptrs, contribution.to(compute_type), mask=grad_input_mask)


@triton.jit
def fused_moe_backward_weight_kernel(
    grad_output_ptr,
    input_ptr,
    grad_weight_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K,
    EM,
    num_valid_tokens,
    stride_gom,
    stride_gon,
    stride_im,
    stride_ik,
    stride_gwe,
    stride_gwn,
    stride_gwk,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    expert_id = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if expert_id == -1:
        return

    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_token_id = pid_m * BLOCK_SIZE_M + offs_m.to(tl.int64)
    offs_token = tl.load(
        sorted_token_ids_ptr + offs_token_id,
        mask=offs_token_id < num_tokens_post_padded,
        other=num_valid_tokens,
    )
    offs_token = offs_token.to(tl.int64)
    token_mask = (offs_token_id < num_tokens_post_padded) & (offs_token < num_valid_tokens)
    offs_token_clamped = tl.where(token_mask, offs_token, 0)

    if MUL_ROUTED_WEIGHT:
        input_token_idx = offs_token_clamped
        input_mask = token_mask
    else:
        input_token_idx = offs_token_clamped // top_k
        num_input_tokens = num_valid_tokens // top_k
        input_mask = token_mask & (input_token_idx < num_input_tokens)

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token_clamped, mask=token_mask, other=0.0)

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    grad_output_ptrs = grad_output_ptr + (
        offs_token_clamped[:, None] * stride_gom + offs_n[None, :] * stride_gon
    )
    grad_out = tl.load(
        grad_output_ptrs,
        mask=token_mask[:, None] & (offs_n[None, :] < N),
        other=0.0,
    )

    if MUL_ROUTED_WEIGHT:
        grad_out = grad_out * moe_weight[:, None]

    grad_out = grad_out * token_mask[:, None]

    for k_block in range(tl.cdiv(K, BLOCK_SIZE_K)):
        offs_k = k_block * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
        input_ptrs = input_ptr + (input_token_idx[:, None] * stride_im + offs_k[None, :] * stride_ik)
        inp = tl.load(
            input_ptrs,
            mask=input_mask[:, None] & (offs_k[None, :] < K),
            other=0.0,
        )
        inp = inp * input_mask[:, None]

        grad_w_contribution = tl.dot(grad_out.T, inp)
        grad_weight_ptrs = (
            grad_weight_ptr + expert_id * stride_gwe + offs_n[:, None] * stride_gwn + offs_k[None, :] * stride_gwk
        )
        grad_weight_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        tl.atomic_add(grad_weight_ptrs, grad_w_contribution.to(compute_type), mask=grad_weight_mask)


@triton.jit
def fused_moe_backward_weight_kernel_v1(
    grad_output_ptr,
    input_ptr,
    grad_weight_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    sorted_block_ids_ptr,
    expert_offsets_ptr,
    N,
    K,
    num_tokens_post_padded,
    num_valid_tokens,
    stride_gom,
    stride_gon,
    stride_im,
    stride_ik,
    stride_gwe,
    stride_gwn,
    stride_gwk,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    """Expert-parallel backward_weight kernel (V1).

    Grid: (num_experts * num_N_blocks,). Each thread block owns ALL M-blocks
    for one expert. Accumulates grad_out.T @ inp in registers — zero atomics.
    """
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    expert_id = (pid // num_pid_n).to(tl.int64)
    pid_n = pid % num_pid_n

    block_start = tl.load(expert_offsets_ptr + expert_id).to(tl.int32)
    block_end = tl.load(expert_offsets_ptr + expert_id + 1).to(tl.int32)
    num_blocks = block_end - block_start
    if num_blocks == 0:
        return

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    offs_m = tl.arange(0, BLOCK_SIZE_M)

    for k_block in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        offs_k = k_block * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
        acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)

        block_idx = block_start
        while block_idx < block_end:
            m_block = tl.load(sorted_block_ids_ptr + block_idx).to(tl.int32)
            offs_token_id = m_block * BLOCK_SIZE_M + offs_m.to(tl.int64)
            offs_token = tl.load(
                sorted_token_ids_ptr + offs_token_id,
                mask=offs_token_id < num_tokens_post_padded,
                other=num_valid_tokens,
            )
            offs_token = offs_token.to(tl.int64)
            token_mask = (offs_token_id < num_tokens_post_padded) & (offs_token < num_valid_tokens)
            offs_token_clamped = tl.where(token_mask, offs_token, 0)

            # Load grad_output slice
            grad_output_ptrs = grad_output_ptr + (
                offs_token_clamped[:, None] * stride_gom + offs_n[None, :] * stride_gon
            )
            grad_out = tl.load(
                grad_output_ptrs,
                mask=token_mask[:, None] & (offs_n[None, :] < N),
                other=0.0,
            )

            if MUL_ROUTED_WEIGHT:
                moe_weight = tl.load(topk_weights_ptr + offs_token_clamped, mask=token_mask, other=0.0)
                grad_out = grad_out * moe_weight[:, None]

            grad_out = grad_out * token_mask[:, None]

            # Load input slice
            if MUL_ROUTED_WEIGHT:
                input_token_idx = offs_token_clamped
                input_mask = token_mask
            else:
                input_token_idx = offs_token_clamped // top_k
                num_input_tokens = num_valid_tokens // top_k
                input_mask = token_mask & (input_token_idx < num_input_tokens)

            input_ptrs = input_ptr + (input_token_idx[:, None] * stride_im + offs_k[None, :] * stride_ik)
            inp = tl.load(
                input_ptrs,
                mask=input_mask[:, None] & (offs_k[None, :] < K),
                other=0.0,
            )
            inp = inp * input_mask[:, None]

            acc += tl.dot(grad_out.T, inp)
            block_idx += 1

        # Single store — no atomics
        grad_weight_ptrs = (
            grad_weight_ptr + expert_id * stride_gwe + offs_n[:, None] * stride_gwn + offs_k[None, :] * stride_gwk
        )
        grad_weight_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        tl.store(grad_weight_ptrs, acc.to(compute_type), mask=grad_weight_mask)


@triton.jit
def fused_moe_backward_topk_weights_kernel(
    grad_output_ptr,
    input_ptr,
    weight_ptr,
    grad_topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K,
    EM,
    num_valid_tokens,
    stride_gom,
    stride_gon,
    stride_im,
    stride_ik,
    stride_we,
    stride_wn,
    stride_wk,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid * BLOCK_SIZE_M < num_tokens_post_padded:
        offs_token_id = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        offs_token = tl.load(
            sorted_token_ids_ptr + offs_token_id,
            mask=offs_token_id < num_tokens_post_padded,
            other=num_valid_tokens,
        )
        offs_token = offs_token.to(tl.int64)
        token_mask = (offs_token_id < num_tokens_post_padded) & (offs_token < num_valid_tokens)
        offs_token_clamped = tl.where(token_mask, offs_token, 0)

        off_experts = tl.load(expert_ids_ptr + pid).to(tl.int64)
        if off_experts != -1:
            offs_n = tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            accumulator = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

            for n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
                curr_offs_n = n * BLOCK_SIZE_N + offs_n
                grad_output_ptrs = grad_output_ptr + (
                    offs_token_clamped[:, None] * stride_gom + curr_offs_n[None, :] * stride_gon
                )
                grad_out = tl.load(
                    grad_output_ptrs,
                    mask=token_mask[:, None] & (curr_offs_n[None, :] < N),
                    other=0.0,
                )
                forward_output_n = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

                for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                    curr_offs_k = k * BLOCK_SIZE_K + offs_k
                    input_ptrs = input_ptr + (
                        (offs_token_clamped[:, None] // top_k) * stride_im + curr_offs_k[None, :] * stride_ik
                    )
                    inp = tl.load(
                        input_ptrs,
                        mask=token_mask[:, None] & (curr_offs_k[None, :] < K),
                        other=0.0,
                    )
                    weight_ptrs = (
                        weight_ptr
                        + off_experts * stride_we
                        + curr_offs_n[:, None] * stride_wn
                        + curr_offs_k[None, :] * stride_wk
                    )
                    w = tl.load(
                        weight_ptrs,
                        mask=(curr_offs_n[:, None] < N) & (curr_offs_k[None, :] < K),
                        other=0.0,
                    )
                    forward_output_n += tl.dot(inp, w.T)

                accumulator += tl.sum(grad_out * forward_output_n, axis=1)

            tl.atomic_add(grad_topk_weights_ptr + offs_token_clamped, accumulator.to(compute_type), mask=token_mask)


def invoke_fused_moe_backward_kernel(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    grad_input: torch.Tensor,
    grad_weight: torch.Tensor,
    grad_topk_weights: torch.Tensor | None,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
    compute_type: tl.dtype,
    use_expert_parallel: bool = False,
) -> None:
    del topk_ids
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    if grad_output.ndim == 3:
        grad_output = grad_output.reshape(-1, grad_output.shape[-1])

    num_experts, N, K = weight.shape

    # --- backward_input (unchanged, still uses atomics) ---
    def grid_input(meta):
        return (triton.cdiv(sorted_token_ids.shape[0], meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),)

    fused_moe_backward_input_kernel[grid_input](
        grad_output,
        weight,
        grad_input,
        grad_topk_weights if grad_topk_weights is not None else grad_input,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        sorted_token_ids.shape[0],
        grad_output.shape[0],
        grad_output.stride(0),
        grad_output.stride(1),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        grad_input.stride(0),
        grad_input.stride(1),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        **config,
    )

    # --- backward_weight ---
    if use_expert_parallel:
        # V1: expert-parallel grid, zero atomics, no zeroing needed
        sorted_block_ids, expert_offsets = build_expert_block_mapping(expert_ids, num_experts)

        def grid_weight_v1(meta):
            return (num_experts * triton.cdiv(N, meta["BLOCK_SIZE_N"]),)

        # Load num_tokens_post_padded as a Python int for the V1 kernel
        ntp = num_tokens_post_padded.item()

        fused_moe_backward_weight_kernel_v1[grid_weight_v1](
            grad_output,
            input,
            grad_weight,
            topk_weights,
            sorted_token_ids,
            sorted_block_ids,
            expert_offsets,
            N,
            K,
            ntp,
            grad_output.shape[0],
            grad_output.stride(0),
            grad_output.stride(1),
            input.stride(0),
            input.stride(1),
            grad_weight.stride(0),
            grad_weight.stride(1),
            grad_weight.stride(2),
            MUL_ROUTED_WEIGHT=mul_routed_weight,
            top_k=top_k,
            compute_type=compute_type,
            BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
            BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
        )
    else:
        # V0: original atomic-based backward_weight
        grad_weight.zero_()

        def grid_weight(meta):
            return (triton.cdiv(sorted_token_ids.shape[0], meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),)

        fused_moe_backward_weight_kernel[grid_weight](
            grad_output,
            input,
            grad_weight,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            N,
            K,
            sorted_token_ids.shape[0],
            grad_output.shape[0],
            grad_output.stride(0),
            grad_output.stride(1),
            input.stride(0),
            input.stride(1),
            grad_weight.stride(0),
            grad_weight.stride(1),
            grad_weight.stride(2),
            MUL_ROUTED_WEIGHT=mul_routed_weight,
            top_k=top_k,
            compute_type=compute_type,
            **config,
        )

    # --- backward_topk_weights (unchanged) ---
    if mul_routed_weight and grad_topk_weights is not None:

        def grid_topk(meta):
            return (triton.cdiv(sorted_token_ids.shape[0], meta["BLOCK_SIZE_M"]),)

        fused_moe_backward_topk_weights_kernel[grid_topk](
            grad_output,
            input,
            weight,
            grad_topk_weights.view(-1),
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            N,
            K,
            sorted_token_ids.shape[0],
            grad_output.shape[0],
            grad_output.stride(0),
            grad_output.stride(1),
            input.stride(0),
            input.stride(1),
            weight.stride(0),
            weight.stride(1),
            weight.stride(2),
            top_k=top_k,
            compute_type=compute_type,
            BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
            BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
        )
