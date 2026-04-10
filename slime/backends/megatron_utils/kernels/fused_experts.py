from __future__ import annotations

import torch
import triton.language as tl
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
    invoke_fused_moe_kernel,
    moe_align_block_size,
    moe_sum_reduce,
    silu_and_mul,
)

from .fused_moe_triton_backward_kernels import invoke_fused_moe_backward_kernel

DEFAULT_CONFIG = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 8,
}
CHUNK_SIZE = 64 * 1024

# V1 expert-parallel backward_weight: eliminates tl.atomic_add contention.
# Set to False to revert to V0 (atomic-based) for A/B benchmarking.
USE_EXPERT_PARALLEL = True


class GateUpProjFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, w1, topk_weights, topk_ids):
        num_tokens, _ = hidden_states.shape
        num_experts, intermediate_size_x2, _ = w1.shape
        topk = topk_ids.shape[1]

        intermediate_cache1 = torch.empty(
            (num_tokens * topk, intermediate_size_x2),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        for chunk in range((num_tokens // CHUNK_SIZE) + 1):
            begin_chunk_idx = chunk * CHUNK_SIZE
            end_chunk_idx = min((chunk + 1) * CHUNK_SIZE, num_tokens)
            if begin_chunk_idx == end_chunk_idx:
                continue

            curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
            curr_intermediate_cache1 = intermediate_cache1[begin_chunk_idx * topk : end_chunk_idx * topk]
            curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
            curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                curr_topk_ids,
                DEFAULT_CONFIG["BLOCK_SIZE_M"],
                num_experts,
            )

            invoke_fused_moe_kernel(
                curr_hidden_states,
                w1,
                None,
                curr_intermediate_cache1,
                None,
                None,
                None,
                curr_topk_weights,
                curr_topk_ids,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                False,
                topk,
                DEFAULT_CONFIG,
                compute_type=tl.bfloat16,
                use_fp8_w8a8=False,
                use_int8_w8a8=False,
                use_int8_w8a16=False,
                use_int4_w4a16=False,
                per_channel_quant=False,
                block_shape=None,
                c_sorted=False,
                filter_expert=True,
            )

        ctx.save_for_backward(hidden_states, w1, topk_weights, topk_ids)
        ctx.num_tokens = num_tokens
        ctx.topk = topk
        return intermediate_cache1

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, w1, topk_weights, topk_ids = ctx.saved_tensors
        num_tokens = ctx.num_tokens
        topk = ctx.topk

        grad_hidden_states = torch.zeros_like(hidden_states)
        # V1 writes all locations via tl.store — no zeroing needed
        grad_w1 = torch.empty_like(w1) if USE_EXPERT_PARALLEL else torch.zeros_like(w1)
        grad_topk_weights = torch.zeros_like(topk_weights)

        for chunk in range((num_tokens // CHUNK_SIZE) + 1):
            begin_chunk_idx = chunk * CHUNK_SIZE
            end_chunk_idx = min((chunk + 1) * CHUNK_SIZE, num_tokens)
            curr_num_tokens = end_chunk_idx - begin_chunk_idx
            if curr_num_tokens == 0:
                continue

            curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
            curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
            curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]
            curr_grad_output = grad_output[begin_chunk_idx * topk : end_chunk_idx * topk]

            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                curr_topk_ids,
                DEFAULT_CONFIG["BLOCK_SIZE_M"],
                w1.shape[0],
            )

            curr_grad_hidden_states = torch.zeros_like(curr_hidden_states)
            curr_grad_w1 = torch.empty_like(w1) if USE_EXPERT_PARALLEL else torch.zeros_like(w1)

            invoke_fused_moe_backward_kernel(
                grad_output=curr_grad_output,
                input=curr_hidden_states,
                weight=w1,
                grad_input=curr_grad_hidden_states,
                grad_weight=curr_grad_w1,
                grad_topk_weights=None,
                topk_weights=curr_topk_weights,
                topk_ids=curr_topk_ids,
                sorted_token_ids=sorted_token_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_padded,
                mul_routed_weight=False,
                top_k=topk,
                config=DEFAULT_CONFIG,
                compute_type=tl.bfloat16,
                use_expert_parallel=USE_EXPERT_PARALLEL,
            )

            grad_hidden_states[begin_chunk_idx:end_chunk_idx] += curr_grad_hidden_states
            if USE_EXPERT_PARALLEL:
                # V1: curr_grad_w1 is the complete gradient (no accumulation needed for single chunk)
                if chunk == 0:
                    grad_w1 = curr_grad_w1
                else:
                    grad_w1 += curr_grad_w1
            else:
                grad_w1 += curr_grad_w1

        return grad_hidden_states, grad_w1, grad_topk_weights, None


class SiluAndMulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, intermediate_cache1):
        num_tokens, intermediate_size_x2 = intermediate_cache1.shape
        intermediate_cache2 = torch.empty(
            (num_tokens, intermediate_size_x2 // 2),
            device=intermediate_cache1.device,
            dtype=intermediate_cache1.dtype,
        )
        silu_and_mul(intermediate_cache1.view(-1, intermediate_size_x2), intermediate_cache2)
        ctx.save_for_backward(intermediate_cache1)
        return intermediate_cache2

    @staticmethod
    def backward(ctx, grad_output):
        (intermediate_cache1,) = ctx.saved_tensors
        intermediate_size_x2 = intermediate_cache1.shape[-1]
        x1, x2 = intermediate_cache1.view(-1, intermediate_size_x2).chunk(2, dim=-1)
        silu_x1 = torch.nn.functional.silu(x1)
        sig = torch.sigmoid(x1)
        dsilu_dx1 = sig + x1 * sig * (1 - sig)
        grad_x1 = grad_output * x2 * dsilu_dx1
        grad_x2 = grad_output * silu_x1
        grad_input = torch.cat([grad_x1, grad_x2], dim=-1)
        return grad_input.view_as(intermediate_cache1)


class DownProjFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, intermediate_cache2, w2, topk_weights, topk_ids):
        num_tokens_x_topk, _ = intermediate_cache2.shape
        topk = topk_ids.shape[1]
        num_tokens = num_tokens_x_topk // topk
        num_experts = w2.shape[0]

        intermediate_cache3 = torch.empty(
            (num_tokens, topk, w2.shape[1]),
            device=intermediate_cache2.device,
            dtype=intermediate_cache2.dtype,
        )

        for chunk in range((num_tokens // CHUNK_SIZE) + 1):
            begin_chunk_idx = chunk * CHUNK_SIZE
            end_chunk_idx = min((chunk + 1) * CHUNK_SIZE, num_tokens)
            if begin_chunk_idx == end_chunk_idx:
                continue

            curr_intermediate_cache2 = intermediate_cache2[begin_chunk_idx * topk : end_chunk_idx * topk]
            curr_intermediate_cache3 = intermediate_cache3[begin_chunk_idx:end_chunk_idx]
            curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
            curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                curr_topk_ids,
                DEFAULT_CONFIG["BLOCK_SIZE_M"],
                num_experts,
            )
            invoke_fused_moe_kernel(
                curr_intermediate_cache2,
                w2,
                None,
                curr_intermediate_cache3,
                None,
                None,
                None,
                curr_topk_weights,
                curr_topk_ids,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                True,
                1,
                DEFAULT_CONFIG,
                compute_type=tl.bfloat16,
                use_fp8_w8a8=False,
                use_int8_w8a8=False,
                use_int8_w8a16=False,
                use_int4_w4a16=False,
                per_channel_quant=False,
                block_shape=None,
                a_use_tma=False,
                b_use_tma=False,
            )

        ctx.save_for_backward(intermediate_cache2, w2, topk_weights, topk_ids)
        ctx.num_tokens = num_tokens
        ctx.topk = topk
        return intermediate_cache3

    @staticmethod
    def backward(ctx, grad_output):
        intermediate_cache2, w2, topk_weights, topk_ids = ctx.saved_tensors
        num_tokens = ctx.num_tokens
        topk = ctx.topk

        grad_intermediate_cache2 = torch.zeros_like(intermediate_cache2)
        grad_w2 = torch.empty_like(w2) if USE_EXPERT_PARALLEL else torch.zeros_like(w2)
        grad_topk_weights = torch.zeros_like(topk_weights)

        for chunk in range((num_tokens // CHUNK_SIZE) + 1):
            begin_chunk_idx = chunk * CHUNK_SIZE
            end_chunk_idx = min((chunk + 1) * CHUNK_SIZE, num_tokens)
            curr_num_tokens = end_chunk_idx - begin_chunk_idx
            if curr_num_tokens == 0:
                continue

            curr_intermediate_cache2 = intermediate_cache2[begin_chunk_idx * topk : end_chunk_idx * topk]
            curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
            curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]
            curr_grad_output = grad_output[begin_chunk_idx:end_chunk_idx]

            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                curr_topk_ids,
                DEFAULT_CONFIG["BLOCK_SIZE_M"],
                w2.shape[0],
            )

            curr_grad_intermediate_cache2 = torch.zeros_like(curr_intermediate_cache2)
            curr_grad_w2 = torch.empty_like(w2) if USE_EXPERT_PARALLEL else torch.zeros_like(w2)
            curr_grad_topk_weights = torch.zeros_like(curr_topk_weights)

            invoke_fused_moe_backward_kernel(
                grad_output=curr_grad_output,
                input=curr_intermediate_cache2,
                weight=w2,
                grad_input=curr_grad_intermediate_cache2,
                grad_weight=curr_grad_w2,
                grad_topk_weights=curr_grad_topk_weights,
                topk_weights=curr_topk_weights,
                topk_ids=curr_topk_ids,
                sorted_token_ids=sorted_token_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_padded,
                mul_routed_weight=True,
                top_k=1,
                config=DEFAULT_CONFIG,
                compute_type=tl.bfloat16,
                use_expert_parallel=USE_EXPERT_PARALLEL,
            )

            grad_intermediate_cache2[begin_chunk_idx * topk : end_chunk_idx * topk] = curr_grad_intermediate_cache2
            if USE_EXPERT_PARALLEL:
                if chunk == 0:
                    grad_w2 = curr_grad_w2
                else:
                    grad_w2 += curr_grad_w2
            else:
                grad_w2 += curr_grad_w2
            grad_topk_weights[begin_chunk_idx:end_chunk_idx] = curr_grad_topk_weights

        return grad_intermediate_cache2, grad_w2, grad_topk_weights, None


class MoeSumReduceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, intermediate_cache3, hidden_states_shape):
        out_hidden_states = torch.empty(
            hidden_states_shape,
            device=intermediate_cache3.device,
            dtype=intermediate_cache3.dtype,
        )
        moe_sum_reduce(intermediate_cache3, out_hidden_states, 1.0)
        ctx.save_for_backward(intermediate_cache3)
        return out_hidden_states

    @staticmethod
    def backward(ctx, grad_output):
        (intermediate_cache3,) = ctx.saved_tensors
        return grad_output.unsqueeze(1).expand_as(intermediate_cache3), None


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden states must be contiguous"
    assert w1.is_contiguous(), "Expert gate/up weights must be contiguous"
    assert w2.is_contiguous(), "Expert down weights must be contiguous"
    assert hidden_states.dtype in [torch.bfloat16]

    intermediate_cache1 = GateUpProjFunction.apply(hidden_states, w1, topk_weights, topk_ids)
    intermediate_cache2 = SiluAndMulFunction.apply(intermediate_cache1)
    intermediate_cache3 = DownProjFunction.apply(intermediate_cache2, w2, topk_weights, topk_ids)
    return MoeSumReduceFunction.apply(intermediate_cache3, hidden_states.shape)
