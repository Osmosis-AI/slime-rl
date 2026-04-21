from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
import triton
import triton.language as tl


_BLOCK_N = 256
_BACKWARD_VOCAB_TILE = 1024


def _get_tp_rank_and_world_size(tp_group: dist.ProcessGroup | None) -> tuple[int, int]:
    if tp_group is None or not dist.is_initialized():
        return 0, 1
    return dist.get_rank(group=tp_group), dist.get_world_size(group=tp_group)


def _all_reduce_max_(tensor: torch.Tensor, tp_group: dist.ProcessGroup | None) -> torch.Tensor:
    if tp_group is not None and dist.is_initialized() and dist.get_world_size(group=tp_group) > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=tp_group)
    return tensor


def _all_reduce_sum_(tensor: torch.Tensor, tp_group: dist.ProcessGroup | None) -> torch.Tensor:
    if tp_group is not None and dist.is_initialized() and dist.get_world_size(group=tp_group) > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=tp_group)
    return tensor


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 8, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    ],
    key=["M", "K"],
)
@triton.jit
def _partial_row_max_kernel(
    hidden_ptr,
    weight_ptr,
    bias_ptr,
    partial_max_ptr,
    stride_hm,
    stride_hk,
    stride_wn,
    stride_wk,
    stride_pm,
    stride_pn,
    M,
    N,
    K,
    inv_temperature,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr = _BLOCK_N,
):
    pid_m = tl.cast(tl.program_id(axis=0), tl.int64)
    pid_n = tl.cast(tl.program_id(axis=1), tl.int64)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        hidden = tl.load(
            hidden_ptr + offs_m[:, None] * stride_hm + offs_k[None, :] * stride_hk,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        weight = tl.load(
            weight_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        )
        acc = tl.dot(hidden, weight, acc)

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        acc += bias[None, :]

    acc *= inv_temperature
    acc = tl.where(mask_n[None, :], acc, -float("inf"))
    row_max = tl.max(acc, axis=1)

    tl.store(
        partial_max_ptr + offs_m * stride_pm + pid_n * stride_pn,
        row_max,
        mask=mask_m,
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 8, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    ],
    key=["M", "K", "WITH_ENTROPY"],
)
@triton.jit
def _partial_sum_selected_kernel(
    hidden_ptr,
    weight_ptr,
    bias_ptr,
    global_max_ptr,
    tokens_ptr,
    partial_sum_ptr,
    selected_ptr,
    weighted_sum_ptr,
    stride_hm,
    stride_hk,
    stride_wn,
    stride_wk,
    stride_psm,
    stride_psn,
    stride_wsm,
    stride_wsn,
    M,
    N,
    K,
    vocab_start_index,
    inv_temperature,
    HAS_BIAS: tl.constexpr,
    WITH_ENTROPY: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr = _BLOCK_N,
):
    pid_m = tl.cast(tl.program_id(axis=0), tl.int64)
    pid_n = tl.cast(tl.program_id(axis=1), tl.int64)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        hidden = tl.load(
            hidden_ptr + offs_m[:, None] * stride_hm + offs_k[None, :] * stride_hk,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        weight = tl.load(
            weight_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        )
        acc = tl.dot(hidden, weight, acc)

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        acc += bias[None, :]

    scaled_logits = acc * inv_temperature
    global_max = tl.load(global_max_ptr + offs_m, mask=mask_m, other=0.0)
    shifted = scaled_logits - global_max[:, None]
    shifted = tl.where(mask_n[None, :], shifted, -float("inf"))
    exp_logits = tl.exp(shifted)
    partial_sum = tl.sum(exp_logits, axis=1)
    tl.store(
        partial_sum_ptr + offs_m * stride_psm + pid_n * stride_psn,
        partial_sum,
        mask=mask_m,
    )

    if WITH_ENTROPY:
        weighted_sum = tl.sum(exp_logits * scaled_logits, axis=1)
        tl.store(
            weighted_sum_ptr + offs_m * stride_wsm + pid_n * stride_wsn,
            weighted_sum,
            mask=mask_m,
        )

    tokens = tl.load(tokens_ptr + offs_m, mask=mask_m, other=0).to(tl.int64)
    local_tokens = tokens - vocab_start_index
    tile_start = pid_n * BLOCK_N
    tile_end = tile_start + BLOCK_N
    token_in_tile = mask_m & (local_tokens >= tile_start) & (local_tokens < tile_end)
    matches = offs_n[None, :] == local_tokens[:, None]
    selected_values = tl.sum(tl.where(matches, scaled_logits, 0.0), axis=1)
    tl.store(selected_ptr + offs_m, selected_values, mask=token_in_tile)


def _compute_forward_stats(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    tokens: torch.Tensor,
    tp_group: dist.ProcessGroup | None,
    inv_temperature: float,
    return_entropy: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if hidden_states.ndim != 2:
        raise ValueError(f"Expected hidden_states to be 2D, got {tuple(hidden_states.shape)}")
    if weight.ndim != 2:
        raise ValueError(f"Expected weight to be 2D, got {tuple(weight.shape)}")
    if tokens.ndim != 1:
        raise ValueError(f"Expected tokens to be 1D, got {tuple(tokens.shape)}")
    if hidden_states.size(0) != tokens.size(0):
        raise ValueError(
            "Chunked fused selected TP logprob expects hidden/token rows to match. "
            f"Got hidden={tuple(hidden_states.shape)} tokens={tuple(tokens.shape)}"
        )
    if hidden_states.size(1) != weight.size(1):
        raise ValueError(
            "Chunked fused selected TP logprob expects hidden size to match output weight. "
            f"Got hidden={tuple(hidden_states.shape)} weight={tuple(weight.shape)}"
        )
    if hidden_states.numel() == 0:
        empty = hidden_states.new_zeros((0,), dtype=torch.float32)
        return empty, empty, empty, empty
    if not hidden_states.is_cuda or not weight.is_cuda:
        raise RuntimeError("Fused selected TP logprob Triton path requires CUDA tensors.")

    hidden_states = hidden_states.contiguous()
    weight = weight.contiguous()
    tokens = tokens.contiguous()
    bias_tensor = bias.contiguous() if bias is not None else hidden_states.new_empty((0,), dtype=hidden_states.dtype)
    has_bias = bias is not None

    num_rows, hidden_size = hidden_states.shape
    local_vocab_size = weight.size(0)
    num_vocab_blocks = triton.cdiv(local_vocab_size, _BLOCK_N)
    partial_max = torch.empty((num_rows, num_vocab_blocks), device=hidden_states.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(num_rows, meta["BLOCK_M"]), num_vocab_blocks)
    _partial_row_max_kernel[grid](
        hidden_states,
        weight,
        bias_tensor,
        partial_max,
        hidden_states.stride(0),
        hidden_states.stride(1),
        weight.stride(0),
        weight.stride(1),
        partial_max.stride(0),
        partial_max.stride(1),
        num_rows,
        local_vocab_size,
        hidden_size,
        float(inv_temperature),
        HAS_BIAS=has_bias,
    )

    global_max = partial_max.max(dim=1).values
    _all_reduce_max_(global_max, tp_group)

    partial_sum = torch.empty_like(partial_max)
    selected_local = torch.zeros((num_rows,), device=hidden_states.device, dtype=torch.float32)
    weighted_sum = (
        torch.empty_like(partial_max) if return_entropy else torch.empty((0,), device=hidden_states.device, dtype=torch.float32)
    )

    tp_rank, _tp_world_size = _get_tp_rank_and_world_size(tp_group)
    vocab_start_index = tp_rank * local_vocab_size
    weighted_stride_0 = weighted_sum.stride(0) if return_entropy else 0
    weighted_stride_1 = weighted_sum.stride(1) if return_entropy else 0

    _partial_sum_selected_kernel[grid](
        hidden_states,
        weight,
        bias_tensor,
        global_max,
        tokens,
        partial_sum,
        selected_local,
        weighted_sum,
        hidden_states.stride(0),
        hidden_states.stride(1),
        weight.stride(0),
        weight.stride(1),
        partial_sum.stride(0),
        partial_sum.stride(1),
        weighted_stride_0,
        weighted_stride_1,
        num_rows,
        local_vocab_size,
        hidden_size,
        vocab_start_index,
        float(inv_temperature),
        HAS_BIAS=has_bias,
        WITH_ENTROPY=return_entropy,
    )

    global_sum = partial_sum.sum(dim=1)
    _all_reduce_sum_(global_sum, tp_group)
    _all_reduce_sum_(selected_local, tp_group)

    row_lse = global_max + global_sum.log()
    log_prob = selected_local - row_lse

    entropy = hidden_states.new_empty((0,), dtype=torch.float32)
    row_expected_scaled = hidden_states.new_empty((0,), dtype=torch.float32)
    if return_entropy:
        global_weighted_sum = weighted_sum.sum(dim=1)
        _all_reduce_sum_(global_weighted_sum, tp_group)
        row_expected_scaled = global_weighted_sum / global_sum
        entropy = row_lse - row_expected_scaled

    return log_prob, entropy, row_lse, row_expected_scaled


class _FusedSelectedTPLogProbFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        tokens: torch.Tensor,
        inv_temperature: float,
        tp_group: dist.ProcessGroup | None,
        return_entropy: bool,
        need_entropy_grad: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if need_entropy_grad and not return_entropy:
            raise ValueError("need_entropy_grad=True requires return_entropy=True.")
        hidden_input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(weight.dtype)
        log_prob, entropy, row_lse, row_expected_scaled = _compute_forward_stats(
            hidden_states,
            weight,
            bias,
            tokens,
            tp_group,
            inv_temperature,
            return_entropy=return_entropy,
        )
        bias_tensor = bias if bias is not None else weight.new_empty((0,))
        ctx.save_for_backward(hidden_states, weight, bias_tensor, tokens, row_lse, row_expected_scaled)
        ctx.has_bias = bias is not None
        ctx.tp_group = tp_group
        ctx.inv_temperature = float(inv_temperature)
        ctx.hidden_input_dtype = hidden_input_dtype
        ctx.need_hidden_grad = ctx.needs_input_grad[0]
        ctx.need_weight_grad = ctx.needs_input_grad[1]
        ctx.need_bias_grad = bias is not None and ctx.needs_input_grad[2]
        ctx.need_entropy_grad = bool(need_entropy_grad)
        if not ctx.need_entropy_grad:
            ctx.mark_non_differentiable(entropy)
        return log_prob, entropy

    @staticmethod
    def backward(ctx, grad_log_prob: torch.Tensor, grad_entropy: torch.Tensor | None):
        hidden_states, weight, bias_tensor, tokens, row_lse, row_expected_scaled = ctx.saved_tensors
        bias = bias_tensor if ctx.has_bias else None
        need_hidden_grad = ctx.need_hidden_grad
        need_weight_grad = ctx.need_weight_grad
        need_bias_grad = ctx.need_bias_grad

        if grad_log_prob is not None:
            grad_log_prob = grad_log_prob.contiguous().to(torch.float32)
        if grad_entropy is not None:
            grad_entropy = grad_entropy.contiguous().to(torch.float32)

        num_rows = hidden_states.size(0)
        if num_rows == 0:
            grad_hidden = (
                hidden_states.new_zeros(hidden_states.shape, dtype=ctx.hidden_input_dtype) if need_hidden_grad else None
            )
            grad_weight = torch.zeros_like(weight) if need_weight_grad else None
            grad_bias = torch.zeros_like(bias) if need_bias_grad else None
            return grad_hidden, grad_weight, grad_bias, None, None, None, None, None

        if not (need_hidden_grad or need_weight_grad or need_bias_grad):
            return None, None, None, None, None, None, None, None
        if grad_log_prob is None and (grad_entropy is None or not ctx.need_entropy_grad):
            return None, None, None, None, None, None, None, None

        grad_hidden = (
            torch.zeros(hidden_states.shape, device=hidden_states.device, dtype=torch.float32) if need_hidden_grad else None
        )
        grad_weight = torch.empty_like(weight) if need_weight_grad else None
        grad_bias = torch.empty_like(bias) if need_bias_grad else None

        local_vocab_size = weight.size(0)
        tp_rank, _tp_world_size = _get_tp_rank_and_world_size(ctx.tp_group)
        vocab_start_index = tp_rank * local_vocab_size
        tokens = tokens.to(torch.int64)
        row_lse = row_lse.to(torch.float32)
        row_lse = row_lse.unsqueeze(1)
        row_expected_scaled = row_expected_scaled.to(torch.float32)
        row_expected_scaled = row_expected_scaled.unsqueeze(1)
        grad_log_prob_scale = None
        if grad_log_prob is not None:
            grad_log_prob_scale = grad_log_prob.unsqueeze(1) * ctx.inv_temperature
        grad_entropy_scale = None
        if ctx.need_entropy_grad and grad_entropy is not None:
            grad_entropy_scale = grad_entropy.unsqueeze(1) * ctx.inv_temperature
        hidden_states_fp32 = hidden_states.float() if need_weight_grad else None

        for tile_start in range(0, local_vocab_size, _BACKWARD_VOCAB_TILE):
            tile_end = min(tile_start + _BACKWARD_VOCAB_TILE, local_vocab_size)
            weight_tile = weight[tile_start:tile_end]
            bias_tile = bias[tile_start:tile_end] if bias is not None else None

            scaled_logits_tile = F.linear(hidden_states, weight_tile, bias_tile).float()
            if ctx.inv_temperature != 1.0:
                scaled_logits_tile.mul_(ctx.inv_temperature)
            probs_tile = torch.exp(scaled_logits_tile - row_lse)
            grad_logits_tile = torch.zeros_like(probs_tile)

            if grad_log_prob_scale is not None:
                grad_logits_tile.add_(probs_tile.mul(-grad_log_prob_scale))

            local_targets = tokens - vocab_start_index - tile_start
            target_mask = (local_targets >= 0) & (local_targets < (tile_end - tile_start))
            if grad_log_prob_scale is not None and target_mask.any():
                grad_logits_tile[target_mask, local_targets[target_mask]] += grad_log_prob_scale[target_mask, 0]

            if grad_entropy_scale is not None:
                grad_logits_tile.add_(
                    probs_tile * grad_entropy_scale * (row_expected_scaled - scaled_logits_tile)
                )

            if grad_hidden is not None:
                grad_hidden.add_(grad_logits_tile @ weight_tile.float())
            if grad_weight is not None:
                grad_weight_tile = grad_logits_tile.transpose(0, 1) @ hidden_states_fp32
                grad_weight[tile_start:tile_end] = grad_weight_tile.to(weight.dtype)
            if grad_bias is not None:
                grad_bias[tile_start:tile_end] = grad_logits_tile.sum(dim=0).to(bias.dtype)

        if grad_hidden is not None:
            _all_reduce_sum_(grad_hidden, ctx.tp_group)
            grad_hidden = grad_hidden.to(ctx.hidden_input_dtype)

        return grad_hidden, grad_weight, grad_bias, None, None, None, None, None


def fused_selected_tp_logprob(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    tokens: torch.Tensor,
    tp_group: dist.ProcessGroup | None,
    rollout_temperature: float,
    with_entropy: bool,
    need_entropy_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if rollout_temperature <= 0:
        raise ValueError(f"rollout_temperature must be > 0, got {rollout_temperature}")
    if need_entropy_grad and not with_entropy:
        raise ValueError("need_entropy_grad=True requires with_entropy=True.")
    inv_temperature = 1.0 / float(rollout_temperature)
    log_prob, entropy = _FusedSelectedTPLogProbFunction.apply(
        hidden_states,
        weight,
        bias,
        tokens,
        inv_temperature,
        tp_group,
        with_entropy,
        need_entropy_grad,
    )
    if with_entropy:
        return log_prob, entropy
    return log_prob, None
