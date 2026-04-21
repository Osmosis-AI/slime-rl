from __future__ import annotations

import inspect
import types
from argparse import Namespace
import torch
import torch.nn.functional as F
from megatron.core import tensor_parallel


_BYPASS_ENABLED_ATTR = "_slime_chunked_tp_logprob_enabled"
_ORIGINAL_FORWARD_ATTR = "_slime_chunked_tp_original_forward"
_HAS_RUNTIME_GATHER_ATTR = "_slime_chunked_tp_has_runtime_gather_output_arg"
_HAS_WEIGHT_ATTR = "_slime_chunked_tp_has_weight_arg"


def should_enable_chunked_tp_logprob(args: Namespace, role: str) -> bool:
    return role == "actor" and (
        getattr(args, "use_chunked_tp_logprob_loss", False) or getattr(args, "use_fused_selected_tp_logprob", False)
    )


def validate_chunked_tp_logprob_config(args: Namespace) -> None:
    if getattr(args, "use_fused_selected_tp_logprob", False) and not getattr(
        args, "use_chunked_tp_logprob_loss", False
    ):
        raise ValueError("--use-fused-selected-tp-logprob requires --use-chunked-tp-logprob-loss.")
    if getattr(args, "qkv_format", None) != "bshd":
        raise ValueError(
            "--use-chunked-tp-logprob-loss currently supports only --qkv-format bshd. "
            f"Got: {getattr(args, 'qkv_format', None)}"
        )
    if getattr(args, "context_parallel_size", 1) != 1:
        raise ValueError(
            "--use-chunked-tp-logprob-loss currently supports only --context-parallel-size 1. "
            f"Got: {getattr(args, 'context_parallel_size', None)}"
        )
    if getattr(args, "allgather_cp", False):
        raise ValueError("--use-chunked-tp-logprob-loss does not support --allgather-cp yet.")
    if getattr(args, "chunked_tp_logprob_seq_chunk_size", 0) <= 0:
        raise ValueError(
            "--use-chunked-tp-logprob-loss requires --chunked-tp-logprob-seq-chunk-size > 0. "
            f"Got: {getattr(args, 'chunked_tp_logprob_seq_chunk_size', None)}"
        )


def output_layer_uses_hidden_state_bypass(output_layer: torch.nn.Module | None) -> bool:
    return bool(output_layer is not None and getattr(output_layer, _BYPASS_ENABLED_ATTR, False))


def should_use_fused_selected_tp_logprob(
    args: Namespace,
    output_layer: torch.nn.Module | None,
    *,
    with_entropy: bool,
    need_entropy_grad: bool,
) -> bool:
    del with_entropy, need_entropy_grad
    return bool(
        getattr(args, "use_fused_selected_tp_logprob", False)
        and output_layer_uses_hidden_state_bypass(output_layer)
        and output_layer is not None
        and hasattr(output_layer, "weight")
        and output_layer.weight is not None
    )


def patch_output_layer_for_hidden_state_bypass(output_layer: torch.nn.Module) -> bool:
    if output_layer_uses_hidden_state_bypass(output_layer):
        return False

    original_forward = output_layer.forward
    signature = inspect.signature(original_forward)

    def _hidden_state_passthrough(self, input_: torch.Tensor, *args, **kwargs) -> tuple[torch.Tensor, None]:
        del args, kwargs
        return input_, None

    setattr(output_layer, _ORIGINAL_FORWARD_ATTR, original_forward)
    setattr(output_layer, _HAS_RUNTIME_GATHER_ATTR, "runtime_gather_output" in signature.parameters)
    setattr(output_layer, _HAS_WEIGHT_ATTR, "weight" in signature.parameters)
    setattr(output_layer, _BYPASS_ENABLED_ATTR, True)
    output_layer.forward = types.MethodType(_hidden_state_passthrough, output_layer)
    return True


def gather_hidden_states_for_output_layer(
    hidden_states: torch.Tensor,
    output_layer: torch.nn.Module,
) -> torch.Tensor:
    if not getattr(output_layer, "sequence_parallel", False):
        return hidden_states
    if hidden_states.dim() != 3:
        raise ValueError(
            "Chunked TP logprob expects 3D hidden states before sequence-parallel gather. "
            f"Got: {tuple(hidden_states.shape)}"
        )

    # Megatron sequence-parallel collectives operate on the leading dimension,
    # so transpose bshd -> sbhd, gather, then restore the original layout.
    hidden_states = hidden_states.transpose(0, 1).contiguous()
    hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
        hidden_states,
        tensor_parallel_output_grad=False,
    )
    return hidden_states.transpose(0, 1).contiguous()


def _load_fused_selected_tp_logprob_impl():
    from .kernels.selected_tp_logprob_triton import fused_selected_tp_logprob

    return fused_selected_tp_logprob


def compute_fused_selected_tp_logprob(
    hidden_states: torch.Tensor,
    tokens: torch.Tensor,
    *,
    output_layer: torch.nn.Module,
    tp_group,
    rollout_temperature: float,
    with_entropy: bool,
    need_entropy_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if not hasattr(output_layer, "weight") or output_layer.weight is None:
        raise ValueError("Fused selected TP logprob requires output_layer.weight.")

    hidden_states = hidden_states.to(output_layer.weight.dtype)
    bias = getattr(output_layer, "bias", None)
    fused_impl = _load_fused_selected_tp_logprob_impl()
    return fused_impl(
        hidden_states=hidden_states,
        weight=output_layer.weight,
        bias=bias,
        tokens=tokens,
        tp_group=tp_group,
        rollout_temperature=rollout_temperature,
        with_entropy=with_entropy,
        need_entropy_grad=need_entropy_grad,
    )


def call_output_layer_linear(output_layer: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    original_forward = getattr(output_layer, _ORIGINAL_FORWARD_ATTR, output_layer.forward)
    kwargs = {}
    if getattr(output_layer, _HAS_WEIGHT_ATTR, False) and hasattr(output_layer, "weight"):
        kwargs["weight"] = output_layer.weight
    if getattr(output_layer, _HAS_RUNTIME_GATHER_ATTR, False):
        kwargs["runtime_gather_output"] = False

    # Hidden states may be fp32 (from Float16Module unwrap / grad accumulation)
    # while output_layer weight is bf16. Cast to match.
    if hasattr(output_layer, "weight") and output_layer.weight is not None:
        hidden_states = hidden_states.to(output_layer.weight.dtype)

    # Replay only the local vocab projection here. The chunked path gathers
    # hidden states explicitly when sequence parallel is enabled, so letting the
    # output layer forward all-gather sequence again would double the row count
    # and break token alignment.
    if getattr(output_layer, "sequence_parallel", False):
        if not hasattr(output_layer, "weight") or output_layer.weight is None:
            raise ValueError("Sequence-parallel chunked TP logprob replay requires output_layer.weight.")
        output = F.linear(hidden_states, output_layer.weight, getattr(output_layer, "bias", None))
        return output.float()

    output = original_forward(hidden_states, **kwargs)
    if isinstance(output, tuple):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Expected output_layer to return a Tensor or tuple[Tensor, ...], got {type(output)!r}")
    return output
