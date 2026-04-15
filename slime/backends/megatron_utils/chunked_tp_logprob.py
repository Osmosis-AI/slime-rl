from __future__ import annotations

import inspect
import types
from argparse import Namespace
import torch


_BYPASS_ENABLED_ATTR = "_slime_chunked_tp_logprob_enabled"
_ORIGINAL_FORWARD_ATTR = "_slime_chunked_tp_original_forward"
_HAS_RUNTIME_GATHER_ATTR = "_slime_chunked_tp_has_runtime_gather_output_arg"
_HAS_WEIGHT_ATTR = "_slime_chunked_tp_has_weight_arg"


def should_enable_chunked_tp_logprob(args: Namespace, role: str) -> bool:
    return role == "actor" and getattr(args, "use_chunked_tp_logprob_loss", False)


def validate_chunked_tp_logprob_config(args: Namespace) -> None:
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

    output = original_forward(hidden_states, **kwargs)
    if isinstance(output, tuple):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Expected output_layer to return a Tensor or tuple[Tensor, ...], got {type(output)!r}")
    return output
