from __future__ import annotations

import logging
import re
import types
from collections.abc import Iterable, Sequence

import torch

logger = logging.getLogger(__name__)

_WEIGHT_SUFFIX_RE = re.compile(r"^weight(\d+)$")


def _load_fused_experts_impl():
    from .fused_experts import fused_experts_impl

    return fused_experts_impl


def _extract_topk_output(topk_output) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(topk_output, "topk_weights") and hasattr(topk_output, "topk_ids"):
        return topk_output.topk_weights, topk_output.topk_ids
    if isinstance(topk_output, tuple) and len(topk_output) >= 2:
        return topk_output[0], topk_output[1]
    raise TypeError("Unsupported top-k output type for fused MoE integration.")


def _stack_grouped_expert_weights(linear_module: torch.nn.Module) -> torch.Tensor:
    expert_weights: list[tuple[int, torch.Tensor]] = []

    for attr_name in dir(linear_module):
        match = _WEIGHT_SUFFIX_RE.match(attr_name)
        if match is None:
            continue
        weight = getattr(linear_module, attr_name)
        if isinstance(weight, torch.nn.Parameter):
            expert_weights.append((int(match.group(1)), weight))

    if not expert_weights:
        raise ValueError(f"No grouped expert weights found on {type(linear_module).__name__}.")

    expert_weights.sort(key=lambda item: item[0])
    expected = list(range(len(expert_weights)))
    actual = [index for index, _ in expert_weights]
    if actual != expected:
        raise ValueError(f"Expected contiguous expert weights {expected}, found {actual}.")

    return torch.stack([weight for _, weight in expert_weights], dim=0).contiguous()


def _has_expert_lora_params(module: torch.nn.Module) -> bool:
    for name, _ in module.named_parameters():
        if "lora_" in name or ".adapter." in name:
            return True
    return False


def _reshape_hidden_states(hidden_states: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    if hidden_states.ndim < 2:
        raise ValueError(f"Expected hidden states with rank >= 2, got {hidden_states.shape}.")
    original_shape = tuple(hidden_states.shape)
    return hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous(), original_shape


def _restore_hidden_states(hidden_states: torch.Tensor, original_shape: tuple[int, ...]) -> torch.Tensor:
    return hidden_states.reshape(*original_shape)


def patch_grouped_experts_module(module: torch.nn.Module) -> bool:
    if getattr(module, "_slime_fused_moe_patched", False):
        return False

    linear_fc1 = getattr(module, "linear_fc1", None)
    linear_fc2 = getattr(module, "linear_fc2", None)
    if linear_fc1 is None or linear_fc2 is None:
        return False

    try:
        _stack_grouped_expert_weights(linear_fc1)
        _stack_grouped_expert_weights(linear_fc2)
    except ValueError:
        return False

    if _has_expert_lora_params(module):
        raise RuntimeError(
            "Fused MoE backward does not yet support LoRA adapters on grouped expert weights."
        )

    original_forward = module.forward

    def fused_forward(self, hidden_states, topk_output, *args, **kwargs):
        del args, kwargs

        routed_hidden_states, original_shape = _reshape_hidden_states(hidden_states)
        topk_weights, topk_ids = _extract_topk_output(topk_output)

        topk_weights = topk_weights.reshape(routed_hidden_states.shape[0], -1).contiguous()
        topk_ids = topk_ids.reshape(routed_hidden_states.shape[0], -1).to(dtype=torch.int64).contiguous()

        if (topk_ids < 0).any():
            raise RuntimeError("Fused MoE backward only supports local expert ids in the EP=1 path.")

        fused_experts_impl = _load_fused_experts_impl()
        w1 = _stack_grouped_expert_weights(self.linear_fc1)
        w2 = _stack_grouped_expert_weights(self.linear_fc2)
        output = fused_experts_impl(
            routed_hidden_states.to(dtype=torch.bfloat16),
            w1,
            w2,
            topk_weights.to(dtype=routed_hidden_states.dtype),
            topk_ids,
        )
        output = output.to(dtype=hidden_states.dtype)
        return _restore_hidden_states(output, original_shape)

    module._slime_fused_moe_original_forward = original_forward
    module.forward = types.MethodType(fused_forward, module)
    module._slime_fused_moe_patched = True
    logger.info("Patched grouped expert module %s with fused MoE forward/backward.", type(module).__name__)
    return True


def patch_model_for_fused_moe(model_chunks: Sequence[torch.nn.Module] | torch.nn.Module) -> int:
    if not isinstance(model_chunks, Iterable) or isinstance(model_chunks, torch.nn.Module):
        model_chunks = [model_chunks]

    patched = 0
    for model_chunk in model_chunks:
        for module in model_chunk.modules():
            patched += int(patch_grouped_experts_module(module))
    return patched
