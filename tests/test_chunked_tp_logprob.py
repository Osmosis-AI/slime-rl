from __future__ import annotations

from argparse import Namespace

import torch
import torch.nn.functional as F

from slime.backends.megatron_utils import chunked_tp_logprob
from slime.backends.megatron_utils import loss as megatron_loss
from slime.backends.megatron_utils.chunked_tp_logprob import (
    call_output_layer_linear,
    output_layer_uses_hidden_state_bypass,
    patch_output_layer_for_hidden_state_bypass,
)


class _FakeOutputLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(7, 5, dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.randn(7, dtype=torch.float32))
        self.forward_calls: list[tuple[bool, bool | None]] = []

    def forward(
        self,
        input_: torch.Tensor,
        weight: torch.Tensor | None = None,
        runtime_gather_output: bool | None = None,
    ) -> tuple[torch.Tensor, None]:
        self.forward_calls.append((weight is self.weight, runtime_gather_output))
        if weight is None:
            weight = self.weight
        return F.linear(input_, weight, self.bias), None


class _FakeSequenceParallelOutputLayer(_FakeOutputLayer):
    def __init__(self) -> None:
        super().__init__()
        self.sequence_parallel = True

    def forward(
        self,
        input_: torch.Tensor,
        weight: torch.Tensor | None = None,
        runtime_gather_output: bool | None = None,
    ) -> tuple[torch.Tensor, None]:
        output, _ = super().forward(input_, weight=weight, runtime_gather_output=runtime_gather_output)
        return torch.cat([output, output], dim=0), None


def test_hidden_state_bypass_preserves_original_output_layer_math():
    torch.manual_seed(0)
    layer = _FakeOutputLayer()
    hidden_states = torch.randn(13, 5, dtype=torch.float32)

    baseline, _ = layer(hidden_states)

    assert patch_output_layer_for_hidden_state_bypass(layer) is True
    assert patch_output_layer_for_hidden_state_bypass(layer) is False
    assert output_layer_uses_hidden_state_bypass(layer)

    passthrough, aux = layer(hidden_states)
    replay = call_output_layer_linear(layer, hidden_states)

    assert aux is None
    assert torch.equal(passthrough, hidden_states)
    assert torch.allclose(replay, baseline)
    assert layer.forward_calls[-1] == (True, False)


def _naive_calculate_log_probs_and_entropy(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    tp_group,
    with_entropy: bool = False,
    chunk_size: int = -1,
    need_entropy_grad: bool = False,
):
    del tp_group, chunk_size, need_entropy_grad
    if logits.numel() == 0:
        empty = logits.new_zeros((0,), dtype=torch.float32)
        return empty, (empty if with_entropy else None)

    log_probs = F.log_softmax(logits.float(), dim=-1)
    selected = log_probs.gather(dim=-1, index=tokens.view(-1, 1)).squeeze(-1)

    entropy = None
    if with_entropy:
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)

    return selected, entropy


def test_chunked_hidden_state_path_matches_full_logits_for_logprobs_and_grads(monkeypatch):
    monkeypatch.setattr(megatron_loss.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(megatron_loss.mpu, "get_tensor_model_parallel_group", lambda: None)
    monkeypatch.setattr(megatron_loss, "calculate_log_probs_and_entropy", _naive_calculate_log_probs_and_entropy)

    args = Namespace(
        qkv_format="bshd",
        allgather_cp=False,
        chunked_tp_logprob_seq_chunk_size=5,
        rollout_temperature=1.0,
        entropy_coef=0.0,
        log_probs_chunk_size=-1,
    )

    unconcat_tokens = [
        torch.tensor([1, 3, 2, 4, 5], dtype=torch.long),
        torch.tensor([2, 6, 1, 3, 4, 0, 2], dtype=torch.long),
    ]
    total_lengths = [5, 7]
    response_lengths = [2, 3]
    max_seq_lens = [8, 8]

    torch.manual_seed(7)
    baseline_hidden_states = torch.randn(2, 8, 5, dtype=torch.float32, requires_grad=True)
    chunked_hidden_states = baseline_hidden_states.detach().clone().requires_grad_(True)

    baseline_layer = _FakeOutputLayer()
    chunked_layer = _FakeOutputLayer()
    chunked_layer.load_state_dict(baseline_layer.state_dict())
    patch_output_layer_for_hidden_state_bypass(chunked_layer)

    baseline_logits = F.linear(baseline_hidden_states, baseline_layer.weight, baseline_layer.bias)
    _, baseline_res = megatron_loss.get_log_probs_and_entropy(
        baseline_logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=True,
        max_seq_lens=max_seq_lens,
    )

    _, chunked_res = megatron_loss.get_log_probs_and_entropy(
        chunked_hidden_states,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=True,
        max_seq_lens=max_seq_lens,
        output_layer=chunked_layer,
    )

    baseline_log_probs = torch.cat(baseline_res["log_probs"], dim=0)
    chunked_log_probs = torch.cat(chunked_res["log_probs"], dim=0)
    baseline_entropy = torch.cat(baseline_res["entropy"], dim=0)
    chunked_entropy = torch.cat(chunked_res["entropy"], dim=0)

    assert torch.allclose(chunked_log_probs, baseline_log_probs, atol=1e-6, rtol=1e-6)
    assert torch.allclose(chunked_entropy, baseline_entropy, atol=1e-6, rtol=1e-6)

    baseline_loss = -baseline_log_probs.mean()
    chunked_loss = -chunked_log_probs.mean()
    baseline_loss.backward()
    chunked_loss.backward()

    assert torch.allclose(chunked_hidden_states.grad, baseline_hidden_states.grad, atol=1e-6, rtol=1e-6)
    assert torch.allclose(chunked_layer.weight.grad, baseline_layer.weight.grad, atol=1e-6, rtol=1e-6)
    assert torch.allclose(chunked_layer.bias.grad, baseline_layer.bias.grad, atol=1e-6, rtol=1e-6)


def test_chunked_hidden_state_path_handles_sequence_parallel_replay(monkeypatch):
    monkeypatch.setattr(megatron_loss.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(megatron_loss.mpu, "get_tensor_model_parallel_group", lambda: None)
    monkeypatch.setattr(megatron_loss, "calculate_log_probs_and_entropy", _naive_calculate_log_probs_and_entropy)
    monkeypatch.setattr(
        chunked_tp_logprob.tensor_parallel,
        "gather_from_sequence_parallel_region",
        lambda x, tensor_parallel_output_grad=False: torch.cat([x, x], dim=0),
    )

    args = Namespace(
        qkv_format="bshd",
        allgather_cp=False,
        chunked_tp_logprob_seq_chunk_size=4,
        rollout_temperature=1.0,
        entropy_coef=0.0,
        log_probs_chunk_size=-1,
    )

    unconcat_tokens = [torch.tensor([0, 1, 2, 3, 4, 5, 6, 1], dtype=torch.long)]
    total_lengths = [8]
    response_lengths = [3]
    max_seq_lens = [8]

    torch.manual_seed(11)
    baseline_local_hidden = torch.randn(1, 4, 5, dtype=torch.float32, requires_grad=True)
    chunked_local_hidden = baseline_local_hidden.detach().clone().requires_grad_(True)

    baseline_layer = _FakeOutputLayer()
    chunked_layer = _FakeSequenceParallelOutputLayer()
    chunked_layer.load_state_dict(baseline_layer.state_dict())
    patch_output_layer_for_hidden_state_bypass(chunked_layer)

    baseline_hidden = torch.cat(
        [
            baseline_local_hidden,
            baseline_local_hidden,
        ],
        dim=1,
    )
    baseline_logits = F.linear(baseline_hidden, baseline_layer.weight, baseline_layer.bias)
    _, baseline_res = megatron_loss.get_log_probs_and_entropy(
        baseline_logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=True,
        max_seq_lens=max_seq_lens,
    )

    _, chunked_res = megatron_loss.get_log_probs_and_entropy(
        chunked_local_hidden,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=True,
        max_seq_lens=max_seq_lens,
        output_layer=chunked_layer,
    )

    baseline_log_probs = torch.cat(baseline_res["log_probs"], dim=0)
    chunked_log_probs = torch.cat(chunked_res["log_probs"], dim=0)
    baseline_entropy = torch.cat(baseline_res["entropy"], dim=0)
    chunked_entropy = torch.cat(chunked_res["entropy"], dim=0)

    assert torch.allclose(chunked_log_probs, baseline_log_probs, atol=1e-6, rtol=1e-6)
    assert torch.allclose(chunked_entropy, baseline_entropy, atol=1e-6, rtol=1e-6)

    baseline_loss = -baseline_log_probs.mean()
    chunked_loss = -chunked_log_probs.mean()
    baseline_loss.backward()
    chunked_loss.backward()

    assert torch.allclose(chunked_local_hidden.grad, baseline_local_hidden.grad, atol=1e-6, rtol=1e-6)
    assert torch.allclose(chunked_layer.weight.grad, baseline_layer.weight.grad, atol=1e-6, rtol=1e-6)
    assert torch.allclose(chunked_layer.bias.grad, baseline_layer.bias.grad, atol=1e-6, rtol=1e-6)

