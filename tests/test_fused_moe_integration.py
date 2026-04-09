from __future__ import annotations

import types

import pytest
import torch

from slime.backends.megatron_utils.kernels import fused_moe_integration


class _FakeGroupedLinear(torch.nn.Module):
    def __init__(self, weights: list[torch.Tensor]):
        super().__init__()
        for index, weight in enumerate(weights):
            setattr(self, f"weight{index}", torch.nn.Parameter(weight.clone()))


class _FakeGroupedExperts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_fc1 = _FakeGroupedLinear(
            [
                torch.full((4, 3), 1.0, dtype=torch.bfloat16),
                torch.full((4, 3), 2.0, dtype=torch.bfloat16),
            ]
        )
        self.linear_fc2 = _FakeGroupedLinear(
            [
                torch.full((3, 2), 3.0, dtype=torch.bfloat16),
                torch.full((3, 2), 4.0, dtype=torch.bfloat16),
            ]
        )

    def forward(self, hidden_states, topk_output):
        del topk_output
        return hidden_states + 100


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = _FakeGroupedExperts()


def test_patch_model_for_fused_moe_monkeypatches_grouped_experts(monkeypatch):
    recorded = {}

    def fake_fused_impl(hidden_states, w1, w2, topk_weights, topk_ids):
        recorded["hidden_states_shape"] = tuple(hidden_states.shape)
        recorded["w1_shape"] = tuple(w1.shape)
        recorded["w2_shape"] = tuple(w2.shape)
        recorded["topk_weights"] = topk_weights.clone()
        recorded["topk_ids"] = topk_ids.clone()
        return torch.zeros_like(hidden_states) + 7

    monkeypatch.setattr(fused_moe_integration, "_load_fused_experts_impl", lambda: fake_fused_impl)

    model = _FakeModel()
    patched = fused_moe_integration.patch_model_for_fused_moe([model])

    assert patched == 1

    hidden_states = torch.arange(12, dtype=torch.bfloat16).reshape(2, 2, 3)
    topk_output = types.SimpleNamespace(
        topk_weights=torch.tensor(
            [
                [0.7, 0.3],
                [0.4, 0.6],
                [0.8, 0.2],
                [0.1, 0.9],
            ],
            dtype=torch.bfloat16,
        ),
        topk_ids=torch.tensor(
            [
                [0, 1],
                [1, 0],
                [0, 1],
                [1, 0],
            ],
            dtype=torch.int64,
        ),
    )

    output = model.experts(hidden_states, topk_output)

    assert tuple(output.shape) == (2, 2, 3)
    assert torch.all(output == 7)
    assert recorded["hidden_states_shape"] == (4, 3)
    assert recorded["w1_shape"] == (2, 4, 3)
    assert recorded["w2_shape"] == (2, 3, 2)


def test_patch_grouped_experts_module_rejects_remote_expert_ids(monkeypatch):
    monkeypatch.setattr(
        fused_moe_integration,
        "_load_fused_experts_impl",
        lambda: lambda hidden_states, w1, w2, topk_weights, topk_ids: hidden_states,
    )

    module = _FakeGroupedExperts()
    assert fused_moe_integration.patch_grouped_experts_module(module)

    with pytest.raises(RuntimeError, match="local expert ids"):
        module(
            torch.zeros((2, 3), dtype=torch.bfloat16),
            types.SimpleNamespace(
                topk_weights=torch.ones((2, 2), dtype=torch.bfloat16),
                topk_ids=torch.tensor([[0, -1], [1, 0]], dtype=torch.int64),
            ),
        )


def test_patch_grouped_experts_module_ignores_non_grouped_modules():
    class _DenseOnly(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_fc1 = torch.nn.Linear(3, 4, bias=False)
            self.linear_fc2 = torch.nn.Linear(2, 3, bias=False)

    assert fused_moe_integration.patch_grouped_experts_module(_DenseOnly()) is False
