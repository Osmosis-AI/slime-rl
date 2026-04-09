from __future__ import annotations

import argparse
import time

import torch

from slime.backends.megatron_utils.kernels.fused_experts import fused_experts_impl


def naive_routed_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    num_tokens, hidden_size = hidden_states.shape
    topk = topk_ids.shape[1]
    output = torch.zeros((num_tokens, hidden_size), device=hidden_states.device, dtype=hidden_states.dtype)

    for token_idx in range(num_tokens):
        token_hidden = hidden_states[token_idx]
        for slot in range(topk):
            expert_idx = int(topk_ids[token_idx, slot].item())
            routed_weight = topk_weights[token_idx, slot]
            gate_up = torch.matmul(w1[expert_idx], token_hidden)
            gate, up = gate_up.chunk(2, dim=0)
            activated = torch.nn.functional.silu(gate) * up
            down = torch.matmul(w2[expert_idx], activated)
            output[token_idx] += routed_weight * down
    return output


def make_inputs(args):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed)

    hidden_states = torch.randn(
        (args.num_tokens, args.hidden_size),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
        requires_grad=True,
    )
    w1 = torch.randn(
        (args.num_experts, args.ffn_hidden_size * 2, args.hidden_size),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
        requires_grad=True,
    )
    w2 = torch.randn(
        (args.num_experts, args.hidden_size, args.ffn_hidden_size),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
        requires_grad=True,
    )
    scores = torch.randn(
        (args.num_tokens, args.topk),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    topk_weights = torch.softmax(scores, dim=-1).to(torch.bfloat16)
    topk_ids = torch.randint(
        low=0,
        high=args.num_experts,
        size=(args.num_tokens, args.topk),
        device="cuda",
        generator=generator,
    )
    return hidden_states, w1, w2, topk_weights, topk_ids


def run_microbenchmark(args):
    hidden_states, w1, w2, topk_weights, topk_ids = make_inputs(args)

    def benchmark(label, fn):
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = fn(hidden_states, w1, w2, topk_weights, topk_ids)
        loss = output.float().square().mean()
        loss.backward()
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000
        peak_alloc_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"{label}: {elapsed_ms:.2f} ms, peak_alloc={peak_alloc_gb:.2f} GB")

    torch.cuda.reset_peak_memory_stats()
    benchmark("baseline", naive_routed_experts)

    hidden_states.grad = None
    w1.grad = None
    w2.grad = None
    torch.cuda.reset_peak_memory_stats()
    benchmark("fused", fused_experts_impl)


def run_layer_parity(args):
    hidden_states, w1, w2, topk_weights, topk_ids = make_inputs(args)

    baseline_out = naive_routed_experts(hidden_states, w1, w2, topk_weights, topk_ids)
    baseline_loss = baseline_out.float().square().mean()
    baseline_loss.backward()
    baseline_hidden_grad = hidden_states.grad.detach().clone()
    baseline_w1_grad = w1.grad.detach().clone()
    baseline_w2_grad = w2.grad.detach().clone()

    hidden_states.grad = None
    w1.grad = None
    w2.grad = None

    fused_out = fused_experts_impl(hidden_states, w1, w2, topk_weights, topk_ids)
    fused_loss = fused_out.float().square().mean()
    fused_loss.backward()

    print("forward_max_abs_diff", (baseline_out - fused_out).abs().max().item())
    print("hidden_grad_max_abs_diff", (baseline_hidden_grad - hidden_states.grad).abs().max().item())
    print("w1_grad_max_abs_diff", (baseline_w1_grad - w1.grad).abs().max().item())
    print("w2_grad_max_abs_diff", (baseline_w2_grad - w2.grad).abs().max().item())


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark or parity-check fused MoE kernels.")
    parser.add_argument("--mode", choices=["micro", "parity"], default="micro")
    parser.add_argument("--num-tokens", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--ffn-hidden-size", type=int, default=512)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")
    if args.mode == "micro":
        run_microbenchmark(args)
    else:
        run_layer_parity(args)
