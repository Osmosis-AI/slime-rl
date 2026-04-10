#!/bin/bash
# Kernel microbenchmark: V0 (atomic) vs V1 (expert-parallel) backward_weight
#
# Sweeps across Qwen3.5-35B and 122B shapes at multiple token counts.
# Also runs parity checks for both variants.
#
# Usage:
#   bash scripts/benchmark/bench_fused_moe_kernels.sh
#   bash scripts/benchmark/bench_fused_moe_kernels.sh 2>&1 | tee kernel_benchmark.log
set -ex

VARIANTS=(v0 v1)
TOKEN_COUNTS=(512 1024 2048 4096 8192)

echo "========================================"
echo "  Parity Checks"
echo "========================================"

for variant in "${VARIANTS[@]}"; do
    echo "--- ${variant}: 35B shapes ---"
    python3 tools/benchmark_fused_moe.py --mode parity --variant "$variant" \
        --num-tokens 256 --hidden-size 2048 --ffn-hidden-size 512 \
        --num-experts 256 --topk 8

    echo "--- ${variant}: 122B shapes ---"
    python3 tools/benchmark_fused_moe.py --mode parity --variant "$variant" \
        --num-tokens 256 --hidden-size 3072 --ffn-hidden-size 1024 \
        --num-experts 256 --topk 8
done

echo ""
echo "========================================"
echo "  Microbenchmark: Qwen3.5-35B-A3B"
echo "  H=2048, FFN=512, E=256, topk=8"
echo "========================================"

for variant in "${VARIANTS[@]}"; do
    for T in "${TOKEN_COUNTS[@]}"; do
        echo "--- ${variant} T=${T} ---"
        python3 tools/benchmark_fused_moe.py --mode micro --variant "$variant" \
            --num-tokens "$T" --hidden-size 2048 --ffn-hidden-size 512 \
            --num-experts 256 --topk 8
    done
done

echo ""
echo "========================================"
echo "  Microbenchmark: Qwen3.5-122B-A10B"
echo "  H=3072, FFN=1024, E=256, topk=8"
echo "========================================"

for variant in "${VARIANTS[@]}"; do
    for T in "${TOKEN_COUNTS[@]}"; do
        echo "--- ${variant} T=${T} ---"
        python3 tools/benchmark_fused_moe.py --mode micro --variant "$variant" \
            --num-tokens "$T" --hidden-size 3072 --ffn-hidden-size 1024 \
            --num-experts 256 --topk 8
    done
done

echo ""
echo "========================================"
echo "  Done"
echo "========================================"
