"""Tensor operations exposed by BioCUDA AutoTune."""

from __future__ import annotations

import time
from typing import Dict

import torch

from .autotune import BioCUDAAutoTune
from .extension import hamming_popc, smith_waterman_score

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _matmul_kernel(a, b, c, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, sa_m: tl.constexpr, sa_k: tl.constexpr, sb_k: tl.constexpr, sb_n: tl.constexpr, sc_m: tl.constexpr, sc_n: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, G: tl.constexpr):
        pid = tl.program_id(0)
        num_m = tl.cdiv(M, BM)
        num_n = tl.cdiv(N, BN)
        group_id = pid // (G * num_n)
        first_m = group_id * G
        group_m = min(num_m - first_m, G)
        pid_m = first_m + ((pid % (G * num_n)) % group_m)
        pid_n = (pid % (G * num_n)) // group_m
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        ap = a + rm[:, None] * sa_m + rk[None, :] * sa_k
        bp = b + rk[:, None] * sb_k + rn[None, :] * sb_n
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in range(0, K, BK):
            av = tl.load(ap, mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
            bv = tl.load(bp, mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
            acc += tl.dot(av, bv)
            ap += BK * sa_k
            bp += BK * sb_k
            rk += BK
        cp = c + rm[:, None] * sc_m + rn[None, :] * sc_n
        tl.store(cp, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def matmul(a: torch.Tensor, b: torch.Tensor, tuner: BioCUDAAutoTune | None = None, use_triton: bool = True) -> torch.Tensor:
    """Matrix multiply with BioCUDA zero-shot config selection.

    Falls back to ``torch.mm`` when Triton/CUDA is unavailable.
    """

    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("matmul expects shapes [M,K] and [K,N]")
    if not (use_triton and HAS_TRITON and a.is_cuda and b.is_cuda):
        return torch.mm(a, b)
    tuner = tuner or BioCUDAAutoTune(verbose=False)
    cfg = tuner.matmul_configs(a, b, top_k=1)[0]
    c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(a.shape[0], cfg.bm) * triton.cdiv(b.shape[1], cfg.bn),)
    _matmul_kernel[grid](
        a, b, c, a.shape[0], b.shape[1], a.shape[1],
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BM=cfg.bm, BN=cfg.bn, BK=cfg.bk, G=cfg.group,
        num_warps=cfg.warps, num_stages=cfg.stages,
    )
    return c


def benchmark_matmul(size: int = 1024, dtype: torch.dtype = torch.float16, iterations: int = 100) -> Dict[str, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_matmul requires CUDA")
    a = torch.randn((size, size), device="cuda", dtype=dtype)
    b = torch.randn((size, size), device="cuda", dtype=dtype)
    tuner = BioCUDAAutoTune(verbose=False)
    for _ in range(10):
        torch.mm(a, b)
        matmul(a, b, tuner=tuner)
    torch.cuda.synchronize()

    def bench(fn):
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1e6 / iterations

    torch_us = bench(lambda: torch.mm(a, b))
    bio_us = bench(lambda: matmul(a, b, tuner=tuner))
    return {"torch_mm_us": torch_us, "biocuda_us": bio_us, "speedup": torch_us / bio_us}


def hamming_distance_words(a: torch.Tensor, b: torch.Tensor, tuner: BioCUDAAutoTune | None = None) -> torch.Tensor:
    tuner = tuner or BioCUDAAutoTune(verbose=False)
    best = tuner.optimizer.optimize_hamming(int(a.numel()))
    return hamming_popc(a, b, block_size=int(best["block"]))


def smith_waterman(a: torch.Tensor, b: torch.Tensor, tuner: BioCUDAAutoTune | None = None, match: int = 2, mismatch: int = -1, gap_open: int = 2, gap_extend: int = 1) -> torch.Tensor:
    tuner = tuner or BioCUDAAutoTune(verbose=False)
    best = tuner.optimizer.optimize_sw(int(a.numel()), int(b.numel()))
    return smith_waterman_score(
        a,
        b,
        match=match,
        mismatch=mismatch,
        gap_open=gap_open,
        gap_extend=gap_extend,
        block_size=int(best["block"]),
    )
