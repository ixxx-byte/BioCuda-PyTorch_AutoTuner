"""Calibration helpers restored from the BioCUDA notebook.

This module is import-safe.  CuPy/NVRTC is imported only inside functions.
The production package does not require CuPy, but the notebook used it for
reference compilation and tensor-core latency calibration.
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, Iterable

from .core import GPU_DB
from .kernel_specs import KERNEL_SPECS


class KernelUnavailable(RuntimeError):
    """Raised when an optional CuPy/NVRTC calibration kernel cannot run."""


TC_CALIBRATION_SOURCES = {
    "volta": "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32",
    "turing": "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32",
    "ampere": "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32",
    "ada": "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32",
    "hopper": "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32",
}


def tensor_core_flops_per_mma(gpu_key: str) -> int:
    m, n, k = KERNEL_SPECS[gpu_key].mma_shape
    return 2 * m * n * k


def estimate_tc_peak_from_tau(gpu_key: str, tau_cycles_per_mma: float, sm_clock_hz: float, sm_count: int | None = None) -> Dict[str, Any]:
    """Estimate FP16 tensor-core peak using the notebook's corrected v39.3 logic."""

    gpu = GPU_DB[gpu_key]
    spec = KERNEL_SPECS[gpu_key]
    sm_count = gpu.n_sm if sm_count is None else int(sm_count)
    flops_per_mma = tensor_core_flops_per_mma(gpu_key)
    if tau_cycles_per_mma <= 0:
        tflops_est = float("nan")
    else:
        mma_per_sec_per_warp = sm_clock_hz / tau_cycles_per_mma
        tflops_uncapped = mma_per_sec_per_warp * gpu.w_max * sm_count * flops_per_mma / 1e12
        # Hardware cap: 512 FMA/cycle/SM = 1024 FLOP/cycle/SM.
        tflops_hw_cap = 512.0 * 2.0 * sm_count * sm_clock_hz / 1e12
        tflops_est = min(tflops_uncapped, tflops_hw_cap)
    return {
        "gpu_key": gpu_key,
        "sm_arch": spec.sm_arch,
        "mma_shape": spec.mma_shape,
        "tau_tc_cycles_per_mma": float(tau_cycles_per_mma),
        "flops_per_mma": int(flops_per_mma),
        "sm_clock_hz": float(sm_clock_hz),
        "sm_count": int(sm_count),
        "w_max_used": int(gpu.w_max),
        "tflops_estimated_peak": float(tflops_est),
    }


def summarize_cycle_samples(gpu_key: str, samples: Iterable[int], iters: int, sm_clock_hz: float, sm_count: int | None = None) -> Dict[str, Any]:
    samples = sorted(int(s) for s in samples)
    if not samples:
        raise ValueError("samples must not be empty")
    cycles_median = int(statistics.median(samples))
    tau = cycles_median / max(int(iters), 1)
    result = estimate_tc_peak_from_tau(gpu_key, tau, sm_clock_hz, sm_count=sm_count)
    result.update({
        "iters": int(iters),
        "cycles_samples": samples,
        "cycles_median": cycles_median,
    })
    return result


def compile_for_cupy(gpu_key: str, src: str, fn_name: str, extra_opts=()):
    """Notebook-compatible CuPy RawKernel compiler for reference experiments."""

    try:
        import cupy as cp
    except Exception as exc:
        raise KernelUnavailable("cupy not importable") from exc
    if cp.cuda.runtime.getDeviceCount() == 0:
        raise KernelUnavailable("no CUDA device visible")
    spec = KERNEL_SPECS[gpu_key]
    opts = tuple(spec.build_opts) + tuple(extra_opts)
    try:
        return cp.RawKernel(src, fn_name, options=opts, backend="nvrtc")
    except cp.cuda.compiler.CompileException as exc:
        raise KernelUnavailable(f"NVRTC failed for {fn_name} on {gpu_key}/{spec.sm_arch}: {exc}") from exc
