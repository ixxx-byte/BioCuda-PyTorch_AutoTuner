"""Notebook-parity per-GPU kernel metadata.

The original BioCUDA v39.12 notebook carried ten per-GPU NVRTC modules.  The
packaged PyTorch extension uses portable C++/CUDA sources for production calls,
but this table preserves the notebook's hardware-specific kernel metadata for
calibration, reporting, and future CuPy/NVRTC reference backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class KernelSpec:
    sm_arch: str
    cc: Tuple[int, int]
    tc_gen: str
    mma_shape: Tuple[int, int, int]
    has_dpx: bool
    build_opts: Tuple[str, ...] = ("--use_fast_math", "-lineinfo")
    sw_variant: str = "base"
    tc_variant: str = "ampere"


KERNEL_SPECS: Dict[str, KernelSpec] = {
    "V100": KernelSpec("sm_70", (7, 0), "volta", (8, 8, 4), False, tc_variant="volta"),
    "T4": KernelSpec("sm_75", (7, 5), "turing", (16, 8, 8), False, tc_variant="turing"),
    "A100": KernelSpec("sm_80", (8, 0), "ampere", (16, 8, 16), False),
    "A10": KernelSpec("sm_86", (8, 6), "ampere", (16, 8, 16), False),
    "RTX3090": KernelSpec("sm_86", (8, 6), "ampere", (16, 8, 16), False),
    "L4": KernelSpec("sm_89", (8, 9), "ada", (16, 8, 16), False),
    "L40": KernelSpec("sm_89", (8, 9), "ada", (16, 8, 16), False),
    "RTX4090": KernelSpec("sm_89", (8, 9), "ada", (16, 8, 16), False),
    "H100_SXM5": KernelSpec("sm_90a", (9, 0), "hopper", (16, 8, 16), True, sw_variant="h100_dpx"),
    "H100_PCIE": KernelSpec("sm_90a", (9, 0), "hopper", (16, 8, 16), True, sw_variant="h100_dpx"),
}


def kernel_modules_count() -> int:
    return len(KERNEL_SPECS)


def dry_run_table() -> list[dict]:
    return [
        {
            "key": key,
            "sm_arch": spec.sm_arch,
            "tc_gen": spec.tc_gen,
            "mma_shape": spec.mma_shape,
            "has_dpx": spec.has_dpx,
            "build_opts": spec.build_opts,
            "sw_variant": spec.sw_variant,
            "tc_variant": spec.tc_variant,
        }
        for key, spec in KERNEL_SPECS.items()
    ]
